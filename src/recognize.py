# src/recognize.py
"""
Multi-face recognition (CPU-friendly) using your now-stable pipeline:
Haar (multi-face) -> FaceMesh 5pt (per-face ROI) -> align_face_5pt (112x112)
-> ArcFace ONNX embedding -> cosine distance to DB -> label each face.
"""

from __future__ import annotations
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import onnxruntime as ort

try:
    import mediapipe as mp
except Exception as e:
    mp = None
    _MP_IMPORT_ERROR = e

from .haar_5pt import align_face_5pt


# -------------------------
# Data
# -------------------------
@dataclass
class FaceDet:
    x1: int
    y1: int
    x2: int
    y2: int
    score: float
    kps: np.ndarray  # (5,2)


@dataclass
class MatchResult:
    name: Optional[str]
    distance: float
    similarity: float
    accepted: bool


# -------------------------
# Math helpers
# -------------------------
def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a.reshape(-1), b.reshape(-1)))


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    return 1.0 - cosine_similarity(a, b)


def _clip_xyxy(
    x1: float, y1: float, x2: float, y2: float, W: int, H: int
) -> Tuple[int, int, int, int]:
    x1 = int(max(0, min(W - 1, round(x1))))
    y1 = int(max(0, min(H - 1, round(y1))))
    x2 = int(max(0, min(W - 1, round(x2))))
    y2 = int(max(0, min(H - 1, round(y2))))

    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1

    return x1, y1, x2, y2


def _bbox_from_5pt(
    kps: np.ndarray,
    pad_x: float = 0.55,
    pad_y_top: float = 0.85,
    pad_y_bot: float = 1.15,
) -> np.ndarray:
    k = kps.astype(np.float32)
    x_min, y_min = k.min(axis=0)
    x_max, y_max = k.max(axis=0)

    w = max(1.0, x_max - x_min)
    h = max(1.0, y_max - y_min)

    return np.array(
        [
            x_min - pad_x * w,
            y_min - pad_y_top * h,
            x_max + pad_x * w,
            y_max + pad_y_bot * h,
        ],
        dtype=np.float32,
    )


def _kps_span_ok(kps: np.ndarray, min_eye_dist: float) -> bool:
    le, re, no, lm, rm = kps
    if np.linalg.norm(re - le) < min_eye_dist:
        return False
    if not (lm[1] > no[1] and rm[1] > no[1]):
        return False
    return True


# -------------------------
# DB helpers
# -------------------------
def load_db_npz(db_path: Path) -> Dict[str, np.ndarray]:
    if not db_path.exists():
        return {}

    data = np.load(str(db_path), allow_pickle=True)
    return {k: data[k].astype(np.float32).reshape(-1) for k in data.files}


# -------------------------
# Embedder
# -------------------------
class ArcFaceEmbedderONNX:
    def __init__(
        self,
        model_path: str = "models/embedder_arcface.onnx",
        input_size: Tuple[int, int] = (112, 112),
        debug: bool = False,
    ):
        self.in_w, self.in_h = input_size
        self.sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        self.in_name = self.sess.get_inputs()[0].name
        self.out_name = self.sess.get_outputs()[0].name
        self.debug = debug

    def _preprocess(self, img: np.ndarray) -> np.ndarray:
        img = cv2.resize(img, (self.in_w, self.in_h))
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
        rgb = (rgb - 127.5) / 128.0
        return np.transpose(rgb, (2, 0, 1))[None, ...]

    def embed(self, img: np.ndarray) -> np.ndarray:
        x = self._preprocess(img)
        y = self.sess.run([self.out_name], {self.in_name: x})[0]
        v = y.reshape(-1).astype(np.float32)
        return v / (np.linalg.norm(v) + 1e-12)


# -------------------------
# Matcher
# -------------------------
class FaceDBMatcher:
    def __init__(self, db: Dict[str, np.ndarray], dist_thresh: float = 0.34):
        self.db = db
        self.dist_thresh = dist_thresh
        self._names = sorted(db.keys())
        self._mat = (
            np.stack([db[n] for n in self._names], axis=0)
            if self._names
            else None
        )

    def reload_from(self, path: Path):
        self.db = load_db_npz(path)
        self._names = sorted(self.db.keys())
        self._mat = (
            np.stack([self.db[n] for n in self._names], axis=0)
            if self._names
            else None
        )

    def match(self, emb: np.ndarray) -> MatchResult:
        if self._mat is None:
            return MatchResult(None, 1.0, 0.0, False)

        sims = self._mat @ emb.reshape(-1, 1)
        best_i = int(np.argmax(sims))
        best_sim = float(sims[best_i])
        best_dist = 1.0 - best_sim

        return MatchResult(
            name=self._names[best_i] if best_dist <= self.dist_thresh else None,
            distance=best_dist,
            similarity=best_sim,
            accepted=best_dist <= self.dist_thresh,
        )


# -------------------------
# Demo
# -------------------------
def main():
    db_path = Path("data/db/face_db.npz")
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        raise RuntimeError("Camera not available")

    embedder = ArcFaceEmbedderONNX()
    matcher = FaceDBMatcher(load_db_npz(db_path))

    print("Recognize running — q to quit")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        cv2.imshow("recognize", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
