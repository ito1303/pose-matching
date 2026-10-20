"""MediaPipe Tasks PoseLandmarker (Python 3.14 / mediapipe 1.x)."""

from __future__ import annotations

import urllib.request
from pathlib import Path

import mediapipe as mp
import numpy as np
from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.tasks.python.vision import PoseLandmarker, PoseLandmarkerOptions, RunningMode

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
)
MODEL_PATH = Path(__file__).resolve().parent / "models" / "pose_landmarker_lite.task"


def ensure_model() -> Path:
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    if MODEL_PATH.exists() and MODEL_PATH.stat().st_size > 100_000:
        return MODEL_PATH
    print("Downloading Pose Landmarker model (~5MB)...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    return MODEL_PATH


def create_landmarker() -> PoseLandmarker:
    ensure_model()
    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return PoseLandmarker.create_from_options(options)


def detect_pose(landmarker: PoseLandmarker, rgb: np.ndarray, timestamp_ms: int):
    rgb = np.ascontiguousarray(rgb)
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    return landmarker.detect_for_video(image, timestamp_ms)
