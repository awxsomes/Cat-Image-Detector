"""Shared helpers: MediaPipe setup and feature extraction."""

import os
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

import config


def make_landmarker(running_mode=vision.RunningMode.VIDEO):
    """Build a MediaPipe FaceLandmarker that outputs blendshape scores."""
    if not os.path.exists(config.LANDMARKER_TASK):
        raise FileNotFoundError(
            f"Missing {config.LANDMARKER_TASK}. Download it with:\n"
            "  curl -o face_landmarker.task https://storage.googleapis.com/"
            "mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
        )
    options = vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=config.LANDMARKER_TASK),
        running_mode=running_mode,
        num_faces=1,
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=False,
    )
    return vision.FaceLandmarker.create_from_options(options)


def bgr_to_mp_image(frame_bgr):
    rgb = np.ascontiguousarray(frame_bgr[:, :, ::-1])
    return mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)


def extract_features(result):
    """Return a (52,) float32 vector of blendshape scores, or None if no face.

    These scores are things like jawOpen, browInnerUp, mouthSmileLeft --
    already a compact description of the expression, and invariant to
    lighting, background, distance and (mostly) head pose. Training on these
    instead of raw pixels is why this needs hundreds of samples and not tens
    of thousands.
    """
    if not result.face_blendshapes:
        return None
    scores = [c.score for c in result.face_blendshapes[0]]
    if len(scores) != config.N_BLENDSHAPES:
        return None
    return np.asarray(scores, dtype=np.float32)


def blendshape_names(result):
    if not result.face_blendshapes:
        return []
    return [c.category_name for c in result.face_blendshapes[0]]