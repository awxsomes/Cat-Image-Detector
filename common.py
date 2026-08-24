"""MediaPipe setup and feature extraction (face blendshapes + hand landmarks)."""

import os
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

import config

HAND_FEATURE_DIM = 2 + 21 * 3


def feature_dim(use_hands):
    return config.N_BLENDSHAPES + (HAND_FEATURE_DIM if use_hands else 0)


def _require(path, url):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing {path}. Download it with:\n  curl -o {path} {url}")


def _make_face():
    _require(config.LANDMARKER_TASK,
             "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
             "face_landmarker/float16/1/face_landmarker.task")
    return vision.FaceLandmarker.create_from_options(
        vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=config.LANDMARKER_TASK),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=False,
        ))


def _make_hand():
    _require(config.HAND_LANDMARKER_TASK,
             "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
             "hand_landmarker/float16/1/hand_landmarker.task")
    return vision.HandLandmarker.create_from_options(
        vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=config.HAND_LANDMARKER_TASK),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=1,
        ))


def bgr_to_mp_image(frame_bgr):
    rgb = np.ascontiguousarray(frame_bgr[:, :, ::-1])
    return mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)


def _face_vec(result):
    """(52,) blendshape scores, or None if no face.

    These are things like jawOpen, cheekPuff, tongueOut -- already a compact
    description of the expression, and invariant to lighting and background.
    """
    if not result.face_blendshapes:
        return None
    scores = [c.score for c in result.face_blendshapes[0]]
    if len(scores) != config.N_BLENDSHAPES:
        return None
    return np.asarray(scores, dtype=np.float32)


def _hand_vec(result, aspect):
    """(65,) canonicalised hand pose, plus a bool for whether a hand was found.

    Raw landmarks are normalised image coordinates, so they encode *where* your
    hand is in frame -- useless as a gesture descriptor. Three fixes:

    1. x is rescaled by the frame aspect ratio, because MediaPipe normalises x
       and y independently and a 16:9 frame otherwise squashes the hand.
    2. Left hands are mirrored onto the right-hand form, so a thumbs up reads
       the same either-handed and you don't have to train both.
    3. The wrist is moved to the origin and everything is divided by the
       wrist-to-middle-knuckle distance, giving translation and scale
       invariance -- near or far from the camera looks the same.

    Rotation is deliberately NOT normalised: thumbs up and thumbs down are the
    same shape at different angles, so the angle has to survive.
    """
    if not result.hand_landmarks:
        return np.zeros(HAND_FEATURE_DIM, dtype=np.float32), False

    lm = result.hand_landmarks[0]
    pts = np.array([[p.x, p.y, p.z] for p in lm], dtype=np.float32)   # (21, 3)

    is_left = 0.0
    if result.handedness and result.handedness[0]:
        is_left = 1.0 if result.handedness[0][0].category_name == "Left" else 0.0

    pts[:, 0] *= aspect
    pts[:, 2] *= aspect
    if is_left:
        pts[:, 0] = -pts[:, 0]
    pts -= pts[0]
    scale = float(np.linalg.norm(pts[9])) + 1e-6     # wrist -> middle finger MCP
    pts /= scale

    return np.concatenate([[1.0, is_left], pts.ravel()]).astype(np.float32), True


class FeatureExtractor:
    """Runs both landmarkers on a frame and returns one concatenated vector."""

    def __init__(self, use_hands=None):
        self.use_hands = config.USE_HANDS if use_hands is None else use_hands
        self.face = _make_face()
        self.hand = _make_hand() if self.use_hands else None
        self.dim = feature_dim(self.use_hands)

    def process(self, frame_bgr, ts_ms):
        """-> (features (dim,) or None if no face, hand_present bool)"""
        image = bgr_to_mp_image(frame_bgr)
        face = _face_vec(self.face.detect_for_video(image, ts_ms))
        if face is None:
            return None, False
        if not self.use_hands:
            return face, False
        h, w = frame_bgr.shape[:2]
        hand, present = _hand_vec(self.hand.detect_for_video(image, ts_ms), w / h)
        return np.concatenate([face, hand]), present

    def close(self):
        self.face.close()
        if self.hand is not None:
            self.hand.close()