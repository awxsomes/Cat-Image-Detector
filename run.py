import os
import time
from collections import deque

import cv2
import numpy as np
import torch
from PIL import Image, ImageSequence

import config
from common import FeatureExtractor
from train import MLP

EXTS = ("gif", "webp", "png", "jpg", "jpeg", "bmp")


class Animation:

    def __init__(self, frames, durations_ms):
        self.frames = frames
        self.cum = np.cumsum(durations_ms)
        self.total = float(self.cum[-1])
        self.animated = len(frames) > 1

    def frame_at(self, elapsed_ms):
        if not self.animated:
            return self.frames[0]
        t = elapsed_ms % self.total
        return self.frames[int(np.searchsorted(self.cum, t, side="right"))]


def load_asset(path, max_side=640):
    im = Image.open(path)
    bg_rgb = config.TRANSPARENT_BACKGROUND
    bg_rgba = (bg_rgb[0], bg_rgb[1], bg_rgb[2], 255)

    frames, durations = [], []
    for f in ImageSequence.Iterator(im):
        rgba = f.convert("RGBA")
        canvas = Image.new("RGBA", rgba.size, bg_rgba)
        canvas.alpha_composite(rgba)
        arr = np.asarray(canvas.convert("RGB"))[:, :, ::-1]   # RGB -> BGR
        frames.append(fit(np.ascontiguousarray(arr), max_side))
        # some encoders write 0; browsers clamp these, so we do too
        durations.append(max(int(f.info.get("duration", 100)), 20))

    if not frames:
        raise ValueError(f"no frames decoded from {path}")
    return Animation(frames, durations)


def load_images():
    """Map label -> Animation. Requires images/default.*"""
    assets = {}
    for name in config.EXPRESSIONS + ["default"]:
        for ext in EXTS:
            path = os.path.join(config.IMAGE_DIR, f"{name}.{ext}")
            if os.path.exists(path):
                try:
                    assets[name] = load_asset(path)
                    n = len(assets[name].frames)
                    print(f"  {name:<12} {os.path.basename(path)} "
                          f"({n} frame{'s' if n > 1 else ''})")
                except Exception as e:
                    print(f"warning: could not load {path}: {e}")
                break
    if "default" not in assets:
        raise SystemExit(
            f"Need a fallback image at {config.IMAGE_DIR}/default.png "
            f"(or .gif/.jpg/.jpeg/.webp/.bmp)")
    for name in config.EXPRESSIONS:
        if name not in assets and name != config.OTHER_CLASS:
            print(f"warning: no image for '{name}', will use default")
    return assets


def fit(img, max_side):
    h, w = img.shape[:2]
    s = min(max_side / max(h, w), 1.0)
    if s < 1.0:
        img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    return img


def main():
    ckpt = torch.load(config.MODEL_PATH, map_location="cpu", weights_only=False)
    classes = ckpt["classes"]
    mean = torch.from_numpy(np.asarray(ckpt["mean"], np.float32))
    std = torch.from_numpy(np.asarray(ckpt["std"], np.float32))
    model = MLP(ckpt["n_in"], len(classes))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    use_hands = ckpt.get("use_hands", False)

    images = load_images()
    cap = cv2.VideoCapture(config.CAMERA_INDEX)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera {config.CAMERA_INDEX}")
    # follow the checkpoint, not config, so features always match the model
    extractor = FeatureExtractor(use_hands=use_hands)
    if extractor.dim != ckpt["n_in"]:
        raise SystemExit("feature/model size mismatch -- retrain with train.py")

    probs_hist = deque(maxlen=config.SMOOTH_WINDOW)
    displayed = "default"
    candidate, candidate_streak = "default", 0
    t0 = time.time()
    shown_since = t0

    cv2.namedWindow("camera", cv2.WINDOW_NORMAL)
    cv2.namedWindow("expression", cv2.WINDOW_NORMAL)

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if config.MIRROR_PREVIEW:
            frame = cv2.flip(frame, 1)

        ts_ms = int((time.time() - t0) * 1000)
        feats, hand_present = extractor.process(frame, ts_ms)

        # ---- three ways to end up on the default image ---------------------
        # 1. no face in frame
        # 2. the model's best guess is the trained "other" class
        # 3. the smoothed confidence is below threshold
        if feats is None:
            probs_hist.clear()
            raw_label, conf = "default", 0.0
        else:
            x = (torch.from_numpy(feats) - mean) / std
            with torch.no_grad():
                p = torch.softmax(model(x.unsqueeze(0)), dim=1)[0].numpy()
            probs_hist.append(p)
            smoothed = np.mean(probs_hist, axis=0)
            top = int(smoothed.argmax())
            conf = float(smoothed[top])
            name = classes[top]
            if conf < config.CONF_THRESHOLD or name == config.OTHER_CLASS:
                raw_label = "default"
            else:
                raw_label = name

        if raw_label == candidate:
            candidate_streak += 1
        else:
            candidate, candidate_streak = raw_label, 1
        now = time.time()
        if candidate_streak >= config.HOLD_FRAMES and candidate != displayed:
            displayed = candidate
            shown_since = now 

        asset = images.get(displayed, images["default"])
        cv2.imshow("expression", asset.frame_at((now - shown_since) * 1000.0))

        cv2.putText(frame, f"{displayed}  ({conf:.2f})", (12, 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        if use_hands:
            cv2.putText(frame, "hand" if hand_present else "no hand",
                        (frame.shape[1] - 130, 34), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 255, 0) if hand_present else (120, 120, 120), 2)
        if feats is not None and probs_hist:
            sm = np.mean(probs_hist, axis=0)
            for i, c in enumerate(classes):
                y = 70 + 24 * i
                cv2.rectangle(frame, (12, y - 12), (12 + int(160 * sm[i]), y + 2),
                              (90, 160, 90), -1)
                cv2.putText(frame, f"{c} {sm[i]:.2f}", (180, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)
        cv2.imshow("camera", frame)

        if (cv2.waitKey(1) & 0xFF) == ord("q"):
            break

    extractor.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()