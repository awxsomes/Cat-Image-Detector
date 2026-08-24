"""Record labelled training samples from your webcam.

Controls
--------
  1..9    select the expression you are about to make
  r       start / stop recording a burst
  u       undo the last burst
  q       quit (data is written continuously, nothing is lost)

Each burst gets its own session id. train.py splits train/val by session,
never by frame -- see the note in train.py for why that matters.
"""

import argparse
import csv
import os
import time
from collections import Counter

import cv2
import numpy as np

import config
from common import FeatureExtractor, feature_dim


def load_existing(path, expected_dim=None):
    """Return (rows, next_session_id, counts_per_label)."""
    if not os.path.exists(path):
        return [], 0, Counter()
    rows = []
    counts = Counter()
    max_session = -1
    with open(path, newline="") as f:
        for row in csv.reader(f):
            if not row or row[0] == "label":
                continue
            if expected_dim is not None and len(row) - 2 != expected_dim:
                raise SystemExit(
                    f"{path} has {len(row) - 2}-dim features but the current "
                    f"config produces {expected_dim}. You changed USE_HANDS "
                    f"after collecting. Move the old file aside and re-record.")
            rows.append(row)
            counts[row[0]] += 1
            max_session = max(max_session, int(row[1]))
    return rows, max_session + 1, counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stride", type=int, default=2,
                    help="keep every Nth frame while recording (consecutive "
                         "frames are nearly identical, so >1 saves disk "
                         "without losing information)")
    args = ap.parse_args()

    dim = feature_dim(config.USE_HANDS)
    os.makedirs(os.path.dirname(config.DATA_CSV) or ".", exist_ok=True)
    _, session_id, counts = load_existing(config.DATA_CSV, dim)

    new_file = not os.path.exists(config.DATA_CSV)
    csv_file = open(config.DATA_CSV, "a", newline="")
    writer = csv.writer(csv_file)
    if new_file:
        writer.writerow(["label", "session"] + [f"f{i}" for i in range(dim)])

    cap = cv2.VideoCapture(config.CAMERA_INDEX)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera {config.CAMERA_INDEX}")

    extractor = FeatureExtractor()

    label_idx = 0
    recording = False
    frame_i = 0
    burst_count = 0
    burst_sessions = []          # for undo
    t0 = time.time()

    print("Recording to", config.DATA_CSV)
    print("Labels:", ", ".join(f"[{i+1}] {n}" for i, n in enumerate(config.EXPRESSIONS)))

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if config.MIRROR_PREVIEW:
            frame = cv2.flip(frame, 1)

        ts_ms = int((time.time() - t0) * 1000)
        feats, hand_present = extractor.process(frame, ts_ms)

        if recording and feats is not None:
            if frame_i % args.stride == 0:
                label = config.EXPRESSIONS[label_idx]
                writer.writerow([label, session_id] + [f"{v:.6f}" for v in feats])
                csv_file.flush()
                counts[label] += 1
                burst_count += 1
            frame_i += 1

        # ---- overlay -------------------------------------------------------
        h = frame.shape[0]
        cv2.putText(frame, f"label: {config.EXPRESSIONS[label_idx]}", (12, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        status = f"RECORDING  n={burst_count}" if recording else "paused  (r to record)"
        colour = (0, 0, 255) if recording else (180, 180, 180)
        cv2.putText(frame, status, (12, 64),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, colour, 2)
        if feats is None:
            cv2.putText(frame, "NO FACE", (12, 96),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
        elif config.USE_HANDS:
            cv2.putText(frame, "hand: yes" if hand_present else "hand: --",
                        (12, 96), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 255, 0) if hand_present else (150, 150, 150), 2)
        for i, name in enumerate(config.EXPRESSIONS):
            cv2.putText(frame, f"{i+1} {name}: {counts[name]}",
                        (12, h - 12 - 22 * (len(config.EXPRESSIONS) - 1 - i)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (0, 255, 0) if i == label_idx else (200, 200, 200), 1)
        cv2.imshow("collect", frame)

        # ---- keys ----------------------------------------------------------
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("r"):
            if recording:
                recording = False
                print(f"  session {session_id} ({config.EXPRESSIONS[label_idx]}): "
                      f"{burst_count} samples")
                burst_sessions.append(session_id)
                session_id += 1
            else:
                recording = True
                burst_count = 0
                frame_i = 0
        elif key == ord("u") and not recording and burst_sessions:
            drop = burst_sessions.pop()
            csv_file.close()
            _undo_session(config.DATA_CSV, drop)
            csv_file = open(config.DATA_CSV, "a", newline="")
            writer = csv.writer(csv_file)
            _, _, counts = load_existing(config.DATA_CSV, dim)
            print(f"  removed session {drop}")
        elif ord("1") <= key <= ord("9"):
            i = key - ord("1")
            if i < len(config.EXPRESSIONS) and not recording:
                label_idx = i

    if recording:
        print(f"  session {session_id}: {burst_count} samples")
    csv_file.close()
    extractor.close()
    cap.release()
    cv2.destroyAllWindows()

    print("\nTotals:")
    for name in config.EXPRESSIONS:
        print(f"  {name:<12} {counts[name]}")


def _undo_session(path, session):
    with open(path, newline="") as f:
        rows = list(csv.reader(f))
    header, body = rows[0], [r for r in rows[1:] if r and int(r[1]) != session]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(body)


if __name__ == "__main__":
    main()