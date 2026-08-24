# Real-time expression → image

Webcam facial expressions and hand gestures drive an image display, in real
time. MediaPipe face blendshapes and hand landmarks feed a small PyTorch MLP;
unrecognised input falls back to a default image.

## Demo

[![Demo video](https://img.youtube.com/vi/2jbFuosY0do/hqdefault.jpg)](https://www.youtube.com/watch?v=2jbFuosY0do)

<!-- Optional inline GIF, so the README shows motion without a click:
     ffmpeg -i demo.mp4 -ss 0 -t 6 -vf "fps=12,scale=640:-1:flags=lanczos,palettegen" palette.png
     ffmpeg -i demo.mp4 -ss 0 -t 6 -i palette.png -lavfi "fps=12,scale=640:-1:flags=lanczos,paletteuse" demo.gif
     then add:  ![demo](demo.gif)                                            -->

## Setup

```bash
pip install -r requirements.txt
curl -o face_landmarker.task https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
curl -o hand_landmarker.task https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
```

The hand model is only needed while `USE_HANDS = True` in `config.py`.
With hands on, each frame is described by 117 numbers: 52 face blendshapes
plus a 65-number hand block (present flag, handedness, and 21 landmarks that
have been made translation- and scale-invariant but deliberately *not*
rotation-invariant, so thumbs up and thumbs down stay distinguishable).

Put your images in `images/`, named after the expressions in `config.py`:

```
images/scuba.gif
images/shocked.jpg
images/thumbsup.png
images/tongue.jpg
images/default.jpg     <- required fallback
```

Animated GIFs work and loop, restarting from frame 0 each time that expression
is triggered. `.gif .webp .png .jpg .jpeg .bmp` are all accepted; if more than
one file matches a name, that's the priority order. Transparency is flattened
onto `TRANSPARENT_BACKGROUND` in `config.py`, since OpenCV windows can't show
an alpha channel.

## 1. Collect

```bash
python collect.py
```

Press `1`–`5` to pick the expression, `r` to start/stop recording a burst,
`q` to quit. Aim for **4–6 separate bursts of ~10 seconds per expression**,
and change something between bursts: move closer/further, turn your head,
switch on a lamp, take your glasses off, record again tomorrow. Variation
across bursts is what makes it work outside the exact conditions you recorded
in — more frames of the same static pose adds almost nothing.

For gestures, move the hand around while recording: near the face, off to the
side, high, low, both hands, partially out of frame. The `hand` indicator in
the corner shows whether MediaPipe currently sees one -- if it flickers off,
that frame is being recorded with a zeroed hand block, which is worth avoiding.

Give the `other` class the most data. Record yourself talking, looking away,
scrolling on your phone, resting your face, and *mid-transition* between
expressions. With hands on, also record your hand doing nothing in particular
in frame -- otherwise "any hand at all" becomes a shortcut the model latches
onto, and every stray gesture fires one of your real classes. This class is what makes the default image actually appear.

## 2. Train

```bash
python train.py
```

Takes a few seconds on CPU. Read the confusion matrix — if two expressions
get mixed up, they probably look too similar and are worth merging or
exaggerating.

## 3. Run

```bash
python run.py
```

## Tuning

In `config.py`:

- image flickers between two expressions → raise `SMOOTH_WINDOW` or `HOLD_FRAMES`
- stuck on default too often → lower `CONF_THRESHOLD`, or record more data
- fires on expressions you aren't really making → raise `CONF_THRESHOLD`, or
  record more `other` data covering whatever it's false-firing on
- gesture classes fire without the gesture → you need more hands-visible
  `other` data; the model is keying on your face alone
- changing `USE_HANDS` changes the feature width, so existing data becomes
  unusable — move `data/samples.csv` aside and re-record