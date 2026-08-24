# The expressions the model can choose from.
# IMPORTANT: keep "other" in this list. It is a real, trained class that soaks
# up neutral faces, talking, mid-transition frames, looking away, etc. Without
# it a softmax classifier is forced to pick one of your "real" expressions for
# every single frame, and the fallback image will almost never appear.
EXPRESSIONS = [
    "scuba",
    "shocked",
    "thumbsup",
    "tongue",
    "smile",
    "other",
]

OTHER_CLASS = "other"

# Paths
LANDMARKER_TASK = "face_landmarker.task"        # downloaded once, see README
HAND_LANDMARKER_TASK = "hand_landmarker.task"   # only needed if USE_HANDS
DATA_CSV = "data/samples.csv"
MODEL_PATH = "expression_model.pt"
IMAGE_DIR = "images"

USE_HANDS = True

# Camera
CAMERA_INDEX = 0
MIRROR_PREVIEW = True

# Display
# OpenCV windows cannot show transparency, so transparent pixels in GIFs and
# PNGs are flattened onto this colour. RGB, 0-255.
TRANSPARENT_BACKGROUND = (0, 0, 0)

# Live inference behaviour
SMOOTH_WINDOW = 8        # average the probabilities over this many frames
CONF_THRESHOLD = 0.60    # smoothed probability required to commit to a label
HOLD_FRAMES = 5          # a new label must win this many frames before switching

# Number of blendshape coefficients MediaPipe returns. Do not change.
N_BLENDSHAPES = 52