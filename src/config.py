from pathlib import Path

# Root directory of the project
ROOT = Path(__file__).resolve().parents[1]

# Data paths
DATA_DIR = ROOT / "data"
SPLITS_DIR = DATA_DIR / "splits"

# Experiment paths
EXPERIMENTS_DIR = ROOT / "experiments"
CHECKPOINTS_DIR = EXPERIMENTS_DIR / "checkpoints"

# Training configuration
TRAIN_CONFIG = {
    "batch_size": 64,
    "num_epochs": 30,
    "lr": 1e-3,
    "img_size": 48,
    "device": "cuda" if __import__("torch").cuda.is_available() else "cpu",
    "num_workers": 4
}

# Emotion labels
EMOTION_MAP = ["angry", "disgusted", "fearful", "happy", "sad", "surprised", "neutral"]
