import sys
import os

# Ensure project root is visible
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import shutil
import random
from pathlib import Path


def make_val_split(val_ratio=0.1):
    DATA_ROOT = Path(ROOT) / "data" / "splits"
    train_dir = DATA_ROOT / "train"
    val_dir = DATA_ROOT / "val"

    print("Project root:", ROOT)
    print("Training directory:", train_dir)

    if not train_dir.exists():
        raise FileNotFoundError(f"❌ ERROR: Train folder not found: {train_dir}")

    # Create val directory structure
    val_dir.mkdir(parents=True, exist_ok=True)

    # Iterate over class folders
    for class_dir in sorted(train_dir.iterdir()):
        if not class_dir.is_dir():
            continue

        class_name = class_dir.name
        class_val_dir = val_dir / class_name
        class_val_dir.mkdir(parents=True, exist_ok=True)

        images = list(class_dir.glob("*"))
        random.shuffle(images)

        val_count = int(len(images) * val_ratio)
        val_images = images[:val_count]

        for img in val_images:
            shutil.move(str(img), str(class_val_dir / img.name))

        print(f"[{class_name}] moved {val_count} images → val")


if __name__ == "__main__":
    make_val_split()
