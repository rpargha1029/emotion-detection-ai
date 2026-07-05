from pathlib import Path
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
from src.config import TRAIN_CONFIG, EMOTION_MAP


class FERDataset(Dataset):
    def __init__(self, root_dir, split="train", transform=None):
        self.root = Path(root_dir)
        self.split = split
        self.transform = transform
        self.samples = []

        split_dir = self.root / split
        if not split_dir.exists():
            raise FileNotFoundError(f"Split folder not found: {split_dir}")

        # Map class names → index
        name_to_idx = {name: i for i, name in enumerate(EMOTION_MAP)}

        # Iterate through class folders named "angry", "sad", etc.
        for class_dir in sorted(split_dir.iterdir()):
            if not class_dir.is_dir():
                continue

            class_name = class_dir.name.lower().strip()

            if class_name not in name_to_idx:
                raise ValueError(f"Unknown class folder name: {class_name}")

            class_label = name_to_idx[class_name]

            for img_path in class_dir.glob("*"):
                if img_path.suffix.lower() in [".jpg", ".png", ".jpeg", ".bmp"]:
                    self.samples.append((img_path, class_label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]

        img = Image.open(img_path).convert("L")

        if self.transform:
            img = self.transform(img)
        else:
            default_transform = transforms.Compose([
                transforms.Resize((TRAIN_CONFIG["img_size"], TRAIN_CONFIG["img_size"])),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5])
            ])
            img = default_transform(img)

        return img, label
