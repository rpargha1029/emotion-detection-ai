import sys
import os

# Ensure project root is visible for imports
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import torch
from torchvision import transforms
from PIL import Image

from src.model import SimpleCNN, get_resnet18
from src.config import TRAIN_CONFIG, EMOTION_MAP


class LiveInference:
    def __init__(self, weights_path, use_resnet=False):
        self.device = TRAIN_CONFIG["device"]
        self.img_size = TRAIN_CONFIG["img_size"]

        # -------------------- Load Model --------------------
        if use_resnet:
            self.model = get_resnet18(
                num_classes=len(EMOTION_MAP),
                pretrained=False,
                input_channels=1
            )
        else:
            self.model = SimpleCNN(
                num_classes=len(EMOTION_MAP),
                input_channels=1
            )

        checkpoint = torch.load(weights_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

        # -------------------- Transform Pipeline --------------------
        self.transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

    def predict_pil(self, pil_img):
        """Run inference on a PIL image."""
        img_tensor = self.transform(pil_img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(img_tensor)
            probs = torch.softmax(output, dim=1)[0]

        pred_idx = torch.argmax(probs).item()
        label = EMOTION_MAP[pred_idx]
        confidence = float(probs[pred_idx])

        return label, confidence

    def predict_path(self, image_path):
        """Run inference on an image file path."""
        pil_img = Image.open(image_path).convert("RGB")
        return self.predict_pil(pil_img)
