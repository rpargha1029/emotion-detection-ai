import sys
import os

# Ensure root path for importing src.*
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import torch
from torchvision import transforms
from PIL import Image

from src.model import SimpleCNN, get_resnet18
from src.config import TRAIN_CONFIG, EMOTION_MAP


def predict(image_path, weights_path, use_resnet=False):
    device = TRAIN_CONFIG["device"]
    img_size = TRAIN_CONFIG["img_size"]

    # -------------------- TRANSFORM --------------------
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    img = Image.open(image_path).convert("RGB")
    img = transform(img).unsqueeze(0).to(device)

    # -------------------- LOAD MODEL --------------------
    if use_resnet:
        model = get_resnet18(num_classes=len(EMOTION_MAP),
                             pretrained=False,
                             input_channels=1)
    else:
        model = SimpleCNN(num_classes=len(EMOTION_MAP), input_channels=1)

    checkpoint = torch.load(weights_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    model.to(device)
    model.eval()

    # -------------------- INFERENCE --------------------
    with torch.no_grad():
        output = model(img)
        probs = torch.softmax(output, dim=1)[0]
        pred_idx = torch.argmax(probs).item()

    predicted_label = EMOTION_MAP[pred_idx]
    confidence = float(probs[pred_idx])

    return predicted_label, confidence


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--use_resnet", action="store_true")
    args = parser.parse_args()

    label, conf = predict(args.image, args.weights, use_resnet=args.use_resnet)
    print(f"Prediction: {label} ({conf:.4f})")
