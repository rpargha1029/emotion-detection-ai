import sys
import os

# Ensure root path for importing src.*
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from src.dataset import FERDataset
from src.model import SimpleCNN, get_resnet18
from src.config import TRAIN_CONFIG, EMOTION_MAP, EXPERIMENTS_DIR


def plot_confusion_matrix(cm, labels, out_path):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.savefig(out_path)
    plt.close()


def evaluate(weights_path, use_resnet=False):
    device = TRAIN_CONFIG["device"]
    img_size = TRAIN_CONFIG["img_size"]

    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    test_dataset = FERDataset("data/splits", split="test", transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

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

    y_true = []
    y_pred = []

    # -------------------- INFERENCE --------------------
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            preds = outputs.argmax(1)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    # -------------------- REPORT --------------------
    print("\nClassification Report:\n")
    print(classification_report(y_true, y_pred, target_names=EMOTION_MAP))

    cm = confusion_matrix(y_true, y_pred)

    results_path = EXPERIMENTS_DIR / "results"
    os.makedirs(results_path, exist_ok=True)

    cm_path = results_path / "confusion_matrix.png"
    plot_confusion_matrix(cm, EMOTION_MAP, cm_path)

    print(f"\nConfusion matrix saved to: {cm_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--use_resnet", action="store_true")
    args = parser.parse_args()

    evaluate(args.weights, use_resnet=args.use_resnet)
