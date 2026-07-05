import sys
import os

# Ensure root path for importing src.*
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
import argparse

from src.dataset import FERDataset
from src.model import SimpleCNN, get_resnet18
from src.utils import save_checkpoint
from src.config import TRAIN_CONFIG, CHECKPOINTS_DIR, EMOTION_MAP


def train(args):
    device = TRAIN_CONFIG["device"]
    print("🚀 Training will use device:", device)
    img_size = TRAIN_CONFIG["img_size"]

    # -------------------- TRANSFORMS --------------------
    transform_train = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    transform_val = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    # -------------------- DATASETS --------------------
    train_dataset = FERDataset("data/splits", split="train", transform=transform_train)
    val_dataset   = FERDataset("data/splits", split="val",   transform=transform_val)

    train_loader = DataLoader(train_dataset, batch_size=TRAIN_CONFIG["batch_size"],
                              shuffle=True, num_workers=TRAIN_CONFIG["num_workers"])
    val_loader   = DataLoader(val_dataset, batch_size=TRAIN_CONFIG["batch_size"],
                              shuffle=False, num_workers=TRAIN_CONFIG["num_workers"])

    # -------------------- MODEL --------------------
    if args.use_resnet:
        model = get_resnet18(num_classes=len(EMOTION_MAP),
                             pretrained=args.pretrained,
                             input_channels=1)
    else:
        model = SimpleCNN(num_classes=len(EMOTION_MAP), input_channels=1)

    model = model.to(device)

    # -------------------- OPTIMIZER + LOSS --------------------
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=TRAIN_CONFIG["lr"])
    scheduler = StepLR(optimizer, step_size=7, gamma=0.1)

    best_val_acc = 0
    os.makedirs(CHECKPOINTS_DIR, exist_ok=True)

    # -------------------- TRAINING LOOP --------------------
    for epoch in range(TRAIN_CONFIG["num_epochs"]):
        model.train()
        running_loss = 0
        correct = 0
        total = 0

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)

            outputs = model(imgs)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * imgs.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)

        train_loss = running_loss / total
        train_acc = correct / total

        scheduler.step()

        # -------------------- VALIDATION --------------------
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)

                outputs = model(imgs)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * imgs.size(0)
                val_correct += (outputs.argmax(1) == labels).sum().item()
                val_total += labels.size(0)

        val_loss /= val_total
        val_acc = val_correct / val_total

        print(f"Epoch [{epoch+1}/{TRAIN_CONFIG['num_epochs']}] "
              f"| Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} "
              f"| Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        # -------------------- SAVE BEST MODEL --------------------
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(
                {
                    "model_state_dict": model.state_dict(),
                    "val_acc": val_acc,
                },
                CHECKPOINTS_DIR,
                filename="best_model.pth"
            )
            print(f"🔥 New best model saved! Val Acc = {val_acc:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_resnet", action="store_true")
    parser.add_argument("--pretrained", action="store_true")
    args = parser.parse_args()

    train(args)
