# src/train_max_v2.py
import sys
import os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torchvision import transforms, models
from collections import Counter

# Try Albumentations
USE_ALB = False
try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    USE_ALB = True
except Exception:
    USE_ALB = False

from src.dataset import FERDataset
from src.config import TRAIN_CONFIG, EMOTION_MAP, CHECKPOINTS_DIR
from src.utils import save_checkpoint


# ======================================================
# Utilities
# ======================================================
def count_class_samples(root="data/splits", split="train"):
    path = Path(root) / split
    counts = {}
    for c in sorted(path.iterdir()):
        if c.is_dir():
            imgs = list(c.glob("*"))
            counts[c.name.lower()] = len(imgs)
    return counts


# ======================================================
# Focal Loss with class weights
# ======================================================
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, weight=None):
        super().__init__()
        self.gamma = gamma
        self.ce = nn.CrossEntropyLoss(weight=weight, reduction="none")

    def forward(self, logits, targets):
        ce_loss = self.ce(logits, targets)
        pt = torch.exp(-ce_loss)
        loss = (1 - pt) ** self.gamma * ce_loss
        return loss.mean()


# ======================================================
# EMA (Exponential Moving Average)
# ======================================================
class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {}
        self.model = model
        for name, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[name] = p.data.clone()

    def update(self):
        for name, p in self.model.named_parameters():
            if p.requires_grad:
                new = (1.0 - self.decay) * p.data + self.decay * self.shadow[name]
                self.shadow[name] = new.clone()

    def apply_shadow(self):
        self.backup = {}
        for name, p in self.model.named_parameters():
            if p.requires_grad:
                self.backup[name] = p.data.clone()
                p.data = self.shadow[name]

    def restore(self):
        for name, p in self.model.named_parameters():
            if p.requires_grad and name in self.backup:
                p.data = self.backup[name]


# ======================================================
# Backbone
# ======================================================
def get_backbone(name="efficientnet", pretrained=True, num_classes=7, input_channels=1):
    name = name.lower()

    if "efficient" in name:
        model = models.efficientnet_b0(weights="IMAGENET1K_V1" if pretrained else None)

        # Fix first conv for 1 channel
        try:
            old = model.features[0][0]
            model.features[0][0] = nn.Conv2d(
                input_channels, old.out_channels,
                kernel_size=old.kernel_size, stride=old.stride,
                padding=old.padding, bias=False
            )
        except:
            model.features[0] = nn.Conv2d(input_channels, 32, kernel_size=3, stride=2, padding=1)

        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        return model

    if "resnet" in name:
        if name == "resnet34":
            model = models.resnet34(weights="IMAGENET1K_V1" if pretrained else None)
        elif name == "resnet50":
            model = models.resnet50(weights="IMAGENET1K_V1" if pretrained else None)
        else:
            model = models.resnet18(weights="IMAGENET1K_V1" if pretrained else None)

        if input_channels != 3:
            old = model.conv1
            model.conv1 = nn.Conv2d(
                input_channels, old.out_channels,
                kernel_size=old.kernel_size, stride=old.stride,
                padding=old.padding, bias=old.bias
            )

        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    # fallback
    from src.model import SimpleCNN
    return SimpleCNN(num_classes=num_classes, input_channels=input_channels)


# ======================================================
# Transforms
# ======================================================
def get_transforms(img_size=48, mode="train"):
    if USE_ALB and mode == "train":
        return A.Compose([
            A.RandomResizedCrop(img_size, img_size, scale=(0.7, 1.0)),
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=12, p=0.6),
            A.RandomBrightnessContrast(p=0.5),
            A.GaussNoise(p=0.3),
            A.CLAHE(p=0.3),
            A.Normalize(mean=(0.5,), std=(0.5,)),
            ToTensorV2()
        ])

    if mode == "train":
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(12),
            transforms.RandomErasing(p=0.25),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])


# Albumentations dataset wrapper
if USE_ALB:
    class AlbFERDataset(torch.utils.data.Dataset):
        def __init__(self, root, split="train", aug=None):
            self.root = Path(root) / split
            self.samples = []
            name_to_idx = {n:i for i,n in enumerate(EMOTION_MAP)}

            for c in sorted(self.root.iterdir()):
                if not c.is_dir(): continue
                label = name_to_idx[c.name.lower()]
                for f in c.glob("*"):
                    if f.suffix.lower() in [".jpg",".png",".jpeg",".bmp"]:
                        self.samples.append((str(f), label))

            self.aug = aug

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, idx):
            p, label = self.samples[idx]
            import cv2
            img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
            aug = self.aug(image=img)
            return aug["image"], label


# ======================================================
# Mixed Precision (Version Adaptive)
# ======================================================
def get_autocast(device):
    if not device.startswith("cuda"):
        return torch.cpu.amp.autocast()

    try:
        # New API (PyTorch ≥ 2.1)
        from torch import amp
        return amp.autocast("cuda", enabled=True)
    except Exception:
        # Old API (PyTorch ≤ 2.0)
        return torch.cuda.amp.autocast(enabled=True)


def get_scaler(device):
    if not device.startswith("cuda"):
        return None
    try:
        from torch import amp
        return amp.GradScaler()
    except Exception:
        return torch.cuda.amp.GradScaler()


# ======================================================
# Training Loop
# ======================================================
def train_max_v2(args):
    device = TRAIN_CONFIG["device"]
    img_size = TRAIN_CONFIG["img_size"]
    batch_size = args.batch_size
    epochs = args.epochs
    num_classes = len(EMOTION_MAP)

    # Dataset
    if USE_ALB:
        train_ds = AlbFERDataset("data/splits", split="train", aug=get_transforms(img_size,"train"))
        val_ds   = FERDataset("data/splits", split="val", transform=get_transforms(img_size,"val"))
    else:
        train_ds = FERDataset("data/splits", split="train", transform=get_transforms(img_size,"train"))
        val_ds   = FERDataset("data/splits", split="val", transform=get_transforms(img_size,"val"))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # Class weights
    counts = count_class_samples()
    weight_list = [1 / max(1, counts[c]) for c in EMOTION_MAP]
    weight_list = torch.tensor(weight_list, dtype=torch.float).to(device)

    # Model
    model = get_backbone(args.model, pretrained=args.pretrained, num_classes=num_classes, input_channels=1)
    model = model.to(device)

    # Loss
    if args.use_focal:
        loss_fn = FocalLoss(gamma=2.0, weight=weight_list)
    else:
        loss_fn = nn.CrossEntropyLoss(weight=weight_list, label_smoothing=args.label_smoothing)

    # Optimizer + Scheduler
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

    # EMA
    ema = EMA(model, decay=args.ema_decay) if args.use_ema else None

    # AMP
    scaler = get_scaler(device)

    # Logging
    os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
    writer = SummaryWriter(log_dir=f"experiments/runs/maxv2_{int(time.time())}")

    best_val = 0.0

    # ======================================================
    # Epoch Loop
    # ======================================================
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        total_correct = 0
        total_n = 0

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()

            with get_autocast(device):
                outputs = model(imgs)
                loss = loss_fn(outputs, labels)

            if scaler:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            if ema:
                ema.update()

            total_loss += loss.item() * imgs.size(0)
            total_correct += (outputs.argmax(1) == labels).sum().item()
            total_n += labels.size(0)

        train_loss = total_loss / total_n
        train_acc = total_correct / total_n

        # Validation
        model.eval()
        if ema:
            ema.apply_shadow()

        val_loss = 0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)

                outputs = model(imgs)
                loss = loss_fn(outputs, labels)

                val_loss += loss.item() * imgs.size(0)
                val_correct += (outputs.argmax(1) == labels).sum().item()
                val_total += labels.size(0)

        val_loss /= val_total
        val_acc = val_correct / val_total

        if ema:
            ema.restore()

        # Logging
        writer.add_scalar("train/loss", train_loss, epoch)
        writer.add_scalar("train/acc", train_acc, epoch)
        writer.add_scalar("val/loss", val_loss, epoch)
        writer.add_scalar("val/acc", val_acc, epoch)

        print(f"Epoch {epoch+1}/{epochs} | TrainAcc {train_acc:.4f} | ValAcc {val_acc:.4f}")

        scheduler.step(epoch + 1)

        # Save Best
        if val_acc > best_val:
            best_val = val_acc
            ckpt = {
                "model_state_dict": model.state_dict(),
                "val_acc": val_acc,
                "epoch": epoch + 1
            }
            save_checkpoint(ckpt, CHECKPOINTS_DIR, filename="best_model_max_v2.pth")
            print(f"🔥 Best Model Updated: {best_val:.4f}")

    writer.close()
    print("Training complete. Best Acc =", best_val)


# ======================================================
# CLI
# ======================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="efficientnet", choices=["efficientnet","resnet34","resnet18","resnet50"])
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--use_focal", action="store_true")
    parser.add_argument("--use_ema", action="store_true")
    parser.add_argument("--ema_decay", type=float, default=0.999)
    parser.add_argument("--label_smoothing", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=75)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-3)
    args = parser.parse_args()

    train_max_v2(args)
