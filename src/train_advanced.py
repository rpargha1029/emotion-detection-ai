# src/train_advanced.py
import sys, os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import time
import shutil
from pathlib import Path
import argparse
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import OneCycleLR
from torchvision import transforms, models
from torchvision.datasets import ImageFolder

from src.config import TRAIN_CONFIG, EMOTION_MAP, CHECKPOINTS_DIR
from src.utils import save_checkpoint
from src.dataset import FERDataset

# Optional: attempt to import albumentations for better augmentations
USE_ALBUMENTATIONS = False
try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    USE_ALBUMENTATIONS = True
except Exception:
    USE_ALBUMENTATIONS = False

# -------------------------
# Losses
# -------------------------
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.ce = nn.CrossEntropyLoss(reduction='none')

    def forward(self, inputs, targets):
        logp = self.ce(inputs, targets)  # (N,)
        p = torch.exp(-logp)
        loss = ((1 - p) ** self.gamma) * logp
        if self.alpha is not None:
            at = self.alpha[targets].to(inputs.device)
            loss = at * loss
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

# -------------------------
# Model helpers
# -------------------------
def get_backbone(name='resnet34', pretrained=False, num_classes=7, input_channels=1):
    name = name.lower()
    if 'resnet' in name:
        if name == 'resnet34':
            model = models.resnet34(pretrained=pretrained)
        elif name == 'resnet18':
            model = models.resnet18(pretrained=pretrained)
        else:
            model = models.resnet50(pretrained=pretrained)
        # adapt first conv if input_channels != 3
        if input_channels != 3:
            old = model.conv1
            model.conv1 = nn.Conv2d(input_channels, old.out_channels,
                                    kernel_size=old.kernel_size,
                                    stride=old.stride,
                                    padding=old.padding,
                                    bias=old.bias)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    if 'efficientnet' in name or 'efficient' in name:
        # torchvision efficientnet_b0 exists on modern torchvision
        model = models.efficientnet_b0(pretrained=pretrained)
        if input_channels != 3:
            # replace features[0] stem conv
            # efficientnet_b0 has .features[0][0] as Conv2d in some versions — we do a robust replace
            try:
                model.features[0][0] = nn.Conv2d(input_channels, model.features[0][0].out_channels,
                                                 kernel_size=model.features[0][0].kernel_size,
                                                 stride=model.features[0][0].stride,
                                                 padding=model.features[0][0].padding,
                                                 bias=False)
            except Exception:
                # fallback: try model.features[0]
                model.features[0] = nn.Conv2d(input_channels, 32, kernel_size=3, stride=2, padding=1, bias=False)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        return model

    # fallback: simple cnn
    from src.model import SimpleCNN
    return SimpleCNN(num_classes=num_classes, input_channels=input_channels)

# -------------------------
# Augmentations
# -------------------------
def build_transforms(img_size=48, mode='train'):
    if USE_ALBUMENTATIONS and mode == 'train':
        aug = A.Compose([
            A.RandomResizedCrop(img_size, img_size, scale=(0.7, 1.0)),
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.6),
            A.RandomBrightnessContrast(p=0.5),
            A.GaussNoise(p=0.3),
            A.Normalize(mean=0.5, std=0.5),
            ToTensorV2()
        ])
        return aug
    else:
        if mode == 'train':
            return transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(12),
                transforms.ColorJitter(brightness=0.2, contrast=0.2) if TRAIN_CONFIG.get("input_channels",1)==3 else transforms.RandomErasing(p=0.2),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5])
            ])
        else:
            return transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5])
            ])

# -------------------------
# Dataset wrapper for albumentations (optional)
# -------------------------
if USE_ALBUMENTATIONS:
    from torch.utils.data import Dataset as TorchDataset
    class AlbFERDataset(TorchDataset):
        def __init__(self, root, split='train', alb_aug=None):
            self.root = Path(root) / split
            self.samples = []
            self.alb = alb_aug
            name_to_idx = {n:i for i,n in enumerate(EMOTION_MAP)}
            for c in sorted(self.root.iterdir()):
                if not c.is_dir(): continue
                label = name_to_idx[c.name.lower().strip()]
                for p in c.glob("*"):
                    if p.suffix.lower() in [".jpg",".png",".jpeg",".bmp"]:
                        self.samples.append((str(p), label))
        def __len__(self): return len(self.samples)
        def __getitem__(self, idx):
            p, label = self.samples[idx]
            import cv2
            img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
            if img is None: raise FileNotFoundError(p)
            img = img.astype("uint8")
            augmented = self.alb(image=img)
            tensor = augmented['image']  # already tensor from ToTensorV2
            return tensor, label

# -------------------------
# Training loop
# -------------------------
def train(args):
    device = TRAIN_CONFIG["device"]
    img_size = TRAIN_CONFIG["img_size"]
    num_workers = TRAIN_CONFIG.get("num_workers", 4)
    batch_size = args.batch_size or TRAIN_CONFIG["batch_size"]
    epochs = args.epochs or TRAIN_CONFIG.get("num_epochs", 30)
    num_classes = len(EMOTION_MAP)
    input_channels = 1  # grayscale

    # Transforms
    if USE_ALBUMENTATIONS:
        train_alb = build_transforms(img_size=img_size, mode='train')
        train_dataset = AlbFERDataset("data/splits", split="train", alb_aug=train_alb)
        val_transform = build_transforms(img_size=img_size, mode='val')
        val_dataset = FERDataset("data/splits", split="val", transform=val_transform)
    else:
        train_transform = build_transforms(img_size=img_size, mode='train')
        val_transform = build_transforms(img_size=img_size, mode='val')
        train_dataset = FERDataset("data/splits", split="train", transform=train_transform)
        val_dataset = FERDataset("data/splits", split="val", transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)

    # Model
    model = get_backbone(args.model, pretrained=args.pretrained, num_classes=num_classes, input_channels=input_channels)
    model = model.to(device)

    # Loss
    if args.use_focal:
        alpha = None
        # optional: class-balanced alpha vector (uniform for now)
        loss_fn = FocalLoss(gamma=args.focal_gamma, alpha=alpha)
    else:
        # label_smoothing supported in modern PyTorch
        loss_fn = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    # Optimizer + scheduler (OneCycle)
    optimizer = SGD(model.parameters(), lr=args.max_lr, momentum=0.9, weight_decay=1e-4) if args.opt == 'sgd' else Adam(model.parameters(), lr=args.max_lr)
    scheduler = OneCycleLR(optimizer, max_lr=args.max_lr, steps_per_epoch=len(train_loader), epochs=epochs)

    scaler = torch.cuda.amp.GradScaler() if device.startswith('cuda') else None

    os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
    best_val = 0.0

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{epochs}] Train", leave=False)
        for batch in loop:
            imgs, labels = batch
            imgs = imgs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            if scaler is not None:
                with torch.cuda.amp.autocast():
                    outputs = model(imgs)
                    loss = loss_fn(outputs, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(imgs)
                loss = loss_fn(outputs, labels)
                loss.backward()
                optimizer.step()

            running_loss += loss.item() * imgs.size(0)
            preds = outputs.argmax(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            loop.set_postfix(loss=running_loss/total, acc=correct/total)

            scheduler.step()

        train_loss = running_loss / total
        train_acc = correct / total

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for imgs, labels in tqdm(val_loader, desc="Valid", leave=False):
                imgs = imgs.to(device)
                labels = labels.to(device)
                outputs = model(imgs)
                loss = loss_fn(outputs, labels)
                val_loss += loss.item() * imgs.size(0)
                val_correct += (outputs.argmax(1) == labels).sum().item()
                val_total += labels.size(0)

        val_loss = val_loss / val_total
        val_acc = val_correct / val_total

        print(f"Epoch {epoch+1}/{epochs} - Train loss: {train_loss:.4f}, Train acc: {train_acc:.4f} | Val loss: {val_loss:.4f}, Val acc: {val_acc:.4f}")

        # Save best
        if val_acc > best_val:
            best_val = val_acc
            ckpt = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_acc,
                "epoch": epoch+1
            }
            save_checkpoint(ckpt, CHECKPOINTS_DIR, filename="best_model_adv.pth")
            print(f"🔥 New best model saved: {best_val:.4f}")

    print("Training complete. Best val acc:", best_val)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="resnet34", choices=["resnet18","resnet34","resnet50","efficientnet"], help="backbone")
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--use_focal", action="store_true")
    parser.add_argument("--focal_gamma", type=float, default=2.0)
    parser.add_argument("--label_smoothing", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_lr", type=float, default=1e-3)
    parser.add_argument("--opt", default="adam", choices=["adam","sgd"])
    args = parser.parse_args()

    train(args)
