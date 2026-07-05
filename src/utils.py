import os
import torch


def save_checkpoint(state, checkpoint_dir, filename="checkpoint.pth"):
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    path = os.path.join(checkpoint_dir, filename)
    torch.save(state, path)


def load_checkpoint(path, model, optimizer=None, map_location="cpu"):
    checkpoint = torch.load(path, map_location=map_location)
    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return checkpoint
