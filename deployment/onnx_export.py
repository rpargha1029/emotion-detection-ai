import sys
import os

# Ensure project root is visible
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import torch
from src.model import SimpleCNN, get_resnet18
from src.config import TRAIN_CONFIG, EMOTION_MAP


def export_to_onnx(weights_path, out_path="model.onnx", use_resnet=False):
    img_size = TRAIN_CONFIG["img_size"]

    # -------------------- Load Model --------------------
    if use_resnet:
        model = get_resnet18(
            num_classes=len(EMOTION_MAP),
            pretrained=False,
            input_channels=1
        )
    else:
        model = SimpleCNN(
            num_classes=len(EMOTION_MAP),
            input_channels=1
        )

    checkpoint = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Dummy input
    dummy = torch.randn(1, 1, img_size, img_size)

    # -------------------- Export --------------------
    torch.onnx.export(
        model,
        dummy,
        out_path,
        input_names=["input"],
        output_names=["output"],
        opset_version=12
    )

    print(f"✅ Exported model to ONNX: {out_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--out", default="model.onnx")
    parser.add_argument("--use_resnet", action="store_true")
    args = parser.parse_args()

    export_to_onnx(args.weights, out_path=args.out, use_resnet=args.use_resnet)
