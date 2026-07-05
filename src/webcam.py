import sys
import os
import cv2
import torch
import time
from collections import deque
from torchvision import transforms
from PIL import Image

# Ensure project root is visible
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.model import SimpleCNN, get_resnet18
from src.config import TRAIN_CONFIG, EMOTION_MAP


# 🎨 Color map for emotions
COLOR_MAP = {
    "angry": (0, 0, 255),
    "disgusted": (0, 128, 0),
    "fearful": (128, 0, 128),
    "happy": (0, 255, 0),
    "sad": (255, 0, 0),
    "surprised": (0, 255, 255),
    "neutral": (200, 200, 200)
}


def run_webcam(weights_path, use_resnet=False):

    device = TRAIN_CONFIG["device"]
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

    checkpoint = torch.load(weights_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    # -------------------- Transform Pipeline --------------------
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    # -------------------- Face Detector --------------------
    detector = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Could not access webcam.")
        return

    print("🎥 Webcam running... Press 'q' to quit.")

    # FPS calculation
    prev_time = time.time()
    fps = 0

    # Emotion smoothing buffer
    smooth_buffer = deque(maxlen=7)   # keeps last 7 predictions

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(gray, 1.3, 5)

        # FPS update
        current_time = time.time()
        fps = 1 / (current_time - prev_time)
        prev_time = current_time

        # Draw FPS on screen
        cv2.putText(frame, f"FPS: {fps:.1f}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2)

        for (x, y, w, h) in faces:
            # Bounding box styling
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 180, 255), 2)

            # Extract face region
            face = frame[y:y+h, x:x+w]
            face_gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            pil_img = Image.fromarray(face_gray).convert("L")
            img_tensor = transform(pil_img).unsqueeze(0).to(device)

            # Inference
            with torch.no_grad():
                output = model(img_tensor)
                probs = torch.softmax(output, dim=1)[0]

            pred_idx = torch.argmax(probs).item()
            label = EMOTION_MAP[pred_idx]
            confidence = float(probs[pred_idx])

            # Push to smoothing buffer
            smooth_buffer.append(label)
            # Use the most frequent label in last N frames
            stable_label = max(set(smooth_buffer), key=smooth_buffer.count)

            # Choose color
            color = COLOR_MAP.get(stable_label, (255, 255, 255))

            # Emotion label text
            cv2.putText(
                frame,
                f"{stable_label} ({confidence:.2f})",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                color,
                2
            )

        cv2.imshow("Emotion Detection AI - Enhanced", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--use_resnet", action="store_true")
    args = parser.parse_args()

    run_webcam(args.weights, use_resnet=args.use_resnet)
