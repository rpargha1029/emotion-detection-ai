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


# Emojis for emotions
EMOJI_MAP = {
    "angry": "😡",
    "disgusted": "🤢",
    "fearful": "😱",
    "happy": "😊",
    "sad": "😢",
    "surprised": "😮",
    "neutral": "😐"
}

# Emotion colors
COLOR_MAP = {
    "angry": (0, 0, 255),
    "disgusted": (0, 128, 0),
    "fearful": (128, 0, 128),
    "happy": (0, 255, 0),
    "sad": (255, 0, 0),
    "surprised": (0, 255, 255),
    "neutral": (200, 200, 200)
}


def draw_text_with_shadow(frame, text, pos, color, shadow=(0, 0, 0)):
    x, y = pos
    cv2.putText(frame, text, (x + 2, y + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.80, shadow, 3)
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.80, color, 2)


def run_webcam_premium(weights_path, use_resnet=False):

    device = TRAIN_CONFIG["device"]
    img_size = TRAIN_CONFIG["img_size"]

    # -------------------- Load Model --------------------
    if use_resnet:
        model = get_resnet18(num_classes=len(EMOTION_MAP), pretrained=False, input_channels=1)
    else:
        model = SimpleCNN(num_classes=len(EMOTION_MAP), input_channels=1)

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
    detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Could not access webcam.")
        return

    print("✨ Premium Webcam UI running... Press 'q' to quit.")

    prev_time = time.time()
    smooth_buffer = deque(maxlen=7)
    timeline = deque(maxlen=40)

    tracker = None
    tracking = False

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # FPS
        now = time.time()
        fps = 1 / (now - prev_time)
        prev_time = now
        draw_text_with_shadow(frame, f"FPS: {fps:.1f}", (10, 30), (255, 255, 255))

        # -------------------- DETECTION OR TRACKING --------------------
        if not tracking:
            faces = detector.detectMultiScale(gray, 1.3, 5)

            if len(faces) > 0:
                x, y, w, h = faces[0]

                # Use CSRT tracker (best)
                tracker = cv2.legacy.TrackerCSRT_create()
                tracker.init(frame, (x, y, w, h))
                tracking = True

        else:
            ok, box = tracker.update(frame)

            if not ok:
                tracking = False
                tracker = None
                continue

            x, y, w, h = [int(v) for v in box]

        # If tracking, process the face
        if tracking:
            # Clamp bounding box
            x = max(0, x)
            y = max(0, y)
            w = max(1, w)
            h = max(1, h)

            # Draw bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (100, 255, 180), 3)

            face = frame[y:y+h, x:x+w]

            # SAFETY CHECK
            if face.size == 0:
                continue

            face_gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            pil_img = Image.fromarray(face_gray).convert("L")
            img_tensor = transform(pil_img).unsqueeze(0).to(device)

            with torch.no_grad():
                output = model(img_tensor)
                probs = torch.softmax(output, dim=1)[0]

            pred_idx = torch.argmax(probs).item()
            label = EMOTION_MAP[pred_idx]
            confidence = float(probs[pred_idx])

            # Smooth emotion
            smooth_buffer.append(label)
            stable_label = max(set(smooth_buffer), key=smooth_buffer.count)

            # Timeline strip
            timeline.append(stable_label)

            # Emoji + label
            emoji = EMOJI_MAP.get(stable_label, "🙂")

            draw_text_with_shadow(
                frame,
                f"{emoji} {stable_label.upper()} ({confidence:.2f})",
                (x, y - 15),
                COLOR_MAP.get(stable_label, (255, 255, 255))
            )

            # Confidence bar
            bar_x = x
            bar_y = y + h + 10
            bar_w = int(w * confidence)

            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + w, bar_y + 12), (50, 50, 50), -1)
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + 12),
                          COLOR_MAP[stable_label], -1)

        # -------------------- Timeline Bar (Bottom Strip) --------------------
        timeline_y = frame.shape[0] - 20
        if len(timeline) > 0:
            block_w = int(frame.shape[1] / len(timeline))
            for i, lbl in enumerate(timeline):
                cv2.rectangle(
                    frame,
                    (i * block_w, timeline_y),
                    (i * block_w + block_w, timeline_y + 20),
                    COLOR_MAP.get(lbl, (255, 255, 255)),
                    -1
                )

        cv2.imshow("Premium Emotion AI", frame)

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

    run_webcam_premium(args.weights, use_resnet=args.use_resnet)
