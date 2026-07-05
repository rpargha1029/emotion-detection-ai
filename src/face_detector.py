import cv2

try:
    from mtcnn import MTCNN
    MTCNN_AVAILABLE = True
except Exception:
    MTCNN_AVAILABLE = False


class HaarFaceDetector:
    def __init__(self):
        self.detector = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

    def detect(self, gray_frame):
        faces = self.detector.detectMultiScale(
            gray_frame,
            scaleFactor=1.1,
            minNeighbors=5
        )
        return faces


class MTCNNFaceDetector:
    def __init__(self):
        if not MTCNN_AVAILABLE:
            raise RuntimeError(
                "MTCNN is not installed. Install using: pip install mtcnn"
            )
        self.mtcnn = MTCNN()

    def detect(self, rgb_frame):
        detections = self.mtcnn.detect_faces(rgb_frame)
        faces = []

        for d in detections:
            x, y, w, h = d["box"]
            x = max(0, x)
            y = max(0, y)
            faces.append((x, y, w, h))

        return faces
