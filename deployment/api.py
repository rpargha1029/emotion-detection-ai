import sys
import os

# Ensure project root is visible
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from PIL import Image
import io

from src.predict import predict

app = FastAPI()

DEFAULT_WEIGHTS = "experiments/checkpoints/best_model.pth"


@app.post("/predict")
async def predict_api(file: UploadFile = File(...)):
    # Read image from request
    contents = await file.read()
    img = Image.open(io.BytesIO(contents)).convert("RGB")

    # Save temporarily because predict() expects a file path
    temp_path = "temp_input.jpg"
    img.save(temp_path)

    label, confidence = predict(temp_path, DEFAULT_WEIGHTS)

    os.remove(temp_path)

    return JSONResponse({
        "emotion": label,
        "confidence": confidence
    })


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
