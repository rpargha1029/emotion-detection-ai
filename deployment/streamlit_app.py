import sys
import os

# Ensure project root is visible
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import streamlit as st
from PIL import Image
import tempfile
from src.predict import predict

WEIGHTS = "experiments/checkpoints/best_model.pth"

st.title("Emotion Detection App")
st.write("Upload an image to detect emotion.")

uploaded = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    if st.button("Predict Emotion"):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp:
            img.save(temp.name)
            label, conf = predict(temp.name, WEIGHTS)

        st.success(f"Prediction: **{label}** ({conf:.2f})")
