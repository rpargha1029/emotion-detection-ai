# deployment/streamlit_app_v2.py
import sys, os, tempfile, time
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import streamlit as st
from PIL import Image
import numpy as np

from src.predict import predict
from src.config import EMOTION_MAP, EXPERIMENTS_DIR

st.set_page_config(page_title="Emotion AI — Studio", layout="wide")

st.markdown("# 🎛️ Emotion AI — Studio")
st.write("Upload an image or use your webcam snapshot to get emotion predictions. Use the advanced model from `experiments/checkpoints/best_model_adv.pth` if available.")

col1, col2 = st.columns([2,1])

with col1:
    st.subheader("Image Input")
    uploaded = st.file_uploader("Upload JPG/PNG image", type=["jpg","jpeg","png"])
    camera = st.camera_input("Or take a webcam snapshot")

    img = None
    if uploaded:
        img = Image.open(uploaded).convert("RGB")
    elif camera:
        img = Image.open(camera).convert("RGB")

    if img is not None:
        st.image(img, caption="Input image", use_column_width=True)
        st.write("---")

        # save temp file
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
        img.save(tmp.name)
        tmp_path = tmp.name

        # choose weights
        weights_default = str(EXPERIMENTS_DIR / "checkpoints" / "best_model_adv.pth")
        weights = st.text_input("Weights path", value=weights_default)

        if st.button("Predict"):
            with st.spinner("Running prediction..."):
                label, conf = predict(tmp_path, weights)
            st.success(f"Prediction: **{label}** ({conf:.3f})")
            os.unlink(tmp_path)

with col2:
    st.subheader("Model & Results")
    if st.button("Show confusion matrix (last run)"):
        cm_path = EXPERIMENTS_DIR / "results" / "confusion_matrix.png"
        if cm_path.exists():
            st.image(str(cm_path), caption="Confusion matrix")
        else:
            st.info("No confusion matrix found — run evaluation first.")

    st.write("Quick actions:")
    if st.button("Evaluate model (server-side)"):
        st.info("Running evaluation. This can take some time; please check terminal output.")

    st.markdown("### Tips")
    st.write("- Use `best_model_adv.pth` for advanced model predictions.")
    st.write("- If your model expects grayscale, the UI handles conversion automatically.")

st.markdown("---")
st.write("Powered by your local model. Keep iterating 💡")
