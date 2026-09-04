"""
streamlit_app.py — Web demo for the Arabic character recognizer (v0.1).

Upload an image of a single handwritten Arabic character; the app preprocesses
it (via the SAME shared pipeline used in training) and shows the model's top
predictions.

Run with:  streamlit run app/streamlit_app.py

Design notes for reviewers:
    - The app is a thin UI wrapper around predict.py — no ML logic here.
    - Predictions are shown as class IDs + confidence, honestly. Human-readable
      letter names are a v0.2 addition (the shipped label mapping is unreliable).
    - v0.1 KNOWN LIMITATION: the model expects a tightly-cropped character that
      fills the frame; upload a cropped image for best results.
"""

import sys
from pathlib import Path

import streamlit as st
from PIL import Image

# Make the src/ modules importable when running from the project root.
SRC_DIR = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))

from predict import predict  # noqa: E402  (import after sys.path tweak)


# --- Page setup ------------------------------------------------------------
st.set_page_config(page_title="Arabic Character Recognizer", page_icon="✒️")

st.title("✒️ Arabic Handwritten Character Recognizer")
st.caption(
    "Upload an image of a single handwritten Arabic character. "
    "For best results, crop the image so the character fills most of the frame."
)

# Honest disclaimer — sets expectations for a v0.1 baseline.
st.info(
    "This is a v0.1 baseline model. Accuracy is limited and predictions are "
    "shown as class IDs (readable letter names are planned for v0.2)."
)

# --- File uploader ---------------------------------------------------------
uploaded = st.file_uploader(
    "Choose an image", type=["png", "jpg", "jpeg", "bmp"]
)

if uploaded is not None:
    # Show what was uploaded.
    image = Image.open(uploaded)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Your image")
        st.image(image, use_container_width=True)

    # Run the prediction (predict.py handles preprocessing internally).
    with col2:
        st.subheader("Prediction")
        with st.spinner("Analyzing..."):
            result = predict(image, top_k=3)

        # Headline result.
        st.metric(
            label="Predicted class",
            value=f"Class {result['class_id']}",
            delta=f"{result['confidence']:.1%} confidence",
        )

        # Top-3 guesses as labelled progress bars.
        st.write("**Top 3 guesses:**")
        for class_id, prob in result["top_k"]:
            st.write(f"Class {class_id}")
            st.progress(min(prob, 1.0))
            st.caption(f"{prob:.1%}")
else:
    st.write("👆 Upload an image to get a prediction.")