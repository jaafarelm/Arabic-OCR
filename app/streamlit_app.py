"""
streamlit_app.py — Web demo for the Arabic character recognizer (v0.2).

Two ways to input a character:
  1. DRAW it on a canvas (recommended) — you control background & ink, so the
     input is crisp black-on-white, matching the training data. This avoids the
     contrast/texture problems that photos suffer from.
  2. UPLOAD an image — works best with a clean, high-contrast photo.

Both paths feed the SAME predict() -> preprocess_image() pipeline.

Run with:  streamlit run app/streamlit_app.py
Requires:  pip install "streamlit-drawable-canvas[image]"
"""

import sys
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image
from streamlit_drawable_canvas import st_canvas

# Make src/ importable.
SRC_DIR = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))

from predict import predict  # noqa: E402


st.set_page_config(page_title="Arabic Character Recognizer", page_icon="✒️")
st.title("✒️ Arabic Handwritten Character Recognizer")
st.info(
    "v0.2 baseline. Predictions are shown as class IDs (readable letter names "
    "are planned). Draw a single character below, or upload an image."
)


def show_prediction(result):
    """Render a prediction result (shared by both input paths)."""
    st.metric(
        label="Predicted letter",
        value=result["name"],
        delta=f"{result['confidence']:.1%} confidence",
    )
    st.write("**Top 3 guesses:**")
    for class_id, name, prob in result["top_k"]:   # note: now 3 values
        st.write(f"**{name}**")
        st.progress(min(prob, 1.0))
        st.caption(f"{prob:.1%}")

tab_draw, tab_upload = st.tabs(["✏️ Draw", "📁 Upload"])

# --- Draw tab -------------------------------------------------------------
with tab_draw:
    st.caption("Draw a single Arabic character, then it predicts automatically.")

    col1, col2 = st.columns(2)
    with col1:
        # White canvas, black stroke: matches the training convention
        # (dark ink on light background) at the SOURCE.
        canvas = st_canvas(
            fill_color="rgba(0,0,0,1)",
            stroke_width=6,             # thick, so it resembles training strokes
            stroke_color="#000000",      # black ink
            background_color="#FFFFFF",  # white background
            height=280,
            width=280,
            drawing_mode="freedraw",
            key="canvas",
            return_image_data=True,      # required so we can read canvas.image_data
        )

    with col2:
        if canvas.image_data is not None:
            drawn = canvas.image_data  # RGBA numpy array
            # A pixel is "ink" if it's dark. On a white canvas, blank = all 255.
            has_ink = (drawn[:, :, :3] < 128).any()

            if has_ink:
                # Convert RGBA canvas -> RGB PIL image and predict.
                img = Image.fromarray(drawn.astype("uint8")).convert("RGB")
                result = predict(img, top_k=3)
                show_prediction(result)
            else:
                st.write("👈 Draw a character to see a prediction.")
        else:
            st.write("👈 Draw a character to see a prediction.")

# --- Upload tab -----------------------------------------------------------
with tab_upload:
    st.caption(
        "Upload a clean, high-contrast image of a single character "
        "(white background, dark ink works best)."
    )
    uploaded = st.file_uploader("Choose an image", type=["png", "jpg", "jpeg", "bmp"])

    if uploaded is not None:
        image = Image.open(uploaded)
        c1, c2 = st.columns(2)
        with c1:
            st.image(image, use_container_width=True)
        with c2:
            with st.spinner("Analyzing..."):
                result = predict(image, top_k=3)
            show_prediction(result)