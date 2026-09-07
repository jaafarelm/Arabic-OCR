"""
streamlit_app.py — Web demo for the Arabic character recognizer.

Two input paths:
  1. DRAW  — recommended. The canvas controls background (white) and ink
     (black), so input closely matches the training data.
  2. UPLOAD — for testing. Photos bring paper texture and uneven lighting,
     which after downscaling can leave a low-contrast blob the model reads
     poorly. Works best with a clean, high-contrast, tightly-cropped image.

Both paths feed the SAME predict() -> preprocess_image() pipeline.

Run with:  streamlit run app/streamlit_app.py
Requires:  pip install streamlit-drawable-canvas
"""

import sys
from pathlib import Path

import streamlit as st
from PIL import Image
from streamlit_drawable_canvas import st_canvas

# Make src/ importable.
SRC_DIR = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))

from predict import predict  # noqa: E402


st.set_page_config(page_title="Arabic Character Recognizer", page_icon="✒️")

st.title("✒️ Arabic Handwritten Character Recognizer")
st.caption("Draw a single Arabic character, or upload an image of one.")

with st.expander("About this model"):
    st.markdown(
        """
        A ResNet-style CNN trained on **70,910** handwritten samples across
        **46 base letter classes**, combining two datasets (HMBD and AHCD) so
        the model sees more than one writing style.

        **92.1% accuracy** on a held-out test set.

        Positional forms (initial/medial/final/isolated) are collapsed into one
        class per letter: in an isolated drawing those forms are visually
        near-identical, so distinguishing them is not a well-posed task.

        Most of the remaining error sits in letters that differ only by a
        diacritic dot (dal/zal/raa), which at 32×32 survives downscaling as
        only a few faint pixels.
        """
    )

st.divider()


def show_prediction(result):
    """Render a prediction result (shared by both input paths)."""
    st.metric(
        label="Predicted letter",
        value=result["name"],
        delta=f"{result['confidence']:.1%} confidence",
    )
    st.write("**Top 3 guesses**")
    for class_id, name, prob in result["top_k"]:
        st.write(f"{name}")
        st.progress(min(prob, 1.0))
        st.caption(f"{prob:.1%}")


tab_draw, tab_upload = st.tabs(["✏️ Draw", "📁 Upload"])

# --- Draw tab -------------------------------------------------------------
with tab_draw:
    col_draw, col_result = st.columns(2)

    with col_draw:
        st.caption("Tip: draw with a thin, clear stroke, centred in the box.")

        # White background + black stroke matches the training convention
        # (dark ink on light paper) at the SOURCE.
        canvas = st_canvas(
            fill_color="rgba(0,0,0,1)",
            stroke_width=8,              # thin: closer to the dataset's style
            stroke_color="#000000",
            background_color="#FFFFFF",
            height=280,
            width=280,
            drawing_mode="freedraw",
            key="canvas",
            return_image_data=True,      # required to read canvas.image_data
        )
        st.caption("Use the 🗑️ icon above the canvas to clear it.")

    with col_result:
        if canvas.image_data is not None:
            drawn = canvas.image_data                      # RGBA array
            has_ink = (drawn[:, :, :3] < 128).any()        # blank = all 255

            if has_ink:
                img = Image.fromarray(drawn.astype("uint8")).convert("RGB")
                show_prediction(predict(img, top_k=3))
            else:
                st.info("👈 Draw a character to see a prediction.")
        else:
            st.info("👈 Draw a character to see a prediction.")

# --- Upload tab -----------------------------------------------------------
with tab_upload:
    st.caption(
        "Upload an image of a single character. Best results: white paper, "
        "dark pen, good even lighting, cropped so the character fills the frame."
    )
    uploaded = st.file_uploader(
        "Choose an image", type=["png", "jpg", "jpeg", "bmp"]
    )

    if uploaded is not None:
        image = Image.open(uploaded)
        c1, c2 = st.columns(2)
        with c1:
            st.image(image, use_container_width=True)
        with c2:
            with st.spinner("Analyzing..."):
                result = predict(image, top_k=3)
            show_prediction(result)
    else:
        st.info("👆 Upload an image to get a prediction.")
