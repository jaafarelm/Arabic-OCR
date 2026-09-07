"""
streamlit_app.py — Web demo for the Arabic character recognizer.

DRAW-ONLY by design. Photo upload was removed deliberately: on a drawing
canvas we control the background (pure white) and the ink (pure black), so the
input closely matches the training data. Phone photos bring paper texture,
uneven lighting and low contrast, which after downscaling leave the model a
washed-out grey blob it cannot read. Rather than ship a path that fails, the
demo keeps the one that works. (Photo support needs adaptive thresholding —
noted as future work.)

Run with:  streamlit run app/streamlit_app.py
Requires:  pip install "streamlit-drawable-canvas[image]"
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
st.caption(
    "Draw a single Arabic character in the box. The model predicts it live."
)

with st.expander("About this model"):
    st.markdown(
        """
        A ResNet-style CNN trained on **69,508** handwritten samples across
        **46 base letter classes**, combining two datasets (HMBD and AHCD) so
        the model sees more than one writing style.

        **~90% accuracy** on a held-out test set.

        Positional forms (initial/medial/final/isolated) are collapsed into one
        class per letter: in an isolated drawing those forms are visually
        near-identical, so distinguishing them is not a well-posed task.
        """
    )

st.divider()

col_draw, col_result = st.columns([1, 1])

with col_draw:
    st.subheader("Draw here")
    st.caption("Tip: draw with a thin, clear stroke, centred in the box.")

    # White background + black stroke matches the training convention
    # (dark ink on light paper) at the SOURCE, which is exactly why the
    # canvas works where photos do not.
    canvas = st_canvas(
        fill_color="rgba(0,0,0,1)",
        stroke_width=8,              # thin: closer to the dataset's stroke style
        stroke_color="#000000",      # black ink
        background_color="#FFFFFF",  # white background
        height=280,
        width=280,
        drawing_mode="freedraw",
        key="canvas",
        return_image_data=True,      # required to read canvas.image_data
    )

    st.caption("Use the 🗑️ icon above the canvas to clear it.")

with col_result:
    st.subheader("Prediction")

    if canvas.image_data is not None:
        drawn = canvas.image_data                       # RGBA array
        has_ink = (drawn[:, :, :3] < 128).any()         # blank canvas = all 255

        if has_ink:
            img = Image.fromarray(drawn.astype("uint8")).convert("RGB")
            result = predict(img, top_k=3)

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
        else:
            st.info("👈 Draw a character to see a prediction.")
    else:
        st.info("👈 Draw a character to see a prediction.")
