"""
predict.py — Load the trained model and predict the class of a single image.

This is the bridge between the trained model and the Streamlit app. Given any
user image, it:
    1. preprocesses it (SAME shared function training used) via preprocess.py
    2. runs the model
    3. returns the predicted class and a confidence score

Design notes for reviewers:
    - Uses the shared preprocess_image() so inference matches training exactly.
    - Returns the integer class id plus confidence. Human-readable letter names
      are a v0.2 concern (the shipped label_mapping.csv is unreliable; correct
      names should be regenerated from the Dataset/ folder order).
    - The model is loaded once and cached, so repeated predictions are fast.
"""

from pathlib import Path

import numpy as np
from tensorflow.keras.models import load_model

from preprocess import preprocess_image


MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "best_model.keras"

# Module-level cache so we don't reload the model on every call.
_model = None


def get_model():
    """Load the trained model once and reuse it."""
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"No trained model at {MODEL_PATH}. Run train.py first."
            )
        _model = load_model(MODEL_PATH)
    return _model


def predict(img, top_k=3):
    """Predict the class of a single image.

    Parameters
    ----------
    img : PIL.Image.Image or np.ndarray
        The raw user image (any size / mode).
    top_k : int
        How many of the top guesses to return.

    Returns
    -------
    dict with:
        'class_id'   : int, the most likely class
        'confidence' : float, probability of that class (0-1)
        'top_k'      : list of (class_id, probability) tuples, best first
    """
    model = get_model()

    # Shared preprocessing -> (1, 32, 32, 1) tensor, identical to training.
    x = preprocess_image(img)

    # Model returns a (1, num_classes) array of probabilities (softmax output).
    probs = model.predict(x, verbose=0)[0]

    # Best class = highest probability.
    class_id = int(np.argmax(probs))
    confidence = float(probs[class_id])

    # Top-k guesses, sorted best-first (useful for showing runner-ups).
    top_idx = np.argsort(probs)[::-1][:top_k]
    top = [(int(i), float(probs[i])) for i in top_idx]

    return {
        "class_id": class_id,
        "confidence": confidence,
        "top_k": top,
    }


# Quick manual check: predict on a blank dummy image (just proves it runs).
if __name__ == "__main__":
    from PIL import Image
    dummy = Image.new("L", (100, 100), color=255)
    result = predict(dummy)
    print(f"Predicted class: {result['class_id']}")
    print(f"Confidence: {result['confidence']:.2%}")
    print(f"Top-3: {result['top_k']}")