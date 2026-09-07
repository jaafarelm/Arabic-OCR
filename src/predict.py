"""
predict.py — Load the trained PyTorch model and predict a single image.

Bridge between the trained model and the Streamlit app:
    1. preprocess the image (shared preprocess_image, unchanged)
    2. run the model on GPU (or CPU if unavailable)
    3. return the predicted class id, its readable NAME, and confidence

PyTorch notes:
    - The model is a class, so we rebuild the architecture then load the saved
      state_dict (weights) into it.
    - model.eval() puts BatchNorm/Dropout into inference mode — essential, or
      predictions will be wrong.
    - The model outputs raw logits, so we apply softmax here to get
      probabilities.
"""

from pathlib import Path
from label_map import num_classes

import numpy as np
import torch
import torch.nn.functional as F

from preprocess import preprocess_image
from label_map import name_for
from model import build_model


MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "best_model.pt"
NUM_CLASSES = num_classes()          # must match what train.py used

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_model = None


def get_model():
    """Build the architecture and load the trained weights once."""
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"No trained model at {MODEL_PATH}. Run train.py first."
            )
        m = build_model(num_classes=NUM_CLASSES)
        m.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        m.to(DEVICE)
        m.eval()                      # inference mode (BatchNorm/Dropout off)
        _model = m
    return _model


def predict(img, top_k=3):
    """Predict the class of a single image.

    Returns dict with 'class_id', 'name', 'confidence', and 'top_k'
    (a list of (class_id, name, probability) tuples, best first).
    """
    model = get_model()

    # Shared preprocessing -> (1, 32, 32, 1) NHWC numpy, values in [0, 1].
    x = preprocess_image(img)

    # NHWC -> NCHW tensor on the right device.
    x = torch.from_numpy(x).float().permute(0, 3, 1, 2).to(DEVICE)

    with torch.no_grad():                     # no gradients needed for inference
        logits = model(x)
        probs = F.softmax(logits, dim=1)[0].cpu().numpy()

    class_id = int(np.argmax(probs))
    confidence = float(probs[class_id])

    top_idx = np.argsort(probs)[::-1][:top_k]
    top = [(int(i), name_for(int(i)), float(probs[i])) for i in top_idx]

    return {
        "class_id": class_id,
        "name": name_for(class_id),
        "confidence": confidence,
        "top_k": top,
    }


if __name__ == "__main__":
    from PIL import Image
    dummy = Image.new("L", (100, 100), color=255)
    result = predict(dummy)
    print(f"Device: {DEVICE}")
    print(f"Predicted: {result['name']} (class {result['class_id']})")
    print(f"Confidence: {result['confidence']:.2%}")
    print(f"Top-3: {result['top_k']}")
