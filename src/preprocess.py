import numpy as np
from PIL import Image, ImageOps

IMG_SIZE = 32


def preprocess_image(img, invert_if_needed=True):

    if isinstance(img, np.ndarray):
        img = Image.fromarray(img)

    img = img.convert("L")

    img = img.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)

    arr = np.asarray(img, dtype="float32") / 255.0

    if invert_if_needed and arr.mean() < 0.5:
        arr = 1.0 - arr

    arr = arr.reshape(1, IMG_SIZE, IMG_SIZE, 1)

    return arr


# Quick manual check: feed it a dummy image and confirm the output shape/range.
if __name__ == "__main__":
    dummy = Image.new("RGB", (100, 120), color=(255, 255, 255))
    out = preprocess_image(dummy)
    print(f"Output shape: {out.shape}")          # expect (1, 32, 32, 1)
    print(f"Pixel range: [{out.min():.2f}, {out.max():.2f}]")
    
