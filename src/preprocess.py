"""
preprocess.py — Shared image preprocessing for training AND inference. (v0.2)

Still the SPINE of the project: training and inference must transform pixels
identically, or a good model fails on real input.

v0.2 change — CROP TO INK:
    v0.1 resized the WHOLE image to 32x32. If a user uploaded a small letter on
    a big background (a phone photo, a canvas drawing), the letter shrank to a
    few pixels and the model saw mostly background -> garbage predictions.
    v0.2 finds the letter's bounding box, crops to it, centers it with a margin,
    THEN resizes -> the letter fills the frame like the training data does.

Also fixes the inversion heuristic: orientation is now decided from the CORNER
pixels (reliable background) instead of the whole-image mean (fooled by letter
size).

Target format (must match data.py's prepare_images output):
    grayscale, 32x32, dark character on light background, values in [0, 1].
"""

import numpy as np
from PIL import Image

IMG_SIZE = 32
# When cropping, leave this fraction of padding around the letter so it isn't
# jammed against the edges (training letters have a little breathing room).
PAD_FRAC = 0.15
# A pixel is considered "ink" if it's darker than this (on a 0-1 scale, after
# ensuring dark-on-light orientation).
INK_THRESHOLD = 0.5


def _ensure_dark_on_light(arr):
    """Flip the image if it's light-on-dark, using the CORNER pixels.

    The four corners are almost always background. If they're dark, the image
    is inverted relative to training, so we flip it. (More reliable than the
    old whole-image mean, which a large letter could fool.)
    """
    corners = [arr[0, 0], arr[0, -1], arr[-1, 0], arr[-1, -1]]
    if np.mean(corners) < 0.5:      # corners dark => inverted background
        arr = 1.0 - arr
    return arr


def _crop_to_ink(arr):
    """Crop the image down to the bounding box of the letter's ink.

    Returns the cropped array, or the original if no ink is found (blank image).
    """
    # "Ink" = pixels darker than the threshold (letter is dark on light bg).
    ink = arr < INK_THRESHOLD
    if not ink.any():
        return arr  # nothing to crop (blank) — return as-is

    # Find rows/cols that contain any ink, take the bounding box.
    rows = np.any(ink, axis=1)
    cols = np.any(ink, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    return arr[rmin:rmax + 1, cmin:cmax + 1]


def _pad_to_square(arr):
    """Pad the (usually non-square) crop into a square with a light-background
    margin, so the letter ends up centered and not touching the edges.
    """
    h, w = arr.shape
    side = max(h, w)
    pad = int(side * PAD_FRAC)
    canvas_size = side + 2 * pad

    # Start with a light (1.0) canvas — matches the training background.
    canvas = np.ones((canvas_size, canvas_size), dtype="float32")

    # Center the crop on the canvas.
    top = (canvas_size - h) // 2
    left = (canvas_size - w) // 2
    canvas[top:top + h, left:left + w] = arr
    return canvas


def preprocess_image(img):
    """Turn an arbitrary user image into a model-ready 32x32x1 tensor.

    Returns np.ndarray shape (1, 32, 32, 1), values in [0, 1].
    """
    # Accept a NumPy array or a PIL image.
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img)

    # Grayscale, to array, scale to [0, 1].
    img = img.convert("L")
    arr = np.asarray(img, dtype="float32") / 255.0

    # Fix orientation, then crop to the letter, then center in a square.
    arr = _ensure_dark_on_light(arr)
    arr = _crop_to_ink(arr)
    arr = _pad_to_square(arr)

    # Now resize the centered letter to the model's input size.
    pil = Image.fromarray((arr * 255).astype("uint8"))
    pil = pil.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
    arr = np.asarray(pil, dtype="float32") / 255.0

    # Batch of one: (1, 32, 32, 1).
    return arr.reshape(1, IMG_SIZE, IMG_SIZE, 1)


if __name__ == "__main__":
    # Quick check: a small letter on a big background should still work now.
    canvas = Image.new("L", (400, 400), color=255)   # big white background
    # draw a small dark blob near one corner
    px = canvas.load()
    for y in range(60, 120):
        for x in range(60, 90):
            px[x, y] = 0
    out = preprocess_image(canvas)
    print(f"Output shape: {out.shape}")           # (1, 32, 32, 1)
    print(f"Pixel range: [{out.min():.2f}, {out.max():.2f}]")
    # The letter should now fill much of the 32x32 frame, not be a tiny dot.
    print(f"Fraction of dark pixels: {(out < 0.5).mean():.2%}")