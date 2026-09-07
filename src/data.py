"""
data.py — Load, collapse, MERGE two datasets, and split.

Datasets combined:
  1. HMBD  (data/images.csv, data/labels.csv)
     53,184 samples, 115 positional-form classes, black ink on white.
  2. AHCD  (data/ahcd_images.csv, data/ahcd_labels.csv)
     16,800 samples, 28 base-letter classes, WHITE ink on BLACK, and stored
     TRANSPOSED (letters appear rotated until you transpose each 32x32 image).

WHY MERGE:
  The model was overfitting to HMBD's single writing style, so it failed on
  any other style (canvas drawings, photos). AHCD is a different collection
  with different writers and stroke styles. Training on both forces the model
  to learn what makes a letter that letter, rather than memorising one
  dataset's look.

  The 115 -> 46 base-letter collapse is what makes the merge possible: AHCD
  has no positional forms, so the two label systems only line up once HMBD's
  four forms per letter are collapsed into one.

AHCD FIXES APPLIED (both are essential — verified visually):
  - TRANSPOSE each image (AHCD is stored row/column swapped).
  - INVERT pixels (255 - x) so it matches HMBD's black-on-white convention.
  - Remap AHCD labels 1..28 -> the matching base-class ids.

Design notes for reviewers:
  - CSVs have NO header row; loaded with header=None throughout.
  - Image/label alignment is asserted, never assumed.
  - Three-way stratified split; the test set is touched ONCE, at final eval.
  - KNOWN LIMITATION: neither dataset carries writer identifiers, so a
    writer-independent split is not possible; some writer leakage may remain.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from label_map import OLD_TO_BASE, AHCD_TO_BASE, NUM_BASE_CLASSES


IMG_SIZE = 32
DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# HMBD
IMAGES_CSV = DATA_DIR / "images.csv"
LABELS_CSV = DATA_DIR / "labels.csv"

# AHCD (optional — merged only if the files are present)
AHCD_IMAGES = DATA_DIR / "ahcd_images.csv"
AHCD_LABELS = DATA_DIR / "ahcd_labels.csv"

RANDOM_SEED = 42


# --- HMBD ------------------------------------------------------------------

def load_hmbd():
    """Load HMBD, drop the junk class, collapse 115 labels -> 46 base classes.

    Returns X (N, 1024) uint8-ish floats and y (N,) collapsed labels.
    """
    X = pd.read_csv(IMAGES_CSV, header=None).values
    y = pd.read_csv(LABELS_CSV, header=None).values.ravel()
    assert X.shape[0] == y.shape[0], "HMBD image/label row mismatch"

    # Keep only labels that have a base mapping. The junk class (24, the
    # ingested ".ipynb_checkpoints" folder) has none, so its rows are dropped
    # from BOTH arrays, preserving alignment.
    keep = np.array([lbl in OLD_TO_BASE for lbl in y])
    dropped = int((~keep).sum())
    X, y = X[keep], y[keep]
    y = np.array([OLD_TO_BASE[l] for l in y], dtype=np.int64)

    print(f"HMBD: {X.shape[0]} samples "
          f"({dropped} junk rows dropped), {len(np.unique(y))} base classes")
    return X, y


# --- AHCD ------------------------------------------------------------------

def load_ahcd():
    """Load AHCD, fix orientation/inversion, remap labels to base classes.

    Returns (X, y) or (None, None) if the files are not present.
    """
    if not (AHCD_IMAGES.exists() and AHCD_LABELS.exists()):
        print("AHCD: files not found, skipping merge")
        return None, None

    X = pd.read_csv(AHCD_IMAGES, header=None).values
    y = pd.read_csv(AHCD_LABELS, header=None).values.ravel()
    assert X.shape[0] == y.shape[0], "AHCD image/label row mismatch"

    # --- FIX 1: transpose. AHCD stores each image row/column swapped, so
    # letters appear rotated. Reshape to 32x32, transpose, flatten back.
    X = X.reshape(-1, IMG_SIZE, IMG_SIZE).transpose(0, 2, 1).reshape(-1, IMG_SIZE * IMG_SIZE)

    # --- FIX 2: invert. AHCD is white ink on black; HMBD is black on white.
    X = 255 - X

    # --- FIX 3: remap labels 1..28 -> base class ids.
    keep = np.array([lbl in AHCD_TO_BASE for lbl in y])
    X, y = X[keep], y[keep]
    y = np.array([AHCD_TO_BASE[l] for l in y], dtype=np.int64)

    print(f"AHCD: {X.shape[0]} samples (transposed + inverted), "
          f"{len(np.unique(y))} base classes")
    return X, y


# --- Combine, scale, split -------------------------------------------------

def prepare_images(X):
    """Flat pixel rows -> normalized (N, 32, 32, 1) float tensors in [0, 1]."""
    X = X.astype("float32") / 255.0
    return X.reshape(-1, IMG_SIZE, IMG_SIZE, 1)


def split_data(X, y, val_size=0.10, test_size=0.10):
    """Three-way stratified split: train / val / test.

    Test is carved out first and left untouched until final evaluation.
    val_size is rescaled so it remains the intended fraction of the whole set.
    """
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=RANDOM_SEED
    )
    val_relative = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_relative, stratify=y_temp,
        random_state=RANDOM_SEED
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def get_datasets():
    """Load both datasets, merge, prepare, and split. Returns six arrays."""
    X_h, y_h = load_hmbd()
    X_a, y_a = load_ahcd()

    if X_a is not None:
        X = np.concatenate([X_h, X_a], axis=0)
        y = np.concatenate([y_h, y_a], axis=0)
        print(f"MERGED: {X.shape[0]} total samples, "
              f"{len(np.unique(y))} classes")
    else:
        X, y = X_h, y_h

    X = prepare_images(X)
    return split_data(X, y)


if __name__ == "__main__":
    X_train, X_val, X_test, y_train, y_val, y_test = get_datasets()
    print()
    print(f"Train: {X_train.shape}, {y_train.shape}")
    print(f"Val:   {X_val.shape}, {y_val.shape}")
    print(f"Test:  {X_test.shape}, {y_test.shape}")
    print(f"Classes: {len(np.unique(y_train))} (expected {NUM_BASE_CLASSES})")
    print(f"Pixel range: [{X_train.min():.2f}, {X_train.max():.2f}]")

    _, counts = np.unique(y_train, return_counts=True)
    print(f"Train samples per class: min {counts.min()}, "
          f"max {counts.max()}, avg {counts.mean():.0f}")
