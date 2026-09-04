"""
data.py — Data loading and splitting for the Arabic handwritten character recognizer.

Responsibilities (and ONLY these — keep this file focused):
  1. Load the flattened image pixels and their integer labels from CSV.
  2. Reshape the flat pixel rows back into 32x32 single-channel images.
  3. Normalize pixel values to the [0, 1] range.
  4. Split into train / validation / test sets, honestly.

Design notes for reviewers:
  - The raw CSVs have NO header row, so we load with header=None. Loading with a
    header would silently consume the first real image row as column names.
  - images.csv and labels.csv are row-aligned (row i in one corresponds to row i
    in the other). We never break that alignment.
  - We use a three-way split (train/val/test). The test set is touched ONCE, at
    final evaluation, so it stays an honest measure of unseen performance.
  - KNOWN LIMITATION: HMBD as distributed carries no writer identifiers, so a
    writer-independent split is not possible. Some writer leakage may remain.
    This is documented rather than hidden.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


# --- Constants -------------------------------------------------------------

# Each image is a flattened 32x32 grayscale square (1024 pixel values per row).
IMG_SIZE = 32
N_PIXELS = IMG_SIZE * IMG_SIZE  # 1024

# Where the data lives, relative to the project root.
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
IMAGES_CSV = DATA_DIR / "images.csv"
LABELS_CSV = DATA_DIR / "labels.csv"

# Fixed seed so the split is reproducible run-to-run.
RANDOM_SEED = 42


# --- Loading ---------------------------------------------------------------

def load_raw():
    """Load pixel data and labels from CSV as aligned NumPy arrays.

    Returns
    -------
    X : np.ndarray, shape (n_samples, 1024), dtype float
        Raw flattened pixel values (0-255), not yet normalized or reshaped.
    y : np.ndarray, shape (n_samples,), dtype int
        Integer class label for each image.
    """
    # header=None is REQUIRED: the CSVs start straight at data, no column names.
    X = pd.read_csv(IMAGES_CSV, header=None).values
    y = pd.read_csv(LABELS_CSV, header=None).values.ravel()  # flatten to 1-D

    # Sanity check: the two files must stay row-aligned. If they ever drift,
    # every label would point at the wrong image — so we fail loudly here.
    assert X.shape[0] == y.shape[0], (
        f"Row mismatch: {X.shape[0]} images vs {y.shape[0]} labels"
    )

    return X, y


# --- Shaping and scaling ---------------------------------------------------

def prepare_images(X):
    """Turn flat pixel rows into normalized 32x32x1 image tensors.

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, 1024)

    Returns
    -------
    np.ndarray, shape (n_samples, 32, 32, 1), values in [0, 1]
    """
    # Scale 0-255 -> 0-1. Neural nets train more stably on small inputs.
    X = X.astype("float32") / 255.0

    # Restore the 2-D spatial shape the CNN needs. The trailing 1 is the single
    # grayscale channel (Keras Conv2D expects a channel dimension).
    X = X.reshape(-1, IMG_SIZE, IMG_SIZE, 1)

    return X


# --- Splitting -------------------------------------------------------------

def split_data(X, y, val_size=0.10, test_size=0.10):
    """Three-way stratified split into train / validation / test.

    Stratify=y keeps each class's proportion roughly equal across all three
    splits, which matters here because the classes are imbalanced.

    The test set is carved out first and then left alone until final
    evaluation, so it remains an honest estimate of unseen performance.
    """
    # First split off the test set (its fraction of the whole).
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=RANDOM_SEED,
    )

    # From what's left, split off the validation set. We rescale val_size so
    # it still ends up as the intended fraction of the ORIGINAL dataset.
    val_relative = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=val_relative,
        stratify=y_temp,
        random_state=RANDOM_SEED,
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


# --- Public entry point ----------------------------------------------------

def get_datasets():
    """Load, prepare, and split the data in one call.

    Returns six arrays: X_train, X_val, X_test, y_train, y_val, y_test.
    This is the function train.py imports.
    """
    X, y = load_raw()
    X = prepare_images(X)
    return split_data(X, y)


# Quick manual check: run `python src/data.py` to verify shapes look right.
if __name__ == "__main__":
    X_train, X_val, X_test, y_train, y_val, y_test = get_datasets()
    print(f"Train: {X_train.shape}, {y_train.shape}")
    print(f"Val:   {X_val.shape}, {y_val.shape}")
    print(f"Test:  {X_test.shape}, {y_test.shape}")
    print(f"Classes: {len(np.unique(y_train))}")
    print(f"Pixel range: [{X_train.min():.2f}, {X_train.max():.2f}]")