"""
data.py — Load the cached HMBD arrays, merge AHCD, and split.

DATA SOURCES:
  1. HMBD  — data/hmbd_cache.npz, produced by build_dataset.py from the
     ORIGINAL image folders. Labels come from folder names, so they are
     correct by construction. (The shipped images.csv/labels.csv are NOT used:
     their label_mapping.csv was corrupt, which silently trained the model on
     wrong letter names.)
  2. AHCD  — data/ahcd_images.csv + data/ahcd_labels.csv. Correctly labelled,
     but stored TRANSPOSED and with WHITE ink on BLACK, so both are fixed here.

Both are collapsed to the same 46 base-letter classes, which is what makes the
merge possible: AHCD has no positional forms, so the label systems only align
once HMBD's four forms per letter are merged into one.

Design notes for reviewers:
  - Class names are read from the cache, not a hand-maintained file, so the
    mapping cannot drift out of sync with the data again.
  - Three-way stratified split; the test set is evaluated ONCE.
  - KNOWN LIMITATION: neither dataset carries writer identifiers, so a
    writer-independent split is not possible; some writer leakage may remain.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


IMG_SIZE = 32
DATA_DIR = Path(__file__).resolve().parent.parent / "data"

HMBD_CACHE = DATA_DIR / "hmbd_cache.npz"
AHCD_IMAGES = DATA_DIR / "ahcd_images.csv"
AHCD_LABELS = DATA_DIR / "ahcd_labels.csv"

RANDOM_SEED = 42

# AHCD labels run 1..28 in standard Arabic alphabet order (verified by
# rendering: labels 2/3/4 show one/two/three dots = beh/teh/theh).
# Mapped here to the matching HMBD base-letter NAME; the numeric id is looked
# up from the cache's own name list, so it stays correct automatically.
AHCD_TO_NAME = {
    1: "Alf",  2: "Baa",  3: "Taa",   4: "Thaa", 5: "Gem",  6: "Ha",
    7: "Khaa", 8: "Dal",  9: "Zal",  10: "Raa", 11: "Zin", 12: "Sin",
    13: "Shen", 14: "Saad", 15: "Daad", 16: "Tah", 17: "Zah", 18: "Ain",
    19: "Gen", 20: "Faa", 21: "Qaf", 22: "Kaf", 23: "Lam", 24: "Mem",
    25: "Non", 26: "Haa", 27: "Waw", 28: "Yaa",
}


def load_hmbd():
    """Load the cached HMBD arrays and the class-name list."""
    if not HMBD_CACHE.exists():
        raise FileNotFoundError(
            f"{HMBD_CACHE} not found. Run:  python src/build_dataset.py"
        )
    d = np.load(HMBD_CACHE, allow_pickle=False)
    X, y, names = d["X"], d["y"], [str(n) for n in d["names"]]
    print(f"HMBD: {X.shape[0]} samples, {len(np.unique(y))} base classes")
    return X, y, names


def load_ahcd(names):
    """Load AHCD, fix orientation/inversion, map labels into HMBD's class ids.

    Returns (None, None) if the files are absent, so training still works with
    HMBD alone.
    """
    if not (AHCD_IMAGES.exists() and AHCD_LABELS.exists()):
        print("AHCD: files not found, skipping merge")
        return None, None

    X = pd.read_csv(AHCD_IMAGES, header=None).values
    y = pd.read_csv(AHCD_LABELS, header=None).values.ravel()
    assert X.shape[0] == y.shape[0], "AHCD image/label row mismatch"

    # FIX 1: AHCD stores each image row/column swapped, so letters appear
    # rotated until transposed.
    X = (X.reshape(-1, IMG_SIZE, IMG_SIZE)
           .transpose(0, 2, 1)
           .reshape(-1, IMG_SIZE * IMG_SIZE))

    # FIX 2: AHCD is white ink on black; HMBD is black ink on white.
    X = 255 - X

    # FIX 3: map AHCD's 1..28 to HMBD's base-class ids VIA NAMES, so the
    # numbering always follows the cache rather than a hardcoded table.
    name_to_id = {n: i for i, n in enumerate(names)}
    lut = {}
    for ahcd_label, nm in AHCD_TO_NAME.items():
        if nm in name_to_id:
            lut[ahcd_label] = name_to_id[nm]
        else:
            print(f"  WARNING: AHCD name '{nm}' not in HMBD classes, dropping")

    keep = np.array([lbl in lut for lbl in y])
    X, y = X[keep], y[keep]
    y = np.array([lut[l] for l in y], dtype=np.int64)

    print(f"AHCD: {X.shape[0]} samples (transposed + inverted), "
          f"{len(np.unique(y))} classes")
    return X, y


def prepare_images(X):
    """Flat pixel rows -> normalized (N, 32, 32, 1) float tensors in [0, 1]."""
    X = X.astype("float32") / 255.0
    return X.reshape(-1, IMG_SIZE, IMG_SIZE, 1)


def split_data(X, y, val_size=0.10, test_size=0.10):
    """Three-way stratified split. Test is carved out first and left alone."""
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=RANDOM_SEED
    )
    val_relative = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_relative, stratify=y_temp,
        random_state=RANDOM_SEED
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def get_class_names():
    """Class-name list straight from the cache (used by predict/app)."""
    d = np.load(HMBD_CACHE, allow_pickle=False)
    return [str(n) for n in d["names"]]


def get_datasets():
    """Load both sources, merge, prepare, split. Returns six arrays."""
    X_h, y_h, names = load_hmbd()
    X_a, y_a = load_ahcd(names)

    if X_a is not None:
        X = np.concatenate([X_h, X_a], axis=0)
        y = np.concatenate([y_h, y_a], axis=0)
        print(f"MERGED: {X.shape[0]} total samples, {len(np.unique(y))} classes")
    else:
        X, y = X_h, y_h

    X = prepare_images(X)
    return split_data(X, y)


if __name__ == "__main__":
    X_train, X_val, X_test, y_train, y_val, y_test = get_datasets()
    names = get_class_names()
    print()
    print(f"Train: {X_train.shape}, {y_train.shape}")
    print(f"Val:   {X_val.shape}, {y_val.shape}")
    print(f"Test:  {X_test.shape}, {y_test.shape}")
    print(f"Classes: {len(np.unique(y_train))}")
    print(f"Pixel range: [{X_train.min():.2f}, {X_train.max():.2f}]")
    _, counts = np.unique(y_train, return_counts=True)
    print(f"Train samples per class: min {counts.min()}, "
          f"max {counts.max()}, avg {counts.mean():.0f}")
