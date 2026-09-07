"""
build_dataset.py — ONE-TIME script: build a clean HMBD cache from the original
image folders.

WHY THIS EXISTS:
    The flattened images.csv/labels.csv shipped alongside HMBD had a CORRUPT
    label_mapping.csv — verified by rendering samples: folders and CSV labels
    did not agree (e.g. a junk ".ipynb_checkpoints" entry occupied a class id,
    and letter names did not match their images). Training on it meant the
    model learned correct image groupings under WRONG names.

    The original Dataset/ folders are the ground truth: the folder name IS the
    label. Verified visually — Ain_Isolated contains ain, Kaf_Isolated contains
    kaf, and so on. Building from folders makes mislabelling impossible by
    construction.

WHAT IT DOES:
    - Walks data/Dataset/<Letter>_<Form>/ (115 folders).
    - Strips the positional suffix (_Start/_Middle/_End/_Isolated) so all forms
      of a letter collapse into ONE base class (46 total). Isolated 32x32
      glyphs cannot reliably distinguish positional forms, so this is the
      well-posed task.
    - Loads each JPG, converts to grayscale, resizes to 32x32.
    - Caches arrays + the generated name mapping to data/hmbd_cache.npz.

Run ONCE:  python src/build_dataset.py
(~4 minutes for 54k images. data.py then loads the cache instantly.)
"""

import os
import glob
from pathlib import Path

import numpy as np
from PIL import Image


IMG_SIZE = 32
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATASET_DIR = DATA_DIR / "Dataset"
CACHE_PATH = DATA_DIR / "hmbd_cache.npz"

# Positional-form suffixes stripped to get the base letter.
SUFFIXES = ["_Start", "_Middle", "_End", "_Isolated"]


def base_name(folder):
    """'Ain_Middle' -> 'Ain'. Digits ('Zero'..'Nine') pass through unchanged."""
    for s in SUFFIXES:
        if folder.endswith(s):
            return folder[: -len(s)]
    return folder


def main():
    if not DATASET_DIR.exists():
        raise FileNotFoundError(
            f"{DATASET_DIR} not found. Unzip the HMBD Dataset folder to data/Dataset/"
        )

    # Real character folders only — skip hidden dirs so junk like
    # .ipynb_checkpoints can never become a class again.
    folders = sorted(
        d for d in os.listdir(DATASET_DIR)
        if (DATASET_DIR / d).is_dir() and not d.startswith(".")
    )
    print(f"Found {len(folders)} character folders")

    # Build base-letter classes from the folder names themselves.
    bases = sorted({base_name(f) for f in folders})
    base_to_id = {b: i for i, b in enumerate(bases)}
    print(f"Collapsed to {len(bases)} base classes")

    X_list, y_list = [], []
    for folder in folders:
        label = base_to_id[base_name(folder)]
        files = glob.glob(str(DATASET_DIR / folder / "*"))

        for fp in files:
            try:
                img = Image.open(fp).convert("L")           # grayscale
                img = img.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
                X_list.append(np.asarray(img, dtype=np.uint8).ravel())
                y_list.append(label)
            except Exception as e:                          # skip unreadable files
                print(f"  skipped {fp}: {e}")

        print(f"  {folder:28} -> class {label:2} ({bases[label]}), {len(files)} images")

    X = np.stack(X_list)
    y = np.array(y_list, dtype=np.int64)

    print()
    print(f"Loaded {X.shape[0]} images, shape {X.shape}")
    print(f"Classes: {len(np.unique(y))}")
    print(f"Pixel range: {X.min()} - {X.max()} (mean {X.mean():.1f})")

    # Save arrays AND the name list, so data.py / label_map never need a
    # separate hand-maintained mapping file again.
    np.savez_compressed(CACHE_PATH, X=X, y=y, names=np.array(bases))
    print(f"\nCached -> {CACHE_PATH}")
    print("Base class names (id: name):")
    for i, b in enumerate(bases):
        print(f"  {i:2}: {b}")


if __name__ == "__main__":
    main()
