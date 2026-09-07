"""
label_map.py — Class-id -> letter-name lookup.

The names are read from data/hmbd_cache.npz, which build_dataset.py generated
from the ORIGINAL folder names. This replaces the shipped label_mapping.csv,
which was verified corrupt (its names did not match the images).

Because the names come from the same artifact as the labels, the mapping cannot
drift out of sync with the data.
"""

from pathlib import Path

import numpy as np

CACHE_PATH = Path(__file__).resolve().parent.parent / "data" / "hmbd_cache.npz"

_names = None


def _load():
    global _names
    if _names is None:
        if not CACHE_PATH.exists():
            raise FileNotFoundError(
                f"{CACHE_PATH} not found. Run:  python src/build_dataset.py"
            )
        d = np.load(CACHE_PATH, allow_pickle=False)
        _names = [str(n) for n in d["names"]]
    return _names


def name_for(class_id):
    """Return the letter name for a class id (falls back to 'Class N')."""
    names = _load()
    if 0 <= class_id < len(names):
        return names[class_id]
    return f"Class {class_id}"


def num_classes():
    """Number of base classes, read from the cache."""
    return len(_load())
