"""
train.py — Train the Arabic character recognizer and save the best model.

This ties the whole pipeline together:
    data.py  -> load & split
    model.py -> build the CNN
    here     -> augment, train (with callbacks), evaluate, save, plot

Run with:  python src/train.py

Outputs (written to models/):
    - best_model.keras   : the best model seen during training (by val loss)
    - training_history.png : train/val accuracy & loss curves

Design notes for reviewers:
    - num_classes is computed from the data, never hardcoded.
    - Augmentation does NOT flip images: Arabic letters are not flip-invariant,
      so a mirrored glyph is a different or invalid letter.
    - EarlyStopping halts when validation stops improving (good on CPU: avoids
      grinding through fixed epochs). ModelCheckpoint keeps the BEST model, not
      just the last one.
    - The test set is NOT touched here beyond a single final evaluation.
"""

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")  # no display needed; we save the plot to a file
import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from data import get_datasets
from model import build_model


# Where to save outputs.
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
MODELS_DIR.mkdir(exist_ok=True)
BEST_MODEL_PATH = MODELS_DIR / "best_model.keras"
HISTORY_PLOT_PATH = MODELS_DIR / "training_history.png"

# Training settings. Batch size and a generous epoch cap; EarlyStopping will
# usually stop well before the cap.
BATCH_SIZE = 64
MAX_EPOCHS = 50


def make_augmenter():
    """Build the training-time image augmenter.

    NOTE: no horizontal/vertical flips — Arabic script is not flip-invariant.
    We only apply small geometric perturbations that a real handwritten letter
    could plausibly show.
    """
    return ImageDataGenerator(
        rotation_range=10,      # small tilt, degrees
        width_shift_range=0.1,  # slide horizontally up to 10%
        height_shift_range=0.1, # slide vertically up to 10%
        zoom_range=0.1,         # zoom in/out up to 10%
        shear_range=0.1,        # slight slant
        # horizontal_flip / vertical_flip intentionally OMITTED (default False).
    )


def plot_history(history):
    """Save train/val accuracy and loss curves to a PNG."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(history.history["accuracy"], label="train")
    ax1.plot(history.history["val_accuracy"], label="val")
    ax1.set_title("Accuracy")
    ax1.set_xlabel("epoch")
    ax1.legend()

    ax2.plot(history.history["loss"], label="train")
    ax2.plot(history.history["val_loss"], label="val")
    ax2.set_title("Loss")
    ax2.set_xlabel("epoch")
    ax2.legend()

    fig.tight_layout()
    fig.savefig(HISTORY_PLOT_PATH, dpi=90)
    print(f"Saved training curves -> {HISTORY_PLOT_PATH}")


def main():
    # --- 1. Load the data -------------------------------------------------
    X_train, X_val, X_test, y_train, y_val, y_test = get_datasets()
    print(f"Train {X_train.shape} | Val {X_val.shape} | Test {X_test.shape}")

    # Compute class count from the data — never hardcode it.
    num_classes = len(np.unique(y_train))
    print(f"Classes: {num_classes}")

    # --- 2. Build the model ----------------------------------------------
    model = build_model(num_classes=num_classes)

    # --- 3. Set up augmentation ------------------------------------------
    augmenter = make_augmenter()
    train_flow = augmenter.flow(X_train, y_train, batch_size=BATCH_SIZE)

    # --- 4. Callbacks -----------------------------------------------------
    callbacks = [
        # Stop when val_loss hasn't improved for 5 epochs; restore best weights.
        EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
        ),
        # Save the best model (lowest val_loss) to disk during training.
        ModelCheckpoint(
            BEST_MODEL_PATH,
            monitor="val_loss",
            save_best_only=True,
        ),
    ]

    # --- 5. Train ---------------------------------------------------------
    history = model.fit(
        train_flow,
        validation_data=(X_val, y_val),
        epochs=MAX_EPOCHS,
        callbacks=callbacks,
    )

    # --- 6. Final honest evaluation on the untouched test set ------------
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nTEST accuracy: {test_acc:.4f}  |  TEST loss: {test_loss:.4f}")

    # --- 7. Save the history plot ----------------------------------------
    plot_history(history)
    print(f"Best model saved -> {BEST_MODEL_PATH}")


if __name__ == "__main__":
    main()