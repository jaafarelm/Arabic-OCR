"""
train.py — Train the Arabic character recognizer on GPU with PyTorch.

Ties the pipeline together:
    data.py  -> load & split (unchanged, pure NumPy)
    model.py -> build the ResNet CNN (PyTorch)
    here     -> augment, train on GPU, evaluate, save, plot

Run with:  python src/train.py

Outputs (models/):
    best_model.pt          - best weights by validation loss
    training_history.png   - train/val accuracy & loss curves

PyTorch notes (the main difference from Keras):
    - There is no .fit(). You write the training loop explicitly:
      for each batch -> forward -> loss -> backward -> optimizer step.
    - Tensors and the model must be moved to the GPU with .to(device).
    - model.train() / model.eval() switch BatchNorm and Dropout behaviour.
    - CrossEntropyLoss expects RAW LOGITS (no softmax in the model).
"""

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import torchvision.transforms as T

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data import get_datasets
from model import build_model


MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
MODELS_DIR.mkdir(exist_ok=True)
BEST_MODEL_PATH = MODELS_DIR / "best_model.pt"
HISTORY_PLOT_PATH = MODELS_DIR / "training_history.png"

BATCH_SIZE = 128          # larger than the CPU run: the GPU can handle it
MAX_EPOCHS = 50
LEARNING_RATE = 1e-4
PATIENCE = 8              # early-stopping patience on val loss

# Use the GPU if available, else fall back to CPU.
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --- Augmentation ----------------------------------------------------------
# Applied on-GPU per batch. NOTE: no horizontal/vertical flips — Arabic script
# is not flip-invariant, so a mirrored glyph is a different or invalid letter.
augment = T.Compose([
    T.RandomAffine(
        degrees=10,               # small tilt
        translate=(0.1, 0.1),     # shift up to 10%
        scale=(0.9, 1.1),         # zoom in/out 10%
        shear=10,                 # slight slant
        fill=1.0,                 # pad with white (background colour)
    ),
])


def make_loaders(X_train, y_train, X_val, y_val, X_test, y_test):
    """Wrap the NumPy arrays from data.py into PyTorch DataLoaders.

    data.py gives (N, 32, 32, 1) NHWC float arrays in [0, 1].
    PyTorch wants (N, 1, 32, 32) NCHW, so we permute the axes.
    """
    def to_tensor(X, y):
        X_t = torch.from_numpy(X).float().permute(0, 3, 1, 2)  # NHWC -> NCHW
        y_t = torch.from_numpy(y).long()
        return TensorDataset(X_t, y_t)

    train_ds = to_tensor(X_train, y_train)
    val_ds = to_tensor(X_val, y_val)
    test_ds = to_tensor(X_test, y_test)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=2, pin_memory=True)
    return train_loader, val_loader, test_loader


def run_epoch(model, loader, criterion, optimizer=None, use_augment=False):
    """Run one pass over a loader. If optimizer is given, this is training."""
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss, correct, total = 0.0, 0, 0

    # No gradient tracking during validation/test — faster and uses less memory.
    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for images, labels in loader:
            images = images.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)

            if use_augment:
                images = augment(images)

            outputs = model(images)                 # raw logits
            loss = criterion(outputs, labels)

            if is_train:
                optimizer.zero_grad()               # clear old gradients
                loss.backward()                     # backprop
                optimizer.step()                    # update weights

            total_loss += loss.item() * labels.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)

    return total_loss / total, correct / total


def plot_history(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(history["train_acc"], label="train")
    ax1.plot(history["val_acc"], label="val")
    ax1.set_title("Accuracy"); ax1.set_xlabel("epoch"); ax1.legend()
    ax2.plot(history["train_loss"], label="train")
    ax2.plot(history["val_loss"], label="val")
    ax2.set_title("Loss"); ax2.set_xlabel("epoch"); ax2.legend()
    fig.tight_layout()
    fig.savefig(HISTORY_PLOT_PATH, dpi=90)
    print(f"Saved training curves -> {HISTORY_PLOT_PATH}")


def main():
    print(f"Device: {DEVICE}")
    if DEVICE.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # --- 1. Data ----------------------------------------------------------
    X_train, X_val, X_test, y_train, y_val, y_test = get_datasets()
    print(f"Train {X_train.shape} | Val {X_val.shape} | Test {X_test.shape}")

    num_classes = len(np.unique(y_train))     # computed, never hardcoded
    print(f"Classes: {num_classes}")

    train_loader, val_loader, test_loader = make_loaders(
        X_train, y_train, X_val, y_val, X_test, y_test
    )

    # --- 2. Model, loss, optimizer ---------------------------------------
    model = build_model(num_classes=num_classes).to(DEVICE)
    criterion = nn.CrossEntropyLoss()          # expects raw logits
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # --- 3. Training loop with early stopping -----------------------------
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_loss = float("inf")
    epochs_without_improvement = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion,
                                    optimizer, use_augment=True)
        va_loss, va_acc = run_epoch(model, val_loader, criterion)

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(va_loss)
        history["val_acc"].append(va_acc)

        print(f"Epoch {epoch:2}/{MAX_EPOCHS} - "
              f"loss: {tr_loss:.4f} - acc: {tr_acc:.4f} - "
              f"val_loss: {va_loss:.4f} - val_acc: {va_acc:.4f}")

        # Save the best model (lowest validation loss), like ModelCheckpoint.
        if va_loss < best_val_loss:
            best_val_loss = va_loss
            epochs_without_improvement = 0
            torch.save(model.state_dict(), BEST_MODEL_PATH)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= PATIENCE:
                print(f"Early stopping at epoch {epoch}")
                break

    # --- 4. Final honest evaluation on the untouched test set -------------
    model.load_state_dict(torch.load(BEST_MODEL_PATH))   # restore best weights
    test_loss, test_acc = run_epoch(model, test_loader, criterion)
    print(f"\nTEST accuracy: {test_acc:.4f}  |  TEST loss: {test_loss:.4f}")

    plot_history(history)
    print(f"Best model saved -> {BEST_MODEL_PATH}")


if __name__ == "__main__":
    main()
