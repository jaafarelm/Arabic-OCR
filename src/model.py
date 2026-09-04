"""
model.py — CNN architecture (BatchNorm variant) for Arabic character recognition.

This is the stronger of the two architectures considered: four convolution
blocks with BatchNormalization, followed by a dense classifier head.

Why BatchNormalization:
  It re-normalizes each layer's activations (roughly mean 0, std 1) per batch,
  keeping the scale of the numbers stable as they flow through the network.
  That makes training faster and more stable, and usually improves accuracy.

Why the order is Conv -> BatchNorm -> Activation('relu'):
  ReLU discards all negative values. If we normalized AFTER relu, we'd be
  normalizing data that's already been clipped. Putting BatchNorm BEFORE relu
  normalizes the full signal (positive AND negative) first, then relu clips
  well-scaled data. So: extract features -> normalize them -> apply nonlinearity.

Design notes for reviewers:
  - num_classes is passed in, never hardcoded (the old notebook hardcoded it).
  - Loss is sparse_categorical_crossentropy (labels are integers 0..N-1).
  - BatchNorm uses batch statistics during training and running averages at
    inference; Keras handles that switch automatically inside model.predict().
"""

from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam


def build_model(num_classes, input_shape=(32, 32, 1)):
    """Build and compile the BatchNorm CNN.

    Parameters
    ----------
    num_classes : int
        Number of output classes; pass len(np.unique(y_train)), never hardcode.
    input_shape : tuple
        Shape of one input image. (32, 32, 1) = 32x32 grayscale.

    Returns
    -------
    A compiled tf.keras.Model, ready for .fit().
    """
    model = models.Sequential([
        layers.Input(shape=input_shape),

        # --- Block 1: 32 filters ------------------------------------------
        # Conv extracts features -> BatchNorm normalizes them -> ReLU clips.
        layers.Conv2D(32, (3, 3), padding="same"),
        layers.BatchNormalization(momentum=0.9),
        layers.Activation("relu"),
        layers.MaxPooling2D((2, 2)),

        # --- Block 2: 64 filters ------------------------------------------
        layers.Conv2D(64, (3, 3), padding="same"),
        layers.BatchNormalization(momentum=0.9),
        layers.Activation("relu"),
        layers.MaxPooling2D((2, 2)),

        # --- Block 3: 128 filters -----------------------------------------
        layers.Conv2D(128, (3, 3), padding="same"),
        layers.BatchNormalization(momentum=0.9),
        layers.Activation("relu"),
        layers.MaxPooling2D((2, 2)),

        # --- Block 4: 256 filters (no pooling after this one) -------------
        # A final, wider feature-extraction block for richer representations.
        layers.Conv2D(256, (3, 3), padding="same"),
        layers.BatchNormalization(momentum=0.9),
        layers.Activation("relu"),

        # --- Classifier head ----------------------------------------------
        # Flatten: 3D feature maps -> 1D vector (only 2D->1D step, at the end).
        layers.Flatten(),
        layers.Dense(256),
        layers.BatchNormalization(momentum=0.9),
        layers.Activation("relu"),
        # Dropout 0.6: switch off 60% of neurons during training only, to fight
        # overfitting (a bit stronger than the usual 0.5 given the added depth).
        layers.Dropout(0.6),
        # One output per class; softmax -> probabilities that sum to 1.
        layers.Dense(num_classes, activation="softmax"),
    ])

    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


def build_model_(num_classes, input_shape=(32, 32, 1)):
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax"),
    ])
    model.compile(
        optimizer=Adam(learning_rate=0.0001),   # keep the LR fix
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model

# Quick manual check: build the model and print its layer summary.
if __name__ == "__main__":
    m = build_model(num_classes=115)
    m.summary()
    