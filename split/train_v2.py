from pathlib import Path

import numpy as np
import tensorflow as tf

from model import build_model


# ============================================================
# Configuration
# ============================================================

BASE = Path.home() / "split"
FEATURES = BASE / "features"

INPUT_SHAPE = (99, 40, 1)

BATCH_SIZE = 32
EPOCHS = 50

SEED = 42

tf.random.set_seed(SEED)
np.random.seed(SEED)


# ============================================================
# Load features
# ============================================================

def load_split(split_name):

    split_dir = FEATURES / split_name

    positive_files = sorted(
        (split_dir / "positive").glob("*.npy")
    )

    negative_files = sorted(
        (split_dir / "negative").glob("*.npy")
    )

    X = []
    y = []

    # Label convention:
    # 0 = negative
    # 1 = Dracarys

    for path in negative_files:

        feature = np.load(path).astype(
            np.float32
        )

        X.append(feature)
        y.append(0)

    for path in positive_files:

        feature = np.load(path).astype(
            np.float32
        )

        X.append(feature)
        y.append(1)

    X = np.asarray(
        X,
        dtype=np.float32
    )

    y = np.asarray(
        y,
        dtype=np.int32
    )

    if X.ndim != 3:
        raise ValueError(
            f"{split_name}: expected 3D feature array, "
            f"got {X.shape}"
        )

    if X.shape[1:] != INPUT_SHAPE[:2]:
        raise ValueError(
            f"{split_name}: expected "
            f"{INPUT_SHAPE[:2]}, got {X.shape[1:]}"
        )

    return X, y


print("=" * 70)
print("Dracarys KWS Baseline Training")
print("=" * 70)
print()


X_train, y_train = load_split("train")
X_val, y_val = load_split("validation")


print("Training data:")
print("  X:", X_train.shape)
print("  y:", y_train.shape)

print()

print("Validation data:")
print("  X:", X_val.shape)
print("  y:", y_val.shape)

print()

print(
    "Train positive:",
    int(np.sum(y_train == 1))
)

print(
    "Train negative:",
    int(np.sum(y_train == 0))
)

print(
    "Validation positive:",
    int(np.sum(y_val == 1))
)

print(
    "Validation negative:",
    int(np.sum(y_val == 0))
)

# Add channel dimension.
X_train = X_train[..., np.newaxis]
X_val = X_val[..., np.newaxis]

print()
print("Model input:", X_train.shape[1:])


# ============================================================
# Build model
# ============================================================

model = build_model(
    input_shape=INPUT_SHAPE,
    num_classes=2
)

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.0005
    ),
    loss="sparse_categorical_crossentropy",
    metrics=[
        "accuracy"
    ]
)


# ============================================================
# Callbacks
# ============================================================

BASE.mkdir(
    parents=True,
    exist_ok=True
)

checkpoint_path = (
    BASE / "dracarys_best.keras"
)

callbacks = [

    tf.keras.callbacks.ModelCheckpoint(
        filepath=str(checkpoint_path),
        monitor="val_accuracy",
        save_best_only=True,
        mode="max",
        verbose=1
    ),

    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=12,
        restore_best_weights=True,
        verbose=1
    ),

    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=3,
        min_lr=1e-6,
        verbose=1
    )
]


# ============================================================
# Train
# ============================================================

history = model.fit(
    X_train,
    y_train,
    validation_data=(
        X_val,
        y_val
    ),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    shuffle=True,
    callbacks=callbacks,
    verbose=1
)


# ============================================================
# Save final model
# ============================================================

model.save(
    BASE / "dracarys_final.keras"
)

print()
print("=" * 70)
print("TRAINING COMPLETE")
print("=" * 70)

print(
    "Best model:",
    checkpoint_path
)

print(
    "Final model:",
    BASE / "dracarys_final.keras"
)

print(
    "Best validation accuracy:",
    max(
        history.history["val_accuracy"]
    )
)
