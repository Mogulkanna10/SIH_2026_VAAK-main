from pathlib import Path

import numpy as np
import tensorflow as tf

from model import build_model


BASE = Path.home() / "split"
FEATURES = BASE / "features"

INPUT_SHAPE = (49, 40, 1)

BATCH_SIZE = 32
EPOCHS = 60

SEED = 42

np.random.seed(SEED)
tf.random.set_seed(SEED)


# ============================================================
# Dataset loading
# ============================================================

def load_split(split_name):

    X = []
    y = []

    split_dir = FEATURES / split_name

    for label, class_name in [
        (0, "negative"),
        (1, "positive"),
    ]:

        files = sorted(
            (split_dir / class_name).glob("*.npy")
        )

        for path in files:

            feature = np.load(
                path
            ).astype(np.float32)

            if feature.shape != (
                INPUT_SHAPE[0],
                INPUT_SHAPE[1]
            ):
                raise ValueError(
                    f"{path}: unexpected shape "
                    f"{feature.shape}"
                )

            X.append(feature)
            y.append(label)

    X = np.asarray(
        X,
        dtype=np.float32
    )

    y = np.asarray(
        y,
        dtype=np.int32
    )

    return X, y


# ============================================================
# Training-only feature augmentation
# ============================================================

def augment_batch(x, training=False):

    if not training:
        return x

    x = tf.convert_to_tensor(
        x,
        dtype=tf.float32
    )

    # --------------------------------------------------------
    # Small random time shift.
    # --------------------------------------------------------

    shift = tf.random.uniform(
        [],
        minval=-3,
        maxval=4,
        dtype=tf.int32
    )

    x = tf.roll(
        x,
        shift=shift,
        axis=1
    )

    # --------------------------------------------------------
    # Small random frequency shift.
    # --------------------------------------------------------

    freq_shift = tf.random.uniform(
        [],
        minval=-1,
        maxval=2,
        dtype=tf.int32
    )

    x = tf.roll(
        x,
        shift=freq_shift,
        axis=2
    )

    # --------------------------------------------------------
    # Frequency masking.
    # --------------------------------------------------------

    mask_probability = tf.random.uniform([])

    def apply_freq_mask():

        width = tf.random.uniform(
            [],
            minval=1,
            maxval=5,
            dtype=tf.int32
        )

        start = tf.random.uniform(
            [],
            minval=0,
            maxval=40,
            dtype=tf.int32
        )

        positions = tf.range(
            40
        )

        mask = tf.logical_and(
            positions >= start,
            positions < start + width
        )

        mask = tf.reshape(
            mask,
            [1, 1, 40, 1]
        )

        return tf.where(
            mask,
            tf.zeros_like(x),
            x
        )

    x = tf.cond(
        mask_probability < 0.20,
        apply_freq_mask,
        lambda: x
    )

    # --------------------------------------------------------
    # Time masking.
    # --------------------------------------------------------

    mask_probability = tf.random.uniform([])

    def apply_time_mask():

        width = tf.random.uniform(
            [],
            minval=1,
            maxval=6,
            dtype=tf.int32
        )

        start = tf.random.uniform(
            [],
            minval=0,
            maxval=49,
            dtype=tf.int32
        )

        positions = tf.range(
            49
        )

        mask = tf.logical_and(
            positions >= start,
            positions < start + width
        )

        mask = tf.reshape(
            mask,
            [1, 49, 1, 1]
        )

        return tf.where(
            mask,
            tf.zeros_like(x),
            x
        )

    x = tf.cond(
        mask_probability < 0.20,
        apply_time_mask,
        lambda: x
    )

    return x


# ============================================================
# Load data
# ============================================================

print("=" * 70)
print("DRACARYS KWS ROBUST TRAINING")
print("=" * 70)
print()

X_train, y_train = load_split(
    "train"
)

X_val, y_val = load_split(
    "validation"
)

print(
    "Training data:",
    X_train.shape
)

print(
    "Validation data:",
    X_val.shape
)

print(
    "Train positive:",
    np.sum(y_train == 1)
)

print(
    "Train negative:",
    np.sum(y_train == 0)
)

print(
    "Validation positive:",
    np.sum(y_val == 1)
)

print(
    "Validation negative:",
    np.sum(y_val == 0)
)

# Add channel dimension.
X_train = X_train[..., np.newaxis]
X_val = X_val[..., np.newaxis]


# ============================================================
# TensorFlow datasets
# ============================================================

train_ds = tf.data.Dataset.from_tensor_slices(
    (X_train, y_train)
)

train_ds = train_ds.shuffle(
    buffer_size=len(X_train),
    seed=SEED,
    reshuffle_each_iteration=True
)

train_ds = train_ds.batch(
    BATCH_SIZE
)

train_ds = train_ds.map(
    lambda x, y: (
        augment_batch(x, training=True),
        y
    ),
    num_parallel_calls=tf.data.AUTOTUNE
)

train_ds = train_ds.prefetch(
    tf.data.AUTOTUNE
)


val_ds = tf.data.Dataset.from_tensor_slices(
    (X_val, y_val)
)

val_ds = val_ds.batch(
    BATCH_SIZE
)

val_ds = val_ds.prefetch(
    tf.data.AUTOTUNE
)


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
    metrics=["accuracy"]
)


# ============================================================
# Callbacks
# ============================================================

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

    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        verbose=1
    ),

    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=12,
        restore_best_weights=True,
        verbose=1
    )
]


# ============================================================
# Train
# ============================================================

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks,
    verbose=1
)


# ============================================================
# Save
# ============================================================

model.save(
    BASE / "dracarys_final.keras"
)

best_val = max(
    history.history["val_accuracy"]
)

print()
print("=" * 70)
print("TRAINING COMPLETE")
print("=" * 70)

print(
    "Best validation accuracy:",
    best_val
)

print(
    "Best model:",
    checkpoint_path
)

print(
    "Final model:",
    BASE / "dracarys_final.keras"
)

