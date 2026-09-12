from pathlib import Path
import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from model import build_model


BASE = Path.home() / "split"
FEATURES = BASE / "features"

X = []
y = []

for label, folder in [
    (0, FEATURES / "train" / "negative"),
    (1, FEATURES / "train" / "positive"),
]:

    for p in sorted(folder.glob("*.npy")):

        X.append(
            np.load(p).astype(np.float32)
        )
        y.append(label)


X = np.asarray(X, dtype=np.float32)
y = np.asarray(y, dtype=np.int32)

print("Full training pool:", X.shape)
print("Positive:", np.sum(y == 1))
print("Negative:", np.sum(y == 0))

X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

X_train = X_train[..., np.newaxis]
X_val = X_val[..., np.newaxis]

print()
print("Random training:", X_train.shape)
print("Random validation:", X_val.shape)

model = build_model(
    input_shape=(49, 40, 1),
    num_classes=2
)

model.compile(
    optimizer=tf.keras.optimizers.Adam(0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        patience=8,
        mode="max",
        restore_best_weights=True,
        verbose=1
    )
]

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=50,
    batch_size=32,
    shuffle=True,
    callbacks=callbacks,
    verbose=1
)

pred = np.argmax(
    model.predict(X_val, verbose=0),
    axis=1
)

acc = accuracy_score(
    y_val,
    pred
)

print()
print("=" * 60)
print("RANDOM SPLIT RESULT")
print("=" * 60)
print("Best validation accuracy:",
      max(history.history["val_accuracy"]))
print("Final measured accuracy:",
      acc)

