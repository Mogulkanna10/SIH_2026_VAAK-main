import numpy as np
import tensorflow as tf
from pathlib import Path

from model import build_model

BASE = Path.home() / "split"

X = []
y = []

# 10 positive + 10 negative
for label, folder in [
    (1, BASE / "features/train/positive"),
    (0, BASE / "features/train/negative"),
]:

    files = sorted(folder.glob("*.npy"))[:10]

    for p in files:
        X.append(
            np.load(p).astype(np.float32)
        )
        y.append(label)

X = np.asarray(X, dtype=np.float32)[..., np.newaxis]
y = np.asarray(y, dtype=np.int32)

print("X shape:", X.shape)
print("y shape:", y.shape)

model = build_model(
    input_shape=(49, 40, 1),
    num_classes=2
)

model.compile(
    optimizer=tf.keras.optimizers.Adam(0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.fit(
    X,
    y,
    epochs=100,
    batch_size=4,
    shuffle=True,
    verbose=1
)

loss, acc = model.evaluate(
    X,
    y,
    verbose=0
)

print()
print("Tiny-subset accuracy:", acc)

probs = model.predict(X, verbose=0)
pred = np.argmax(probs, axis=1)

print("True labels:")
print(y)

print("Predicted:")
print(pred)
