from pathlib import Path
import numpy as np
import tensorflow as tf

BASE = Path.home() / "split"

MODEL = BASE / "dracarys_best.keras"
VAL = BASE / "features" / "validation"

model = tf.keras.models.load_model(MODEL)

X = []
y = []
names = []

for label, folder in [
    (0, VAL / "negative"),
    (1, VAL / "positive"),
]:
    for path in sorted(folder.glob("*.npy")):

        x = np.load(path).astype(np.float32)

        X.append(x)
        y.append(label)
        names.append(path.name)

X = np.asarray(X)[..., np.newaxis]
y = np.asarray(y)

probs = model.predict(X, verbose=0)
pred = np.argmax(probs, axis=1)

print("=" * 60)
print("VALIDATION DIAGNOSTIC")
print("=" * 60)

print("Samples:", len(y))
print()

print("Actual:")
print("  Negative :", np.sum(y == 0))
print("  Dracarys :", np.sum(y == 1))

print()

print("Predicted:")
print("  Negative :", np.sum(pred == 0))
print("  Dracarys :", np.sum(pred == 1))

print()

for cls in [0, 1]:

    mask = y == cls

    print(
        f"Actual class {cls} predicted as:"
    )

    print(
        "  Negative:",
        np.sum(pred[mask] == 0)
    )

    print(
        "  Dracarys:",
        np.sum(pred[mask] == 1)
    )

    print()

print("Mean probabilities:")
print(
    "Negative probability:",
    probs[:, 0].mean()
)

print(
    "Dracarys probability:",
    probs[:, 1].mean()
)

print()

print("First 20 predictions:")

for i in range(min(20, len(y))):

    print(
        names[i],
        "actual=", y[i],
        "pred=", pred[i],
        "P(Dracarys)=",
        f"{probs[i,1]:.4f}"
    )
