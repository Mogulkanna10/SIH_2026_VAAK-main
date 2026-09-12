from pathlib import Path
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix


BASE = Path.home() / "split"

MODEL_PATH = BASE / "dracarys_best.keras"
FEATURE_DIR = BASE / "features" / "validation"


def load_data():
    X = []
    y = []

    for label, folder_name in [
        (0, "negative"),
        (1, "positive"),
    ]:

        folder = FEATURE_DIR / folder_name

        for path in sorted(folder.glob("*.npy")):

            x = np.load(path).astype(np.float32)

            if x.shape != (49, 40):
                raise ValueError(
                    f"{path}: unexpected shape {x.shape}"
                )

            X.append(x)
            y.append(label)

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int32)

    return X[..., np.newaxis], y


model = tf.keras.models.load_model(
    MODEL_PATH
)

X, y = load_data()

probs = model.predict(
    X,
    verbose=0
)[:, 1]


print("=" * 75)
print("DRACARYS VALIDATION THRESHOLD ANALYSIS")
print("=" * 75)
print()
print(
    "Threshold | TPR    | FPR    | FRR    | Accuracy | FP | FN"
)
print("-" * 70)


for threshold in np.arange(
    0.20,
    1.00,
    0.05
):

    pred = (
        probs >= threshold
    ).astype(np.int32)

    tn, fp, fn, tp = confusion_matrix(
        y,
        pred,
        labels=[0, 1]
    ).ravel()

    tpr = tp / (tp + fn)
    fpr = fp / (fp + tn)
    frr = fn / (tp + fn)
    accuracy = (tp + tn) / len(y)

    print(
        f"{threshold:9.2f} | "
        f"{tpr:6.3f} | "
        f"{fpr:6.3f} | "
        f"{frr:6.3f} | "
        f"{accuracy:8.3f} | "
        f"{fp:2d} | "
        f"{fn:2d}"
    )

