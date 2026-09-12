from pathlib import Path

import numpy as np
import tensorflow as tf

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)


BASE = Path.home() / "split"
FEATURE_DIR = BASE / "features" / "test"
MODEL_PATH = BASE / "dracarys_best.keras"

CLASS_NAMES = [
    "negative",
    "Dracarys",
]


def load_test_data():
    X = []
    y = []
    paths = []

    for label, folder_name in [
        (0, "negative"),
        (1, "positive"),
    ]:
        folder = FEATURE_DIR / folder_name

        files = sorted(folder.glob("*.npy"))

        for path in files:
            feature = np.load(path).astype(np.float32)

            if feature.shape != (49, 40):
                raise ValueError(
                    f"{path}: unexpected shape {feature.shape}"
                )

            X.append(feature)
            y.append(label)
            paths.append(path)

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int32)

    X = X[..., np.newaxis]

    return X, y, paths


print("=" * 70)
print("DRACARYS KWS TEST EVALUATION")
print("=" * 70)
print()

model = tf.keras.models.load_model(MODEL_PATH)

X_test, y_test, paths = load_test_data()

print("Test shape:", X_test.shape)
print("Test labels:", y_test.shape)
print("Positive:", int(np.sum(y_test == 1)))
print("Negative:", int(np.sum(y_test == 0)))
print()

probabilities = model.predict(
    X_test,
    verbose=0
)

predictions = np.argmax(
    probabilities,
    axis=1
)

accuracy = accuracy_score(
    y_test,
    predictions
)

precision = precision_score(
    y_test,
    predictions,
    zero_division=0
)

recall = recall_score(
    y_test,
    predictions,
    zero_division=0
)

f1 = f1_score(
    y_test,
    predictions,
    zero_division=0
)

cm = confusion_matrix(
    y_test,
    predictions
)

tn, fp, fn, tp = cm.ravel()

tpr = tp / (tp + fn) if (tp + fn) else 0
fpr = fp / (fp + tn) if (fp + tn) else 0
fnr = fn / (tp + fn) if (tp + fn) else 0

print("=" * 70)
print("METRICS")
print("=" * 70)

print(f"Accuracy           : {accuracy:.4f}")
print(f"Precision          : {precision:.4f}")
print(f"Recall             : {recall:.4f}")
print(f"F1 Score           : {f1:.4f}")

print()
print("Confusion Matrix:")
print(cm)

print()
print("=" * 70)
print("KWS METRICS")
print("=" * 70)

print(f"True Positives     : {tp}")
print(f"False Positives    : {fp}")
print(f"True Negatives     : {tn}")
print(f"False Negatives    : {fn}")

print()
print(f"True Positive Rate : {tpr:.4f}")
print(f"False Positive Rate: {fpr:.4f}")
print(f"False Reject Rate  : {fnr:.4f}")

print()
print("=" * 70)
print("CLASSIFICATION REPORT")
print("=" * 70)

print(
    classification_report(
        y_test,
        predictions,
        target_names=CLASS_NAMES,
        zero_division=0
    )
)

report_path = BASE / "dracarys_test_predictions.csv"

with open(report_path, "w") as f:
    f.write(
        "file,true_label,predicted_label,"
        "negative_probability,dracarys_probability\n"
    )

    for path, true_label, pred, probs in zip(
        paths,
        y_test,
        predictions,
        probabilities
    ):
        f.write(
            f"{path},"
            f"{CLASS_NAMES[true_label]},"
            f"{CLASS_NAMES[pred]},"
            f"{probs[0]:.6f},"
            f"{probs[1]:.6f}\n"
        )

print()
print("Prediction report:", report_path)
