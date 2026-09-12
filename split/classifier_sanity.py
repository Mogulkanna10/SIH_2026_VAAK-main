from pathlib import Path
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


BASE = Path.home() / "split"
FEATURES = BASE / "features"

CLASS_MAP = {
    "negative": 0,
    "positive": 1,
}


def load_split(split):

    X = []
    y = []

    for class_name, label in CLASS_MAP.items():

        folder = FEATURES / split / class_name

        for path in sorted(folder.glob("*.npy")):

            x = np.load(path).astype(np.float32)

            if x.shape != (49, 40):
                raise ValueError(
                    f"{path}: unexpected shape {x.shape}"
                )

            X.append(x.reshape(-1))
            y.append(label)

    return (
        np.asarray(X, dtype=np.float32),
        np.asarray(y, dtype=np.int32)
    )


X_train, y_train = load_split("train")
X_val, y_val = load_split("validation")

print("=" * 60)
print("CLASSIFIER SANITY TEST")
print("=" * 60)

print("Train:", X_train.shape)
print("Val  :", X_val.shape)

model = make_pipeline(
    StandardScaler(),
    LogisticRegression(
        max_iter=2000,
        random_state=42
    )
)

model.fit(
    X_train,
    y_train
)

train_pred = model.predict(X_train)
val_pred = model.predict(X_val)

train_acc = accuracy_score(
    y_train,
    train_pred
)

val_acc = accuracy_score(
    y_val,
    val_pred
)

print()
print("Train accuracy:", train_acc)
print("Validation accuracy:", val_acc)

print()
print("Validation confusion matrix:")
print(confusion_matrix(y_val, val_pred))

print()
print("Validation predicted counts:")
print("Negative :", np.sum(val_pred == 0))
print("Positive :", np.sum(val_pred == 1))
