"""Q2 (Easy): Train a classifier, compute accuracy, precision, recall and F1,
and plot its confusion matrix.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # save the plot to a file; no window needed
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, confusion_matrix,
                             f1_score, precision_score, recall_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

FIG_DIR = Path(__file__).resolve().parents[1] / "figures"

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Positive class = 1 (benign). Precision: of predicted benign, how many were benign.
# Recall: of actual benign, how many we found.
print(f"Accuracy:  {accuracy_score(y_test, y_pred):.3f}")
print(f"Precision: {precision_score(y_test, y_pred):.3f}")
print(f"Recall:    {recall_score(y_test, y_pred):.3f}")
print(f"F1-score:  {f1_score(y_test, y_pred):.3f}")

FIG_DIR.mkdir(exist_ok=True)
disp = ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred),
                              display_labels=["malignant", "benign"])
disp.plot(cmap="Blues")
plt.title("Confusion matrix – logistic regression")
plt.savefig(FIG_DIR / "q02_confusion_matrix.png", dpi=120, bbox_inches="tight")
print(f"Saved plot to {FIG_DIR / 'q02_confusion_matrix.png'}")
