"""Q5 (Medium): Plot ROC curves and AUC for logistic regression and random forest
on one chart.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import auc, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

FIG_DIR = Path(__file__).resolve().parents[1] / "figures"

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

models = {
    "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
}

plt.figure(figsize=(6, 5))
for name, model in models.items():
    model.fit(X_train, y_train)
    # ROC needs a score, not a hard label: use the predicted probability of class 1
    scores = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, scores)
    roc_auc = auc(fpr, tpr)
    print(f"{name}: AUC = {roc_auc:.3f}")
    plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.3f})")

plt.plot([0, 1], [0, 1], "k--", label="Random guess")
plt.xlabel("False positive rate")
plt.ylabel("True positive rate")
plt.title("ROC curves")
plt.legend(loc="lower right")
FIG_DIR.mkdir(exist_ok=True)
plt.savefig(FIG_DIR / "q05_roc_curves.png", dpi=120, bbox_inches="tight")
print(f"Saved plot to {FIG_DIR / 'q05_roc_curves.png'}")
