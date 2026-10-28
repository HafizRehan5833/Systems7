"""Q11 (Medium): Handle an imbalanced dataset using class_weight and compare
recall before and after.
"""
import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, recall_score
from sklearn.model_selection import train_test_split

# 95% of samples are class 0, 5% are class 1 (e.g. fraud, rare disease)
X, y = make_classification(n_samples=4000, n_features=10, weights=[0.95, 0.05],
                           random_state=0, flip_y=0.01)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
counts = dict(zip(*np.unique(y_test, return_counts=True)))
print("Class counts in test set:", {int(k): int(v) for k, v in counts.items()})

for label, weight in [("Before (no class_weight)", None), ("After (class_weight='balanced')", "balanced")]:
    model = LogisticRegression(max_iter=1000, class_weight=weight).fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"\n{label}")
    print(f"  accuracy = {accuracy_score(y_test, y_pred):.3f}  (looks good even if it finds no rare cases)")
    print(f"  recall for rare class (1) = {recall_score(y_test, y_pred):.3f}")
