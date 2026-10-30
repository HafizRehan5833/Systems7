from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

FIG_DIR = Path(__file__).resolve().parents[1] / "figures"

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)

depths = range(1, 16)
train_scores, test_scores = [], []
for d in depths:
    tree = DecisionTreeClassifier(max_depth=d, random_state=0).fit(X_train, y_train)
    train_scores.append(tree.score(X_train, y_train))
    test_scores.append(tree.score(X_test, y_test))
    print(f"depth={d:>2}  train={train_scores[-1]:.3f}  test={test_scores[-1]:.3f}")

plt.plot(depths, train_scores, marker="o", label="Training score")
plt.plot(depths, test_scores, marker="o", label="Test score")
plt.xlabel("max_depth (model complexity)")
plt.ylabel("Accuracy")
plt.title("Overfitting: training keeps rising, test stops improving")
plt.legend()
plt.grid(alpha=0.3)
FIG_DIR.mkdir(exist_ok=True)
plt.savefig(FIG_DIR / "q06_depth_vs_score.png", dpi=120, bbox_inches="tight")
print(f"Saved plot to {FIG_DIR / 'q06_depth_vs_score.png'}")
