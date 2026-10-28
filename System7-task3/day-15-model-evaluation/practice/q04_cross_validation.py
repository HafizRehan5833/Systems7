"""Q4 (Medium): Run 5-fold cross-validation on three models and print the
mean and standard deviation of each score.
"""
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

X, y = load_breast_cancer(return_X_y=True)

models = {
    "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
    "Decision Tree": DecisionTreeClassifier(max_depth=4, random_state=0),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=0),
}

print(f"{'Model':<22}{'Mean acc':>10}{'Std':>10}")
for name, model in models.items():
    # cv=5 splits the data into 5 folds; each fold is the test set once
    scores = cross_val_score(model, X, y, cv=5, scoring="accuracy")
    print(f"{name:<22}{scores.mean():>10.3f}{scores.std():>10.3f}")
    print(f"  fold scores: {np.round(scores, 3)}")
