"""Module A practical task: compare multiple models and pick the best one."""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

HOUSES = Path(__file__).resolve().parents[1] / "data" / "houses.csv"


def classification_comparison() -> None:
    X, y = load_breast_cancer(return_X_y=True)
    models = {
        "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
        "KNN (k=5)": make_pipeline(StandardScaler(), KNeighborsClassifier()),
        "Decision Tree": DecisionTreeClassifier(max_depth=4, random_state=0),
        "SVM (RBF)": make_pipeline(StandardScaler(), SVC(probability=False)),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=0),
    }
    scoring = ["accuracy", "f1", "roc_auc"]
    rows = {}
    for name, model in models.items():
        res = cross_validate(model, X, y, cv=5, scoring=scoring)
        rows[name] = {m: res[f"test_{m}"].mean() for m in scoring}

    table = pd.DataFrame(rows).T.sort_values("roc_auc", ascending=False)
    print("=== Classification (breast cancer, 5-fold CV mean) ===")
    print(table.round(3).to_string())
    print(f"Best by ROC-AUC: {table.index[0]}\n")


def regression_comparison() -> None:
    df = pd.read_csv(HOUSES).dropna()
    X = df.drop(columns="price")
    y = df["price"]
    models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(max_depth=5, random_state=0),
        "Random Forest": RandomForestRegressor(n_estimators=200, random_state=0),
    }
    scoring = {"MAE": "neg_mean_absolute_error",
               "RMSE": "neg_root_mean_squared_error",
               "R2": "r2"}
    rows = {}
    for name, model in models.items():
        res = cross_validate(model, X, y, cv=5, scoring=scoring)
        rows[name] = {
            "MAE": -res["test_MAE"].mean(),
            "RMSE": -res["test_RMSE"].mean(),
            "R2": res["test_R2"].mean(),
        }
    table = pd.DataFrame(rows).T.sort_values("RMSE")
    print("=== Regression (house prices, 5-fold CV mean) ===")
    print(table.round(3).to_string())
    print(f"Best by RMSE (lower is better): {table.index[0]}")


if __name__ == "__main__":
    classification_comparison()
    regression_comparison()
