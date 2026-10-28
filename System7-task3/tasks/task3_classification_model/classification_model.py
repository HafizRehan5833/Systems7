"""Task 3: classify houses as expensive or not expensive."""
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "houses.csv"
FEATURES = ["area_sqft", "bedrooms", "bathrooms", "age_years", "distance_km"]


def main() -> None:
    df = pd.read_csv(DATA_PATH).drop_duplicates()
    df = df.fillna(df.median(numeric_only=True))

    train_val, test = train_test_split(df, test_size=0.20, random_state=42)
    train, val = train_test_split(train_val, test_size=0.25, random_state=42)

    # Label from the training median only (no leakage from val/test)
    threshold = train["price"].median()
    for d in (train, val, test):
        d["is_expensive"] = (d["price"] > threshold).astype(int)

    X_train, y_train = train[FEATURES], train["is_expensive"]
    X_val, y_val = val[FEATURES], val["is_expensive"]
    X_test, y_test = test[FEATURES], test["is_expensive"]

    models = {
        "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
        "KNN (k=5)": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5)),
        "Decision Tree": DecisionTreeClassifier(max_depth=4, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
    }

    print(f"Expensive threshold: {threshold:,.0f}\n")
    print("=== Validation accuracy ===")
    scores = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        scores[name] = model.score(X_val, y_val)
        print(f"{name:<22} {scores[name]:.3f}")

    best_name = max(scores, key=scores.get)
    best = models[best_name]
    print(f"\nBest model on validation: {best_name}")

    # Final evaluation on the untouched test set
    y_pred = best.predict(X_test)
    print("\n=== Test set ===")
    print("Confusion matrix (rows = actual, cols = predicted):")
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred, target_names=["not expensive", "expensive"]))

    # Feature importance from the random forest
    forest = models["Random Forest"]
    importance = pd.Series(forest.feature_importances_, index=FEATURES).sort_values(ascending=False)
    print("=== Random forest feature importance ===")
    print(importance.round(3).to_string())


if __name__ == "__main__":
    main()
