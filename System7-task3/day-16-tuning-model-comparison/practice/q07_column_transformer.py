"""Q7 (Medium): Use ColumnTransformer to scale numeric columns and one-hot encode
categorical columns in one step.

The house dataset has no text columns, so this script adds a 'neighbourhood'
category (seeded, so it is the same every run) to show encoding.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA = Path(__file__).resolve().parents[2] / "tasks" / "data" / "houses.csv"
df = pd.read_csv(DATA)
rng = np.random.default_rng(0)
df["neighbourhood"] = rng.choice(["north", "south", "east", "west"], size=len(df))

numeric = ["area_sqft", "bedrooms", "bathrooms", "age_years", "distance_km"]
categorical = ["neighbourhood"]

preprocess = ColumnTransformer(
    transformers=[
        ("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), numeric),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
    ]
)
model = make_pipeline(preprocess, Ridge(alpha=1.0))

X_train, X_test, y_train, y_test = train_test_split(
    df[numeric + categorical], df["price"], test_size=0.2, random_state=42)
model.fit(X_train, y_train)

print("Transformed feature names:")
print(preprocess.fit(X_train).get_feature_names_out())
print(f"\nTest R² = {model.score(X_test, y_test):.3f}")
