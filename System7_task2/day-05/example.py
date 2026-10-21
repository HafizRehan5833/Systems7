"""Day 5: simple vs multiple linear regression with scikit-learn."""
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

houses = pd.read_csv(Path(__file__).resolve().parents[1] / "tasks/data/houses.csv")
houses = houses.dropna()  # keep the example simple; Task 1 handles missing data properly

# Step 1: split
train, test = train_test_split(houses, test_size=0.2, random_state=42)

# Simple linear regression: one feature
X_simple_train, X_simple_test = train[["area_sqft"]], test[["area_sqft"]]

# Step 2: fit
simple = LinearRegression().fit(X_simple_train, train["price"])
# Step 3: predict
pred = simple.predict(X_simple_test)
# Step 4: evaluate
print("Simple linear regression (area only)")
print(f"  coefficient = {simple.coef_[0]:.2f} per sq ft")
print(f"  MAE = {mean_absolute_error(test['price'], pred):,.0f}   R² = {r2_score(test['price'], pred):.3f}")

# Multiple linear regression: several features
features = ["area_sqft", "bedrooms", "bathrooms", "age_years", "distance_km"]
multi = LinearRegression().fit(train[features], train["price"])
pred = multi.predict(test[features])
print("\nMultiple linear regression")
for name, coef in zip(features, multi.coef_):
    print(f"  {name:<12} {coef:>12,.1f}")
print(f"  MAE = {mean_absolute_error(test['price'], pred):,.0f}   R² = {r2_score(test['price'], pred):.3f}")
