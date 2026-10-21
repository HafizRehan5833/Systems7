"""Day 7: coefficients and feature importance on the house data."""
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

houses = pd.read_csv(Path(__file__).resolve().parents[1] / "tasks/data/houses.csv").dropna()
features = ["area_sqft", "bedrooms", "bathrooms", "age_years", "distance_km"]
X, y = houses[features], houses["price"]

# Coefficients: how much the price changes per one extra unit of each feature
linear = LinearRegression().fit(X, y)
coefs = pd.Series(linear.coef_, index=features).sort_values(key=abs, ascending=False)
print("Linear coefficients (price change per unit):")
print(coefs.round(1))
print(f"Example: each extra sq ft adds about {linear.coef_[0]:.0f} to the price.")

# Feature importance: how much each feature helped a tree-based model
forest = RandomForestRegressor(n_estimators=200, random_state=0).fit(X, y)
importance = pd.Series(forest.feature_importances_, index=features).sort_values(ascending=False)
print("\nRandom forest feature importance (sums to 1):")
print(importance.round(3))
