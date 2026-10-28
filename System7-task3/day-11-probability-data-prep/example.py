"""Day 4: probability, correlation and train/validation/test splitting."""
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

# Probability: favourable outcomes / total outcomes
rolls = [1, 6, 3, 6, 2, 6, 4, 5, 6, 1]
p_six = rolls.count(6) / len(rolls)
print(f"P(rolling a 6) from data = {p_six:.2f}")

# Correlation: how strongly two columns move together (-1 to 1)
houses = pd.read_csv(Path(__file__).resolve().parents[1] / "tasks/data/houses.csv")
print("\nCorrelation with price:")
print(houses.corr(numeric_only=True)["price"].sort_values(ascending=False).round(2))

# Data splitting: 60% train, 20% validation, 20% test
X = houses.drop(columns="price")
y = houses["price"]
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42)

print(f"\nTrain: {len(X_train)} | Validation: {len(X_val)} | Test: {len(X_test)}")
