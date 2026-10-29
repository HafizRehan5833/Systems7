"""Q3 (Easy): Calculate MAE, MSE, RMSE and R² for a house-price model with NumPy,
then verify the results with scikit-learn.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
try:
    print("data loading successfull.")
    DATA = Path(__file__).resolve().parents[2] / "syllabus Modules" / "data" / "houses.csv"
except Exception as e:
    print("Error to loading the data:",e)
# if not DATA.exists():
#     DATA = Path(__file__).resolve().parents[2] / "sylla" / "data" / "houses.csv"
df = pd.read_csv(DATA).dropna()
features = ["area_sqft", "bedrooms", "bathrooms", "age_years", "distance_km"]

X_train, X_test, y_train, y_test = train_test_split(
    df[features], df["price"], test_size=0.2, random_state=42)

model = LinearRegression().fit(X_train, y_train)
y_true = y_test.to_numpy()
y_pred = model.predict(X_test)

# NumPy versions of the formulas
errors = y_true - y_pred
mae = np.mean(np.abs(errors))
mse = np.mean(errors ** 2)
rmse = np.sqrt(mse)
r2 = 1 - np.sum(errors ** 2) / np.sum((y_true - y_true.mean()) ** 2)

print("NumPy results")
print(f"  MAE  = {mae:,.2f}\n  MSE  = {mse:,.2f}\n  RMSE = {rmse:,.2f}\n  R²   = {r2:.4f}")

# scikit-learn check
print("\nscikit-learn results")
print(f"  MAE  = {mean_absolute_error(y_true, y_pred):,.2f}")
print(f"  MSE  = {mean_squared_error(y_true, y_pred):,.2f}")
print(f"  RMSE = {np.sqrt(mean_squared_error(y_true, y_pred)):,.2f}")
print(f"  R²   = {r2_score(y_true, y_pred):.4f}")

assert np.isclose(mae, mean_absolute_error(y_true, y_pred))
assert np.isclose(r2, r2_score(y_true, y_pred))
print("\nAll values match.")
