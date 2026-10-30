from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "houses.csv"
FEATURES = ["area_sqft", "bedrooms", "bathrooms", "age_years", "distance_km"]
TARGET = "price"


def evaluate(name, model, X_val, y_val):
    pred = model.predict(X_val)
    mae = mean_absolute_error(y_val, pred)
    rmse = np.sqrt(mean_squared_error(y_val, pred))
    r2 = r2_score(y_val, pred)
    print(f"{name:<22} MAE={mae:>10,.0f}  RMSE={rmse:>10,.0f}  R²={r2:.3f}")
    return r2


def main() -> None:
    df = pd.read_csv(DATA_PATH).drop_duplicates()
    train_val, test = train_test_split(df, test_size=0.20, random_state=42)
    train, val = train_test_split(train_val, test_size=0.25, random_state=42)

    # Pipeline: fill missing values with the median, then fit the regression.
    # Putting the imputer inside the pipeline stops validation data leaking into training.
    def make_model():
        return make_pipeline(SimpleImputer(strategy="median"), LinearRegression())

    print("=== Validation results ===")
    simple = make_model().fit(train[["area_sqft"]], train[TARGET])
    evaluate("Simple (area only)", simple, val[["area_sqft"]], val[TARGET])

    multi = make_model().fit(train[FEATURES], train[TARGET])
    r2_multi = evaluate("Multiple (5 features)", multi, val[FEATURES], val[TARGET])

    # Coefficients: the change in price for a one-unit increase in each feature,
    # holding the other features fixed. Example reading: if the area coefficient
    # is 120, each extra square foot adds about 120 to the predicted price.
    coefs = pd.Series(multi.named_steps["linearregression"].coef_, index=FEATURES)
    print("\n=== Coefficients (multiple regression) ===")
    print(coefs.round(1).to_string())
    print(f"Intercept: {multi.named_steps['linearregression'].intercept_:,.1f}")

    # Final check on the untouched test set
    print("\n=== Test set (multiple regression) ===")
    pred_test = multi.predict(test[FEATURES])
    print(f"MAE={mean_absolute_error(test[TARGET], pred_test):,.0f}  R²={r2_score(test[TARGET], pred_test):.3f}")

    # Predict a new house
    new_house = pd.DataFrame([{
        "area_sqft": 1800, "bedrooms": 3, "bathrooms": 2,
        "age_years": 8, "distance_km": 5.0,
    }])
    price = multi.predict(new_house)[0]
    print(f"\nPredicted price for a 1800 sq ft, 3-bed, 2-bath, 8-year-old house 5 km out: {price:,.0f}")

    print("\nStudent note: write one sentence explaining the sign (+/-) of the age and distance coefficients.")


if __name__ == "__main__":
    main()
