# Task 2 – House-Price Predictor

**Skills used:** Linear and multiple regression, Scikit-learn, coefficients

## Goal
Predict the price of a house from its features using linear regression, compare a simple model with a multiple regression model, and explain the coefficients.

## Steps the script performs
1. Load the house data and handle missing values inside a Scikit-learn pipeline.
2. Train a **simple** linear regression (area only) and a **multiple** linear regression (all features).
3. Evaluate both on the validation set with MAE, RMSE and R².
4. Evaluate the chosen model on the test set.
5. Print the coefficients and predict the price of a new house.

## Run
```bash
python tasks/task2_house_price_predictor/house_price_predictor.py
```

## Deliverables
- Metric comparison (simple vs multiple regression)
- Coefficient table with a written interpretation (see the comments in the script)
