# Day 1 – Model Evaluation (Module A)

**Commit:** `day-15-model-evaluation`

## Topics
- Train-test split and cross-validation
- Accuracy, precision, recall, F1-score
- Confusion matrix and ROC-AUC
- MAE, MSE, RMSE, and R²

## Practice questions
| # | Level | File | Question |
| --- | --- | --- | --- |
| 1 | Easy | `practice/q01_split_stratify.py` | 80/20 stratified split, shape and class balance |
| 2 | Easy | `practice/q02_classification_metrics.py` | Accuracy, precision, recall, F1 and confusion matrix |
| 3 | Easy | `practice/q03_regression_metrics.py` | MAE, MSE, RMSE, R² with NumPy, verified with scikit-learn |
| 4 | Medium | `practice/q04_cross_validation.py` | 5-fold CV on three models: mean and std |
| 5 | Medium | `practice/q05_roc_curves.py` | ROC curves and AUC for logistic regression and random forest |

Run any question with:
```bash
python day-15-model-evaluation/practice/q01_split_stratify.py
```

## Learning notes
Write a short note in your own words for each topic. Use these prompts:
- **Choosing the right metric:** When does recall matter more than precision? Why can accuracy mislead on imbalanced data?
  - My note:

## Completed today
- [ ] Solved Q1 to Q5 (each in its own file)
- [ ] Wrote the learning note above
