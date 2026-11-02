# Module A Practical Task – Compare Multiple Models and Justify the Best

## Goal
Compare several models on a classification problem and a regression problem, using the same evaluation method for each, then choose a best model and justify the choice.

## Steps the script performs
1. **Classification** (breast cancer dataset, built into scikit-learn): 5-fold cross-validation of logistic regression, KNN, decision tree, SVM and random forest on accuracy, F1 and ROC-AUC.
2. **Regression** (house data from `tasks/data/`): 5-fold cross-validation of linear regression, decision tree and random forest on MAE and RMSE.
3. Print a comparison table for each problem and name the best model by the main metric.

## Run
```bash
python tasks/moduleA_model_comparison/compare_models.py
```

## Deliverables
- The printed comparison tables
- `JUSTIFICATION.md` in this folder: one paragraph per problem explaining why the best model wins, with at least one trade-off (speed, interpretability, or risk of overfitting). Fill it in yourself after running the script.
