# Task 3 – Basic Classification Model

**Skills used:** Logistic regression, KNN, decision trees, feature importance

## Goal
Classify houses as **expensive** or **not expensive** (label `is_expensive`, created in Task 1). Compare four classifiers and report the best one.

## Steps the script performs
1. Create the label using the median price of the **training** set only.
2. Train logistic regression, KNN, decision tree and random forest.
3. Compare them on the validation set and pick the best.
4. Evaluate the chosen model on the test set with a confusion matrix and a classification report.
5. Show feature importance from the random forest.

## Run
```bash
python tasks/task3_classification_model/classification_model.py
```

## Deliverables
- Model comparison table
- Test-set confusion matrix and classification report
- Feature importance ranking and a short written conclusion
