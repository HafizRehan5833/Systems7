# System 7 Repository

Mentor: **Saad Hassan Saadi**
Program: System 7 · Week 2 · Beginner Level
Focus: Mathematics foundations and core machine learning algorithms

Work every day, commit every day. Each day has its own folder, and each day
ends with one commit named after that folder.

## Daily Plan

| Day | Topics | Folder (commit name) |
| --- | --- | --- |
| Day 1 | What is AI, ML and Deep Learning; supervised, unsupervised, reinforcement learning; dataset terms | `day-08-ml-foundations` |
| Day 2 | Mean, median, mode, variance, standard deviation | `day-09-descriptive-statistics` |
| Day 3 | Linear algebra basics: vectors, matrices, dot products | `day-10-linear-algebra` |
| Day 4 | Probability, correlation, train/validation/test split; dataset exploration task | `day-11-probability-data-prep` |
| Day 5 | Linear and multiple linear regression; classification vs. regression; Scikit-learn workflow | `day-12-regression-sklearn` |
| Day 6 | Logistic regression, KNN, decision trees, random forests | `day-13-classification-models` |
| Day 7 | Predictions, coefficients, feature importance; house-price and classification tasks | `day-14-house-price-classifier` |

## Practical Tasks

| # | Task | Folder |
| --- | --- | --- |
| 1 | Explore and prepare a dataset using Pandas and NumPy | `tasks/task1_dataset_preparation` |
| 2 | Build a house-price predictor | `tasks/task2_house_price_predictor` |
| 3 | Build a basic classification model | `tasks/task3_classification_model` |

## Folder Structure

```
System7-Repository/
├── README.md                      <- this file (overview and daily log)
├── requirements.txt               <- Python packages used in the repo
├── glossary/
│   └── DEFINITIONS.md             <- definitions of every term in the syllabus
├── day-08-ml-foundations/         <- Day 1
├── day-09-descriptive-statistics/ <- Day 2
├── day-10-linear-algebra/         <- Day 3
├── day-11-probability-data-prep/  <- Day 4
├── day-12-regression-sklearn/     <- Day 5
├── day-13-classification-models/  <- Day 6
├── day-14-house-price-classifier/ <- Day 7
└── tasks/
    ├── data/                      <- shared dataset (houses.csv) and generator
    ├── task1_dataset_preparation/
    ├── task2_house_price_predictor/
    └── task3_classification_model/
```

## Setup

```bash
pip install -r requirements.txt
python tasks/data/generate_houses.py          # creates tasks/data/houses.csv (already included)
python tasks/task1_dataset_preparation/prepare_dataset.py
python tasks/task2_house_price_predictor/house_price_predictor.py
python tasks/task3_classification_model/classification_model.py
```

## Daily Log

Update this list at the end of each day.

- [ ] Day 1 – ML foundations
- [ ] Day 2 – Descriptive statistics
- [ ] Day 3 – Linear algebra
- [ ] Day 4 – Probability, correlation, data splitting
- [ ] Day 5 – Regression and Scikit-learn workflow
- [ ] Day 6 – Classification models
- [ ] Day 7 – Predictions, coefficients, feature importance, Tasks 2 and 3
