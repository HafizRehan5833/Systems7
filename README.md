# System7 Project Summary

This repository contains the work completed across two internship tracks: a Python/data-analysis track and a machine-learning workflow track. The material is split between the root project documentation, the Task 1 exercises, and the Task 2 machine-learning tasks.

## Repository overview

- Root project: [README.md](./README.md)
- Python fundamentals and data analysis: [system7_task1/README.md](./system7_task1/README.md)
- Machine-learning study materials and tasks: [System7_task2/README.md](./System7_task2/README.md)
- ML glossary and terminology: [System7_task2/Definitions/DEFINITIONS.md](./System7_task2/Definitions/DEFINITIONS.md)

## 1) Internship foundation

The root README describes the internship as a collection of learning exercises and experiments, with a primary focus on Python fundamentals, data analysis, and applied ML. The work is organized by task and day, and the project emphasizes clean scripts, small runnable exercises, and generated outputs stored next to the code that produces them.

Core ideas from the root documentation:

- Task 1 is built around Python and data-analysis foundations.
- Training uses `uv` for dependency management.
- Scripts are designed to be small, readable, and runnable from their documented commands.
- Generated charts and datasets are kept alongside the source files.
- TensorFlow is optional and not required for the default setup.

## 2) Task 1: Python and data analysis

The detailed documentation for the first track is in [system7_task1/README.md](./system7_task1/README.md).

This module includes:

- Python basics: variables, types, operators, input, loop logic, and functions
- Data structures: lists, dictionaries, tuples, and sets
- Object-oriented programming with small apps such as bank account and student management
- NumPy operations and numerical computing
- Data cleaning with Pandas and CSV processing
- Visualization with Matplotlib and Seaborn
- Summary statistics and correlation analysis
- A short TensorFlow introduction

### Task 1 exercise table

| Day | File | Topic |
| --- | --- | --- |
| 1 | [system7_task1/day1/basics.py](./system7_task1/day1/basics.py) | Python variables, operators, input, and output |
| 2 | [system7_task1/day2/calculator_game.py](./system7_task1/day2/calculator_game.py) | Calculator game and control flow |
| 3 | [system7_task1/day3/data_structures.py](./system7_task1/day3/data_structures.py) | Python data structures |
| 4 | [system7_task1/day4/bank_account.py](./system7_task1/day4/bank_account.py) | Classes and bank-account operations |
| 4 | [system7_task1/day4/student_manager.py](./system7_task1/day4/student_manager.py) | Student management |
| 5 | [system7_task1/day5/numpy_ops.py](./system7_task1/day5/numpy_ops.py) | NumPy operations |
| 6 | [system7_task1/day6/clean_data.py](./system7_task1/day6/clean_data.py) | Create and clean a student dataset |
| 7 | [system7_task1/day7/analysis.py](./system7_task1/day7/analysis.py) | CSV analysis, statistics, and charts |
| 7 | [system7_task1/day7/tensorflow_intro.py](./system7_task1/day7/tensorflow_intro.py) | TensorFlow tensors and variables |

### Day 7 results captured in the docs

The Task 1 documentation reports the following analysis results from the cleaned student dataset:

- 190 cleaned student records were analyzed
- Mean score: 74.12
- Highest department mean: AI, with 75.40
- Study-hours-to-score correlation: 0.82

Generated outputs include summary reports and charts in [system7_task1/day7](./system7_task1/day7).

## 3) Task 2: Machine learning and model-building workflow

The task 2 materials are centered on a machine-learning study path covering probability, correlation, data splitting, regression, and classification.

### Day 4: Probability, correlation, and data splitting

From [System7_task2/day-04/README.md](./System7_task2/day-04/README.md):

- Topic: probability, correlation, basic statistics
- Focus: training, validation, and testing datasets
- Example file: [System7_task2/day-04/example.py](./System7_task2/day-04/example.py)
- Goal: estimate a probability, compute a correlation matrix, and split data into 60/20/20 train-validation-test groups

### Day 5: Regression and the scikit-learn workflow

From [System7_task2/day-05/README.md](./System7_task2/day-05/README.md):

- Topic: linear and multiple regression
- Focus: classification versus regression
- Example file: [System7_task2/day-05/example.py](./System7_task2/day-05/example.py)
- Goal: follow the four-stage scikit-learn workflow: split, fit, predict, evaluate

### Day 6: Classification models

From [System7_task2/day-06/README.md](./System7_task2/day-06/README.md):

- Topic: logistic regression, KNN, decision trees, and random forests
- Example file: [System7_task2/day-06/example.py](./System7_task2/day-06/example.py)
- Goal: train multiple classifiers and compare their accuracy

### Day 7: Predictions, coefficients, and feature importance

From [System7_task2/day-07/README.md](./System7_task2/day-07/README.md):

- Topic: prediction, coefficients, feature importance
- Focus: work for the Task 2 house-price predictor and the Task 3 classification model
- Example file: [System7_task2/day-07/example.py](./System7_task2/day-07/example.py)
- Goal: read model coefficients and tree feature importance on the housing dataset

## 4) Task 2 data and project tasks

The weekday exercises lead into the end-of-module task implementations under [System7_task2/tasks](./System7_task2/tasks).

### Task 1: Explore and prepare a dataset

From [System7_task2/tasks/task1_dataset_preparation/README.md](./System7_task2/tasks/task1_dataset_preparation/README.md):

- Goal: load and clean the house dataset, generate a useful classification label, and split data into train/validation/test sets
- Script: [System7_task2/tasks/task1_dataset_preparation/prepare_dataset.py](./System7_task2/tasks/task1_dataset_preparation/prepare_dataset.py)
- Output files: [System7_task2/tasks/data/processed/train.csv](./System7_task2/tasks/data/processed/train.csv), [System7_task2/tasks/data/processed/val.csv](./System7_task2/tasks/data/processed/val.csv), [System7_task2/tasks/data/processed/test.csv](./System7_task2/tasks/data/processed/test.csv)
- Workflow:
  1. Read the house dataset
  2. Inspect shape, dtypes, and summary statistics
  3. Fill missing values with median values
  4. Remove duplicates
  5. Create `is_expensive` label
  6. Save processed splits

### Task 2: House-price predictor

From [System7_task2/tasks/task2_house_price_predictor/README.md](./System7_task2/tasks/task2_house_price_predictor/README.md):

- Goal: predict house price using linear regression models
- Script: [System7_task2/tasks/task2_house_price_predictor/house_price_predictor.py](./System7_task2/tasks/task2_house_price_predictor/house_price_predictor.py)
- Workflow:
  1. Load the dataset and handle missing values with a scikit-learn pipeline
  2. Train a simple linear model using area only
  3. Train a multiple regression model with all features
  4. Compare performance (MAE, RMSE, R²)
  5. Print coefficients and predict a new house price

### Task 3: Basic classification model

From [System7_task2/tasks/task3_classification_model/README.md](./System7_task2/tasks/task3_classification_model/README.md):

- Goal: classify houses as expensive or not expensive
- Script: [System7_task2/tasks/task3_classification_model/classification_model.py](./System7_task2/tasks/task3_classification_model/classification_model.py)
- Workflow:
  1. Create the label using the training median price
  2. Train logistic regression, KNN, decision tree, and random forest models
  3. Compare performance on validation data
  4. Evaluate the selected model on the test set
  5. Display confusion matrix, classification report, and feature importance

## 5) ML glossary and definitions

The definitions file, [System7_task2/Definitions/DEFINITIONS.md](./System7_task2/Definitions/DEFINITIONS.md), provides a compact glossary covering:

- ML foundations: AI, ML, deep learning, supervised learning, unsupervised learning, reinforcement learning
- Statistics: mean, median, mode, variance, standard deviation, probability, correlation
- Linear algebra: vectors, matrices, dot products
- Data splitting: training, validation, and testing sets
- Models: linear regression, multiple regression, logistic regression, KNN, decision trees, random forests
- Scikit-learn workflow: estimator, `fit(X, y)`, `predict(X)`, coefficients, feature importance, pipelines

This glossary acts as a quick reference for the concepts used throughout the ML exercises.

## 6) Practical commands

The docs reference the following commands for setup and execution.

### Task 1 setup

```powershell
cd system7_task1
uv sync
uv run python .\day1\basics.py
uv run python .\day6\clean_data.py
uv run python .\day7\analysis.py
```

### Task 2 script execution

```bash
python tasks/task1_dataset_preparation/prepare_dataset.py
python tasks/task2_house_price_predictor/house_price_predictor.py
python tasks/task3_classification_model/classification_model.py
```

## Final takeaway

Together, these files show a complete internship journey:

- Python fundamentals and data analysis in Task 1
- Statistical and ML concepts in the day-by-day learning path
- Real ML workflows in the Task 2 dataset preparation, regression, and classification tasks

This single summary consolidates the documentation into one place for easier review and navigation.
