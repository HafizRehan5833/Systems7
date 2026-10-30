# Systems 7 — Machine Learning & Deep Learning Engineering Repository

This repository represents the comprehensive work, practice exercises, and end-to-end practical tasks completed across all three progressive tracks of the **Systems 7** program:
1. **Track 1 (`system7_task1`)**: Python Foundations, Data Structures, OOP, and Data Analysis
2. **Track 2 (`System7_task2`)**: Mathematics, Descriptive Statistics, and Classical Machine Learning Workflows
3. **Track 3 (`System7-task3`)**: Advanced Model Evaluation, Hyperparameter Tuning, Deep Learning, and CNNs

> [!IMPORTANT]
> **Execution Records (`Run_file/`):**
> For every executed script and practice question, output records and terminal screenshots are organized inside each day's dedicated `Run_file/` folder (e.g. `day-15-model-evaluation/Run_file/`, `day-16-tuning-model-comparison/Run_file/`), providing visual proof and verified results for every run.

---

## Repository Architecture & Directory Structure

```
systems7/
├── README.md                                  <- Root repository guide and summary
├── .gitignore
│
├── system7_task1/                             <- TRACK 1: Python & Data Analysis Foundations
│   ├── README.md                              <- Task 1 detailed documentation
│   ├── pyproject.toml / uv.lock               <- Dependency management (uv)
│   ├── day1/                                  <- Python variables, syntax, I/O (basics.py)
│   ├── day2/                                  <- Control flow & game loops (calculator_game.py)
│   ├── day3/                                  <- Lists, dictionaries, tuples, sets (data_structures.py)
│   ├── day4/                                  <- OOP: Bank Account & Student Manager
│   ├── day5/                                  <- Numerical computing with NumPy (numpy_ops.py)
│   ├── day6/                                  <- Data cleaning & preprocessing with Pandas (clean_data.py)
│   └── day7/                                  <- EDA, visualizations & TensorFlow intro (analysis.py)
│
├── System7_task2/                             <- TRACK 2: ML Foundations & Core Algorithms
│   ├── README.md                              <- Task 2 detailed documentation
│   ├── Definitions/
│   │   └── DEFINITIONS.md                     <- Comprehensive ML & mathematical glossary
│   ├── day-01/                                <- ML Foundations: Supervised, Unsupervised, RL
│   ├── day-02/                                <- Descriptive Statistics: Mean, Variance, Std Dev
│   ├── day-03/                                <- Linear Algebra: Vectors, Matrices, Dot Products
│   ├── day-04/                                <- Probability, Correlation, Train/Val/Test Split
│   ├── day-05/                                <- Linear & Multiple Regression, Scikit-Learn Workflow
│   ├── day-06/                                <- Classification: Logistic Regression, KNN, Trees, Forests
│   ├── day-07/                                <- Predictions, Coefficients, Feature Importance
│   └── tasks/                                 <- End-of-week practical tasks
│       ├── data/                              <- Raw and processed datasets (train.csv, val.csv, test.csv)
│       ├── task1_dataset_preparation/         <- Exploration, median imputation, labeling
│       ├── task2_house_price_predictor/       <- Multi-feature regression pipeline
│       └── task3_classification_model/        <- Multi-classifier comparison & evaluation
│
└── System7-task3/                             <- TRACK 3: Advanced ML, Tuning, Deep Learning & Vision
    ├── README.md                              <- Task 3 detailed syllabus and daily logs
    ├── requirements.txt / pyproject.toml      <- Dependencies for Week 3
    ├── day-15-model-evaluation/               <- Day 15: Cross-validation, Metrics, ROC-AUC
    │   ├── practice/                          <- Questions Q01–Q05
    │   ├── figures/                           <- ROC and PR curve plots
    │   └── Run_file/                          <- Execution screenshots: q01.png – q05.png
    ├── day-16-tuning-model-comparison/        <- Day 16: ColumnTransformer, GridSearchCV, Overfitting
    │   ├── README.md                          <- Day 16 notes & task guide
    │   ├── practice/                          <- Questions Q06–Q08
    │   ├── figures/                           <- Overfitting vs. tree depth curves
    │   └── Run_file/                          <- Execution screenshots: q06.png, Q07.png, Q08.png
    ├── day-17-svm-clustering/                 <- Day 17: SVM, Naive Bayes, K-Means, DBSCAN
    │   ├── practice/                          <- Questions Q09–Q10
    │   └── figures/                           <- Elbow method & clustering visualizer
    ├── day-18-pca-pipelines/                  <- Day 18: PCA, Class Imbalance, Pipeline Leakage
    │   ├── practice/                          <- Questions Q11–Q13
    │   └── figures/                           <- PCA 2D projections
    ├── day-19-neural-networks/                <- Day 19: Perceptrons, Backprop, Keras Optimizers
    │   ├── practice/                          <- Questions Q14–Q15
    │   └── figures/                           <- Loss curves (SGD vs. Adam)
    ├── day-20-cnn-basics/                     <- Day 20: Preprocessing, Augmentation, CIFAR-10 CNN
    │   └── practice/                          <- Questions Q16–Q17
    ├── day-21-image-classifier/               <- Day 21: Transfer Learning vs. Scratch, MobileNetV2
    │   └── practice/                          <- Question Q18
    └── syllabus Modules/                      <- Week 3 Capstone Modules & Datasets
        ├── data/                              <- Shared datasets (houses.csv, customers.csv, students.csv)
        ├── moduleA_model_comparison/          <- Module A: Model comparison & JUSTIFICATION.md
        ├── moduleB_customer_segmentation/     <- Module B: Clustering & customer profiling
        ├── moduleC_student_results/           <- Module C: Deep Learning student grade predictor
        ├── moduleD_image_classifier/          <- Module D: Image classification with CNNs
        ├── task1_dataset_preparation/         <- Reference preparation scripts
        ├── task2_house_price_predictor/       <- Reference regression scripts
        └── task3_classification_model/        <- Reference classification scripts
```

---

## Track Breakdown & Syllabus Guidance

### Track 1: Python & Data Analysis Foundations (`system7_task1/`)

Track 1 covers software engineering fundamentals in Python, scientific computing, data manipulation, and exploratory data analysis.

| Day | Primary Script | Core Concept & Output |
| :--- | :--- | :--- |
| **Day 1** | [`system7_task1/day1/basics.py`](./system7_task1/day1/basics.py) | Dynamic typing, input sanitization, math operators |
| **Day 2** | [`system7_task1/day2/calculator_game.py`](./system7_task1/day2/calculator_game.py) | Game loops, conditionals, defensive input validation |
| **Day 3** | [`system7_task1/day3/data_structures.py`](./system7_task1/day3/data_structures.py) | Collections: Lists, Tuples, Sets, Dictionaries, lookups |
| **Day 4** | [`system7_task1/day4/bank_account.py`](./system7_task1/day4/bank_account.py)<br>[`system7_task1/day4/student_manager.py`](./system7_task1/day4/student_manager.py) | Object-Oriented Programming (encapsulation, state, methods) |
| **Day 5** | [`system7_task1/day5/numpy_ops.py`](./system7_task1/day5/numpy_ops.py) | Vectorized math, multi-dimensional array slicing, broadcasting |
| **Day 6** | [`system7_task1/day6/clean_data.py`](./system7_task1/day6/clean_data.py) | Handling missing values, deduplication, filtering in Pandas |
| **Day 7** | [`system7_task1/day7/analysis.py`](./system7_task1/day7/analysis.py)<br>[`system7_task1/day7/tensorflow_intro.py`](./system7_task1/day7/tensorflow_intro.py) | Statistical EDA, distributions, heatmaps, introductory tensors |

---

### Track 2: Classical Machine Learning & Statistical Foundations (`System7_task2/`)

Track 2 establishes the core mathematical concepts and classical ML workflows through Scikit-Learn.

#### Daily Syllabus:
- **Day 01:** AI vs. Machine Learning vs. Deep Learning; Supervised, Unsupervised, Reinforcement Learning.
- **Day 02:** Descriptive Statistics: Mean, Median, Mode, Variance, Standard Deviation.
- **Day 03:** Linear Algebra: Vector spaces, matrix-vector multiplication, dot products.
- **Day 04:** Probability, correlation matrices, and 60/20/20 train-val-test dataset splitting.
- **Day 05:** Linear and multiple regression; the 4-step Scikit-Learn API: `split -> fit -> predict -> evaluate`.
- **Day 06:** Classification algorithms: Logistic Regression, K-Nearest Neighbors, Decision Trees, Random Forests.
- **Day 07:** Model inference, inspecting coefficients ($\beta$), and feature importance.
- **ML Glossary:** Comprehensive definitions available in [`System7_task2/Definitions/DEFINITIONS.md`](./System7_task2/Definitions/DEFINITIONS.md).

#### Practical Tasks:
1. **Dataset Preparation (`tasks/task1_dataset_preparation/`):** Missing-value imputation with medians, price thresholding, and generating `train.csv`, `val.csv`, `test.csv`.
2. **House Price Predictor (`tasks/task2_house_price_predictor/`):** Simple vs. Multiple Linear Regression comparison evaluated on MAE, RMSE, and $R^2$.
3. **Classification Model (`tasks/task3_classification_model/`):** Comparing tree ensembles and linear classifiers with confusion matrices and classification reports.

---

### Track 3: Advanced ML, Tuning, Neural Networks & Vision (`System7-task3/`)

Track 3 focuses on production-grade machine learning pipelines, hyperparameter search, deep neural networks, and computer vision.

#### Practice Questions & Verified Execution Records:
Every practice question script is paired with an output screenshot in its day's `Run_file/` folder:

| # | Difficulty | Day | Script Path | Output Record (`Run_file/`) |
| :-: | :--- | :-: | :--- | :--- |
| **Q01** | Easy | 15 | [`day-15-model-evaluation/practice/q01_split_stratify.py`](./System7-task3/day-15-model-evaluation/practice/q01_split_stratify.py) | [`day-15-model-evaluation/Run_file/q01.png`](./System7-task3/day-15-model-evaluation/Run_file/q01.png) |
| **Q02** | Easy | 15 | [`day-15-model-evaluation/practice/q02_classification_metrics.py`](./System7-task3/day-15-model-evaluation/practice/q02_classification_metrics.py) | [`day-15-model-evaluation/Run_file/q02.png`](./System7-task3/day-15-model-evaluation/Run_file/q02.png) |
| **Q03** | Easy | 15 | [`day-15-model-evaluation/practice/q03_regression_metrics.py`](./System7-task3/day-15-model-evaluation/practice/q03_regression_metrics.py) | [`day-15-model-evaluation/Run_file/q03.png`](./System7-task3/day-15-model-evaluation/Run_file/q03.png) |
| **Q04** | Medium | 15 | [`day-15-model-evaluation/practice/q04_cross_validation.py`](./System7-task3/day-15-model-evaluation/practice/q04_cross_validation.py) | [`day-15-model-evaluation/Run_file/q04.png`](./System7-task3/day-15-model-evaluation/Run_file/q04.png) |
| **Q05** | Medium | 15 | [`day-15-model-evaluation/practice/q05_roc_curves.py`](./System7-task3/day-15-model-evaluation/practice/q05_roc_curves.py) | [`day-15-model-evaluation/Run_file/q05.png`](./System7-task3/day-15-model-evaluation/Run_file/q05.png) |
| **Q06** | Medium | 16 | [`day-16-tuning-model-comparison/practice/q06_tree_depth_overfitting.py`](./System7-task3/day-16-tuning-model-comparison/practice/q06_tree_depth_overfitting.py) | [`day-16-tuning-model-comparison/Run_file/q06.png`](./System7-task3/day-16-tuning-model-comparison/Run_file/q06.png) |
| **Q07** | Medium | 16 | [`day-16-tuning-model-comparison/practice/q07_column_transformer.py`](./System7-task3/day-16-tuning-model-comparison/practice/q07_column_transformer.py) | [`day-16-tuning-model-comparison/Run_file/Q07.png`](./System7-task3/day-16-tuning-model-comparison/Run_file/Q07.png) |
| **Q08** | Medium | 16 | [`day-16-tuning-model-comparison/practice/q08_svm_tuning.py`](./System7-task3/day-16-tuning-model-comparison/practice/q08_svm_tuning.py) | [`day-16-tuning-model-comparison/Run_file/Q08.png`](./System7-task3/day-16-tuning-model-comparison/Run_file/Q08.png) |
| **Q09** | Medium | 17 | `day-17-svm-clustering/practice/q09_kmeans_choose_k.py` | `day-17-svm-clustering/Run_file/` |
| **Q10** | Medium | 17 | `day-17-svm-clustering/practice/q10_naive_bayes_vs_svm.py` | `day-17-svm-clustering/Run_file/` |
| **Q11** | Medium | 18 | `day-18-pca-pipelines/practice/q11_imbalanced_class_weight.py` | `day-18-pca-pipelines/Run_file/` |
| **Q12** | Hard | 18 | `day-18-pca-pipelines/practice/q12_pca_2d.py` | `day-18-pca-pipelines/Run_file/` |
| **Q13** | Hard | 18 | `day-18-pca-pipelines/practice/q13_pipeline_leakage.py` | `day-18-pca-pipelines/Run_file/` |
| **Q14** | Easy | 19 | `day-19-neural-networks/practice/q14_perceptron_and.py` | `day-19-neural-networks/Run_file/` |
| **Q15** | Medium | 19 | `day-19-neural-networks/practice/q15_keras_sgd_vs_adam.py` | `day-19-neural-networks/Run_file/` |
| **Q16** | Medium | 20 | `day-20-cnn-basics/practice/q16_resize_augment.py` | `day-20-cnn-basics/Run_file/` |
| **Q17** | Hard | 20 | `day-20-cnn-basics/practice/q17_cnn_cifar10.py` | `day-20-cnn-basics/Run_file/` |
| **Q18** | Hard | 21 | `day-21-image-classifier/practice/q18_transfer_vs_scratch.py` | `day-21-image-classifier/Run_file/` |

#### End-of-Week Capstone Tasks (`syllabus Modules/`):
- **Module A — Model Comparison:** Systematic comparison across multiple regression and classification estimators with benchmark write-up in `JUSTIFICATION.md`.
- **Module B — Customer Segmentation:** Unsupervised clustering (K-Means, Hierarchical, DBSCAN) combined with 2D/3D PCA visualizations.
- **Module C — Neural Network Predictor:** Multi-layer Perceptron (MLP) built with Keras/TensorFlow to predict student academic outcomes.
- **Module D — Image Classifier:** End-to-end vision pipeline utilizing Convolutional Neural Networks and MobileNetV2 transfer learning on CIFAR-10.

---

## Execution & Setup Guidance

This repository uses modern, fast Python package tooling via [`uv`](https://github.com/astral-sh/uv).

### 1. Environment Setup

#### Track 1 (`system7_task1`):
```powershell
cd system7_task1
uv sync
```

#### Track 3 (`System7-task3`):
```powershell
cd System7-task3
uv sync
```

---

### 2. Running Practice Scripts

To run any practice script inside its isolated virtual environment, use `uv run`:

```powershell
# Example: Run Day 16 ColumnTransformer practice
cd System7-task3
uv run .\day-16-tuning-model-comparison\practice\q07_column_transformer.py

# Example: Run Day 16 Overfitting inspection
uv run .\day-16-tuning-model-comparison\practice\q06_tree_depth_overfitting.py

# Example: Run Day 16 SVM tuning comparison
uv run .\day-16-tuning-model-comparison\practice\q08_svm_tuning.py
```

---

### 3. Running Module Capstone Tasks

```powershell
cd System7-task3

# Module A: Model Comparison
uv run ".\syllabus Modules\moduleA_model_comparison\compare_models.py"

# Module B: Customer Segmentation
uv run ".\syllabus Modules\moduleB_customer_segmentation\segment_customers.py"

# Module C: Student Results Deep Learning
uv run ".\syllabus Modules\moduleC_student_results\predict_student_results.py"

# Module D: CNN Image Classifier
uv run ".\syllabus Modules\moduleD_image_classifier\image_classifier.py"
```

---

## Output Verification & Record Keeping

When running scripts in this repository:
1. **Plots & Visualizations:** Auto-saved into each day's `figures/` directory (e.g. `day-15-model-evaluation/figures/`, `day-16-tuning-model-comparison/figures/`).
2. **Terminal Records & Screenshots:** Stored in each day's `Run_file/` directory (e.g. `Run_file/q06.png`, `Run_file/Q07.png`, `Run_file/Q08.png`) to preserve reproducible execution proof.
