# System 7 Repository

Mentor: **Saad Hassan Saadi**
Program: System 7 · Beginner to Intermediate
Weeks covered: Week 2 (Mathematics and ML Basics) and Week 3 (Model Evaluation, Advanced ML and Deep Learning)

Work every day, commit every day. Each day has its own folder, and each day
ends with one commit named after that folder.

## Daily Plan

### Week 2 – Mathematics and Machine Learning Basics

| Day | Topics | Folder (commit name) |
| --- | --- | --- |
| Day 1 | What is AI, ML and Deep Learning; supervised, unsupervised, reinforcement learning; dataset terms | `day-08-ml-foundations` |
| Day 2 | Mean, median, mode, variance, standard deviation | `day-09-descriptive-statistics` |
| Day 3 | Linear algebra basics: vectors, matrices, dot products | `day-10-linear-algebra` |
| Day 4 | Probability, correlation, train/validation/test split; dataset exploration task | `day-11-probability-data-prep` |
| Day 5 | Linear and multiple linear regression; classification vs. regression; Scikit-learn workflow | `day-12-regression-sklearn` |
| Day 6 | Logistic regression, KNN, decision trees, random forests | `day-13-classification-models` |
| Day 7 | Predictions, coefficients, feature importance; house-price and classification tasks | `day-14-house-price-classifier` |

### Week 3 – Model Evaluation, Advanced ML and Deep Learning

| Day | Module | Topics | Folder (commit name) |
| --- | --- | --- | --- |
| Day 1 | A | Train-test split, cross-validation, classification and regression metrics, confusion matrix, ROC-AUC | `day-15-model-evaluation` |
| Day 2 | A | Over/underfitting, scaling, encoding, GridSearchCV, RandomizedSearchCV; model comparison task | `day-16-tuning-model-comparison` |
| Day 3 | B | SVM, Naive Bayes, K-Means, hierarchical clustering, DBSCAN | `day-17-svm-clustering` |
| Day 4 | B | PCA, feature engineering, imbalanced data, leakage, pipelines; segmentation task | `day-18-pca-pipelines` |
| Day 5 | C | Perceptrons, backpropagation, gradient descent, optimizers, Keras workflow; student results task | `day-19-neural-networks` |
| Day 6 | D | Image preprocessing, augmentation, convolution, pooling, CNN basics | `day-20-cnn-basics` |
| Day 7 | D | Multiclass classification, transfer learning, evaluation; image classifier task | `day-21-image-classifier` |

## Practical Tasks

### Week 2

| # | Task | Folder |
| --- | --- | --- |
| 1 | Explore and prepare a dataset using Pandas and NumPy | `tasks/task1_dataset_preparation` |
| 2 | Build a house-price predictor | `tasks/task2_house_price_predictor` |
| 3 | Build a basic classification model | `tasks/task3_classification_model` |

### Week 3

| Module | Task | Folder |
| --- | --- | --- |
| A | Compare multiple models and justify the best-performing model | `tasks/moduleA_model_comparison` |
| B | Customer segmentation | `tasks/moduleB_customer_segmentation` |
| C | Predict student results using a neural network | `tasks/moduleC_student_results` |
| D | Develop an image classifier | `tasks/moduleD_image_classifier` |

## Practice Questions (Week 3)

Each question is a separate file in its day's `practice/` folder. Execution output screenshots and records for running files are stored inside each day's `Run_file/` folder.

| # | Level | Day | File | Run Record |
| --- | --- | --- | --- | --- |
| 1 | Easy | 15 | `day-15-model-evaluation/practice/q01_split_stratify.py` | [`day-15-model-evaluation/Run_file/q01.png`](day-15-model-evaluation/Run_file/q01.png) |
| 2 | Easy | 15 | `day-15-model-evaluation/practice/q02_classification_metrics.py` | [`day-15-model-evaluation/Run_file/q02.png`](day-15-model-evaluation/Run_file/q02.png) |
| 3 | Easy | 15 | `day-15-model-evaluation/practice/q03_regression_metrics.py` | [`day-15-model-evaluation/Run_file/q03.png`](day-15-model-evaluation/Run_file/q03.png) |
| 4 | Medium | 15 | `day-15-model-evaluation/practice/q04_cross_validation.py` | [`day-15-model-evaluation/Run_file/q04.png`](day-15-model-evaluation/Run_file/q04.png) |
| 5 | Medium | 15 | `day-15-model-evaluation/practice/q05_roc_curves.py` | [`day-15-model-evaluation/Run_file/q05.png`](day-15-model-evaluation/Run_file/q05.png) |
| 6 | Medium | 16 | `day-16-tuning-model-comparison/practice/q06_tree_depth_overfitting.py` | [`day-16-tuning-model-comparison/Run_file/q06.png`](day-16-tuning-model-comparison/Run_file/q06.png) |
| 7 | Medium | 16 | `day-16-tuning-model-comparison/practice/q07_column_transformer.py` | [`day-16-tuning-model-comparison/Run_file/Q07.png`](day-16-tuning-model-comparison/Run_file/Q07.png) |
| 8 | Medium | 16 | `day-16-tuning-model-comparison/practice/q08_svm_tuning.py` | [`day-16-tuning-model-comparison/Run_file/Q08.png`](day-16-tuning-model-comparison/Run_file/Q08.png) |
| 9 | Medium | 17 | `day-17-svm-clustering/practice/q09_kmeans_choose_k.py` | — |
| 10 | Medium | 17 | `day-17-svm-clustering/practice/q10_naive_bayes_vs_svm.py` | — |
| 11 | Medium | 18 | `day-18-pca-pipelines/practice/q11_imbalanced_class_weight.py` | — |
| 12 | Hard | 18 | `day-18-pca-pipelines/practice/q12_pca_2d.py` | — |
| 13 | Hard | 18 | `day-18-pca-pipelines/practice/q13_pipeline_leakage.py` | — |
| 14 | Easy | 19 | `day-19-neural-networks/practice/q14_perceptron_and.py` | — |
| 15 | Medium | 19 | `day-19-neural-networks/practice/q15_keras_sgd_vs_adam.py` | — |
| 16 | Medium | 20 | `day-20-cnn-basics/practice/q16_resize_augment.py` | — |
| 17 | Hard | 20 | `day-20-cnn-basics/practice/q17_cnn_cifar10.py` | — |
| 18 | Hard | 21 | `day-21-image-classifier/practice/q18_transfer_vs_scratch.py` | — |

## Folder Structure

```
System7-Repository/
├── README.md                          <- this file (overview and daily log)
├── requirements.txt                   <- Python packages used in the repo
├── .gitignore
├── glossary/
│   └── DEFINITIONS.md                 <- definitions of every term in Weeks 2 and 3
├── day-08-ml-foundations/             <- Week 2, Day 1
├── day-09-descriptive-statistics/     <- Week 2, Day 2
├── day-10-linear-algebra/             <- Week 2, Day 3
├── day-11-probability-data-prep/      <- Week 2, Day 4
├── day-12-regression-sklearn/         <- Week 2, Day 5
├── day-13-classification-models/      <- Week 2, Day 6
├── day-14-house-price-classifier/     <- Week 2, Day 7
├── day-15-model-evaluation/           <- Week 3, Day 1 (Module A)
│   ├── practice/                      <- Questions 1–5, one file each
│   ├── figures/                       <- plots created by the practice scripts
│   └── Run_file/                      <- execution records / screenshots of outputs (q01–q05)
├── day-16-tuning-model-comparison/    <- Week 3, Day 2 (Module A)
│   ├── README.md                      <- topics, practice list, learning notes
│   ├── practice/                      <- Questions 6–8, one file each
│   ├── figures/                       <- plots created by the practice scripts
│   └── Run_file/                      <- execution records / screenshots of outputs (q06–q08)
├── day-17-svm-clustering/             <- Week 3, Day 3 (Module B)
├── day-18-pca-pipelines/              <- Week 3, Day 4 (Module B)
├── day-19-neural-networks/            <- Week 3, Day 5 (Module C)
├── day-20-cnn-basics/                 <- Week 3, Day 6 (Module D)
├── day-21-image-classifier/           <- Week 3, Day 7 (Module D)
└── tasks/
    ├── data/                          <- shared datasets and their generators
    │   ├── houses.csv, generate_houses.py
    │   ├── customers.csv, generate_customers.py
    │   ├── students.csv, generate_students.py
    │   ├── image_data.py              <- CIFAR-10 loader (downloads on first use)
    │   └── processed/                 <- Week 2 Task 1 output
    ├── task1_dataset_preparation/     <- Week 2 Task 1
    ├── task2_house_price_predictor/   <- Week 2 Task 2
    ├── task3_classification_model/    <- Week 2 Task 3
    ├── moduleA_model_comparison/      <- Week 3 Module A task
    ├── moduleB_customer_segmentation/ <- Week 3 Module B task
    ├── moduleC_student_results/       <- Week 3 Module C task
    └── moduleD_image_classifier/      <- Week 3 Module D task
```

## Setup

```bash
pip install -r requirements.txt

# Week 2 tasks
python tasks/task1_dataset_preparation/prepare_dataset.py
python tasks/task2_house_price_predictor/house_price_predictor.py
python tasks/task3_classification_model/classification_model.py

# Week 3 tasks
python tasks/moduleA_model_comparison/compare_models.py
python tasks/moduleB_customer_segmentation/segment_customers.py
python tasks/moduleC_student_results/predict_student_results.py
python tasks/moduleD_image_classifier/image_classifier.py
```

Notes on running Week 3:
- The first run of any CIFAR-10 script (Day 6, Day 7, Module D) downloads about 170 MB. MobileNetV2 also downloads its ImageNet weights (about 14 MB) the first time.
- Day 6 Question 17 and Module D take several minutes on a CPU. A GPU makes them much faster.

## Daily Log

Update this list at the end of each day.

### Week 2
- [ ] Day 1 – ML foundations
- [ ] Day 2 – Descriptive statistics
- [ ] Day 3 – Linear algebra
- [ ] Day 4 – Probability, correlation, data splitting
- [ ] Day 5 – Regression and Scikit-learn workflow
- [ ] Day 6 – Classification models
- [ ] Day 7 – Predictions, coefficients, feature importance, Tasks 2 and 3

### Week 3
- [x] Day 1 – Model evaluation (Q1–Q5 completed, output records in `Run_file/`)
- [ ] Day 2 – Tuning and model comparison (Q6–Q8 completed, output records in `Run_file/`; Module A task, bias–variance note)
- [ ] Day 3 – SVM and clustering (Q9–Q10, note on choosing k and DBSCAN)
- [ ] Day 4 – PCA and pipelines (Q11–Q13, Module B segmentation task, leakage and dimensionality notes)
- [ ] Day 5 – Neural networks (Q14–Q15, Module C task, activation and training-settings notes)
- [ ] Day 6 – CNN basics (Q16–Q17, note on why CNNs suit images)
- [ ] Day 7 – Image classifier (Q18, Module D task, transfer learning note)
