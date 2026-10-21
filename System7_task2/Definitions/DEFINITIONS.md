# System 7 Glossary

Definitions of the terms used in the Week 2 syllabus, grouped by topic.

## ML Foundations

**Artificial Intelligence (AI)**
Building systems that perform tasks which normally need human intelligence, such as recognising speech, making decisions or translating text.

**Machine Learning (ML)**
A subset of AI in which a model learns patterns from data instead of being programmed with explicit rules.

**Deep Learning**
A subset of ML that uses neural networks with many layers. It needs large amounts of data and computing power.

**Supervised Learning**
Learning from data where each example has a known answer (label). The model learns to map inputs to that answer. Examples: regression, classification.

**Unsupervised Learning**
Learning from data without labels. The model looks for structure on its own, such as clusters or groups.

**Reinforcement Learning**
An agent learns by trying actions in an environment and receiving rewards or penalties.

**Dataset**
A collection of data used to train or test a model. Usually stored as a table.

**Sample (row / observation / instance)**
One record in the dataset, for example one house.

**Feature (input / independent variable / column)**
A measurable property used as input to the model, for example `area_sqft`.

**Label (target / response / dependent variable)**
The value the model is trying to predict, for example `price`. The terms "label" and "target variable" are used for the same thing in supervised learning.

## Statistics

**Mean**
The sum of values divided by the count. Sensitive to outliers.

**Median**
The middle value when data is sorted. Robust to outliers.

**Mode**
The most frequent value.

**Variance**
The average squared distance of values from the mean. Measures spread.

**Standard Deviation**
The square root of the variance. Measures spread in the same units as the data.

**Probability**
A number between 0 and 1 that describes how likely an event is.

**Correlation**
A value between -1 and 1 that describes how strongly two variables move together. Correlation does not imply causation.

## Linear Algebra

**Vector**
An ordered list of numbers, such as a single sample's features `[1200, 3, 2]`.

**Matrix**
A two-dimensional grid of numbers. A dataset with rows as samples and columns as features is a matrix.

**Dot Product**
Multiply matching entries of two vectors and add the results. A linear model's prediction is a dot product of features and coefficients.

## Data Splitting

**Training Set**
The data the model learns from.

**Validation Set**
Data used to compare models and tune settings, never used for learning the parameters.

**Testing Set**
Data kept aside until the end to estimate how the final model performs on unseen data.

## Regression Models

**Linear Regression**
Predicts a continuous number as a straight-line function of one feature.

**Multiple Linear Regression**
Predicts a continuous number from several features at once.

**Logistic Regression**
Despite the name, a classification model. It outputs the probability that a sample belongs to a class.

## Classification Models

**Classification**
Predicts a category, such as "expensive" or "cheap".

**Regression**
Predicts a continuous number, such as a price.

**K-Nearest Neighbors (KNN)**
Predicts by looking at the K closest training samples and taking their majority class (or average for regression).

**Decision Tree**
A model that splits data with yes/no questions on features, forming a tree of rules.

**Random Forest**
An ensemble of many decision trees trained on random subsets of the data. Their votes are combined.

## Scikit-learn Workflow

**Estimator**
Any object that learns from data with `.fit()`.

**`fit(X, y)`**
Train the model on features `X` and labels `y`.

**`predict(X)`**
Use the trained model to produce outputs for new data.

**Coefficients**
The learned weights of a linear model. Each one shows how much the prediction changes when that feature increases by one unit (holding others fixed), in the model's units.

**Feature Importance**
A score showing how much each feature contributes to a tree-based model's decisions.

**Pipeline**
A chain of steps (for example imputation, scaling, then a model) that is fitted and applied together, which prevents data leakage.
