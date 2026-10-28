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

---

# Week 3 Glossary

## Model Evaluation (Module A)

**Cross-Validation (k-fold)**
Split the training data into k parts. Train on k−1 parts and test on the remaining part, rotating so every part is tested once. The average gives a more reliable score than a single split.

**Stratified Split**
A train/test split that keeps the same class proportions in both parts. Important when classes are imbalanced.

**Accuracy**
The share of all predictions that were correct.

**Precision**
Of the samples predicted positive, the share that really are positive. High precision means few false alarms.

**Recall (sensitivity)**
Of the samples that really are positive, the share that the model found. High recall means few missed cases.

**F1-score**
The harmonic mean of precision and recall. Useful when you need a balance between the two.

**Confusion Matrix**
A table of actual classes against predicted classes, showing true positives, true negatives, false positives and false negatives.

**ROC Curve and AUC**
The ROC curve plots true positive rate against false positive rate at every threshold. AUC (area under the curve) summarizes it: 1.0 is perfect, 0.5 is random guessing.

**MAE, MSE, RMSE**
Average error measures for regression. MAE is the mean absolute error, MSE the mean squared error, and RMSE its square root (in the same units as the target).

**R² (coefficient of determination)**
The share of the target's variation explained by the model. 1.0 is perfect; 0 means no better than predicting the mean.

**Overfitting**
The model learns noise in the training data and performs well there but poorly on new data.

**Underfitting**
The model is too simple to capture the real pattern, so it performs poorly on both training and new data.

**Generalization**
How well a model performs on data it has not seen.

**Bias–variance trade-off**
Simple models have high bias (they miss patterns) and low variance. Complex models have low bias and high variance (they react to noise). Good models balance the two.

**Feature Scaling**
Putting numeric features on a similar range (for example with StandardScaler). Needed for distance-based and gradient-based models.

**One-Hot Encoding**
Turning a categorical column into one 0/1 column per category.

**ColumnTransformer**
A scikit-learn tool that applies different preprocessing to different columns in one step.

**Hyperparameter**
A setting chosen before training, such as `max_depth` or `C`. It is not learned from the data.

**GridSearchCV**
Tries every combination of hyperparameters in a grid and picks the best using cross-validation.

**RandomizedSearchCV**
Tries a fixed number of random hyperparameter combinations. Faster than a full grid when the grid is large.

## Advanced ML (Module B)

**Support Vector Machine (SVM)**
A classifier that finds the boundary with the widest margin between classes. Kernels (for example RBF) allow curved boundaries.

**Naive Bayes**
A probabilistic classifier that assumes features are independent given the class. Fast and often strong on text and simple data.

**K-Means Clustering**
Groups data into k clusters by repeatedly assigning points to the nearest cluster centre and moving the centres.

**Inertia**
K-Means' total within-cluster squared distance. It always falls as k grows, so we look for an "elbow" where the drop levels off.

**Silhouette Score**
For each point, how close it is to its own cluster compared with the nearest other cluster. Ranges from −1 to 1; higher is better.

**Hierarchical Clustering**
Builds a tree of nested clusters by repeatedly merging the closest pair. The tree (dendrogram) can be cut at any level.

**DBSCAN**
Density-based clustering. Groups points that are packed closely together and labels sparse points as noise. Does not need k in advance and can find non-round clusters.

**Principal Component Analysis (PCA)**
Finds new uncorrelated directions (components) that capture the most variance, so many features can be summarized with fewer.

**Explained Variance Ratio**
The share of the total variance captured by each principal component.

**Curse of Dimensionality**
As the number of features grows, data becomes sparse and distances lose meaning, so models need more data and often benefit from dimensionality reduction.

**Feature Selection**
Choosing a subset of the most useful features (for example with SelectKBest).

**Feature Engineering**
Creating new features from existing ones to help a model learn (for example a price per square foot).

**Imbalanced Dataset**
A dataset where one class is much rarer than the others. Accuracy can be misleading on such data.

**class_weight**
A setting that makes errors on rare classes count more during training.

**Data Leakage**
Information from outside the training data (often the test set) influences training, so scores look better than they really are.

**Pipeline**
A scikit-learn object that chains preprocessing and modelling steps, so that each step is fitted only on the training data.

## Deep Learning (Module C)

**Neural Network**
A network of connected layers of artificial neurons. Each layer transforms its inputs with weights and biases and passes the result through an activation function.

**Perceptron**
The simplest neural unit: a weighted sum of inputs, plus a bias, passed through a step function to give a 0 or 1 output.

**Neuron, Weight, Bias**
A neuron combines its inputs with weights (how important each input is) and adds a bias (an offset). Training adjusts the weights and biases.

**Activation Function**
A non-linear function applied to a neuron's output. Without it, stacked layers would collapse into a single linear model.

**ReLU**
Outputs the input if positive, otherwise 0. The usual default for hidden layers.

**Sigmoid**
Squashes any value into the range 0 to 1. Used for binary probabilities.

**Softmax**
Turns a vector of scores into probabilities that sum to 1. Used for the output of multiclass classification.

**Forward Propagation**
Computing a network's output from its inputs, layer by layer.

**Backpropagation**
Computing how much each weight contributed to the error, working backwards from the output, using the chain rule.

**Loss Function**
A measure of how wrong the predictions are, for example mean squared error (regression) or cross-entropy (classification).

**Gradient Descent**
Repeatedly moving the weights a small step in the direction that reduces the loss.

**Learning Rate**
The size of each gradient descent step. Too large and training is unstable; too small and it is slow.

**Batch Size**
The number of samples used for one weight update. Smaller batches are noisier but can generalize better; larger batches are faster per epoch.

**Epoch**
One full pass through the training data.

**SGD (Stochastic Gradient Descent)**
Gradient descent using one sample or a small batch at a time. Simple, but sensitive to the learning rate.

**Adam**
An optimizer that adapts the learning rate for each weight using running averages of past gradients. Often trains well with default settings.

**Keras Sequential Model**
A model built as a simple stack of layers, with workflow: build, compile (choose optimizer and loss), fit (train), evaluate.

**Early Stopping**
Stopping training when the validation loss stops improving, to avoid overfitting.

**Dropout**
Randomly switching off a share of neurons during training so the network does not rely on any single one.

## Computer Vision and CNNs (Module D)

**Image Tensor**
An image stored as a numeric array with height, width and colour channels (for example 32 × 32 × 3 for a colour image).

**Normalization (images)**
Scaling pixel values from 0–255 to 0–1 (or −1 to 1) so training is stable.

**Resizing**
Changing the height and width of an image, for example to 64 × 64 or to the input size a pretrained model expects.

**Data Augmentation**
Creating varied training images by applying random flips, rotations, zooms or shifts. Applied only during training.

**Convolution**
Sliding a small filter (kernel) across an image and computing a weighted sum at each position, producing a feature map.

**Filter (kernel)**
A small learned grid of weights that detects a pattern such as an edge or a colour change.

**Feature Map**
The output of applying one filter across an image. It shows where the filter's pattern appears.

**Pooling (MaxPooling)**
Shrinking a feature map by keeping only the largest value in each small window. Reduces size and makes the network less sensitive to exact positions.

**Convolutional Neural Network (CNN)**
A network that uses convolution and pooling layers to learn spatial patterns, followed by dense layers for the final decision.

**Weight Sharing**
A filter uses the same weights everywhere in the image, so a CNN needs far fewer parameters than a fully connected network on the same image.

**Multiclass Classification**
Predicting one of more than two classes, with one output per class.

**Transfer Learning**
Reusing a model trained on a large dataset (for example ImageNet) as the starting point for a new task.

**Feature Extraction (transfer learning)**
Keeping the pretrained layers frozen and training only a new head on top.

**Fine-tuning (transfer learning)**
Unfreezing some of the top pretrained layers and training them with a small learning rate, so the features adapt to the new task.

**MobileNetV2**
A lightweight pretrained CNN designed to run efficiently, commonly used for transfer learning.
