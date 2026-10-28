"""Q10 (Medium): Train a Naive Bayes model and an SVM on the same data and
compare precision and recall.
"""
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import classification_report, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1, stratify=y)

models = {
    "Naive Bayes": make_pipeline(StandardScaler(), GaussianNB()),
    "SVM (RBF)": make_pipeline(StandardScaler(), SVC(kernel="rbf", C=1.0)),
}

print(f"{'Model':<14}{'Precision':>10}{'Recall':>10}")
for name, model in models.items():
    y_pred = model.fit(X_train, y_train).predict(X_test)
    print(f"{name:<14}{precision_score(y_test, y_pred):>10.3f}{recall_score(y_test, y_pred):>10.3f}")
    print(classification_report(y_test, y_pred, target_names=["malignant", "benign"], digits=3))
