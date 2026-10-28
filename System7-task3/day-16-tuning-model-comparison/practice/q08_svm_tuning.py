"""Q8 (Medium): Tune an SVM with GridSearchCV and RandomizedSearchCV.
Compare the best score and the run time.
"""
import time

from scipy.stats import loguniform
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Pipeline: SVMs are sensitive to feature scale, so scale inside the CV loop
pipe = make_pipeline(StandardScaler(), SVC())

grid = {
    "svc__C": [0.1, 1, 10, 100],
    "svc__gamma": ["scale", 0.01, 0.001],
    "svc__kernel": ["rbf", "linear"],
}
random_dist = {
    "svc__C": loguniform(1e-2, 1e2),
    "svc__gamma": loguniform(1e-4, 1e0),
    "svc__kernel": ["rbf", "linear"],
}

# Grid search: tries every combination (4 * 3 * 2 = 24 settings)
t0 = time.perf_counter()
gs = GridSearchCV(pipe, grid, cv=5, n_jobs=-1).fit(X_train, y_train)
grid_time = time.perf_counter() - t0

# Randomized search: tries 10 random settings
t0 = time.perf_counter()
rs = RandomizedSearchCV(pipe, random_dist, n_iter=10, cv=5, random_state=42, n_jobs=-1).fit(X_train, y_train)
rand_time = time.perf_counter() - t0

print(f"GridSearchCV      best CV = {gs.best_score_:.3f}  test = {gs.score(X_test, y_test):.3f}  "
      f"time = {grid_time:.2f}s  params = {gs.best_params_}")
print(f"RandomizedSearchCV best CV = {rs.best_score_:.3f}  test = {rs.score(X_test, y_test):.3f}  "
      f"time = {rand_time:.2f}s  params = {rs.best_params_}")
print("\nDiscuss: did the random search lose much accuracy for its speed?")
