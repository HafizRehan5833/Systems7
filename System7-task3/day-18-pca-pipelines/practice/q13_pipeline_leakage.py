"""Q13 (Hard): Build a Pipeline (scaler, PCA, classifier). Then show data leakage
by scaling and selecting features BEFORE the split, and compare the scores.

Part A uses real data (small effect). Part B uses pure random noise, where the
leaky version looks clearly better than chance; that is the classic demonstration.
"""
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def correct_pipeline():
    # Everything that learns from data lives INSIDE the pipeline, so cross-validation
    # refits it on each training fold only.
    return make_pipeline(StandardScaler(), PCA(n_components=5), LogisticRegression(max_iter=1000))


print("=== Part A: real data (breast cancer) ===")
X, y = load_breast_cancer(return_X_y=True)

# Leaky: scale using ALL data (including the future test rows) before splitting
X_leaky = StandardScaler().fit_transform(X)
Xa_train, Xa_test, ya_train, ya_test = train_test_split(X_leaky, y, test_size=0.3, random_state=0)
leaky = LogisticRegression(max_iter=1000).fit(Xa_train, ya_train)
print(f"Leaky (scale before split):  test accuracy = {leaky.score(Xa_test, ya_test):.3f}")

# Correct: split first, then the pipeline fits the scaler and PCA on training data only
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
good = correct_pipeline().fit(X_train, y_train)
print(f"Correct (pipeline):          test accuracy = {good.score(X_test, y_test):.3f}")

print("\n=== Part B: pure random noise (no real signal) ===")
rng = np.random.default_rng(0)
X_noise = rng.normal(size=(100, 2000))   # 2000 random features
y_noise = rng.integers(0, 2, size=100)   # random labels: true accuracy should be ~50%

# Leaky: pick the 20 "best" features using ALL labels, then cross-validate
leaky_selector = SelectKBest(f_classif, k=20).fit(X_noise, y_noise)
X_leaky_noise = leaky_selector.transform(X_noise)
leaky_cv = cross_val_score(LogisticRegression(max_iter=1000), X_leaky_noise, y_noise, cv=5).mean()

# Correct: the selector is inside the pipeline, so it only sees each training fold
honest = make_pipeline(SelectKBest(f_classif, k=20), LogisticRegression(max_iter=1000))
honest_cv = cross_val_score(honest, X_noise, y_noise, cv=5).mean()

print(f"Leaky feature selection CV accuracy:  {leaky_cv:.3f}  (looks good, but it is fake)")
print(f"Pipeline feature selection CV accuracy: {honest_cv:.3f}  (close to 0.5, as expected)")
