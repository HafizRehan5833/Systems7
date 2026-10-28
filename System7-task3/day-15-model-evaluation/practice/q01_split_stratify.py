"""Q1 (Easy): Split a dataset 80/20 with a fixed random_state and stratify.
Print the shape and class balance of both parts.
"""
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

X, y = load_breast_cancer(return_X_y=True, as_frame=True)

# stratify=y keeps the same class ratio in train and test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

for name, X_part, y_part in [("Train", X_train, y_train), ("Test", X_test, y_test)]:
    print(f"{name}: X shape = {X_part.shape}, y shape = {y_part.shape}")
    counts = y_part.value_counts().sort_index().to_dict()
    print(f"  class balance (0=malignant, 1=benign): {counts}")
    print(f"  proportion of class 1: {y_part.mean():.3f}")
