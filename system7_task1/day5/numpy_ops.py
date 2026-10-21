"""Day 5 (Task 5): NumPy array operations on a sample matrix."""
import numpy as np

m = np.arange(1, 17).reshape(4, 4)
print("Matrix:\n", m)
print("Shape:", m.shape, "| dtype:", m.dtype)

# Indexing & slicing
print("Element [2,3]:", m[2, 3])
print("Row 1:", m[1], "| Column 2:", m[:, 2])
print("Sub-matrix:\n", m[1:3, 1:3])
print("Even numbers:", m[m % 2 == 0])

# Math operations
print("Sum:", m.sum(), "| Mean:", m.mean(), "| Std:", round(m.std(), 3))
print("Row sums:", m.sum(axis=1), "| Col means:", m.mean(axis=0))
print("Squared:\n", m ** 2)

# Linear algebra
a = np.array([[2, 1], [1, 3]])
b = np.array([[1, 0], [4, 2]])
print("Dot:\n", a @ b)
print("Transpose:\n", a.T)
print("Determinant:", round(np.linalg.det(a), 3))
print("Inverse:\n", np.linalg.inv(a))

# Useful creators
print(np.zeros((2, 3)), np.ones(3), np.linspace(0, 1, 5), sep="\n")
rng = np.random.default_rng(42)
print("Random:", rng.integers(0, 100, size=5))
