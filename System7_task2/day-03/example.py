"""Day 3: vectors, matrices and dot products."""
import numpy as np

house = np.array([1200, 3, 10])

houses = np.array([
    [1200, 3, 10],
    [850,  2, 25],
    [2000, 4,  5],
])
print("Shape of matrix (samples, features):", houses.shape)

weights = np.array([150.0, 5000.0, -800.0])
bias = 20000.0

manual = house[0] * weights[0] + house[1] * weights[1] + house[2] * weights[2]
print("Manual dot product:", manual)
print("np.dot result:     ", np.dot(house, weights))

predictions = houses @ weights + bias
print("\nPredicted prices:", predictions)
