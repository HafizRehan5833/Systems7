"""Day 3: vectors, matrices and dot products."""
import numpy as np

# A vector: one sample's features [area_sqft, bedrooms, age_years]
house = np.array([1200, 3, 10])

# A matrix: each row is a house, each column a feature
houses = np.array([
    [1200, 3, 10],
    [850,  2, 25],
    [2000, 4,  5],
])
print("Shape of matrix (samples, features):", houses.shape)

# Coefficients a linear model might learn (example values)
weights = np.array([150.0, 5000.0, -800.0])
bias = 20000.0

# Dot product: multiply matching entries and add them up
manual = house[0] * weights[0] + house[1] * weights[1] + house[2] * weights[2]
print("Manual dot product:", manual)
print("np.dot result:     ", np.dot(house, weights))

# Predict for every house at once: matrix times vector, then add bias
predictions = houses @ weights + bias
print("\nPredicted prices:", predictions)
