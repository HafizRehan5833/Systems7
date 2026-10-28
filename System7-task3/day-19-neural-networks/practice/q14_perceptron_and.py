"""Q14 (Easy): Implement a single perceptron in NumPy that learns the AND gate.

Perceptron rule: predict = step(w·x + b); if wrong, nudge w and b toward the answer.
"""
import numpy as np

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 0, 0, 1])          # AND truth table

w = np.zeros(2)
b = 0.0
learning_rate = 0.1


def step(z):
    """Activation: fire (1) if the weighted sum is non-negative, else 0."""
    return (z >= 0).astype(int)


for epoch in range(1, 21):
    errors = 0
    for x_i, target in zip(X, y):
        prediction = int(step(np.dot(w, x_i) + b))
        update = learning_rate * (target - prediction)  # 0 if correct
        w += update * x_i
        b += update
        errors += int(update != 0)
    print(f"epoch {epoch:>2}: errors = {errors}, w = {np.round(w, 2)}, b = {b:.2f}")
    if errors == 0:
        print("Converged.")
        break

print("\nLearned predictions:")
for x_i, target in zip(X, y):
    pred = int(step(np.dot(w, x_i) + b))
    print(f"  input {x_i} -> predicted {pred} (target {target})")
