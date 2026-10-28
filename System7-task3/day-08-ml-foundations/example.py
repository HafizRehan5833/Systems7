"""Day 1: dataset anatomy and the three types of learning."""
import pandas as pd

data = pd.DataFrame({
    "hours_studied": [1, 3, 5, 7, 9],
    "sleep_hours":   [4, 6, 7, 6, 8],
    "passed":        [0, 0, 1, 1, 1],   # label: 0 = fail, 1 = pass
})

features = data[["hours_studied", "sleep_hours"]]   # X
label = data["passed"]                              # y

print("Number of samples:", len(data))
print("Features:", list(features.columns))
print("Label (target):", label.name)
print()
print(data)

# Supervised learning: we have labels, so a model can learn X -> y.
# Unsupervised learning: same X but no y; the model looks for groups.
# Reinforcement learning: no table at all; an agent learns from rewards.
print("\nSupervised example: learn a simple rule from labelled data")
threshold = features["hours_studied"][label == 1].min()
print(f"Students who studied at least {threshold} hours passed.")
