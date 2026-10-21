import statistics

import numpy as np

prices = [120, 135, 140, 140, 150, 155, 900]   # 900 is an outlier

mean = sum(prices) / len(prices)
median = statistics.median(prices)
mode = statistics.mode(prices)

variance = sum((x - mean) ** 2 for x in prices) / len(prices)
std_dev = variance ** 0.5

print(f"Mean:     {mean:.2f}  (pulled up by the outlier)")
print(f"Median:   {median}  (robust to the outlier)")
print(f"Mode:     {mode}")
print(f"Variance: {variance:.2f}")
print(f"Std dev:  {std_dev:.2f}")

arr = np.array(prices)
print("\nNumPy check:")
print("mean", arr.mean().round(2), "| var", arr.var().round(2), "| std", arr.std().round(2))
