"""Generate a synthetic, reproducible customer dataset with four hidden segments.

Run:  python tasks/data/generate_customers.py
Output: tasks/data/customers.csv
"""
from pathlib import Path

import numpy as np
import pandas as pd

# Mean profile of each hidden segment (the segments are NOT saved, so the
# clustering task has to discover them).
SEGMENTS = {
    0: dict(age=24, annual_income=28000, spending_score=30, visits_per_month=2, avg_basket=25, tenure_months=6),
    1: dict(age=45, annual_income=90000, spending_score=75, visits_per_month=6, avg_basket=120, tenure_months=48),
    2: dict(age=35, annual_income=40000, spending_score=60, visits_per_month=10, avg_basket=20, tenure_months=24),
    3: dict(age=52, annual_income=70000, spending_score=40, visits_per_month=1, avg_basket=250, tenure_months=12),
}
SPREAD = dict(age=4, annual_income=6000, spending_score=8, visits_per_month=1, avg_basket=15, tenure_months=5)


def make_customers(n: int = 600, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    seg = rng.choice(list(SEGMENTS), size=n, p=[0.3, 0.3, 0.2, 0.2])
    data = {"customer_id": [f"C{i:04d}" for i in range(1, n + 1)]}
    for col, spread in SPREAD.items():
        means = np.array([SEGMENTS[s][col] for s in seg], dtype=float)
        values = rng.normal(means, spread)
        data[col] = np.maximum(values, 0).round(1 if col == "avg_basket" else 0)
    return pd.DataFrame(data)


if __name__ == "__main__":
    out = Path(__file__).resolve().parent / "customers.csv"
    df = make_customers()
    df.to_csv(out, index=False)
    print(f"Saved {len(df)} customers to {out}")
