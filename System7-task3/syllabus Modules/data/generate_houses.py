"""Generate a synthetic, reproducible house-price dataset.

Run:  python tasks/data/generate_houses.py
Output: tasks/data/houses.csv
"""
from pathlib import Path

import numpy as np
import pandas as pd


def make_houses(n: int = 300, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    area_sqft = rng.uniform(600, 3500, n).round(0)
    bedrooms = np.clip(np.round(area_sqft / 700 + rng.normal(0, 0.7, n)), 1, 6).astype(int)
    bathrooms = np.clip(np.round(bedrooms / 2 + rng.normal(0, 0.4, n)), 1, 4).astype(int)
    age_years = rng.integers(0, 50, n).astype(float)
    distance_km = rng.uniform(0.5, 30, n).round(1)

    price = (
        area_sqft * 120
        + bedrooms * 8000
        + bathrooms * 5000
        - age_years * 600
        - distance_km * 1500
        + rng.normal(0, 15000, n)
    )
    price = np.maximum(price, 20000).round(0)

    df = pd.DataFrame({
        "area_sqft": area_sqft,
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "age_years": age_years,
        "distance_km": distance_km,
        "price": price,
    })

    # Inject some missing values so the cleaning step has real work to do.
    missing_idx = rng.choice(n, size=15, replace=False)
    df.loc[missing_idx, "age_years"] = np.nan
    return df


if __name__ == "__main__":
    out_path = Path(__file__).resolve().parent / "houses.csv"
    df = make_houses()
    df.to_csv(out_path, index=False)
    print(f"Saved {len(df)} rows to {out_path}")
