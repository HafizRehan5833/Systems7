"""Task 1: explore, clean and split the house dataset."""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
RAW_PATH = DATA_DIR / "houses.csv"
OUT_DIR = DATA_DIR / "processed"
RANDOM_STATE = 42


def explore(df: pd.DataFrame) -> None:
    print("=== Shape ===")
    print(df.shape)
    print("\n=== Data types ===")
    print(df.dtypes)
    print("\n=== Summary statistics ===")
    print(df.describe().round(2))
    print("\n=== Missing values per column ===")
    print(df.isna().sum())


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop_duplicates().copy()
    # Fill missing numbers with the column median
    for col in df.columns[df.isna().any()]:
        df[col] = df[col].fillna(df[col].median())
    return df


def add_label(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    df = df.copy()
    df["is_expensive"] = (df["price"] > threshold).astype(int)
    return df


def split(df: pd.DataFrame):
    train_val, test = train_test_split(df, test_size=0.20, random_state=RANDOM_STATE)
    train, val = train_test_split(train_val, test_size=0.25, random_state=RANDOM_STATE)
    return train, val, test


def main() -> None:
    raw = pd.read_csv(RAW_PATH)
    explore(raw)

    df = clean(raw)
    print(f"\nRows after cleaning: {len(df)} (removed {len(raw) - len(df)})")

    # The label threshold is the median price, computed on the training split
    # only so that no information leaks from validation/test data.
    train, val, test = split(df)
    threshold = train["price"].median()
    train, val, test = (add_label(d, threshold) for d in (train, val, test))

    print(f"\nTrain: {len(train)} | Validation: {len(val)} | Test: {len(test)}")
    print(f"Expensive threshold (train median price): {threshold:,.0f}")
    print("Class balance in train:", train["is_expensive"].value_counts().to_dict())

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    train.to_csv(OUT_DIR / "train.csv", index=False)
    val.to_csv(OUT_DIR / "val.csv", index=False)
    test.to_csv(OUT_DIR / "test.csv", index=False)
    print(f"\nSaved processed splits to {OUT_DIR}")


if __name__ == "__main__":
    main()
