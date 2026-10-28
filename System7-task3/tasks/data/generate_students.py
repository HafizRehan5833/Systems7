"""Generate a synthetic, reproducible student-performance dataset.

Run:  python tasks/data/generate_students.py
Output: tasks/data/students.csv
"""
from pathlib import Path

import numpy as np
import pandas as pd

SCHOOL_EFFECT = {"public": 0.0, "private": 3.0, "other": -2.0}


def make_students(n: int = 800, seed: int = 11) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    study_hours = rng.uniform(0, 12, n).round(1)
    attendance_pct = np.clip(rng.normal(80, 12, n), 30, 100).round(1)
    previous_score = np.clip(rng.normal(60, 15, n), 0, 100).round(1)
    sleep_hours = np.clip(rng.normal(7, 1.2, n), 3, 10).round(1)
    practice_tests = rng.integers(0, 15, n)
    assignments_on_time_pct = np.clip(rng.normal(75, 18, n), 0, 100).round(1)
    school_type = rng.choice(list(SCHOOL_EFFECT), size=n, p=[0.6, 0.3, 0.1])

    final_score = (
        10
        + 0.30 * previous_score
        + 0.15 * attendance_pct
        + 1.20 * study_hours
        + 0.60 * practice_tests
        + 0.10 * assignments_on_time_pct
        - 1.50 * np.abs(sleep_hours - 7.5)      # best results near 7.5 hours of sleep
        + np.array([SCHOOL_EFFECT[s] for s in school_type])
        + rng.normal(0, 5, n)
    )
    final_score = np.clip(final_score, 0, 100).round(1)

    return pd.DataFrame({
        "study_hours": study_hours,
        "attendance_pct": attendance_pct,
        "previous_score": previous_score,
        "sleep_hours": sleep_hours,
        "practice_tests": practice_tests,
        "assignments_on_time_pct": assignments_on_time_pct,
        "school_type": school_type,
        "final_score": final_score,
    })


if __name__ == "__main__":
    out = Path(__file__).resolve().parent / "students.csv"
    df = make_students()
    df.to_csv(out, index=False)
    print(f"Saved {len(df)} students to {out}")
