"""Day 6 (Task 6): Pandas basics + clean a dataset with missing values.
Plots the data before and after cleaning."""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

rng = np.random.default_rng(7)
n = 200
df = pd.DataFrame({
    "student_id": range(1, n + 1),
    "gender": rng.choice(["Male", "Female"], n),
    "study_hours": rng.normal(5, 2, n).clip(0.5, 12).round(1),
    "attendance": rng.normal(80, 10, n).clip(40, 100).round(0),
})
df["score"] = (35 + 5 * df["study_hours"] + 0.2 * df["attendance"]
               + rng.normal(0, 6, n)).clip(0, 100).round(1)
df["department"] = rng.choice(["CS", "SE", "AI", "DS"], n)

for col, frac in [("study_hours", 0.10), ("attendance", 0.08), ("score", 0.05)]:
    df.loc[df.sample(frac=frac, random_state=1).index, col] = np.nan
df.to_csv("students_messy.csv", index=False)

print(df.head(), "\n")
print("Missing values:\n", df.isna().sum(), "\n")
print("Filter score > 80:", len(df[df["score"] > 80]))
print("Top 5 by score:\n", df.sort_values("score", ascending=False).head(5), "\n")

clean = df.copy()
clean["study_hours"] = clean["study_hours"].fillna(clean["study_hours"].median())
clean["attendance"] = clean["attendance"].fillna(clean["attendance"].mean())
clean = clean.dropna(subset=["score"])
clean.to_csv("students_clean.csv", index=False)
print("After cleaning, missing:", clean.isna().sum().sum(), "| rows:", len(clean))

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
sns.histplot(df["study_hours"].dropna(), kde=True, ax=ax[0], color="tomato")
ax[0].set_title(f"Before cleaning (missing: {df['study_hours'].isna().sum()})")
sns.histplot(clean["study_hours"], kde=True, ax=ax[1], color="seagreen")
ax[1].set_title("After cleaning (missing: 0)")
plt.tight_layout()
plt.savefig("before_after.png", dpi=120)
print("Saved before_after.png")
