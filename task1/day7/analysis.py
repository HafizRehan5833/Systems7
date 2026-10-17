"""Day 7 (Task 1): CSV analysis with charts and summary statistics.
Dataset: students_clean.csv (from Day 6). Swap in any CSV by changing CSV_PATH."""
import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

HERE = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(HERE, "students_clean.csv")
df = pd.read_csv(CSV_PATH)
sns.set_theme(style="whitegrid")

# ---- Summary statistics ----
print("Shape:", df.shape)
print(df.info())
stats = df.describe().round(2)
print(stats)
by_dept = df.groupby("department")["score"].agg(["mean", "median", "max", "count"]).round(2)
by_gender = df.groupby("gender")["score"].mean().round(2)
corr = df[["study_hours", "attendance", "score"]].corr().round(2)
print("\nScore by department:\n", by_dept)
print("\nScore by gender:\n", by_gender)
print("\nCorrelations:\n", corr)

with open(os.path.join(HERE, "summary_statistics.txt"), "w") as f:
    f.write("DESCRIPTIVE STATISTICS\n" + stats.to_string())
    f.write("\n\nSCORE BY DEPARTMENT\n" + by_dept.to_string())
    f.write("\n\nSCORE BY GENDER\n" + by_gender.to_string())
    f.write("\n\nCORRELATIONS\n" + corr.to_string())

# ---- Charts ----
def save(name):
    plt.tight_layout(); plt.savefig(os.path.join(HERE, name), dpi=120); plt.close()

plt.figure(figsize=(7, 4)); sns.histplot(df["score"], kde=True, color="steelblue")
plt.title("Distribution of Scores"); save("chart1_score_hist.png")

plt.figure(figsize=(7, 4)); sns.barplot(data=df, x="department", y="score", errorbar=None, hue="department", palette="viridis", legend=False)
plt.title("Average Score by Department"); save("chart2_dept_bar.png")

plt.figure(figsize=(7, 4)); sns.scatterplot(data=df, x="study_hours", y="score", hue="gender")
sns.regplot(data=df, x="study_hours", y="score", scatter=False, color="black")
plt.title("Study Hours vs Score"); save("chart3_scatter.png")

plt.figure(figsize=(7, 4)); sns.boxplot(data=df, x="department", y="score", hue="gender")
plt.title("Score Spread by Department & Gender"); save("chart4_box.png")

plt.figure(figsize=(5, 4)); sns.heatmap(corr, annot=True, cmap="coolwarm", vmin=-1, vmax=1)
plt.title("Correlation Heatmap"); save("chart5_heatmap.png")

print("\nSaved 5 charts + summary_statistics.txt")
