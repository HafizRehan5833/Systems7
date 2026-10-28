"""Q9 (Medium): Fit K-Means for k = 2 to 10 and plot the elbow curve and
silhouette scores to pick k.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

FIG_DIR = Path(__file__).resolve().parents[1] / "figures"
DATA = Path(__file__).resolve().parents[2] / "tasks" / "data" / "customers.csv"

df = pd.read_csv(DATA)
X = StandardScaler().fit_transform(df.drop(columns="customer_id"))  # clustering needs scaled features

ks = list(range(2, 11))
inertias, silhouettes = [], []
for k in ks:
    km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X)
    inertias.append(km.inertia_)                       # elbow: total within-cluster spread
    silhouettes.append(silhouette_score(X, km.labels_))  # -1 (bad) to 1 (good)
    print(f"k={k:>2}  inertia={inertias[-1]:>9.1f}  silhouette={silhouettes[-1]:.3f}")

best_k = ks[silhouettes.index(max(silhouettes))]
print(f"\nBest k by silhouette: {best_k}")

fig, ax = plt.subplots(1, 2, figsize=(10, 4))
ax[0].plot(ks, inertias, marker="o")
ax[0].set(title="Elbow curve", xlabel="k", ylabel="Inertia")
ax[1].plot(ks, silhouettes, marker="o", color="tab:green")
ax[1].axvline(best_k, ls="--", color="grey")
ax[1].set(title="Silhouette score", xlabel="k", ylabel="Silhouette")
FIG_DIR.mkdir(exist_ok=True)
plt.savefig(FIG_DIR / "q09_choose_k.png", dpi=120, bbox_inches="tight")
print(f"Saved plot to {FIG_DIR / 'q09_choose_k.png'}")
