"""Module B practical task: customer segmentation with clustering and PCA."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data" / "customers.csv"


def score(X, labels):
    """Silhouette needs at least 2 clusters and ignores DBSCAN noise (-1)."""
    mask = labels != -1
    if len(set(labels[mask])) < 2:
        return float("nan")
    return silhouette_score(X[mask], labels[mask])


def main() -> None:
    df = pd.read_csv(DATA)
    features = df.drop(columns="customer_id")
    X = StandardScaler().fit_transform(features)

    # 1. Choose k for K-Means by silhouette
    sil = {}
    for k in range(2, 9):
        labels = KMeans(n_clusters=k, n_init=10, random_state=42).fit_predict(X)
        sil[k] = silhouette_score(X, labels)
    best_k = max(sil, key=sil.get)
    print("K-Means silhouette by k:", {k: round(v, 3) for k, v in sil.items()})
    print(f"Chosen k = {best_k}\n")

    # 2. Compare algorithms
    km_labels = KMeans(n_clusters=best_k, n_init=10, random_state=42).fit_predict(X)
    hc_labels = AgglomerativeClustering(n_clusters=best_k, linkage="ward").fit_predict(X)
    db_labels = DBSCAN(eps=1.0, min_samples=10).fit_predict(X)
    comparison = pd.DataFrame({
        "algorithm": ["K-Means", "Hierarchical (Ward)", "DBSCAN (eps=1.0)"],
        "clusters": [len(set(km_labels)), len(set(hc_labels)), len(set(db_labels) - {-1})],
        "noise_points": [0, 0, int((db_labels == -1).sum())],
        "silhouette": [score(X, km_labels), score(X, hc_labels), score(X, db_labels)],
    })
    print(comparison.round(3).to_string(index=False), "\n")

    # 3. Profile the chosen K-Means segments
    df["segment"] = km_labels
    summary = df.groupby("segment").agg(
        customers=("customer_id", "count"),
        **{col: (col, "mean") for col in features.columns},
    ).round(1)
    print("Segment profiles (means):")
    print(summary.to_string())
    summary.to_csv(HERE / "segments_summary.csv")

    # 4. 2D view with PCA
    X2 = PCA(n_components=2, random_state=0).fit_transform(X)
    plt.figure(figsize=(6, 5))
    plt.scatter(X2[:, 0], X2[:, 1], c=km_labels, cmap="tab10", s=14)
    plt.title(f"Customer segments (K-Means, k={best_k}) in PCA space")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plot_path = HERE / "segments_pca.png"
    plt.savefig(plot_path, dpi=120, bbox_inches="tight")
    print(f"\nSaved {plot_path.name} and segments_summary.csv")


if __name__ == "__main__":
    main()
