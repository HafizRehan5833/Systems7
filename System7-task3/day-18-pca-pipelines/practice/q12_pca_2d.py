"""Q12 (Hard): Apply PCA to reduce the data to 2 components, plot the clusters and
report the explained variance.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import load_breast_cancer
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

FIG_DIR = Path(__file__).resolve().parents[1] / "figures"

X, y = load_breast_cancer(return_X_y=True)
X_scaled = StandardScaler().fit_transform(X)  # PCA is sensitive to scale

pca = PCA(n_components=2, random_state=0)
X2 = pca.fit_transform(X_scaled)

print("Explained variance ratio per component:", np.round(pca.explained_variance_ratio_, 3))
print(f"Total explained variance with 2 components: {pca.explained_variance_ratio_.sum():.1%}")
print(f"Original number of features: {X.shape[1]}")

clusters = KMeans(n_clusters=2, n_init=10, random_state=0).fit_predict(X2)

fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
ax[0].scatter(X2[:, 0], X2[:, 1], c=y, cmap="coolwarm", s=12)
ax[0].set(title="PCA projection – true labels", xlabel="PC1", ylabel="PC2")
ax[1].scatter(X2[:, 0], X2[:, 1], c=clusters, cmap="viridis", s=12)
ax[1].set(title="PCA projection – K-Means clusters (k=2)", xlabel="PC1", ylabel="PC2")
FIG_DIR.mkdir(exist_ok=True)
plt.savefig(FIG_DIR / "q12_pca_clusters.png", dpi=120, bbox_inches="tight")
print(f"Saved plot to {FIG_DIR / 'q12_pca_clusters.png'}")
