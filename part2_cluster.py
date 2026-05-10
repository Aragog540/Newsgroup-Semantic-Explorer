"""Part 2 - Fuzzy clustering using PCA + GaussianMixture.

Performs PCA reduction and fits a GMM; saves PCA, GMM, soft assignments,
and visualisations under `data/`.
"""

import os, pickle, json
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import joblib
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from collections import Counter

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

# Load
print("Loading embeddings...")
embeddings = np.load(f"{DATA_DIR}/embeddings.npy")
with open(f"{DATA_DIR}/texts.pkl","rb") as f: texts = pickle.load(f)
with open(f"{DATA_DIR}/labels.pkl","rb") as f:
  meta = pickle.load(f); doc_labels = meta["labels"]; label_names = meta["label_names"]
print(f"Loaded {len(embeddings)} embeddings of dim {embeddings.shape[1]}")

# PCA
print("PCA to 50 dimensions...")
pca = PCA(n_components=50, random_state=42)
reduced = pca.fit_transform(embeddings)
print(f"PCA variance retained: {pca.explained_variance_ratio_.sum():.1%}")

# BIC-based k selection
print("\nSearching for optimal k via BIC (k=8..24)...")
k_range = range(8, 26, 2)
bics = []
for k in k_range:
  gm = GaussianMixture(n_components=k, covariance_type="diag",
             max_iter=200, random_state=42, n_init=1)
  gm.fit(reduced)
  bic = gm.bic(reduced)
  bics.append(bic)
  print(f"  k={k:2d}  BIC={bic:,.0f}")

bic_best_k = list(k_range)[np.argmin(bics)]
print(f"BIC-optimal k = {bic_best_k}")

# Plot BIC
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(list(k_range), bics, "o-", color="#2196F3")
ax.axvline(bic_best_k, color="#F44336", linestyle="--", label=f"BIC best k={bic_best_k}")
ax.set_xlabel("Number of clusters (k)")
ax.set_ylabel("BIC (lower = better)")
ax.set_title("BIC Curve for GMM Cluster Selection")
ax.legend()
plt.tight_layout()
plt.savefig(f"{DATA_DIR}/bic_curve.png", dpi=120)
plt.close()

# Use k matching known category count for interpretability
best_k = len(label_names)
print(f"Using k={best_k} (matches {best_k} known newsgroup categories)")

# Fit final GMM
print(f"\nFitting final GMM (k={best_k}, covariance=diag, n_init=3)...")
gmm = GaussianMixture(n_components=best_k, covariance_type="diag",
                      max_iter=300, random_state=42, n_init=3)
gmm.fit(reduced)

soft_assignments = gmm.predict_proba(reduced)
hard_assignments = soft_assignments.argmax(axis=1)
print(f"Soft assignment matrix: {soft_assignments.shape}")

# Cluster analysis
print("\nCluster Analysis")
cluster_summaries = {}
for cid in range(best_k):
    members_idx = [i for i, c in enumerate(hard_assignments) if c == cid]
    size = len(members_idx)
    counter = Counter([doc_labels[i] for i in members_idx])
    dominant = [(label_names[l], cnt/size) for l, cnt in counter.most_common(3)]
    avg_entropy = float(np.mean(
        [-np.sum(soft_assignments[i] * np.log(soft_assignments[i] + 1e-10))
         for i in members_idx]
    ))
    cluster_summaries[cid] = {"size": size, "dominant": dominant, "entropy": avg_entropy}
    dom_str = ", ".join(f"{g}({p:.0%})" for g, p in dominant)
    print(f"  C{cid:2d} | n={size:4d} | H={avg_entropy:.2f} | {dom_str}")

# Most ambiguous documents (boundary cases)
print("\nMost Ambiguous Documents (boundary cases)")
doc_entropy = -np.sum(soft_assignments * np.log(soft_assignments + 1e-10), axis=1)
top_ambiguous = np.argsort(doc_entropy)[-5:][::-1]
for idx in top_ambiguous:
    top2 = soft_assignments[idx].argsort()[-2:][::-1]
    print(f"\n  Doc #{idx} H={doc_entropy[idx]:.3f} | True: {label_names[doc_labels[idx]]}")
    print(f"  Top clusters: C{top2[0]}({soft_assignments[idx][top2[0]]:.2f}), "
      f"C{top2[1]}({soft_assignments[idx][top2[1]]:.2f})")
    print(f"  -> {texts[idx][:180]}...")

# t-SNE visualisation
print("\nt-SNE visualisation...")
tsne = TSNE(n_components=2, perplexity=30, random_state=42, max_iter=500)
tsne_coords = tsne.fit_transform(reduced)

fig, ax = plt.subplots(figsize=(12, 9))
colors = cm.tab20(np.linspace(0, 1, best_k))
for cid in range(best_k):
  mask = hard_assignments == cid
  ax.scatter(tsne_coords[mask, 0], tsne_coords[mask, 1],
         c=[colors[cid]], s=15, alpha=0.7, label=f"C{cid}")
ax.set_title(f"t-SNE — {best_k} GMM fuzzy clusters")
ax.legend(loc="upper right", fontsize=7, ncol=4, markerscale=2)
plt.tight_layout()
plt.savefig(f"{DATA_DIR}/tsne_clusters.png", dpi=120)
plt.close()
print("t-SNE saved.")

# Save artefacts
joblib.dump(pca, f"{DATA_DIR}/pca.pkl")
joblib.dump(gmm, f"{DATA_DIR}/gmm.pkl")
np.save(f"{DATA_DIR}/soft_assignments.npy", soft_assignments)
np.save(f"{DATA_DIR}/reduced.npy", reduced)
with open(f"{DATA_DIR}/cluster_summaries.json","w") as f:
    json.dump({
        "k": best_k,
        "bic_optimal_k": bic_best_k,
        "clusters": {
            str(k): {"size": v["size"], "entropy": v["entropy"],
                     "dominant": [(g, float(p)) for g, p in v["dominant"]]}
            for k, v in cluster_summaries.items()
        }
    }, f, indent=2)
print("\nPart 2 complete.")
