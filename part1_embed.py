"""Part 1 - Embedding & NearestNeighbors index builder.

Builds TF-IDF + TruncatedSVD (LSA) embeddings and a sklearn
NearestNeighbors index saved under `data/`.
"""

import json
import os
import pickle
import re

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize

from dataset_utils import load_newsgroups_corpus

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)


# 1. Load dataset
try:
    raw_texts, labels, label_names, manifest = load_newsgroups_corpus()
    with open(f"{DATA_DIR}/dataset_manifest.json", "w", encoding="utf-8") as file_handle:
        json.dump(manifest, file_handle, indent=2)
    print(f"Loaded local 20 Newsgroups corpus: {manifest['document_count']} documents")
except Exception as error:
    print(f"Local dataset unavailable ({error}). Please set NEWSGROUPS_PATH and retry.")
    raise SystemExit(1)


def clean(text: str) -> str:
  text = re.sub(r"(?m)^>.*$", "", text)
  text = re.sub(r"\S+@\S+", "", text)
  text = re.sub(r"http\S+", "", text)
  return re.sub(r"\s+", " ", text).strip()

cleaned = [(clean(t), l) for t, l in zip(raw_texts, labels)]
cleaned = [(t, l) for t, l in cleaned if len(t) >= 50]
texts = [t for t, _ in cleaned]
doc_labels = [l for _, l in cleaned]
print(f"After cleaning: {len(texts)} docs (dropped {len(raw_texts) - len(texts)} short/empty)")


print("Building TF-IDF matrix...")
vec = TfidfVectorizer(max_features=10000, stop_words="english", sublinear_tf=True, ngram_range=(1, 2))
X = vec.fit_transform(texts)
print(f"TF-IDF shape: {X.shape}")

print("LSA dimensionality reduction (300 components)...")
svd = TruncatedSVD(n_components=300, random_state=42)
embeddings = svd.fit_transform(X)
embeddings = normalize(embeddings)
print(f"Embeddings: {embeddings.shape}, variance retained: {svd.explained_variance_ratio_.sum():.1%}")


np.save(f"{DATA_DIR}/embeddings.npy", embeddings)
pickle.dump(vec, open(f"{DATA_DIR}/tfidf_vec.pkl", "wb"))
pickle.dump(svd, open(f"{DATA_DIR}/svd.pkl", "wb"))
with open(f"{DATA_DIR}/texts.pkl", "wb") as f:
  pickle.dump(texts, f)
with open(f"{DATA_DIR}/labels.pkl", "wb") as f:
  pickle.dump({"labels": doc_labels, "label_names": list(label_names)}, f)
print("Embeddings and metadata saved.")


print(f"Building nearest neighbors index ({embeddings.shape[0]} vectors, dim={embeddings.shape[1]})...")
idx = NearestNeighbors(n_neighbors=min(10, len(embeddings)), metric="cosine", algorithm="brute")
idx.fit(embeddings)
pickle.dump(idx, open(f"{DATA_DIR}/nn_index.pkl", "wb"))
print("Nearest neighbors index saved.")


def search(query, k=3):
  q_emb = normalize(svd.transform(vec.transform([query])))
  distances, indices = idx.kneighbors(q_emb, n_neighbors=k)
  return [(label_names[doc_labels[i]], round(1 - d, 3), texts[i][:100])
      for i, d in zip(indices[0], distances[0])]

for q in ["NASA space shuttle", "gun control Congress", "Linux kernel crash"]:
  print(f"\nTop 3 for '{q}':")
  for cat, sim, snippet in search(q):
    print(f"  [{cat}] sim={sim} | {snippet}")

print("\nPart 1 complete.")
