"""Cluster-sharded semantic cache.

Stores query embeddings by GMM cluster and looks up by cosine similarity.
"""

import time
import json
import pickle
import numpy as np
from dataclasses import dataclass, field
from typing import Any, Optional
from collections import defaultdict


@dataclass
class CacheEntry:
    query: str
    embedding: np.ndarray
    result: Any
    dominant_cluster: int
    timestamp: float = field(default_factory=time.time)


class SemanticCache:
    """
    Cluster-sharded semantic cache.

    Parameters
    ----------
    gmm        : fitted sklearn GaussianMixture
    pca        : fitted sklearn PCA  (same as used in clustering)
    encoder    : callable (list[str]) -> np.ndarray  (normalised embeddings)
    theta      : float — cosine similarity threshold for a cache hit
    """

    def __init__(self, gmm, pca, encoder, theta: float = 0.88):
        self.gmm = gmm
        self.pca = pca
        self.encoder = encoder
        self.theta = theta
        self.k = gmm.n_components

        # Cluster-sharded storage: dict[int, list[CacheEntry]]
        self._buckets: dict[int, list[CacheEntry]] = defaultdict(list)

        # Stats
        self._hit_count = 0
        self._miss_count = 0

    # Internal helpers

    def _embed(self, query: str) -> np.ndarray:
        """Embed a single query string and return a normalised vector."""
        emb = self.encoder([query])
        norm = np.linalg.norm(emb, axis=1, keepdims=True)
        return (emb / (norm + 1e-10)).squeeze()

    def _soft_assign(self, emb384: np.ndarray) -> np.ndarray:
        """Return GMM soft assignment probabilities for an embedding."""
        reduced = self.pca.transform(emb384.reshape(1, -1))
        return self.gmm.predict_proba(reduced).squeeze()

    def _top2_clusters(self, soft: np.ndarray):
        """Return (dominant_cluster, secondary_cluster) indices."""
        top2 = soft.argsort()[-2:][::-1]
        return int(top2[0]), int(top2[1])

    def _cosine_sim(self, a: np.ndarray, b: np.ndarray) -> float:
        """Cosine similarity between two normalised vectors (dot product)."""
        return float(np.dot(a, b))

    # Public API

    def lookup(self, query: str) -> Optional[dict]:
        """
        Search the cache for a semantically similar previous query.

        Returns a dict on hit:
          {
            "matched_query": str,
            "similarity_score": float,
            "result": any,
            "dominant_cluster": int,
          }
        Returns None on miss.
        """
        q_emb = self._embed(query)
        soft = self._soft_assign(q_emb)
        c1, c2 = self._top2_clusters(soft)

        best_entry = None
        best_sim = -1.0

        # Search primary + secondary cluster buckets
        for cid in (c1, c2):
            for entry in self._buckets[cid]:
                sim = self._cosine_sim(q_emb, entry.embedding)
                if sim >= self.theta and sim > best_sim:
                    best_sim = sim
                    best_entry = entry

        if best_entry is not None:
            self._hit_count += 1
            return {
                "matched_query": best_entry.query,
                "similarity_score": round(best_sim, 4),
                "result": best_entry.result,
                "dominant_cluster": best_entry.dominant_cluster,
            }

        self._miss_count += 1
        return None

    def store(self, query: str, result: Any) -> int:
        """
        Store a new query→result pair in the cache.
        Returns the dominant cluster the entry was stored in.
        """
        q_emb = self._embed(query)
        soft = self._soft_assign(q_emb)
        c1, _ = self._top2_clusters(soft)

        entry = CacheEntry(
            query=query,
            embedding=q_emb,
            result=result,
            dominant_cluster=c1,
        )
        self._buckets[c1].append(entry)
        return c1

    def get_or_compute(self, query: str, compute_fn) -> dict:
        """
        High-level helper: return cached result or compute + store it.
        compute_fn: callable(query: str) -> Any
        """
        hit = self.lookup(query)
        if hit is not None:
            return {"cache_hit": True, **hit}

        result = compute_fn(query)
        cluster = self.store(query, result)
        return {
            "cache_hit": False,
            "matched_query": None,
            "similarity_score": None,
            "result": result,
            "dominant_cluster": cluster,
        }

    def flush(self):
        """Clear all cache entries and reset stats."""
        self._buckets = defaultdict(list)
        self._hit_count = 0
        self._miss_count = 0

    @property
    def stats(self) -> dict:
        total = self._hit_count + self._miss_count
        total_entries = sum(len(v) for v in self._buckets.values())
        return {
            "total_entries": total_entries,
            "hit_count": self._hit_count,
            "miss_count": self._miss_count,
            "hit_rate": round(self._hit_count / total, 4) if total > 0 else 0.0,
        }

    # Serialisation

    def to_dict(self) -> dict:
        """Serialise cache state (for persistence across restarts)."""
        out = {}
        for cid, entries in self._buckets.items():
            out[str(cid)] = [
                {
                    "query": e.query,
                    "embedding": e.embedding.tolist(),
                    "result": e.result,
                    "dominant_cluster": e.dominant_cluster,
                    "timestamp": e.timestamp,
                }
                for e in entries
            ]
        return {
            "buckets": out,
            "hit_count": self._hit_count,
            "miss_count": self._miss_count,
        }

    def from_dict(self, d: dict):
        """Restore cache state from serialised dict."""
        self._buckets = defaultdict(list)
        for cid_str, entries in d.get("buckets", {}).items():
            cid = int(cid_str)
            for e in entries:
                self._buckets[cid].append(
                    CacheEntry(
                        query=e["query"],
                        embedding=np.array(e["embedding"]),
                        result=e["result"],
                        dominant_cluster=e["dominant_cluster"],
                        timestamp=e["timestamp"],
                    )
                )
        self._hit_count = d.get("hit_count", 0)
        self._miss_count = d.get("miss_count", 0)


# Theta analysis (exploratory)
# Run standalone to generate the analysis; not imported by the API.

if __name__ == "__main__":
    import joblib
    from sentence_transformers import SentenceTransformer

    DATA_DIR = "data"
    print("Loading artefacts for theta analysis...")
    pca = joblib.load(f"{DATA_DIR}/pca.pkl")
    gmm = joblib.load(f"{DATA_DIR}/gmm.pkl")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    encoder = lambda qs: model.encode(qs, normalize_embeddings=True)

    # Paraphrase pairs to probe the cache
    PARAPHRASE_PAIRS = [
        ("What are the health effects of smoking?",
         "How does cigarette smoking affect the human body?"),
        ("Best programming languages for machine learning",
         "Which coding language should I use for AI development?"),
        ("NASA Mars mission updates",
         "Latest news about space exploration on the red planet"),
        ("How do I fix a kernel panic in Linux?",
         "My Linux system is crashing with kernel errors"),
        ("Israeli-Palestinian conflict history",
         "Origins of the Middle East peace dispute"),
    ]

    print("\nTheta Analysis")
    print(f"{'θ':>6} | {'Pair':<60} | Sim  | Hit?")
    print("-" * 90)

    for theta in [0.80, 0.85, 0.88, 0.91, 0.95]:
        cache = SemanticCache(gmm=gmm, pca=pca, encoder=encoder, theta=theta)
        results = []
        for q1, q2 in PARAPHRASE_PAIRS:
            cache.store(q1, f"result_for: {q1}")
            hit = cache.lookup(q2)
            sim = hit["similarity_score"] if hit else None
            results.append((q1[:55], sim, hit is not None))

        hits = sum(1 for _, _, h in results)
        for q, s, h in results:
            sim_str = f"{s:.3f}" if s else "  —  "
            print(f"  {theta:.2f} | {q:<60} | {sim_str} | {'HIT' if h else 'miss'}")
        print(f"  {'':6} {'':60}   Hit rate: {hits}/{len(PARAPHRASE_PAIRS)}")
        print()

    print("""
Interpretation:
  θ = 0.80 → hits nearly all paraphrases but risks false positives on topically
             adjacent but semantically distinct queries.""")
