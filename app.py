from __future__ import annotations

import json
import logging
import os
import pickle
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from sklearn.preprocessing import normalize

from config import get_settings, setup_logging
from dataset_utils import load_newsgroups_corpus, resolve_dataset_root
from part3_cache import SemanticCache

settings = get_settings()
logger = setup_logging(settings)
limiter = Limiter(key_func=get_remote_address)


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
UI_PATH = BASE_DIR / "ui.html"

SAMPLE_QUERIES = [
    "NASA space shuttle mission details",
    "Windows blue screen error troubleshooting",
    "How do I configure gun control legislation?",
    "Best graphics card for 3D rendering",
    "Why do religious debates become political?",
]



class AppState:
    vec = None
    svd = None
    index = None
    texts: list[str] | None = None
    doc_labels: list[int] | None = None
    label_names: list[str] | None = None
    pca = None
    gmm = None
    cache: SemanticCache | None = None
    embedding_type: str = "lsa"
    dataset_manifest: dict = {}
    cluster_summary: dict = {}


state = AppState()


def make_encoder():
    if state.embedding_type == "neural":
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer("all-MiniLM-L6-v2")
        return lambda queries: model.encode(queries, normalize_embeddings=True)

    vec, svd = state.vec, state.svd

    def lsa_encode(queries):
        encoded = vec.transform(queries)
        return normalize(svd.transform(encoded))

    return lsa_encode


def _load_json(path: Path, default: dict) -> dict:
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as file_handle:
        return json.load(file_handle)


def _load_pickle(path: Path):
    with path.open("rb") as file_handle:
        return pickle.load(file_handle)


def load_artefacts():
    DATA_DIR.mkdir(exist_ok=True)

    texts_path = DATA_DIR / "texts.pkl"
    labels_path = DATA_DIR / "labels.pkl"
    manifest_path = DATA_DIR / "dataset_manifest.json"

    if texts_path.exists() and labels_path.exists():
        logger.info("Loading cached corpus artefacts...")
        state.texts = _load_pickle(texts_path)
        label_payload = _load_pickle(labels_path)
        state.doc_labels = label_payload["labels"]
        state.label_names = label_payload["label_names"]
        state.dataset_manifest = _load_json(manifest_path, {
            "dataset_name": "unknown",
            "dataset_root": "unknown",
            "document_count": len(state.texts or []),
            "raw_document_count": len(state.texts or []),
            "category_count": len(state.label_names or []),
            "categories": state.label_names or [],
            "per_category_counts": {},
        })
    else:
        dataset_root = resolve_dataset_root()
        logger.info(f"Loading corpus from {dataset_root}...")
        texts, labels, label_names, manifest = load_newsgroups_corpus()
        state.texts = texts
        state.doc_labels = labels
        state.label_names = label_names
        state.dataset_manifest = manifest

        with texts_path.open("wb") as file_handle:
            pickle.dump(texts, file_handle)
        with labels_path.open("wb") as file_handle:
            pickle.dump({"labels": labels, "label_names": label_names}, file_handle)
        with manifest_path.open("w", encoding="utf-8") as file_handle:
            json.dump(manifest, file_handle, indent=2)

    prefer_neural = settings.embedding_type.lower() == "neural"
    if prefer_neural:
        try:
            from sentence_transformers import SentenceTransformer

            state.neural_model = SentenceTransformer("all-MiniLM-L6-v2")
            state.embedding_type = "neural"
            logger.info("Using neural embeddings (all-MiniLM-L6-v2)")
        except Exception as e:
            state.embedding_type = "lsa"
            logger.warning(f"Neural embeddings unavailable; using LSA instead: {e}")
    else:
        state.vec = _load_pickle(DATA_DIR / "tfidf_vec.pkl")
        state.svd = _load_pickle(DATA_DIR / "svd.pkl")
        state.embedding_type = "lsa"
        logger.info("Using LSA embeddings (TF-IDF + SVD)")

    index_path = DATA_DIR / "nn_index.pkl"
    if index_path.exists():
        state.index = _load_pickle(index_path)
        logger.info("Loaded sklearn NearestNeighbors index")
    else:
        state.index = None
        logger.warning("nn_index.pkl not found. Run part1_embed.py first.")

    logger.info("Loading PCA and GMM...")
    state.pca = joblib.load(DATA_DIR / "pca.pkl")
    state.gmm = joblib.load(DATA_DIR / "gmm.pkl")
    state.cluster_summary = _load_json(DATA_DIR / "cluster_summaries.json", {"k": 0, "clusters": {}})

    theta = settings.cache_theta_neural if state.embedding_type == "neural" else settings.cache_theta_lsa
    logger.info(f"Initialising semantic cache (theta={theta}, embedding={state.embedding_type})...")
    encoder = make_encoder()
    state.cache = SemanticCache(gmm=state.gmm, pca=state.pca, encoder=encoder, theta=theta)
    logger.info("Service ready.")


# Preload models into memory so Gunicorn can share them across workers via CoW
settings.validate_at_startup()
load_artefacts()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events."""
    logger.info(f"Starting {settings.app_name} v{settings.app_version}")
    logger.info(f"Debug mode: {settings.debug}")
    logger.info(f"Embedding type: {settings.embedding_type}")

    logger.info("Application startup complete")
    yield
    logger.info("Application shutting down...")


app = FastAPI(
    title=settings.app_name,
    description="Semantic search with fuzzy GMM clustering and cluster-sharded semantic cache.",
    version=settings.app_version,
    lifespan=lifespan,
    debug=settings.debug,
)

if settings.rate_limit_enabled:
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, lambda request, exc: JSONResponse(
        status_code=429, content={"detail": "Rate limit exceeded"}
    ))


def create_rate_limit_decorator():
    """Create rate limiting decorator based on settings."""
    if settings.rate_limit_enabled:
        return limiter.limit(f"{settings.rate_limit_requests}/{settings.rate_limit_period}seconds")
    else:
        return lambda f: f


rate_limit = create_rate_limit_decorator()

cors_origins = settings.cors_origins.split(",") if settings.cors_origins != "*" else ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(GZipMiddleware, minimum_size=500)


@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    """Add security headers to all responses."""
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; "
        "script-src 'self' 'unsafe-inline'; "
        "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
        "font-src 'self' https://fonts.gstatic.com; "
        "img-src 'self' data:"
    )
    return response


app.mount("/assets", StaticFiles(directory=str(DATA_DIR)), name="assets")


class QueryRequest(BaseModel):
    query: str


class QueryResponse(BaseModel):
    query: str
    cache_hit: bool
    matched_query: Optional[str]
    similarity_score: Optional[float]
    result: str
    dominant_cluster: int


class CacheStatsResponse(BaseModel):
    total_entries: int
    hit_count: int
    miss_count: int
    hit_rate: float


class FlushResponse(BaseModel):
    message: str


def compute_result(query: str) -> str:
    if state.index is None:
        return "Error: index not loaded. Run part1_embed.py first."

    encoder = make_encoder()
    q_emb = encoder([query])
    distances, indices = state.index.kneighbors(q_emb, n_neighbors=min(3, len(state.texts or [])))

    results = []
    for rank, (doc_index, distance) in enumerate(zip(indices[0], distances[0]), 1):
        similarity = round(1 - float(distance), 4)
        category = state.label_names[state.doc_labels[doc_index]]
        snippet = state.texts[doc_index][:300].replace("\n", " ").strip()
        results.append(f"[{rank}] [{category}] (sim={similarity}) {snippet}...")
    return "\n\n".join(results)


@app.get("/", response_class=HTMLResponse)
async def home():
    return UI_PATH.read_text(encoding="utf-8")


@app.get("/health")
async def health_check():
    """Health check endpoint for orchestration (Kubernetes, Docker, etc)."""
    try:
        health_status = {
            "status": "healthy",
            "version": settings.app_version,
            "service": settings.app_name,
            "embedding_type": state.embedding_type,
            "index_loaded": state.index is not None,
            "cache_enabled": state.cache is not None,
        }
        logger.debug("Health check passed")
        return health_status
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "error": str(e)}
        )


@app.get("/api/overview")
async def overview():
    cache_stats = state.cache.stats if state.cache else {"total_entries": 0, "hit_count": 0, "miss_count": 0, "hit_rate": 0.0}
    return {
        "dataset_name": state.dataset_manifest.get("dataset_name", "20_newsgroups"),
        "dataset_root": state.dataset_manifest.get("dataset_root", "unknown"),
        "document_count": len(state.texts or []),
        "category_count": len(state.label_names or []),
        "cluster_count": int(state.gmm.n_components) if state.gmm is not None else 0,
        "embedding_type": state.embedding_type,
        "theta": state.cache.theta if state.cache else None,
        "cache": cache_stats,
        "samples": SAMPLE_QUERIES,
    }


@app.get("/api/cluster-summary")
async def cluster_summary():
    return state.cluster_summary


@app.post("/query", response_model=QueryResponse)
@rate_limit
async def query_endpoint(request: Request, body: QueryRequest):
    if not body.query or not body.query.strip():
        logger.warning("Empty query received")
        raise HTTPException(status_code=400, detail="Query must not be empty.")

    query = body.query.strip()
    logger.info(f"Processing query: {query[:100]}")

    try:
        outcome = state.cache.get_or_compute(query, compute_result)
        logger.info(f"Query processed - Cache hit: {outcome['cache_hit']}")
        return QueryResponse(
            query=query,
            cache_hit=outcome["cache_hit"],
            matched_query=outcome.get("matched_query"),
            similarity_score=outcome.get("similarity_score"),
            result=outcome["result"],
            dominant_cluster=outcome["dominant_cluster"],
        )
    except Exception as e:
        logger.error(f"Query processing failed: {e}")
        raise HTTPException(status_code=500, detail="Query processing failed")


@app.get("/cache/stats", response_model=CacheStatsResponse)
async def cache_stats():
    return CacheStatsResponse(**state.cache.stats)


@app.delete("/cache", response_model=FlushResponse)
async def flush_cache():
    if state.cache:
        state.cache.flush()
        return FlushResponse(message="Cache flushed successfully")
    return FlushResponse(message="Cache not enabled")

