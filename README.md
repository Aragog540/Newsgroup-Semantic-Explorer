# Semantic Search on 20 Newsgroups

This project is a semantic search app built on the 20 Newsgroups corpus. It cleans the raw posts, builds TF-IDF + LSA embeddings, clusters them with a Gaussian Mixture Model, and uses a cluster-sharded semantic cache to speed up repeated or paraphrased queries.

The app includes a FastAPI backend and a browser-based dashboard.

**Status:** ✅ Production-Ready with comprehensive logging, health checks, security, and Docker deployment

## Dataset
Download from the link given - 
https://archive.ics.uci.edu/dataset/113/twenty+newsgroups

## What is included

- `app.py` - FastAPI app and API endpoints
- `main.py` - Uvicorn entry point
- `dataset_utils.py` - Loader for the raw 20 Newsgroups folders
- `part1_embed.py` - Builds embeddings and the nearest-neighbor index
- `part2_cluster.py` - Builds PCA, GMM clustering, and visual artefacts
- `part3_cache.py` - Semantic cache implementation
- `ui.html` - Web dashboard
- `requirements.txt` - Python dependencies

## Prerequisites

- Python 3.10+ recommended
- The raw dataset folders already present on the target system
- Internet access the first time the embedding model is downloaded, if you use neural embeddings or run the cache analysis script

The app expects one of these dataset locations inside the bundle folder or a path provided through `NEWSGROUPS_PATH`:

- `20_newsgroups/20_newsgroups`
- `mini_newsgroups/mini_newsgroups`

## Step-by-step run process

### 1. Open the bundle folder

Open the project folder in a terminal.

### 2. Create a virtual environment

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

On macOS or Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Make sure the dataset folders are available

If the corpus is already inside the bundle folder, no extra setup is needed.

If the corpus is somewhere else, set the path before running the scripts.

PowerShell:

```powershell
$env:NEWSGROUPS_PATH = "C:\path\to\20_newsgroups\20_newsgroups"
```

### 5. Build embeddings and the search index

Run Part 1 first:

```bash
python part1_embed.py
```

This creates the embedding and nearest-neighbor artefacts under `data/`.

### 6. Build clustering artefacts

Run Part 2 next:

```bash
python part2_cluster.py
```

This creates the PCA, GMM, soft assignments, cluster summaries, and the visual files used by the dashboard.

### 7. Start the web app

Start the FastAPI server:

```bash
uvicorn main:app --reload --port 8000
```

### 8. Open the app in the browser

Open:

```text
http://127.0.0.1:8000/
```

The API docs are available at:

```text
http://127.0.0.1:8000/docs
```

## If something is missing

- If `part1_embed.py` fails, check that the dataset path is correct.
- If the server starts but the search result is incomplete, confirm that the `data/` folder contains the generated artefacts.
- If the dashboard images do not appear, rerun `part2_cluster.py` so `bic_curve.png` and `tsne_clusters.png` are created.

## Quick summary

Run these in order:

1. Create and activate a virtual environment
2. `pip install -r requirements.txt`
3. `python part1_embed.py`
4. `python part2_cluster.py`
5. `uvicorn main:app --reload --port 8000`

## Quick Docker Deployment

For quick deployment without complex setup:

```bash
cp .env.example .env
docker-compose up -d
curl http://localhost:8000/health
```

## Production-Ready Features

✅ Comprehensive logging with rotation  
✅ Health check endpoint (`GET /health`)  
✅ Security headers on all responses  
✅ Rate limiting (100 requests/60s)  
✅ GZip compression for API responses  
✅ Graceful shutdown support  
✅ Docker multi-stage build  
✅ Docker Compose orchestration  
✅ Environment-based configuration  

## Documentation

- **README.md** - This file (overview)
- **.env.example** - Configuration template
