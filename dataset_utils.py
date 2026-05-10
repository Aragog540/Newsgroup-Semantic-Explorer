from __future__ import annotations

import os
import re
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent


def _candidate_roots() -> list[Path]:
    env_path = os.getenv("NEWSGROUPS_PATH")
    if env_path:
        return [Path(env_path)]

    full_root = BASE_DIR / "20_newsgroups"
    mini_root = BASE_DIR / "mini_newsgroups"
    legacy_mini_root = BASE_DIR / "mini_groups"
    return [
        full_root / "20_newsgroups",
        full_root,
        mini_root / "mini_newsgroups",
        mini_root,
        legacy_mini_root / "mini_groups",
        legacy_mini_root,
    ]


def _looks_like_dataset_root(path: Path) -> bool:
    if not path.is_dir():
        return False

    category_dirs = [child for child in path.iterdir() if child.is_dir()]
    if len(category_dirs) < 5:
        return False

    sample = category_dirs[:5]
    return all(any(item.is_file() for item in cat.iterdir()) for cat in sample)


def resolve_dataset_root() -> Path:
    for candidate in _candidate_roots():
        if candidate.exists() and _looks_like_dataset_root(candidate):
            return candidate

    for candidate in _candidate_roots():
        if candidate.exists() and candidate.is_dir():
            for child in candidate.iterdir():
                if _looks_like_dataset_root(child):
                    return child

    raise FileNotFoundError(
        "Could not find a 20 Newsgroups dataset folder. Set NEWSGROUPS_PATH to the raw corpus root."
    )


def _strip_headers(text: str) -> str:
    if "\n\n" not in text:
        return text

    header, body = text.split("\n\n", 1)
    header_lower = header.lower()
    if any(marker in header_lower for marker in ("subject:", "from:", "organization:", "lines:", "message-id:")):
        return body
    return text


def clean_text(text: str) -> str:
    text = _strip_headers(text)
    text = re.sub(r"(?m)^>.*$", "", text)
    text = re.sub(r"\S+@\S+", "", text)
    text = re.sub(r"http\S+", "", text)
    return re.sub(r"\s+", " ", text).strip()


def load_newsgroups_corpus(min_chars: int = 50):
    root = resolve_dataset_root()
    category_dirs = sorted([child for child in root.iterdir() if child.is_dir()], key=lambda path: path.name)
    label_names = [path.name for path in category_dirs]

    texts: list[str] = []
    labels: list[int] = []
    raw_count = 0
    kept_count = 0
    per_category: dict[str, int] = {}

    for label_id, category_dir in enumerate(category_dirs):
        print(f"Loading category {label_id + 1}/{len(category_dirs)}: {category_dir.name}...")
        files = sorted([item for item in category_dir.iterdir() if item.is_file()], key=lambda path: path.name)
        raw_count += len(files)
        kept_here = 0
        for file_path in files:
            try:
                raw_text = file_path.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                raw_text = file_path.read_text(errors="ignore")
            cleaned = clean_text(raw_text)
            if len(cleaned) < min_chars:
                continue
            texts.append(cleaned)
            labels.append(label_id)
            kept_here += 1
        kept_count += kept_here
        per_category[category_dir.name] = kept_here

    manifest = {
        "dataset_root": str(root),
        "dataset_name": root.name,
        "document_count": kept_count,
        "raw_document_count": raw_count,
        "category_count": len(label_names),
        "categories": label_names,
        "per_category_counts": per_category,
    }
    return texts, labels, label_names, manifest
