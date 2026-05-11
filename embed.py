#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

CATALOG_PATH = Path("catalog.json")
INDEX_PATH = Path("faiss.index")
ID_MAP_PATH = Path("id_map.json")
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def load_catalog(path: Path) -> List[Dict]:
    if not path.exists():
        raise FileNotFoundError(f"Missing catalog file: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("catalog.json must be a JSON array")
    return data


def build_text_blob(item: Dict) -> str:
    name = str(item.get("name", "")).strip()
    description = str(item.get("description", "")).strip()
    return f"{name}\n{description}".strip()


def build_embeddings(texts: List[str], model_name: str) -> np.ndarray:
    model = SentenceTransformer(model_name)
    vectors = model.encode(
        texts,
        batch_size=64,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    if vectors.dtype != np.float32:
        vectors = vectors.astype(np.float32)
    return vectors


def build_faiss_index(vectors: np.ndarray) -> faiss.Index:
    dim = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)  # cosine via normalized vectors
    index.add(vectors)
    return index


def build_id_map(items: List[Dict], texts: List[str]) -> Dict[str, Dict]:
    id_map: Dict[str, Dict] = {}
    for idx, (item, text_blob) in enumerate(zip(items, texts)):
        id_map[str(idx)] = {
            "name": item.get("name", ""),
            "url": item.get("url", ""),
            "test_type": item.get("test_type", ""),
            "description": item.get("description", ""),
            "text_blob": text_blob,
        }
    return id_map


def main() -> None:
    items = load_catalog(CATALOG_PATH)
    texts = [build_text_blob(item) for item in items]

    if not texts:
        raise ValueError("catalog.json is empty; nothing to embed")

    embeddings = build_embeddings(texts, MODEL_NAME)
    index = build_faiss_index(embeddings)
    id_map = build_id_map(items, texts)

    faiss.write_index(index, str(INDEX_PATH))
    with ID_MAP_PATH.open("w", encoding="utf-8") as f:
        json.dump(id_map, f, ensure_ascii=False, indent=2)

    print(
        f"Saved {len(items)} embeddings to {INDEX_PATH} and metadata to {ID_MAP_PATH}"
    )


if __name__ == "__main__":
    main()
