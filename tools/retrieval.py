import faiss, json, numpy as np
from pathlib import Path
from typing import Dict, Any
from sentence_transformers import SentenceTransformer

FAISS_INDEX = Path(
    r"c:/Users/depaoll/Downloads/chatbot/data/ecb_speeches_embeddings.faiss"
)
META_PATH = Path(
    r"c:/Users/depaoll/Downloads/chatbot/processed_ecb_data/faiss_meta.json"
)
MODEL_PATH = r"C:/Users/depaoll/Downloads/chatbot/all-MiniLM-L6-v2"

_INDEX = None
_ENCODER = None
_META_MAP: Dict[str, Any] = {}


def _load_index():
    global _INDEX
    if _INDEX is None:
        _INDEX = faiss.read_index(str(FAISS_INDEX))
    return _INDEX


def _load_encoder():
    global _ENCODER
    if _ENCODER is None:
        _ENCODER = SentenceTransformer(MODEL_PATH)
    return _ENCODER


def _load_meta():
    global _META_MAP
    if _META_MAP:
        return _META_MAP
    with META_PATH.open(encoding="utf-8") as f:
        raw = json.load(f)
    # Normalize: handle list OR dict
    if isinstance(raw, dict):
        # Already mapping? assume values have speech_id & chunk_index
        _META_MAP = raw
    elif isinstance(raw, list):
        # Expect each element has row_id plus fields
        temp = {}
        for row in raw:
            # Possible keys: row_id / id
            rid = row.get("row_id") or row.get("id")
            if rid is None:
                continue
            temp[str(rid)] = {
                "speech_id": row.get("speech_id"),
                "chunk_index": row.get("chunk_index"),
                **{k: v for k, v in row.items() if k not in ("row_id", "id")},
            }
        _META_MAP = temp
    else:
        raise ValueError("faiss_meta.json format unsupported")
    return _META_MAP


def embed_query(query: str, encoder=None) -> np.ndarray:
    if encoder is None:
        encoder = _load_encoder()
    vec = encoder.encode([query])
    return vec.astype("float32")


def search(query: str, encoder=None, top_k: int = 20):
    index = _load_index()
    meta = _load_meta()
    qv = embed_query(query, encoder)
    D, I = index.search(qv, top_k)
    rows = []
    for score, row_id in zip(D[0], I[0]):
        if row_id == -1:
            continue
        key = str(row_id)
        info = meta.get(key)
        if not info:
            # Fallback placeholder if missing
            info = {"speech_id": None, "chunk_index": None}
        rows.append({**info, "score": float(score), "row_id": int(row_id)})
    return rows
