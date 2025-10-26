# filepath: c:\Users\depaoll\Downloads\chatbot\scripts\check_token_index.py
import json, faiss
from pathlib import Path

INDEX = Path(r"c:/Users/depaoll/Downloads/chatbot/data/ecb_speeches_embedding.faiss")
META = Path(r"c:/Users/depaoll/Downloads/chatbot/processed_ecb_data/faiss_meta.json")
SIDECAR = Path(
    r"c:/Users/depaoll/Downloads/chatbot/processed_ecb_data/speech_sidecar_token_chunks.json"
)

meta = json.load(META.open())
side = json.load(SIDECAR.open())
idx = faiss.read_index(str(INDEX))

print("Index vectors:", idx.ntotal)
print("Meta rows:", len(meta))
total_side_chunks = sum(len(v) for v in side.values())
print("Sidecar chunk count:", total_side_chunks)

assert len(meta) == idx.ntotal, "Meta size mismatch index"
assert total_side_chunks == len(meta), "Sidecar vs meta mismatch"

print("OK: counts align.")
