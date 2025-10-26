# filepath: c:\Users\depaoll\Downloads\chatbot\scripts\create_faiss_index_token.py
import json, faiss, numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer

MODEL_PATH = r"C:/Users/depaoll/Downloads/chatbot/all-MiniLM-L6-v2"
SIDE_PATH = Path(
    r"c:/Users/depaoll/Downloads/chatbot/processed_ecb_data/speech_sidecar_token_chunks.json"
)
OUT_INDEX = Path(
    r"c:/Users/depaoll/Downloads/chatbot/processed_ecb_data/ecb_token_chunks.faiss"
)
OUT_META = Path(
    r"c:/Users/depaoll/Downloads/chatbot/processed_ecb_data/faiss_meta_token.json"
)
EMB_CACHE = Path(
    r"c:/Users/depaoll/Downloads/chatbot/processed_ecb_data/token_embeddings.npy"
)
BATCH = 64


def iterate_chunks(side):
    for sid in sorted(side.keys()):
        for c in sorted(side[sid], key=lambda x: x["chunk_index"]):
            yield sid, c["chunk_index"], c["text"]


def main():
    side = json.load(SIDE_PATH.open(encoding="utf-8"))
    texts = []
    meta = []
    for sid, ci, txt in iterate_chunks(side):
        texts.append(txt)
        meta.append({"speech_id": sid, "chunk_index": ci})
    print("Total token chunks:", len(texts))
    model = SentenceTransformer(MODEL_PATH)
    if EMB_CACHE.exists():
        emb = np.load(EMB_CACHE)
        print("Loaded cache:", EMB_CACHE)
    else:
        emb_parts = []
        for i in range(0, len(texts), BATCH):
            batch = texts[i : i + BATCH]
            e = model.encode(batch, convert_to_numpy=True, normalize_embeddings=True)
            emb_parts.append(e.astype("float32"))
        emb = np.vstack(emb_parts)
        np.save(EMB_CACHE, emb)
        print("Saved embeddings cache:", EMB_CACHE)
    dim = emb.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(emb)
    faiss.write_index(index, str(OUT_INDEX))
    print("Wrote index:", OUT_INDEX, "vectors:", index.ntotal)
    json.dump(meta, OUT_META.open("w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print("Wrote meta:", OUT_META)


if __name__ == "__main__":
    main()
