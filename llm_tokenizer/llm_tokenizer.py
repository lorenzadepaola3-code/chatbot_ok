from transformers import AutoTokenizer
import json, math
from pathlib import Path

TOKENIZER_NAME = (
    "C:/Users/depaoll/Downloads/chatbot/all-MiniLM-L6-v2"  # or model you actually use
)
MAX_TOKENS = 256
OVERLAP = 32  # token overlap
SIDE_IN = Path(
    r"c:/Users/depaoll/Downloads/chatbot/processed_ecb_data/speech_sidecar.json"
)
SIDE_OUT = Path(
    r"c:/Users/depaoll/Downloads/chatbot/processed_ecb_data/speech_sidecar_token_chunks.json"
)

tok = AutoTokenizer.from_pretrained(TOKENIZER_NAME, use_fast=True)


def chunk_text(text: str):
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    ids = enc["input_ids"]
    offsets = enc["offset_mapping"]
    chunks = []
    start = 0
    while start < len(ids):
        end = min(start + MAX_TOKENS, len(ids))
        # Try to extend to sentence end if near boundary
        # (simple heuristic: move end forward until punctuation or limit)
        while end < len(ids) and end - start < MAX_TOKENS + 20:
            char_start, char_end = offsets[end - 1]
            if text[char_end - 1 : char_end] in ".!?":
                break
            end += 1
        sel_ids = ids[start:end]
        sel_offsets = offsets[start:end]
        char_start = sel_offsets[0][0]
        char_end = sel_offsets[-1][1]
        chunk_text = text[char_start:char_end].strip()
        chunks.append(
            {"char_start": char_start, "char_end": char_end, "text": chunk_text}
        )
        if end == len(ids):
            break
        start = end - OVERLAP
    return chunks


def rebuild_sidecar():
    side = json.load(SIDE_IN.open(encoding="utf-8"))
    new_side = {}
    for speech_id, old_chunks in side.items():
        # Rejoin original chunks to full text (if original sidecar is already chunked)
        full = "".join(
            c["text"] for c in sorted(old_chunks, key=lambda x: x["chunk_index"])
        )
        token_chunks = chunk_text(full)
        enriched = []
        # pull metadata from first original chunk
        base_meta = old_chunks[0]
        for idx, ch in enumerate(token_chunks):
            enriched.append(
                {
                    "speech_id": speech_id,
                    "chunk_index": idx,
                    "speaker": base_meta.get("speaker"),
                    "date": base_meta.get("date"),
                    "title": base_meta.get("title"),
                    "text": ch["text"],
                    "char_start": ch["char_start"],
                    "char_end": ch["char_end"],
                    "token_window": f"{MAX_TOKENS}~{OVERLAP}",
                }
            )
        new_side[speech_id] = enriched
    json.dump(new_side, SIDE_OUT.open("w", encoding="utf-8"), ensure_ascii=False)
    print("Wrote token-aware sidecar:", SIDE_OUT)


if __name__ == "__main__":
    rebuild_sidecar()
