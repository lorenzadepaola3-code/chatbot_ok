# filepath: c:\Users\depaoll\Downloads\chatbot\scripts\flatten_token_sidecar.py
import json
from pathlib import Path

SIDE_IN = Path(
    r"c:/Users/depaoll/Downloads/chatbot/processed_ecb_data/speech_sidecar_token_chunks.json"
)
OUT_LIST = Path(
    r"c:/Users/depaoll/Downloads/chatbot/processed_ecb_data/token_chunks_list.json"
)


def main():
    side = json.load(SIDE_IN.open(encoding="utf-8"))
    flat = []
    for sid, chunks in side.items():
        for ch in chunks:
            flat.append(
                {
                    "speech_id": sid,
                    "chunk_index": ch["chunk_index"],
                    "chunk_text": ch["text"],
                    "speaker": ch.get("speaker"),
                    "date": ch.get("date"),
                    "title": ch.get("title"),
                }
            )
    flat.sort(key=lambda x: (x["speech_id"], x["chunk_index"]))
    json.dump(flat, OUT_LIST.open("w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print("Wrote", OUT_LIST, "records:", len(flat))


if __name__ == "__main__":
    main()
