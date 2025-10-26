import json
from pathlib import Path

SIDE = json.load(
    open(
        r"c:/Users/depaoll/Downloads/chatbot/processed_ecb_data/speech_sidecar.json",
        "r",
        encoding="utf-8",
    )
)
INP = Path(r"c:/Users/depaoll/Downloads/chatbot/labeled_queries.jsonl")
OUT = Path(r"c:/Users/depaoll/Downloads/chatbotlabeled_queries_enriched.jsonl")


def first_meta(speech_id):
    arr = SIDE.get(speech_id, [])
    if not arr:
        return {}
    m = arr[0]
    return {
        "speaker": m.get("speaker", ""),
        "date": m.get("date", ""),
        "title": m.get("title", ""),
    }


with INP.open(encoding="utf-8") as fin, OUT.open("w", encoding="utf-8") as fout:
    for line in fin:
        if not line.strip():
            continue
        obj = json.loads(line)
        speakers = set()
        dates = set()
        titles = set()
        for ref in obj["positives"]:
            meta = first_meta(ref["speech_id"])
            if meta.get("speaker"):
                speakers.add(meta["speaker"])
            if meta.get("date"):
                dates.add(meta["date"])
            if meta.get("title"):
                titles.add(meta["title"])
        obj.setdefault("meta", {})["speakers"] = sorted(speakers)
        obj["meta"]["dates"] = sorted(dates)
        obj["meta"]["titles"] = sorted(titles)
        fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
print("Enriched written to", OUT)
