# load_to_es.py
from elasticsearch import Elasticsearch, helpers
import json

ES_URL = "http://localhost:9200"
INDEX = "ecb_speeches"

es = Elasticsearch(ES_URL, headers={"X-elastic-product": "Elasticsearch"})

with open("processed_ecb_data/ecb_speeches_metadata.json", "r", encoding="utf-8") as f:
    docs = json.load(f)

def gen_actions():
    for d in docs:
        # build a stable _id so re-running the script upserts instead of duplicating
        sid = d.get("speech_id")
        cid = d.get("chunk_id")
        doc_id = f"{sid}-{cid}" if sid is not None and cid is not None else None

        yield {
            "_op_type": "index",          # or "create" if you prefer strict no-overwrite
            "_index": INDEX,
            "_id": doc_id,                # lets you safely re-run
            "_source": {
                "speaker": ", ".join(d.get("speakers", [])),
                "date": d.get("date"),    # your mapping expects a date type
                "title": d.get("title"),
                "subtitle": d.get("subtitle"),
                "speech_type": d.get("speech_type"),
                "location": d.get("location"),
                "content": d.get("chunk_text", ""),
                "speech_id": d.get("sppech_id"),
                "chunk_id": d.get("chunk_id"),
                "total_chunks":d.get("total_chunks"),
            }
        }

helpers.bulk(es, gen_actions())
count = es.count(index=INDEX)["count"]
print(f"Done. Index now has {count} documents.")

