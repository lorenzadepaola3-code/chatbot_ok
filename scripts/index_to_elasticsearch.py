import json
from elasticsearch import Elasticsearch, helpers

INDEX_NAME = "ecb_speeches"
DATA_PATH = "processed_ecb_data/ecb_speeches_metadata.json"

# Connect to Elasticsearch
es = Elasticsearch("http://localhost:9200")

# Load the speech metadata
with open(DATA_PATH, "r", encoding="utf-8") as f:
    chunks = json.load(f)

# Prepare documents
actions = [
    {
        "_index": INDEX_NAME,
        "_id": f"{chunk['speech_id']}_{chunk['chunk_id']}",
        "_source": {
            "speaker": chunk.get("speakers", ["Unknown"])[0],
            "date": chunk.get("date", ""),
            "title": chunk.get("title", ""),
            "subtitle": chunk.get("subtitle", ""),
            "speech_type": chunk.get("speech_type", ""),
            "location": chunk.get("location", ""),
            "content": chunk.get("chunk_text", "")
        }
    }
    for chunk in chunks
]

# Bulk insert
print(f"📦 Indexing {len(actions)} speech chunks to Elasticsearch...")
helpers.bulk(es, actions)
print("✅ Indexing complete!")
