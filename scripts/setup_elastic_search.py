from elasticsearch import Elasticsearch, exceptions
import json

# Configurazione
ELASTICSEARCH_URL = "http://localhost:9200"
INDEX_NAME = "ecb_speeches"


# Headers richiesti da Elasticsearch 8+
HEADERS = {"X-elastic-product": "Elasticsearch"}

# Mappatura del nostro indice (puoi modificarla in base ai tuoi dati)
INDEX_MAPPING = {
    "mappings": {
        "properties": {
            "speaker": {"type": "text"},
            "date": {"type": "date"},
            "title": {"type": "text"},
            "subtitle": {"type": "text"},
            "speech_type": {"type": "keyword"},
            "location": {"type": "text"},
            "content": {"type": "text"}
        }
    }
}

def main():
    try:
        es = Elasticsearch(ELASTICSEARCH_URL, headers=HEADERS)

        # Test di connessione
        if not es.ping():
            print("[✗] Could not ping Elasticsearch server")
            return

        print("[✓] Connected to Elasticsearch")

        # Crea indice se non esiste
        if not es.indices.exists(index=INDEX_NAME):
            es.indices.create(index=INDEX_NAME, body=INDEX_MAPPING)
            print(f"[✓] Created index: {INDEX_NAME}")
        else:
            print(f"[•] Index already exists: {INDEX_NAME}")

    except exceptions.ConnectionError as e:
        print("[✗] Could not connect to Elasticsearch")
        print(f"Details: {e}")

    except exceptions.RequestError as e:
        print("[✗] Request error during index creation")
        print(f"Details: {e.info}")

if __name__ == "__main__":
    main()
