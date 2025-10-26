from elasticsearch import Elasticsearch
es = Elasticsearch("http://localhost:9200", headers={"X-elastic-product":"Elasticsearch"})
es.indices.delete(index="ecb_speeches", ignore=[400,404])
print("Deleted index ecb_speeches (if it existed).")