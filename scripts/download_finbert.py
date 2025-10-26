from transformers import AutoTokenizer, AutoModelForSequenceClassification
MODEL = "ProsusAI/finbert"
OUT = "processed_ecb_data/finbert"

# create dir if missing and download+save
import os
os.makedirs(OUT, exist_ok=True)
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)
tok.save_pretrained(OUT)
model.save_pretrained(OUT)
print("FinBERT saved to", OUT)