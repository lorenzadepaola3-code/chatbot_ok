import json
from pathlib import Path
from collections import Counter

DATA = Path(r"c:/Users/depaoll/Downloads/chatbot/labeled_queries.jsonl")

ALLOWED_CATEGORIES = {
    "inflation_dynamics",
    "core_vs_headline",
    "policy_stance",
    "balance_sheet",
    "tltro",
    "wages_labour",
    "digital_euro",
    "payments_innovation",
    "financial_stability",
    "climate",
    "energy_shocks",
    "forward_guidance",
}
ALLOWED_INTENTS = {
    "definition",
    "stance",
    "evolution",
    "mechanism",
    "quantitative",
    "forward_guidance",
    "comparison",
}


def load():
    rows = []
    with DATA.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def validate(rows):
    errors = []
    for i, r in enumerate(rows):
        if not r.get("query") or " " not in r["query"]:
            errors.append((i, "query too short"))
        if not r.get("positives"):
            errors.append((i, "no positives"))
        if not r.get("ideal_summary"):
            errors.append((i, "no ideal_summary"))
        if r.get("ideal_summary") and len(r["ideal_summary"].split()) > 45:
            errors.append((i, "summary too long"))
        if r.get("intent") not in ALLOWED_INTENTS:
            errors.append((i, "invalid intent"))
        if r.get("category") not in ALLOWED_CATEGORIES:
            errors.append((i, "invalid category"))
        for p in r.get("positives", []):
            if "start_char" in p:
                if p["start_char"] >= p["end_char"]:
                    errors.append((i, "invalid span offsets"))
                if "text" not in p or not p["text"].strip():
                    errors.append((i, "span text missing"))
    return errors


def stats(rows):
    cat = Counter(r["category"] for r in rows)
    intent = Counter(r["intent"] for r in rows)
    speakers = Counter()
    years = Counter()
    # Optional: if you add speaker/year in meta later
    return {"categories": cat, "intents": intent, "speakers": speakers, "years": years}


if __name__ == "__main__":
    rows = load()
    errs = validate(rows)
    if errs:
        print("Errors:")
        for idx, msg in errs:
            print(idx, msg)
    else:
        print("No validation errors.")
    print("Stats:", stats(rows))
