import json, os  # readline gives history on *nix; on Windows may ignore
from pathlib import Path
from data_load import load_sidecar, get_chunk_text, sentence_spans
from retrieval import search
from datetime import datetime

INTENT_HELP = """
definition | stance | evolution | mechanism | quantitative | forward_guidance | comparison
"""
CATEGORY_HELP = """
inflation_dynamics | core_vs_headline | policy_stance | balance_sheet | tltro |
wages_labour | digital_euro | payments_innovation | financial_stability |
climate | energy_shocks | forward_guidance
"""
VALID_INTENTS = {
    "definition",
    "stance",
    "evolution",
    "mechanism",
    "quantitative",
    "forward_guidance",
    "comparison",
}
VALID_CATEGORIES = {
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

OUT = Path(r"c:/Users/depaoll/Downloads/chatbot/labeled_queries.jsonl")
DATA = load_sidecar()

ANSI_HI = "\x1b[33m"
ANSI_RESET = "\x1b[0m"


def highlight(text, query):
    for term in query.lower().split():
        if len(term) < 3:
            continue
        text = text.replace(term, f"{ANSI_HI}{term}{ANSI_RESET}")
        text = text.replace(
            term.capitalize(), f"{ANSI_HI}{term.capitalize()}{ANSI_RESET}"
        )
    return text


def pick(indices_line):
    return [int(x) for x in indices_line.strip().split() if x.isdigit()]


def main():
    print("Interactive labeling. Ctrl+C to exit.")
    encoder = None
    while True:
        try:
            query = input("\nQuery: ").strip()
            if not query:
                continue
            # Intent loop
            while True:
                intent = input("Intent (?=help): ").strip().lower()
                if intent == "?":
                    print(INTENT_HELP)
                    continue
                if not intent:
                    intent = "stance"
                if intent in VALID_INTENTS:
                    break
                print("Invalid intent.")
            # Category loop
            while True:
                category = input("Category (?=help): ").strip()
                if category == "?":
                    print(CATEGORY_HELP)
                    continue
                if not category:
                    category = "inflation_dynamics"
                if category in VALID_CATEGORIES:
                    break
                print("Invalid category.")
            results = search(query, encoder, top_k=15)
            if not results:
                print("No results.")
                continue
            print(f"\nTop {len(results)} candidates:")
            for i, r in enumerate(results):
                chunk_text = get_chunk_text(DATA, r["speech_id"], r["chunk_index"])
                display = chunk_text[:250].replace("\n", " ")
                print(
                    f"[{i}] {r['speech_id']}#{r['chunk_index']} score={r['score']:.4f} :: {display}..."
                )
            pos_idxs = pick(input("Select positive result indices (space sep): "))
            neg_idxs = pick(input("Select hard negative indices (space sep): "))

            positives = []
            for idx in pos_idxs:
                r = results[idx]
                chunk_text = get_chunk_text(DATA, r["speech_id"], r["chunk_index"])
                print(f"\nSentence spans for POS {r['speech_id']}#{r['chunk_index']}:")
                spans = sentence_spans(chunk_text)
                for si, (s_start, s_end, sent) in enumerate(spans):
                    print(f"  ({si}) [{s_start}:{s_end}] {sent}")
                sent_sel = input(
                    "Sentence indices to keep as span-level (blank=whole chunk): "
                ).strip()
                if sent_sel:
                    sel = pick(sent_sel)
                    for s in sel:
                        s_start, s_end, sent = spans[s]
                        positives.append(
                            {
                                "speech_id": r["speech_id"],
                                "chunk_index": r["chunk_index"],
                                "start_char": s_start,
                                "end_char": s_end,
                                "text": sent.strip(),
                            }
                        )
                else:
                    positives.append(
                        {"speech_id": r["speech_id"], "chunk_index": r["chunk_index"]}
                    )

            negatives = []
            for idx in neg_idxs:
                r = results[idx]
                negatives.append(
                    {"speech_id": r["speech_id"], "chunk_index": r["chunk_index"]}
                )

            ideal_summary = input("Ideal grounded summary (<=45 words): ").strip()

            record = {
                "query": query,
                "intent": intent,
                "category": category,
                "year_hint": "",
                "positives": positives,
                "negatives": negatives,
                "ideal_summary": ideal_summary,
                "meta": {
                    "created_at": datetime.utcnow().isoformat(),
                    "retrieval_ranks": {
                        f"{results[i]['speech_id']}#{results[i]['chunk_index']}": i
                        for i in pos_idxs
                    },
                },
            }
            with OUT.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
            print("Saved.")
        except KeyboardInterrupt:
            print("\nExiting.")
            break


if __name__ == "__main__":
    main()
