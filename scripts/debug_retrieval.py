import os
import sys
import textwrap

# ensure project root is importable
SCRIPT_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from core import chatbot


def run(query):
    # force deep_mode and full speech retrieval by calling generate_ecb_speech_response
    # trick: pass speaker_filter "Any" and keep query that matches speaker to enable deep_mode detection,
    # or call retrieve_grouped and then run the map-reduce path manually.
    print("Query:", query)
    # get docs, force full_speech_top large so retrieve_grouped returns entire speeches
    docs = chatbot.retrieve_grouped(
        query,
        top_k_chunks=64,
        neighbors=10,
        top_k_speeches=5,
        full_speech_top=5,
        must_keywords=None,
    )
    if not docs:
        print("No docs returned.")
        return

    # For each doc, show chunk decomposition, run per-chunk map_summarize, then merge
    for d in docs:
        print("\n" + "=" * 80)
        print(
            f"Speech: {d['speech_id']} | title={d['meta'].get('title')} | chars={len(d['text'])}"
        )
        chunks = chatbot._chunk_for_map_reduce(d["text"], chunk_size=1800, overlap=200)
        print(f"Chunks: {len(chunks)} (sizes: {[len(c) for c in chunks]})")
        chunk_summaries = []
        for i, ch in enumerate(chunks):
            print("\n" + "-" * 40)
            print(f"Chunk {i+1} chars={len(ch)} head: {ch[:120]!s}")
            # call map_summarize on the chunk (this uses your map_summarize implementation)
            try:
                s = chatbot.map_summarize(
                    query, {"text": ch, "meta": d["meta"]}, max_chars=len(ch)
                )
            except Exception as e:
                print("map_summarize exception, falling back to local extractive:", e)
                s = chatbot._local_semantic_summary(ch, query, max_sentences=4)
            chunk_summaries.append(s)
            print("\nChunk summary:")
            print(textwrap.fill(s[:1500], width=100))
        print("\n" + "=" * 40)
        print("Merging chunk summaries with reduce_merge...")
        merged = chatbot.reduce_merge(query, chunk_summaries, deep_mode=True)
        print("\nMerged summary:")
        print(textwrap.fill(merged[:3000], width=100))
        print("\nSOURCES:", f"{d['meta'].get('speaker')} ({d['meta'].get('date')})")
        print("\n" + "=" * 80)


if __name__ == "__main__":
    q = (
        " ".join(sys.argv[1:])
        if len(sys.argv) > 1
        else "What did Piero Cipollone say about digital euro in 2025?"
    )
    run(q)
