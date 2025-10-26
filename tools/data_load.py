import json, re
from pathlib import Path
from typing import List, Dict, Any

SIDE_CAR = Path(
    r"c:/Users/depaoll/Downloads/chatbot/processed_ecb_data/speech_sidecar.json"
)

_sentence_splitter = re.compile(r"(?<!\b[A-Z])[.?!](?:\s+|$)")


def load_sidecar() -> Dict[str, List[Dict[str, Any]]]:
    with SIDE_CAR.open(encoding="utf-8") as f:
        return json.load(f)


def get_chunk_text(data, speech_id, chunk_index):
    for c in data[speech_id]:
        if c["chunk_index"] == chunk_index:
            return c["text"]
    raise KeyError(f"chunk not found {speech_id} {chunk_index}")


def sentence_spans(text: str):
    spans = []
    start = 0
    for match in re.finditer(r"[^.?!]*[.?!]", text, re.MULTILINE):
        s = match.group(0)
        s_start = match.start()
        s_end = match.end()
        sent = text[s_start:s_end].strip()
        if sent:
            spans.append((s_start, s_end, sent))
    # Fallback if no punctuation-based split
    if not spans:
        spans.append((0, len(text), text))
    return spans
