# filepath: c:\Users\depaoll\Downloads\chatbot\tools\sentiment.py
import json
import os
import re
from typing import Dict, List
import numpy as np

try:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
except ImportError:  # keep lazy failure identical to your logic
    torch = None  # type: ignore[assignment]
    AutoTokenizer = None  # type: ignore[assignment]
    AutoModelForSequenceClassification = None  # type: ignore[assignment]

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

SENTIMENT_MODEL = os.getenv("SENT_MODEL", "ProsusAI/finbert")
USE_SENTIMENT = os.getenv("USE_SENTIMENT", "1").lower() in ("1", "true", "yes")
MAX_SENTENCES = int(os.getenv("SENT_MAX_SENTS", "28"))
MAX_LEN = 128
BATCH_SIZE = int(os.getenv("SENT_BATCH", "16"))
CACHE_FILE = os.getenv("SENT_CACHE_FILE", "processed_ecb_data/sentiment_cache.json")

# Runtime state
_tokenizer = None
_model = None

# Cache
try:
    with open(CACHE_FILE, "r", encoding="utf-8") as f:
        _CACHE = json.load(f)
except Exception:
    _CACHE = {}

# Sentence splitter
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")

# ---------------------------------------------------------------------
# Domain lexicons (minimal starter sets – extend over time)
# ---------------------------------------------------------------------

HAWKISH_WORDS = {
    "tightening",
    "restrictive",
    "normalisation",
    "normalization",
    "inflationary",
    "elevated inflation",
    "withdrawal",
    "rate hikes",
    "higher rates",
    "price pressures",
    "persistent inflation",
}

DOVISH_WORDS = {
    "accommodative",
    "supportive",
    "stimulus",
    "easing",
    "lower rates",
    "growth support",
    "liquidity",
    "flexibility",
    "favourable financing",
    "favorable financing",
}

OPTIMISTIC_WORDS = {
    "resilient",
    "improving",
    "strong",
    "robust",
    "progress",
    "confidence",
    "solid",
    "strengthening",
}

CAUTIOUS_WORDS = {
    "uncertainty",
    "risk",
    "fragile",
    "vulnerable",
    "volatile",
    "headwinds",
    "caution",
    "challenging",
}

SUPPORTIVE_WORDS = {
    "commitment",
    "support",
    "ensure",
    "protect",
    "promote",
    "advance",
    "facilitate",
    "collaboration",
}

CRITICAL_WORDS = {
    "concern",
    "challenge",
    "issue",
    "problem",
    "shortcoming",
    "criticisms",
    "limitations",
    "risk",
}

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


def _save_cache() -> None:
    """Persist the in-memory cache to disk; ignore failures."""
    try:
        with open(CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(_CACHE, f, ensure_ascii=False)
    except Exception:
        pass


def _key(texts: List[str]) -> str:
    """Stable-ish cache key from the head of each text."""
    head = "|".join(t[:80] for t in texts)
    return str(abs(hash(head)))


def _load_model() -> None:
    """Lazy-load tokenizer and model if available and enabled."""
    global _tokenizer, _model

    if _tokenizer is not None or not USE_SENTIMENT:
        return

    if torch is None or AutoTokenizer is None or AutoModelForSequenceClassification is None:
        _tokenizer = None
        _model = None
        return

    try:
        local_only = False
        # if SENTIMENT_MODEL is a local folder, force local_files_only to avoid downloads
        if os.path.isdir(SENTIMENT_MODEL):
            local_only = True
        _tokenizer = AutoTokenizer.from_pretrained(SENTIMENT_MODEL, local_files_only=local_only)
        _model = AutoModelForSequenceClassification.from_pretrained(
            SENTIMENT_MODEL, local_files_only=local_only
        )
        _model.eval()
    except Exception:
        _tokenizer = None
        _model = None


def _prepare_sentences(text: str) -> List[str]:
    """Split into sentences and keep moderately sized ones."""
    sents = [s.strip() for s in _SENT_SPLIT.split(text) if 20 <= len(s.strip()) <= 300]
    return sents[:MAX_SENTENCES]


def _infer_batch(batch: List[str]) -> np.ndarray:
    """Run a batch through the classifier and return summed class probs."""
    if not _tokenizer or not _model:
        return np.zeros((3,), dtype="float32")

    with torch.no_grad():  # type: ignore[union-attr]
        tok = _tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=MAX_LEN,
            return_tensors="pt",
        )
        out = _model(**tok)
        probs = torch.softmax(out.logits, dim=-1).cpu().numpy()
    return probs.sum(axis=0)


# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------


def analyze_texts(raw_texts: List[str]) -> Dict[str, float]:
    """
    Analyze sentiment distribution over a list of texts.
    Returns dict with 'positive', 'negative', 'neutral' in [0,1].
    """
    if not USE_SENTIMENT:
        return {"positive": 0.0, "negative": 0.0, "neutral": 1.0}

    _load_model()
    if _tokenizer is None:
        return {"positive": 0.0, "negative": 0.0, "neutral": 1.0}

    cache_k = _key(raw_texts)
    if cache_k in _CACHE:
        return _CACHE[cache_k]

    agg = np.zeros((3,), dtype="float32")
    sentences: List[str] = []

    for t in raw_texts:
        sentences.extend(_prepare_sentences(t))

    if not sentences:
        dist = {"positive": 0.0, "negative": 0.0, "neutral": 1.0}
        _CACHE[cache_k] = dist
        _save_cache()
        return dist

    batch: List[str] = []
    for s in sentences:
        batch.append(s)
        if len(batch) == BATCH_SIZE:
            agg += _infer_batch(batch)
            batch = []
    if batch:
        agg += _infer_batch(batch)

    total = float(agg.sum()) + 1e-9
    dist = {
        "positive": float(agg[0] / total),
        "negative": float(agg[1] / total),
        "neutral": float(agg[2] / total),
    }

    _CACHE[cache_k] = dist
    _save_cache()
    return dist


def _count_lexicon(words: set, text: str) -> int:
    """Count total occurrences of any lexicon item within text (case-insensitive)."""
    lc = text.lower()
    cnt = 0
    for w in words:
        if w in lc:
            cnt += lc.count(w)
    return cnt

def sentiment_components(raw_texts: List[str]) -> tuple[str, Dict[str, float]]:
    """
    Return (tone_text, facets_dict).
    tone_text: human-readable one-sentence descriptor (no numeric scores).
    facets_dict: the normalized facet scores + raw_counts as in facet_scores().
    """
    dist = analyze_texts(raw_texts)
    facets = facet_scores(raw_texts)
    tone = tone_descriptor(dist, facets)
    return tone, facets

def facet_scores(raw_texts: List[str]) -> Dict[str, float]:
    """
    Compute normalized facet scores and raw counts for the domain lexicons.
    Returns keys: hawkish, dovish, optimistic, cautious, supportive, critical, raw_counts.
    """
    joined = " ".join(raw_texts)

    hawk = _count_lexicon(HAWKISH_WORDS, joined)
    dove = _count_lexicon(DOVISH_WORDS, joined)
    optm = _count_lexicon(OPTIMISTIC_WORDS, joined)
    caut = _count_lexicon(CAUTIOUS_WORDS, joined)
    supp = _count_lexicon(SUPPORTIVE_WORDS, joined)
    crit = _count_lexicon(CRITICAL_WORDS, joined)

    arr = np.array([hawk, dove, optm, caut, supp, crit], dtype="float32")
    total = arr.sum() + 1e-6
    norm = arr / total if total > 0 else arr

    return {
        "hawkish": float(norm[0]),
        "dovish": float(norm[1]),
        "optimistic": float(norm[2]),
        "cautious": float(norm[3]),
        "supportive": float(norm[4]),
        "critical": float(norm[5]),
        "raw_counts": {
            "hawkish": hawk,
            "dovish": dove,
            "optimistic": optm,
            "cautious": caut,
            "supportive": supp,
            "critical": crit,
        },
    }


def tone_descriptor(dist_sent: Dict[str, float], facets: Dict[str, float]) -> str:
    """
    Compose a compact human-readable tone descriptor from sentiment + facets.
    """
    pos, neg, neu = (
        dist_sent["positive"],
        dist_sent["negative"],
        dist_sent["neutral"],
    )

    # Policy stance (hawkish vs dovish)
    hawk = facets["hawkish"]
    dove = facets["dovish"]
    if hawk - dove > 0.15:
        stance = "hawkish tilt"
    elif dove - hawk > 0.15:
        stance = "dovish tilt"
    else:
        stance = "balanced stance"

    # Risk tone (cautious vs optimistic)
    optm = facets["optimistic"]
    caut = facets["cautious"]
    if caut - optm > 0.15:
        risk = "cautious"
    elif optm - caut > 0.15:
        risk = "moderately optimistic"
    else:
        risk = "neutral"

    # Support vs critical
    supp = facets["supportive"]
    crit = facets["critical"]
    if crit - supp > 0.12:
        attitude = "critical notes"
    elif supp - crit > 0.12:
        attitude = "supportive tone"
    else:
        attitude = "mixed"

    # Sentiment polarity (rarely extreme in speeches)
    if max(pos, neg) < 0.35:
        base = "neutral sentiment"
    elif pos - neg > 0.15:
        base = "slightly positive sentiment"
    elif neg - pos > 0.15:
        base = "slightly negative sentiment"
    else:
        base = "mixed sentiment"

    return f"{stance}; {risk}; {attitude}; {base}"


def sentiment_facets_summary(raw_texts: List[str]) -> str:
    """
    One-liner summary combining tone descriptor and normalized facet values.
    """
    dist = analyze_texts(raw_texts)
    facets = facet_scores(raw_texts)
    tone = tone_descriptor(dist, facets)
    return (
        f"Tone facets: {tone} | "
        f"hawkish={facets['hawkish']:.2f}, dovish={facets['dovish']:.2f}, "
        f"optimistic={facets['optimistic']:.2f}, cautious={facets['cautious']:.2f}, "
        f"supportive={facets['supportive']:.2f}, critical={facets['critical']:.2f}"
    )
