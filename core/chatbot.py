import os
import json
import time
import re
import pickle
from collections import defaultdict
import logging
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
from openai import OpenAI
import numpy as np
import faiss
import datetime
from sentence_transformers import SentenceTransformer
from rapidfuzz import process as rf_process
from dotenv import load_dotenv
import hashlib
from pathlib import Path
from tools.sentiment import sentiment_components, USE_SENTIMENT, sentiment_facets_summary
from core.prompts import SUMMARY_PROMPT


LLM_CACHE_PATH = Path("processed_ecb_data/llm_summary_cache.json")
LLM_DAILY_LIMIT = int(os.getenv("LLM_DAILY_LIMIT", "150"))  # max LLM calls/day
LLM_TIMEOUT = int(os.getenv("LLM_TIMEOUT", "40"))
LLM_ENABLED = os.getenv("USE_LLM", "1").lower() in ("1","true","yes")
# Paths & config
ROOT = Path(__file__).resolve().parent.parent
LOCAL_MODEL_PATH = os.getenv("LOCAL_MODEL_PATH", str(ROOT / "all-MiniLM-L6-v2"))
FAISS_INDEX_PATH = "processed_ecb_data/ecb_token_chunks.faiss"
FAISS_META_PATH = "processed_ecb_data/faiss_meta_token.json"
SPEECH_SIDECAR_PATH = "processed_ecb_data/speech_sidecar_token_chunks.json"
CHUNKS_PLK = "processed_ecb_data/ecb_speeches_chunks.plk"

# BATCH_EMBED_SIZE = int(os.getenv("EMBED_BATCH_SIZE", "64"))
HYBRID_ALPHA = 0.75  # semantic weight, keyword overlap weight = 1-alpha

LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
USE_LLM = os.getenv("USE_LLM", "1").lower() in ("1","true","yes")
MAX_SPEECHES_FOR_LLM = int(os.getenv("MAX_SPEECHES_FOR_LLM", "2"))  # limit merged speeches
PARTIAL_MAX_SENTENCES = int(os.getenv("PARTIAL_MAX_SENTENCES", "6"))
FINAL_TARGET_WORDS = "350"  # guidance only
LLM_CACHE_FILE = "processed_ecb_data/llm_final_cache.json"

# load cache
try:
    with LLM_CACHE_PATH.open("r", encoding="utf-8") as f:
        LLM_CACHE = json.load(f)
except Exception:
    LLM_CACHE = {}

def _persist_llm_cache():
    try:
        with LLM_CACHE_PATH.open("w", encoding="utf-8") as f:
            json.dump(LLM_CACHE, f, ensure_ascii=False)
    except Exception:
        pass

# simple daily counter
LLM_COUNTER_PATH = Path("processed_ecb_data/llm_usage_counter.json")
try:
    with LLM_COUNTER_PATH.open("r", encoding="utf-8") as f:
        _cdata = json.load(f)
    if _cdata.get("date") != time.strftime("%Y-%m-%d"):
        LLM_USED_TODAY = 0
    else:
        LLM_USED_TODAY = _cdata.get("count", 0)
except Exception:
    LLM_USED_TODAY = 0

def _inc_llm_used():
    global LLM_USED_TODAY
    LLM_USED_TODAY += 1
    with LLM_COUNTER_PATH.open("w", encoding="utf-8") as f:
        json.dump({"date": time.strftime("%Y-%m-%d"), "count": LLM_USED_TODAY}, f)

def _cache_key(prefix: str, text: str, question: str):
    h = hashlib.sha1((prefix + "|" + text[:2000] + "|" + question).encode("utf-8")).hexdigest()
    return h

def llm_refine(question: str, base_summary: str, meta: dict) -> str:
    if not LLM_ENABLED or LLM_USED_TODAY >= LLM_DAILY_LIMIT:
        return base_summary
    key = _cache_key("refine", base_summary, question)
    if key in LLM_CACHE:
        return LLM_CACHE[key]

    prompt = (
        f"Question: {question}\n"
        f"Speaker: {meta.get('speaker','')}\nDate: {meta.get('date','')}\nTitle: {meta.get('title','')}\n"
        "Draft summary (may be rough):\n"
        f"{base_summary}\n\nRefine this into ONE coherent factual paragraph (80-140 words), "
        "use ONLY info present, no hallucinations, keep neutral tone, mention speaker/date once."
    )
    try:
        resp = client.chat.completions.create(
            model=os.getenv("LLM_MODEL","gpt-3.5-turbo"),
            temperature=0.2,
            max_tokens=380,
            messages=[
                {"role":"system","content":"You refine summaries. No invented facts."},
                {"role":"user","content":prompt}
            ],
            timeout=LLM_TIMEOUT
        )
        out = resp.choices[0].message.content.strip()
        if out:
            LLM_CACHE[key] = out
            _persist_llm_cache()
            _inc_llm_used()
            return out
        return base_summary
    except Exception as e:
        logger.warning("llm_refine failed: %s", e)
        return base_summary

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

try:
    with open(LLM_CACHE_FILE, "r", encoding="utf-8") as _cf:
        _FINAL_CACHE = json.load(_cf)
except Exception:
    _FINAL_CACHE = {}

def _final_cache_key(question: str, audience: str, speech_ids: list[str]) -> str:
    base = question.strip().lower()
    sig = "|".join(sorted(speech_ids))
    return hashlib.sha1(f"{audience}|{base}|{sig}".encode("utf-8")).hexdigest()

def _save_final_cache():
    try:
        with open(LLM_CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(_FINAL_CACHE, f, ensure_ascii=False)
    except Exception:
        pass

def _audience_instructions(audience_level: str) -> str:
    if audience_level.lower().startswith("general"):
        return ("Audience: general public. Use very plain language, short clear sentences (<=20 words), "
            "avoid technical jargon. If you must use a technical term, define it in one short parenthesis. "
            "Prefer simple words (e.g. 'price rise' instead of 'inflationary'), give one short everyday example "
            "if helpful, and keep the tone friendly and explanatory.")
    return ("Audience: professional / analyst. Preserve policy nuance, concise analytical tone, "
            "assume familiarity with basic monetary terms, be specific but very clear.")

def _prepare_local_partials(docs: List[Dict], question: str, max_speeches: int) -> Tuple[List[str], List[Dict]]:
    """Create short local extractive summaries (no LLM) for top speeches."""
    selected_docs = docs[:max_speeches]
    partials = []
    segments = []
    for d in selected_docs:
        seg = {
            "speech_id": d["meta"].get("date","")+"_"+d["meta"].get("title",""),
            "speaker": d["meta"].get("speaker"),
            "date": d["meta"].get("date"),
            "title": d["meta"].get("title"),
        }
        segments.append(seg)
        # local extract (trim)
        extr = _local_semantic_summary(d["text"], question, max_sentences=PARTIAL_MAX_SENTENCES)
        # truncate overly long extract
        if len(extr) > 1200:
            extr = extr[:1200]
        partials.append(extr)
    return partials, segments

def _llm_single_merge(
    question: str,
    audience_level: str,
    partials: List[str],
    segments: List[Dict],
    tone_line: str = ""
) -> str:
    """
    One LLM call merging all partials into a single coherent paragraph.
    IMPORTANT: does NOT inject tone/facet text into the prose. The caller is
    responsible for appending any tone dropdown separately.
    """
    if not partials:
        return ""

    # ---- cache key (stable across same speeches) ----
    speech_ids = [s.get("speech_id", f"{s.get('date','')}_{s.get('title','')}") for s in segments]
    ck = _final_cache_key(question, audience_level, speech_ids)
    if ck in _FINAL_CACHE:
        return _FINAL_CACHE[ck]

    # ---- style / inputs ----
    style = _audience_instructions(audience_level)
    sources_line = "; ".join(f"{s.get('speaker','')} {s.get('date','')}" for s in segments)

    joined_partials = "\n---\n".join(partials)
    if len(joined_partials) > 8000:
        joined_partials = joined_partials[:8000]

    # word target guidance
    try:
        target_words = int(FINAL_TARGET_WORDS)
    except Exception:
        target_words = 300

    # Optional short background blurbs (safe, generic definitions)
    background_phrases = _educational_blurb(_extract_keywords(question), max_items=2)
    bg_instruction = (
        f"\n\nBackground phrases (optional, only if strictly useful): {background_phrases}\n"
        "If you use them, append 1–2 short sentences at the very end labeled 'Background:'. "
        "Do NOT invent facts beyond these phrases."
    ) if background_phrases else ""

    # ---- prompt (NO tone/sentiment mention) ----
    prompt = (
        f"{style}\n"
        f"Question: {question}\n"
        f"Sources (speaker date): {sources_line}\n\n"
        "Extracted passages (already filtered; do not quote bibliographies):\n"
        f"{joined_partials}\n"
        f"{bg_instruction}\n\n"
        f"Task: Produce ONE coherent factual answer of about {target_words} words. "
        "Use ONLY the extracted passages and the provided background phrases (if any). "
        "Do NOT invent numbers, quotes, or names not present in the passages. "
        "Mention each speaker/date at most once. Merge overlapping points, remove duplication and greetings. "
        "Avoid evaluative language about 'tone' or 'sentiment'; do not write anything like 'balanced', 'hawkish', "
        "'dovish', 'optimistic', 'cautious', or facet scores. "
        "If the material only partially answers the question, state that limitation briefly near the start. "
        "Return only the answer paragraph and, if used, a single short 'Background:' line at the end."
    )

    # ---- LLM call ----
    max_tok = int(os.getenv("LLM_MAX_TOKENS", "1200"))
    try:
        resp = client.chat.completions.create(
            model=os.getenv("LLM_MODEL", LLM_MODEL),
            temperature=float(os.getenv("LLM_TEMPERATURE", "0.25")),
            max_tokens=max_tok,
            messages=[
                {"role": "system", "content": "You are a concise, faithful ECB speech summarizer. Do not hallucinate."},
                {"role": "user", "content": prompt},
            ],
            timeout=max(LLM_TIMEOUT, 90),
        )
        out = (resp.choices[0].message.content or "").strip()
        # compact whitespace
        out = re.sub(r"\s+", " ", out)
        if out:
            _FINAL_CACHE[ck] = out
            _save_final_cache()
            try:
                _inc_llm_used()
            except Exception:
                pass
            return out
    except Exception as e:
        logger.warning("FINAL LLM single-merge failed: %s", e)

    # ---- fallback local merge ----
    return _merge_partials(question, partials, max_sentences=12)


def _strip_bibliography(text: str) -> str:
    """
    Remove obvious bibliographic / reference lines and long lists of citations that
    confuse the summarizer. Keeps prose paragraphs.
    """
    if not text:
        return text
    lines = []
    for line in text.splitlines():
        s = line.strip()
        # drop short reference-only lines containing year tokens or "See " / "ECB (" patterns
        if re.search(r"\b(19|20)\d{2}\b", s) and len(s) < 300:
            continue
        if (
            s.startswith("See ")
            or s.startswith("ECB (")
            or s.startswith("Reference")
            or "Position paper" in s
        ):
            continue
        # drop lines that look like a bibliography entry (many commas + years)
        if len(re.findall(r"[,;]", s)) >= 2 and re.search(r"\b(19|20)\d{2}\b", s):
            continue
        lines.append(line)
    cleaned = "\n".join(lines).strip()
    return cleaned if cleaned else text

def generate_conversation_title(user_input: str) -> str:
    kw = user_input.lower()
    if "inflation" in kw:
        return "Inflation"
    if "rate" in kw or "interest" in kw:
        return "Interest Rates"
    if "qe" in kw or "app" in kw or "pepp" in kw:
        return "Asset Purchases"
    if "lagarde" in kw:
        return "Lagarde"
    if "draghi" in kw:
        return "Draghi"
    if "digital euro" in kw:
        return "digital euro"
    return kw[0]


def _load_model():
    global _EMBED_MODEL
    if _EMBED_MODEL is None:
        _EMBED_MODEL = SentenceTransformer(LOCAL_MODEL_PATH)
    return _EMBED_MODEL


def embed_query(text: str) -> np.ndarray:
    m = _load_model()
    v = m.encode([text], convert_to_numpy=True).astype("float32")[0]
    v /= np.linalg.norm(v) + 1e-12
    return v


# Simple helpers
_STOP = set(
    "the a an and or of to for with without in on at by from as about into over after before under above during between against among across more most much many few some any each either neither both per than not no".split()
)


def extract_date_tokens(q: str) -> Dict[str, str]:
    m = re.search(r"\b(20\d{2}-\d{2}-\d{2})\b", q)
    if m:
        return {"date_exact": m.group(1)}
    m2 = re.search(r"\b(20\d{2})-(\d{2})\b", q)
    if m2:
        return {"year": m2.group(1), "month": m2.group(2)}
    m3 = re.search(r"\b(20\d{2})\b", q)
    if m3:
        return {"year": m3.group(1)}
    return {}


# Build speaker list for detection
SPEAKER_LIST = sorted(
    {
        (s.get("speaker") or "").strip()
        for v in SPEECH_SIDECAR.values()
        for s in v
        if s.get("speaker")
    }
)
SPEAKER_LIST = [s for s in SPEAKER_LIST if s]
SPEAKER_LOWER = [s.lower() for s in SPEAKER_LIST]

def _tone_dropdown_html(raw_texts: List[str]) -> str:
    """
    Build a collapsible HTML/markdown block with tone descriptor + facet values.
    Shown *outside* of the answer paragraph (so no numbers pollute the main text).
    """
    try:
        tone_text, facets = sentiment_components(raw_texts)
    except Exception:
        return ""  # safe fallback

    # Pretty-print facet values (2 decimals), hide raw_counts
    rows = []
    for k in ["hawkish","dovish","optimistic","cautious","supportive","critical"]:
        v = facets.get(k, 0.0)
        rows.append(f"<tr><td style='padding:2px 8px'>{k.title()}</td><td>{v:.2f}</td></tr>")
    table = "<table>" + "".join(rows) + "</table>"

    # <details> works in Streamlit's unsafe_allow_html context (we already use HTML containers)
    return (
        "<details style='margin-top:8px'><summary><b>🔎 Tone analysis</b></summary>"
        f"<p style='margin:8px 0 4px'><i>{tone_text}</i></p>{table}"
        "</details>"
    )

def _extract_year_range(q: str) -> List[str]:
    """
    Extracts two endpoint years from patterns like 'from 2019 to 2021' or '2019–2021'.
    Returns [start, end] if found, else [].
    """
    m = re.search(r"\b(20\d{2})\s*(?:\-|–|to|and)\s*(20\d{2})\b", q)
    if m:
        y1, y2 = m.group(1), m.group(2)
        if y1 != y2:
            return [y1, y2]
    m2 = re.search(r"\bfrom\s+(20\d{2})\s+to\s+(20\d{2})\b", q.lower())
    if m2:
        y1, y2 = m2.group(1), m2.group(2)
        if y1 != y2:
            return [y1, y2]
    return []

def detect_compare_topics(q: str) -> List[str]:
    """
    Heuristic: if the query contains 2+ distinct high-signal keywords (from _extract_keywords),
    treat it as a topic comparison request (e.g., 'inflation vs growth' or 'digital euro and inflation').
    """
    kws = _extract_keywords(q)
    # keep distinct, non-overlapping tokens
    unique = []
    for k in kws:
        if all(k.lower() not in u.lower() and u.lower() not in k.lower() for u in unique):
            unique.append(k)
    return unique[:3]  # cap

def detect_speakers(text: str) -> List[str]:
    """
    Heuristically detect ECB speakers mentioned in a query string.

    Looks for known ECB board members / governors mentioned by first or last name
    and returns a list of distinct matches in canonical 'First Last' form.
    This helps the comparison logic decide if the user is asking to compare speakers.
    """

    # ✅ 1. Basic normalization
    t = text.lower()

    # ✅ 2. Common ECB speaker list (extend as needed)
    known_speakers = {
        "christine lagarde": ["lagarde", "christine lagarde"],
        "isabel schnabel": ["schnabel", "isabel schnabel"],
        "piero cipollone": ["cipollone", "piero cipollone"],
        "fabio panetta": ["panetta", "fabio panetta"],
        "luis de guindos": ["guindos", "luis de guindos"],
        "mario draghi": ["draghi", "mario draghi"],
        "willem duisenberg": ["duisenberg", "willem duisenberg"],
        "jean-claude trichet": ["trichet", "jean-claude trichet"],
        "mario centeno": ["centeno", "mario centeno"],
        "philip lane": ["lane", "philip lane"],
        "olivier garnier": ["garnier", "olivier garnier"],
        "yves mersch": ["mersch", "yves mersch"],
        "bostjan jazbec": ["jazbec", "bostjan jazbec"],
        "mario monti": ["monti", "mario monti"],  # optional external references
    }

    found = []
    for full_name, patterns in known_speakers.items():
        for p in patterns:
            if re.search(rf"\b{re.escape(p)}\b", t):
                found.append(full_name)
                break  # avoid duplicates if both first and last matched

    # ✅ 3. Deduplicate while preserving order
    seen = set()
    unique = []
    for s in found:
        if s not in seen:
            unique.append(s)
            seen.add(s)

    return unique


def detect_speaker_from_query(q: str, score_cutoff: int = 75) -> Optional[str]:
    """
    Try to detect a speaker mentioned in the query.

    Behavior:
    - If a known speaker name occurs verbatim (case-insensitive) in the query -> return that name.
    - If the query contains an explicit name-like phrase (e.g. "Mickey Mouse")
      but it does NOT match any known speaker -> return special token "__UNKNOWN::<name>"
      so caller can surface a friendly "no such speaker" message.
    - Otherwise, use fuzzy matching (rapidfuzz) and return candidate if score >= cutoff.
    - Return None if no reasonable detection.
    """
    ql = q.lower()

    # 1) exact substring matches (preferred)
    for s in SPEAKER_LIST:
        if s.lower() in ql:
            return s

    # 2) extract explicit name-like phrases (two or more capitalized words)
    #    if user explicitly typed a name (e.g. "Mickey Mouse"), check exact membership
    name_like = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b", q)
    if name_like:
        # check each found phrase against known speakers (case-insensitive)
        for nm in name_like:
            for s in SPEAKER_LIST:
                if nm.strip().lower() == s.lower():
                    return s
        # user explicitly mentioned a name but none match known speakers -> signal unknown
        # return the first name-like phrase for a clear message
        return f"__UNKNOWN::{name_like[0].strip()}"

    # 3) fuzzy match fallback
    if SPEAKER_LIST:
        match = rf_process.extractOne(q, SPEAKER_LIST)
        if match and len(match) >= 2:
            candidate, score = match[0], match[1]
            if score >= score_cutoff:
                return candidate
    return None



def _extract_years(q: str) -> List[str]:
    return list(dict.fromkeys(re.findall(r"\b(20\d{2})\b", q)))

def _retrieve_topic_for_speaker(
    query: str,
    speaker: str,
    year: Optional[str],
    topic_kws: List[str],
    top_k_chunks: int = 48,
) -> Optional[Dict]:
    """
    More permissive retrieval: attempt full speech first; relax keywords if needed.
    """
    docs = retrieve_grouped(
        query,
        top_k_chunks=top_k_chunks,
        neighbors=0,
        top_k_speeches=1,
        speaker=speaker,
        year=year,
        date_exact=None,
        must_keywords=topic_kws,
        full_speech_top=1,  # take entire speech to avoid keyword guard pruning
    )
    if not docs and year:
        # retry without year restriction
        docs = retrieve_grouped(
            query,
            top_k_chunks=top_k_chunks,
            neighbors=0,
            top_k_speeches=1,
            speaker=speaker,
            must_keywords=topic_kws,
            full_speech_top=1,
        )
    if not docs and topic_kws:
        # final retry without keywords
        docs = retrieve_grouped(
            query,
            top_k_chunks=top_k_chunks,
            neighbors=0,
            top_k_speeches=1,
            speaker=speaker,
            must_keywords=[],
            full_speech_top=1,
        )
    return docs[0] if docs else None

def _build_comparison_paragraph(
    question: str,
    audience_level: str,
    units: List[Tuple[str, str, str, str]],  # (label, speaker, date, summary)
    tone_line: str = "",
) -> str:
    """
    Local fallback comparison paragraph (no LLM) when LLM disabled or fails.
    """
    # Simple synthesis: lead + per-unit sentences.
    lead = f"Comparison for: {question}"
    parts = []
    for label, spk, dt, summ in units:
        parts.append(f"{spk} ({dt}) {label}: {summ}")
    body = " ".join(parts)
    para = f"{lead}. {body}"
    if tone_line:
        para += f" Tone facets: {tone_line}."
    return re.sub(r"\s+", " ", para).strip()
 
def detect_compare_speakers(q: str) -> List[str]:
    """
    Return list of >=2 distinct speaker names found in query (exact or fuzzy).
    Handles possessive apostrophes.
    """
    q_norm = q.lower().replace("’", "'")
    found = []

    for s in SPEAKER_LIST:
        sn = s.lower()
        if sn in q_norm or sn.replace(" ", "") in q_norm:
            found.append(s)

    # Fuzzy backup for tokens separated by punctuation
    if len(found) < 2:
        tokens = re.findall(r"[A-Za-z][A-Za-z]+", q_norm)
        joined = " ".join(tokens)
        for s in SPEAKER_LIST:
            if s.lower() in joined and s not in found:
                found.append(s)

    # Deduplicate
    out = []
    for s in found:
        if s not in out:
            out.append(s)

    return out


def _retrieve_for_speaker(question: str, speaker: str, year_hint: Optional[str]) -> Optional[Dict]:
    """
    Get one grouped document for a specific speaker (optionally filtered by year).
    """
    docs = retrieve_grouped(
        question,
        top_k_chunks=24,
        neighbors=3,
        top_k_speeches=3,
        speaker=speaker,
        year=year_hint,
        date_exact=None,
        must_keywords=_extract_keywords(question),
        full_speech_top=0,
    )
    return docs[0] if docs else None

def _llm_compare_merge(
    question: str,
    audience_level: str,
    per_summaries: List[Tuple[str, str, Dict]],
    tone_line: str = "",
) -> str:
    """
    LLM contrast between multiple speakers.
    per_summaries: list of (speaker_name, summary_text, meta)
    """
    if not per_summaries or len(per_summaries) < 2:
        return "Insufficient distinct speaker material to compare."

    style = _audience_instructions(audience_level)
    blocks = []
    sources = []

    for spk, summ, meta in per_summaries:
        title = meta.get("title", "")
        date = meta.get("date", "")
        sources.append(f"{spk} {date}")
        blocks.append(
            f"Speaker: {spk}\nDate: {date}\nTitle: {title}\nExtractive sentences:\n{summ}"
        )

    joined = "\n\n----\n\n".join(blocks)

    prompt = (
        f"{style}\nQuestion: {question}\nSources: {', '.join(sources)}\n"
        f"{('Tone facets: ' + tone_line) if tone_line else ''}\n\n"
        f"{joined}\n\n"
        "Task: Produce ONE paragraph (200-250 words) comparing and contrasting the speakers' positions "
        "STRICTLY using only the provided sentences. Highlight clear differences and any explicit common ground. "
        "Do NOT guess intentions or add external context; if a difference is not evidenced, omit it. "
        "If evidence for one speaker is sparse, acknowledge the limitation. Mention each speaker once. "
        "Return only the paragraph."
    )

    try:
        resp = client.chat.completions.create(
            model=LLM_MODEL,
            temperature=0.25,
            max_tokens=480,
            messages=[
                {"role": "system", "content": "You compare speakers without hallucinations."},
                {"role": "user", "content": prompt},
            ],
            timeout=50,
        )
        out = resp.choices[0].message.content.strip()
        out = re.sub(r"\s+", " ", out)
        return out

    except Exception as e:
        logger.warning("compare merge failed: %s", e)
        # Fallback: stitch summaries with simple labels
        stitched = " ".join(f"{spk}: {summ}" for spk, summ, _ in per_summaries)
        return stitched[:1500]
    


# FAISS search + hybrid re-ranking
def faiss_search(
    query: str, top_k: int = 16, kws: Optional[List[str]] = None
) -> List[Dict]:
    if faiss_index is None:
        return []
    qv = embed_query(query).reshape(1, -1)
    scores, ids = faiss_index.search(qv, top_k)
    hits = []
    for sc, rid in zip(scores[0].tolist(), ids[0].tolist()):
        if rid < 0:
            continue
        meta = FAISS_META[rid]
        # load chunk text for keyword overlap check
        sid = meta.get("speech_id")
        cidx = meta.get("chunk_index")
        chunk_text = ""
        if sid in SPEECH_SIDECAR:
            for c in SPEECH_SIDECAR[sid]:
                if c.get("chunk_index") == cidx:
                    chunk_text = c.get("text", "")
                    break
        overlap = 0.0
        if kws and chunk_text:
            kcount = sum(1 for k in kws if k.lower() in chunk_text.lower())
            overlap = kcount / (len(kws) + 0.0001)
        combined = HYBRID_ALPHA * float(sc) + (1 - HYBRID_ALPHA) * overlap
        hits.append(
            {
                "row_id": int(rid),
                "score": float(sc),
                "combined_score": combined,
                "speech_id": sid,
                "chunk_index": cidx,
                "text": chunk_text,
            }
        )
    hits.sort(key=lambda x: x["combined_score"], reverse=True)
    return hits

def retrieve_grouped(
    query: str,
    top_k_chunks: int = 32,
    neighbors: int = 3,
    top_k_speeches: int = 3,
    speaker: Optional[str] = None,
    year: Optional[str] = None,
    date_exact: Optional[str] = None,
    must_keywords: Optional[List[str]] = None,
    full_speech_top: int = 0,
) -> List[Dict]:
    kws = must_keywords or _extract_keywords(query)
    raw = faiss_search(query, top_k=top_k_chunks, kws=kws)  # 32 speeches
    if not raw:
        return []

    by_sid = defaultdict(list)
    for h in raw:
        sid = h.get("speech_id")
        if sid:
            by_sid[sid].append(h)

    # sort speeches by best combined score
    ordered = sorted(
        by_sid.items(),
        key=lambda kv: max(x["combined_score"] for x in kv[1]),
        reverse=True,
    )

    results = []
    for rank, (sid, hits) in enumerate(ordered):
        all_chunks = SPEECH_SIDECAR.get(sid, [])
        if not all_chunks:
            continue
        meta0 = all_chunks[0]
        spk = (meta0.get("speaker") or "").lower()
        dt = meta0.get("date") or ""
        yr = dt[:4] if len(dt) >= 4 else ""

        if speaker and speaker.lower() not in spk:
            continue
        if year and yr != year:
            continue
        if date_exact and dt != date_exact:
            continue

        if rank < full_speech_top:
            # take entire speech
            kept = all_chunks
        else:
            # selective expansion around hits
            idxs = set(
                int(h["chunk_index"]) for h in hits if h.get("chunk_index") is not None
            )
            for i in list(idxs):
                for r in range(1, neighbors + 1):
                    idxs.add(i - r)
                    idxs.add(i + r)
            kept = [c for c in all_chunks if c.get("chunk_index") in idxs]
            if not kept:
                continue
            kept.sort(key=lambda x: x.get("chunk_index", 0))

        joined = "\n\n".join(k.get("text", "") for k in kept)

        # keyword guard (only if not full speech OR even full speech but keep relevance if user narrowed topic)
        if kws and rank >= full_speech_top:
            if not any(k.lower() in joined.lower() for k in kws):
                # fallback: test full speech once
                full = "\n\n".join(c.get("text", "") for c in all_chunks)
                if not any(k.lower() in full.lower() for k in kws):
                    continue

        results.append(
            {
                "speech_id": sid,
                "meta": {
                    "title": meta0.get("title", ""),
                    "speaker": meta0.get("speaker", ""),
                    "date": meta0.get("date", ""),
                },
                "text": joined,
            }
        )
        if len(results) >= top_k_speeches:
            break

    if not results and (speaker or year or date_exact or must_keywords):
        return retrieve_grouped(
            query,
            top_k_chunks=top_k_chunks,
            neighbors=1,
            top_k_speeches=top_k_speeches,
            full_speech_top=full_speech_top,
        )
    return results


# Local extractive summarizer (returns concise educational paragraph)
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def _clean_for_summary(text: str) -> str:
    if not text:
        return ""
    # remove CHUNK separators and excessive whitespace
    text = text.replace("---CHUNK_SEPARATOR---", " ")
    text = re.sub(r"\s+", " ", text.strip())

    # drop obvious leading titles/labels like "SPEECH ...", "Speech by ..."
    text = re.sub(
        r"(?i)^\s*(SPEECH[:\-\s].{0,200}?)(?=[A-Z][a-z]{2,})", "", text, flags=re.DOTALL
    )
    text = re.sub(r"(?i)^\s*(Speech by [^\n]*\n?)", "", text)
    # also remove common opening salutations that add noise
    text = re.sub(
        r"(?i)^(dear (colleagues|guests|participants)|ladies and gentlemen|good (morning|afternoon|evening))[^\.\n]{0,200}\.?",
        "",
        text,
    )
    return text.strip()


# Local summarizer: MMR selection, NO per-partial background blurb (blurb added once in merge)
def _local_semantic_summary(text: str, question: str, max_sentences: int = 5) -> str:
    text = _clean_for_summary(text)
    sents = [
        s.strip() for s in _SENTENCE_SPLIT.split(text) if 30 <= len(s.strip()) <= 800
    ]
    if not sents:
        return text[:700] + ("..." if len(text) > 700 else "")

    model = _load_model()
    sent_embs = model.encode(sents, convert_to_numpy=True).astype("float32")
    norms = np.linalg.norm(sent_embs, axis=1, keepdims=True) + 1e-12
    sent_embs = sent_embs / norms
    qv = embed_query(question or "summary")

    # pick via MMR for relevance + diversity
    top_n = min(max_sentences, len(sents))
    sel = _mmr_select(sent_embs, qv, top_n, lamb=0.7)
    # preserve original order for coherence
    sel_sorted = sorted(sel)
    picked = [sents[i] for i in sel_sorted]

    # tighten sentences: remove any that are just metadata/headers
    cleaned = []
    for p in picked:
        if re.match(r"^[A-Z][A-Za-z\s]{0,80}—\s*\d{4}-\d{2}-\d{2}", p):
            continue
        if p in cleaned:
            continue
        cleaned.append(p)
    return " ".join(cleaned)


# Merge partials -> structured, educational answer (single background blurb at end)
def _merge_partials(question: str, partials: List[str], max_sentences: int = 9) -> str:
    # partials are expected to be plain summaries (no headers). if not, strip speaker/date lines
    cleaned_partials = []
    for p in partials:
        p_ = re.sub(
            r"^[^\n]{0,200}\d{4}-\d{2}-\d{2}\:?\s*", "", p
        )  # drop leading "Name — YYYY-MM-DD:" if present
        cleaned_partials.append(_clean_for_summary(p_))

    # collect candidate sentences
    sents = []
    for p in cleaned_partials:
        for s in _SENTENCE_SPLIT.split(p):
            s = s.strip()
            if 30 <= len(s) <= 800 and not s.lower().startswith("speech"):
                sents.append(s)

    if not sents:
        return " ".join(cleaned_partials[:2])

    model = _load_model()
    sent_embs = model.encode(sents, convert_to_numpy=True).astype("float32")
    norms = np.linalg.norm(sent_embs, axis=1, keepdims=True) + 1e-12
    sent_embs = sent_embs / norms
    qv = embed_query(question)

    top_n = min(max_sentences, len(sents))
    sel_idxs = _mmr_select(sent_embs, qv, top_n, lamb=0.65)
    sel_idxs_sorted = sorted(sel_idxs)
    selected = [sents[i] for i in sel_idxs_sorted]

    # Build readable answer: lead + 3 takeaways + short details
    lead = selected[0] if selected else ""
    takeaways = selected[0 : min(3, len(selected))]
    details = selected[3 : min(8, len(selected))]

    out = []
    if lead:
        out.append("Here's what I found about your topic: " + lead)
    if takeaways:
        out.append("\nKey takeaways:")
        for t in takeaways:
            out.append("- " + t)
    if details:
        out.append("\nDetails: " + " ".join(details))

    # add a single educational blurb based on detected keywords
    blurb = _educational_blurb(_extract_keywords(question), max_items=2)
    if blurb:
        out.append("\nBackground: " + blurb)

    return "\n\n".join(out)


def generate_ecb_speech_response(
    user_input: str,
    conversation_history: List[Dict],
    audience_level: str,
    documentation_needed: bool = False,
    mode: str = "deep",
    speaker_filter: str = "Any",
    year_filter: str = "Any",
    topic_filter: str = "Any",
) -> Tuple[str, List[Dict], int, str, str, int]:
    """
    Main orchestration for answering a user query over ECB speeches.

    Returns:
        final_answer (str): the answer text (with Sources + tone dropdown appended)
        segments (List[Dict]): metadata for the evidence speeches used
        steps (int): rough step counter for telemetry
        status (str): status label
        audience_level (str): echoed
        depth (int): rough depth score
    """
    question = user_input.strip()
    if not question:
        return "Please ask a question.", [], 1, "empty", audience_level, 1

    # ---- parse cues from query ------------------------------------------------
    date_tokens = extract_date_tokens(question)             # your existing helper
    years_in_text = _extract_years(question)                # existing year extractor
    year_range = _extract_year_range(question)              # NEW: handles "2019-2021", "from 2020 to 2023"
    if year_range and len(year_range) == 2:
        # extend years list with endpoints; keep unique order
        years_in_text = list(dict.fromkeys(years_in_text + year_range))

    # speakers detected in the query (may be 0, 1, 2+)
    compare_speakers = detect_speakers(question) or []

    # topics from query; also a focused detector for topic-comparisons
    kws = _extract_keywords(question)
    topic_compare = detect_compare_topics(question)  # NEW: up to 3 distinct high-signal keywords
    topic_kws = [k for k in kws if k.lower() not in {s.lower() for s in compare_speakers}]

    # optional UI filters
    if speaker_filter and speaker_filter != "Any":
        # if a filter is set, enforce / prepend it
        if speaker_filter not in compare_speakers:
            compare_speakers = [speaker_filter] + compare_speakers
    if year_filter and year_filter != "Any":
        if year_filter not in years_in_text:
            years_in_text = [year_filter] + years_in_text
    if topic_filter and topic_filter != "Any":
        if topic_filter not in topic_compare:
            topic_compare = [topic_filter] + topic_compare
        if topic_filter not in topic_kws:
            topic_kws = [topic_filter] + topic_kws

    # lightweight "compare" cue detection
    compare_cues = re.search(r"\b(compare|vs\.?|versus|contrast|difference|evolv(e|ed|ing)|change(d)?)\b", question, re.I)
    single_multi_year = len(set(years_in_text)) >= 2
    wants_compare = bool(compare_cues or len(compare_speakers) >= 2 or single_multi_year or len(topic_compare) >= 2)

    # --------------------------------------------------------------------------
    # COMPARISON BRANCH
    # --------------------------------------------------------------------------
    if wants_compare:
        steps = 0
        per_summaries: List[Tuple[str, str, Dict]] = []  # [(label, text, meta), ...]
        segments: List[Dict] = []

        # cap axes to keep prompts manageable
        speakers_axis = compare_speakers[:3] if len(compare_speakers) >= 2 else compare_speakers
        years_axis = years_in_text[:3] if len(years_in_text) >= 2 else []
        topics_axis = [t for t in topic_compare if t not in {s.lower() for s in compare_speakers}]
        topics_axis = topics_axis[:2]

        no_speaker_compare = len(speakers_axis) == 0 and (len(years_axis) >= 2 or len(topics_axis) >= 2)

        def _label(meta, sp_override=None, y_override=None, topic=None):
            sp = sp_override or meta.get("speaker", "Unknown")
            dt = y_override or (meta.get("date", "")[:4] or meta.get("date", ""))
            lab = sp
            if y_override:
                lab += f" {y_override}"
            elif meta.get("date"):
                lab += f" {meta.get('date')}"
            if topic:
                lab += f" — {topic}"
            return lab

        def _fetch_for(speaker, year, topic_hint):
            tk = ([topic_hint] + topic_kws) if topic_hint else topic_kws
            return _retrieve_topic_for_speaker(question, speaker, year, tk, top_k_chunks=64)

        # --- A) multiple speakers (optionally crossed with years or topics)
        if len(speakers_axis) >= 2:
            for spk in speakers_axis:
                if years_axis:
                    for y in years_axis:
                        doc = _fetch_for(spk, y, topics_axis[0] if topics_axis else None)
                        if not doc:
                            continue
                        txt = _strip_bibliography(_clean_for_summary(doc["text"]))
                        if len(txt) > 12000:
                            txt = txt[:12000]
                        meta = doc["meta"].copy()
                        meta["date"] = meta.get("date", f"{y}-??-??")
                        per_summaries.append((_label(meta, sp_override=spk, y_override=y, topic=(topics_axis[0] if topics_axis else None)), txt, meta))
                        segments.append(meta)
                        steps += 1
                elif topics_axis:
                    for t in topics_axis:
                        doc = _fetch_for(spk, year_filter if year_filter != "Any" else None, t)
                        if not doc:
                            continue
                        txt = _strip_bibliography(_clean_for_summary(doc["text"]))
                        if len(txt) > 12000:
                            txt = txt[:12000]
                        meta = doc["meta"]
                        per_summaries.append((_label(meta, sp_override=spk, topic=t), txt, meta))
                        segments.append(meta)
                        steps += 1
                else:
                    doc = _fetch_for(spk, year_filter if year_filter != "Any" else None, None)
                    if doc:
                        txt = _strip_bibliography(_clean_for_summary(doc["text"]))
                        if len(txt) > 12000:
                            txt = txt[:12000]
                        meta = doc["meta"]
                        per_summaries.append((_label(meta, sp_override=spk), txt, meta))
                        segments.append(meta)
                        steps += 1

        # --- B) one speaker across multiple years
        elif single_multi_year and len(compare_speakers) == 1:
            spk_name = compare_speakers[0]
            for y in years_axis:
                doc = _fetch_for(spk_name, y, topics_axis[0] if topics_axis else None)
                if not doc:
                    continue
                txt = _strip_bibliography(_clean_for_summary(doc["text"]))
                if len(txt) > 12000:
                    txt = txt[:12000]
                meta = doc["meta"].copy()
                meta["date"] = meta.get("date", f"{y}-??-??")
                per_summaries.append((_label(meta, sp_override=spk_name, y_override=y, topic=(topics_axis[0] if topics_axis else None)), txt, meta))
                segments.append(meta)
                steps += 1

        # --- C) no explicit speaker: compare by year or by topic
        elif no_speaker_compare:
            if len(years_axis) >= 2:
                for y in years_axis:
                    docs = retrieve_grouped(
                        question, top_k_chunks=64, neighbors=0, top_k_speeches=1,
                        speaker=None, year=y, date_exact=None,
                        must_keywords=topic_kws or _extract_keywords(question),
                        full_speech_top=1
                    )
                    if not docs:
                        continue
                    doc = docs[0]
                    txt = _strip_bibliography(_clean_for_summary(doc["text"]))
                    if len(txt) > 12000:
                        txt = txt[:12000]
                    meta = doc["meta"].copy()
                    meta["date"] = meta.get("date", f"{y}-??-??")
                    per_summaries.append((_label(meta, topic=(topics_axis[0] if topics_axis else None)), txt, meta))
                    segments.append(meta)
                    steps += 1
            elif len(topics_axis) >= 2:
                for t in topics_axis:
                    docs = retrieve_grouped(
                        question, top_k_chunks=64, neighbors=0, top_k_speeches=1,
                        speaker=None, year=year_filter if year_filter != "Any" else None, date_exact=None,
                        must_keywords=[t], full_speech_top=1
                    )
                    if not docs:
                        continue
                    doc = docs[0]
                    txt = _strip_bibliography(_clean_for_summary(doc["text"]))
                    if len(txt) > 12000:
                        txt = txt[:12000]
                    meta = doc["meta"]
                    per_summaries.append((_label(meta, topic=t), txt, meta))
                    segments.append(meta)
                    steps += 1

        # Fallback if nothing gathered
        if not per_summaries:
            # try normal deep mode to at least produce a single answer
            answer, segments, _, _, _, _ = generate_ecb_speech_response(
                question, conversation_history, audience_level, documentation_needed, "deep",
                speaker_filter, year_filter, topic_filter
            )
            return answer, segments, steps + 1, "fallback_deep", audience_level, 4

        # Ask LLM to compare (make it a bit longer 180–240 words)
        answer = _llm_compare_merge(
            question,
            audience_level,
            per_summaries,
            length_hint="180-240"  # << ensure your _llm_compare_merge reads this (or adjust inside it)
        )

        # Append sources and tone dropdown
        tone_html = _tone_dropdown_html([txt for _, txt, _ in per_summaries]) if USE_SENTIMENT else ""
        srcs = "; ".join(f"{m.get('speaker')} ({m.get('date')})" for _, _, m in per_summaries)
        final = f"{answer}\n\nSources: {srcs}{('\n\n' + tone_html) if tone_html else ''}"

        return final, segments, steps + 3, "ok_compare", audience_level, 5

    # --------------------------------------------------------------------------
    # SINGLE (DEEP) ANSWER BRANCH
    # --------------------------------------------------------------------------
    # retrieve a handful of best speeches, then merge
    ctx = retrieve_grouped(
        question,
        top_k_chunks=64,
        neighbors=0,
        top_k_speeches=5,
        speaker=None if speaker_filter == "Any" else speaker_filter,
        year=None if year_filter == "Any" else year_filter,
        date_exact=None,
        must_keywords=_extract_keywords(question),
        full_speech_top=3,
    )

    if not ctx:
        return "Sorry, I couldn’t find relevant ECB speeches for that query.", [], 2, "no_results", audience_level, 2

    # Clean and prep partials
    partials: List[str] = []
    segments: List[Dict] = []
    for doc in ctx[:5]:
        txt = _strip_bibliography(_clean_for_summary(doc["text"]))
        meta = doc["meta"]
        segments.append(meta)
        if len(txt) > 14000:
            txt = txt[:14000]
        partials.append(txt)

    tone_html = _tone_dropdown_html(partials) if USE_SENTIMENT else ""

    if USE_LLM:
        final_answer = _llm_single_merge(
            question,
            audience_level,
            partials,
            segments,
            tone_line=""  # keep prose clean; tone is only in dropdown
        )
    else:
        # very rare fallback
        fallback = partials[0][:900]
        final_answer = fallback

    # Append sources and tone
    srcs = "; ".join(f"{s['speaker']} ({s['date']})" for s in segments)
    final_answer = f"{final_answer}\n\nSources: {srcs}{('\n\n' + tone_html) if tone_html else ''}"

    return final_answer, segments, 5, "ok_deep", audience_level, 4


def simple_answer_for_query(
    query: str,
    top_speeches: int = 2,
    top_k_chunks: int = 16,
    use_llm_merge: bool = False,
):
    """
    Simplified flow:
    1) FAISS -> top_k_chunks
    2) group by speech_id -> pick top `top_speeches` speeches by max combined_score
    3) load full speech text from SPEECH_SIDECAR
    4) produce short extractive summary per speech (local)
    5) optional single LLM merge of the short summaries (use_llm_merge)
    Returns: (final_answer:str, selected_segments:List[dict], partials:List[str])
    """
    dtokens = extract_date_tokens(query)
    if dtokens.get("date_exact") or dtokens.get("year"):
        try:
            today = datetime.date.today()
            if dtokens.get("date_exact"):
                dt = datetime.datetime.strptime(dtokens["date_exact"], "%Y-%m-%d").date()
                if dt > today:
                    return (
                        "⚠️ I cannot provide statements or predictions for future dates. "
                        "This tool summarizes existing ECB speeches. Please ask about past speeches or remove the future date.",
                        [],
                        [],
                    )
            if dtokens.get("year"):
                if int(dtokens["year"]) > today.year:
                    return (
                        "⚠️ I cannot provide statements or predictions for future dates. "
                        "This tool summarizes existing ECB speeches. Please ask about past speeches or remove the future year.",
                        [],
                        [],
                    )
        except Exception:
            pass
    # 1) chunk hits
    hits = faiss_search(query, top_k=top_k_chunks, kws=_extract_keywords(query))
    if not hits:
        return "No relevant chunks found.", [], []

    # 2) group and rank speeches by best hit score
    scores_by_sid = {}
    for h in hits:
        sid = h.get("speech_id")
        if not sid:
            continue
        scores_by_sid[sid] = max(
            scores_by_sid.get(sid, 0.0), h.get("combined_score", 0.0)
        )

    ranked_sids = sorted(
        scores_by_sid.keys(), key=lambda s: scores_by_sid[s], reverse=True
    )[:top_speeches]

    partials = []
    segments = []
    for sid in ranked_sids:
        all_chunks = SPEECH_SIDECAR.get(sid, [])
        if not all_chunks:
            continue
        # meta from first chunk
        meta = {
            "title": all_chunks[0].get("title", ""),
            "speaker": all_chunks[0].get("speaker", ""),
            "date": all_chunks[0].get("date", ""),
        }
        full_text = "\n\n".join(c.get("text", "") for c in all_chunks)
        segments.append(
            {
                "speech_id": sid,
                "speaker": meta["speaker"],
                "date": meta["date"],
                "title": meta["title"],
            }
        )
        # short extractive summary (fast, deterministic)
        short = _local_semantic_summary(full_text, query, max_sentences=5)
        partials.append(short)

    if not partials:
        return "No speeches available for selected chunks.", segments, partials

    # 5) merge: prefer local merge _merge_partials; optionally run reduce_merge (LLM)
    if use_llm_merge:
        try:
            merged = reduce_merge(query, partials, deep_mode=False)
        except Exception:
            merged = _merge_partials(query, partials, max_sentences=8)
    else:
        merged = _merge_partials(query, partials, max_sentences=8)
    # attach human-readable tone (avoid showing raw numeric scores here)
    tone_line = ""
    if USE_SENTIMENT and partials:
        try:
            tone_line, _ = sentiment_components(partials)
        except Exception:
            tone_line = ""

    # add Sources line
    srcs = "; ".join(f"{s['speaker']} ({s['date']})" for s in segments)
    final = merged
    if tone_line:
        final += f"\n\nTone: {tone_line}"
    final += f"\n\nSources: {srcs}"
    return final, segments, partials


def map_summarize(question: str, speech_doc: Dict, max_chars: int = 12000) -> str:
    raw_ctx = speech_doc.get("text", "")[:max_chars]
    ctx = _strip_bibliography(_clean_for_summary(raw_ctx))
    meta = speech_doc["meta"]
    prompt = (
        f"Context: {ctx}\n\n"
        f"Task: In 2-4 short bullet points, state the main findings, positions, or proposals that the speaker "
        f"{meta.get('speaker','')} expresses in this text about the question: {question}. "
        "Do NOT repeat bibliographic references, dates, or lists of citations. Be factual and avoid speculation."
    )
    logger.info(
        "map_summarize: calling LLM for title=%s chars_ctx=%d",
        meta.get("title", "")[:80],
        len(ctx),
    )
    try:
        resp = client.chat.completions.create(
            model="gpt-3.5-turbo",
            temperature=0.0,
            max_tokens=1200,
            messages=[
                {
                    "role": "system",
                    "content": "Be precise, non-speculative, and neutral, avoid bad words and offensive language. Combine and de-duplicate..",
                },
                {"role": "user", "content": prompt},
            ],
            timeout=30,
        )
        out = resp.choices[0].message.content.strip()
        logger.info("map_summarize: LLM returned %d chars", len(out))
        return out
    except Exception:
        return _local_semantic_summary(ctx, question)


def reduce_merge(question: str, partials: List[str], deep_mode: bool = False) -> str:
    """
    Merge multiple per-speech summaries into one coherent paragraph.
    deep_mode = richer (15 sentences) narrative; else 8–12 sentences.
    """
    partials_clean = [p.strip() for p in partials if p and p.strip()]
    if not partials_clean:
        return ""
    # limit number of partials merged to 12 to avoid huge prompt
    max_partials = 12
    parts = partials_clean[:max_partials]
    joined = "\n- " + "\n- ".join([p.strip() for p in partials if p.strip()])
    prompt = (
        f"You are given short summaries (2-4 bullets each) extracted from different parts of one speech.\n\n"
        f"Question: {question}\n\n"
        f"Source summaries:\n{joined}\n\n"
        "Task: Produce ONE coherent paragraph (about 100-200 words) that synthesizes the key findings, the speaker stance, "
        "and any evolution across the speech. Do NOT invent facts or add citations. Keep it concise and factual. "
        "Return only the paragraph."
    )

    # try LLM with retries
    attempts = 2
    for attempt in range(attempts):
        try:
            logger.info(
                "reduce_merge: calling LLM merge attempt=%d partials=%d",
                attempt + 1,
                len(parts),
            )
            resp = client.chat.completions.create(
                model=os.getenv("LLM_MODEL", "gpt-3.5-turbo"),
                temperature=0.2,
                max_tokens=700,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a concise neutral summarizer. Output one paragraph.",
                    },
                    {"role": "user", "content": prompt},
                ],
                timeout=60,
            )
            out = resp.choices[0].message.content.strip()
            logger.info("reduce_merge: LLM merge returned %d chars", len(out))
            return out
        except Exception as e:
            logger.warning("reduce_merge: attempt %d failed: %s", attempt + 1, str(e))
            time.sleep(0.8)
    logger.warning(
        "reduce_merge: all LLM attempts failed, falling back to stitch of top partials"
    )
    # fallback: join best partials (deduplicate short)
    stitched = " ".join(dict.fromkeys(parts))  # preserve order, dedupe
    return stitched if len(stitched) < 3200 else stitched[:3200]


def _extract_keywords(q: str) -> list[str]:
    ql = q.lower()
    phrases = []
    # high-signal known terms
    for k in [
        "digital euro",
        "cbdc",
        "pepp",
        "app",
        "tltro",
        "ngeu",
        "quantitative easing",
        "inflation",
        "interest rate",
        "monetary policy",
        "payments",
    ]:
        if k in ql:
            phrases.append(k)
    # fallback: keep content words (>3 chars) minus stopwords
    words = [w for w in re.findall(r"[a-zA-Z]{4,}", ql) if w not in _STOP]
    # de-duplicate while preserving order
    seen, kws = set(), []
    for w in phrases + words:
        if w not in seen:
            seen.add(w)
            kws.append(w)
    # keep it short
    return kws[:5]

def _mmr_select(
    sent_embs: np.ndarray, query_emb: np.ndarray, top_n: int, lamb: float = 0.7
):
    """
    Simple MMR selector: returns indices of selected sentences.
    sent_embs: (N, D) array, query_emb: (D,)
    """
    if sent_embs.size == 0:
        return []
    sims = (sent_embs @ query_emb).astype(float)  # (N,)
    selected = []
    candidates = set(range(len(sims)))
    # first pick highest sim to query
    first = int(np.argmax(sims))
    selected.append(first)
    candidates.remove(first)
    while len(selected) < top_n and candidates:
        best = None
        best_score = -1e9
        for c in list(candidates):
            sim_q = sims[c]
            sim_to_selected = 0.0
            if selected:
                sim_to_selected = max(
                    float(sent_embs[c] @ sent_embs[s]) for s in selected
                )
            score = lamb * sim_q - (1 - lamb) * sim_to_selected
            if score > best_score:
                best_score = score
                best = c
        if best is None:
            break
        selected.append(best)
        candidates.remove(best)
    return selected


def _educational_blurb(keywords: List[str], max_items: int = 2) -> str:
    """
    Return short factual blurbs for known high-level terms to help educate users.
    This is conservative — only includes generic, widely-known definitions.
    """
    defs = {
        "digital euro": "The digital euro would be a digital form of cash: an electronic means of retail payment issued by us, the European Central Bank. As a form of public money, it would be available free of charge to everyone in the euro area, for any digital payments. Today, people do not have access to public money in digital form. In our increasingly digitalised society, a digital euro would be the next step forward for our single currency. The digital euro would be stored in an electronic wallet set up with your bank or with a public intermediary. This would allow you to make all your usual electronic payments – in your local store, online, to a friend – with your phone or card, online and offline.",
        "cbdc": "A CBDC (central bank digital currency) is a digital liability of a central bank, distinct from commercial bank deposits and designed for general public use.",
        "inflation": "Inflation is the general rise in price levels over time, reducing purchasing power; central banks monitor it to set monetary policy.",
        "interest rate": "Interest rates are the cost of borrowing and the return on savings; central banks set policy rates to influence inflation and economic activity.",
        "payments": "Payments refer to systems and instruments used to transfer money between parties, including cards, bank transfers and digital means.",
        "monetary policy": "Monetary policy concerns the decisions taken by central banks to influence the cost and availability of money in an economy.",
    }
    picked = []
    for k in keywords:
        kk = k.lower()
        if kk in defs and defs[kk] not in picked:
            picked.append(defs[kk])
        if len(picked) >= max_items:
            break
    return " ".join(picked)
