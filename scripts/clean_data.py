import os
import re
import hashlib
import logging
import pandas as pd
from datetime import datetime
from io import StringIO
from pandas.errors import ParserError
import csv

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

INPUT = "data/all_ECB_speeches.csv"  # original CSV
OUT_DATA = "data/all_ECB_speeches_cleaned.csv"  # cleaned CSV (pipe-separated)
OUT_PROC = "processed_ecb_data/cleaned_speeches.csv"  # secondary copy for processing

EXPECTED_COLS = ["date", "speakers", "title", "subtitle", "contents"]


def fix_date(ds: str) -> str:
    if not isinstance(ds, str) or not ds.strip():
        return ""
    s = ds.strip()
    # common typo: "025-06-12" -> "2025-06-12" (3-digit year)
    m = re.match(r"^(\d{2,4})[^\d]?(\d{1,2})[^\d]?(\d{1,2})$", s)
    if m:
        y, mo, day = m.group(1), m.group(2).zfill(2), m.group(3).zfill(2)
        if len(y) == 2:
            y = "20" + y
        if len(y) == 3:
            y = "2" + y
        try:
            return datetime.strptime(f"{y}-{mo}-{day}", "%Y-%m-%d").strftime("%Y-%m-%d")
        except Exception:
            return s
    # try common formats
    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%Y/%m/%d"):
        try:
            return datetime.strptime(s, fmt).strftime("%Y-%m-%d")
        except Exception:
            continue
    return s


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def remove_boilerplate(text: str) -> str:
    if not isinstance(text, str):
        return ""
    t = text
    # remove html
    t = re.sub(r"<[^>]+>", " ", t)
    # common encoding fixes
    t = t.replace("â€™", "'").replace("â€œ", '"').replace("â€", '"')
    # remove explicit "SPEECH" or repeated title labels
    t = re.sub(r"(?i)\bSPEECH\b[:\-\s]*", " ", t)
    t = re.sub(r"(?i)\bSpeech by[^\n]{0,300}\.?", " ", t)
    # remove duplicated long blocks (keep first occurrence)
    t = re.sub(r"(?s)(.{120,2000})\s+\1+", r"\1", t)
    # drop leading title duplication lines that repeat title/metadata
    t = re.sub(r"^\s*[A-Z][A-Za-z0-9\-\s,:]{5,200}\s{2,}", " ", t)
    t = normalize_whitespace(t)
    return t


def safe_read_csv(path: str) -> pd.DataFrame:
    """
    Robust CSV reader:
    1) try normal read_csv
    2) on ParserError try reading with quoting=QUOTE_NONE
    3) on failure replace straight double-quotes with unicode quotes and retry
    After reading, if there are more than 5 columns, join extras into 'contents'.
    """

    # helper to canonicalize columns: ensure EXPECTED_COLS present and extras merged
    def normalize_df(df: pd.DataFrame) -> pd.DataFrame:
        if len(df.columns) >= 5:
            # join any extra columns (5..N) into the contents field
            if len(df.columns) > 5:
                extras = (
                    df.iloc[:, 4:]
                    .astype(str)
                    .apply(lambda r: "|".join(r[r.notna() & (r != "")]), axis=1)
                )
                df = df.iloc[:, :4].copy()
                df["contents"] = extras
            else:
                df = df.iloc[:, :5].copy()
            df.columns = EXPECTED_COLS
        else:
            cols = [f"c{i}" for i in range(len(df.columns))]
            df.columns = cols
            if "contents" not in df.columns:
                df["contents"] = df.iloc[:, -1]
            for c in EXPECTED_COLS:
                if c not in df.columns:
                    df[c] = ""
            df = df[EXPECTED_COLS]
        return df

    # 1) try normal read
    try:
        df = pd.read_csv(
            path, sep="|", engine="python", dtype=str, keep_default_na=False
        )
        df = normalize_df(df)
        return df
    except ParserError as e:
        logger.warning("ParserError reading CSV, trying QUOTE_NONE: %s", e)
    except Exception as e:
        logger.warning("read_csv failed, trying tolerant readers: %s", e)

    # 2) try reading with quoting disabled
    try:
        df = pd.read_csv(
            path,
            sep="|",
            engine="python",
            dtype=str,
            keep_default_na=False,
            quoting=csv.QUOTE_NONE,
            escapechar="\\",
        )
        df = normalize_df(df)
        return df
    except Exception as e:
        logger.warning("QUOTE_NONE read failed: %s", e)

    # 3) last-resort: replace straight double quotes with unicode variant and retry
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            raw = f.read()
        # swap straight quotes to avoid breaking CSV parsing while keeping text visually similar
        repaired = raw.replace('"', "”")
        df = pd.read_csv(
            StringIO(repaired),
            sep="|",
            engine="python",
            dtype=str,
            keep_default_na=False,
        )
        df = normalize_df(df)
        return df
    except Exception as e:
        logger.error("All tolerant read attempts failed: %s", e)
        raise


def fix_malformed_quotes(content: str) -> str:
    """
    Fix common quote malformation issues
    """
    lines = content.split("\n")
    fixed_lines = []

    for line in lines:
        # Count quotes in the line
        quote_count = line.count('"')

        # If odd number of quotes, likely malformed
        if quote_count % 2 == 1:
            # Try to fix by escaping quotes that aren't field delimiters
            # Look for quotes not followed by | or at end of line
            line = re.sub(r'"(?![|\r\n]|$)', '""', line)

        # Fix quotes at the beginning that aren't properly closed
        if line.startswith('"') and not line.endswith('"') and "|" in line:
            # Find the first | and ensure quote is closed before it
            first_pipe = line.find("|")
            if first_pipe > 0 and line[:first_pipe].count('"') % 2 == 1:
                line = line[:first_pipe] + '"' + line[first_pipe:]

        fixed_lines.append(line)

    return "\n".join(fixed_lines)


def text_hash(s: str) -> str:
    return hashlib.sha1((s or "").encode("utf-8")).hexdigest()


def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    # normalize columns
    df["date"] = df["date"].fillna("").astype(str).apply(fix_date)
    df["speakers"] = (
        df["speakers"].fillna("").astype(str).apply(lambda s: normalize_whitespace(s))
    )
    df["title"] = (
        df["title"]
        .fillna("")
        .astype(str)
        .apply(lambda s: normalize_whitespace(re.sub(r"\s{2,}", " ", s)))
    )
    df["subtitle"] = (
        df["subtitle"].fillna("").astype(str).apply(lambda s: normalize_whitespace(s))
    )
    # clean contents
    df["contents"] = df["contents"].fillna("").astype(str).apply(remove_boilerplate)
    # drop rows with too short content
    before = len(df)
    df = df[df["contents"].str.len() > 80].copy()
    logger.info("Dropped %d short/empty rows", before - len(df))
    # create hash and dedupe conservatively by speaker+date+title+content-hash
    df["content_hash"] = df["contents"].apply(text_hash)
    before = len(df)
    df = df.drop_duplicates(
        subset=["speakers", "date", "title", "content_hash"], keep="first"
    )
    logger.info("Removed %d duplicates", before - len(df))
    # reorder columns to expected
    return df[EXPECTED_COLS + ["content_hash"]]


def main():
    os.makedirs(os.path.dirname(OUT_DATA) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(OUT_PROC) or ".", exist_ok=True)

    logger.info("Reading input CSV: %s", INPUT)
    df = safe_read_csv(INPUT)
    logger.info("Rows read: %d", len(df))
    df_clean = clean_df(df)
    logger.info("Rows after cleaning: %d", len(df_clean))

    # write cleaned CSV (pipe-separated) compatible with prepare_data.py
    df_clean.to_csv(OUT_DATA, sep="|", index=False, encoding="utf-8")
    logger.info("Cleaned CSV written to %s", OUT_DATA)

    # also write a copy under processed_ecb_data for convenience
    df_clean.to_csv(OUT_PROC, sep="|", index=False, encoding="utf-8")
    logger.info("Copy written to %s", OUT_PROC)

    # quick diagnostics
    top_dups = df_clean["content_hash"].value_counts().head(10)
    logger.info("Top duplicate counts after cleaning:\n%s", top_dups.to_dict())


if __name__ == "__main__":
    main()
