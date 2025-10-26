#!/usr/bin/env python3

import os
import re
import json
import pickle
import logging
import pandas as pd
from datetime import datetime
from typing import List, Dict, Tuple

# Logging config
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class ECBSpeechPreprocessor:
    def __init__(self, csv_file_path: str, output_dir: str = "processed_ecb_data"):
        self.csv_file_path = csv_file_path
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        logger.info(f"Output directory created/verified: {self.output_dir}")

    def load_csv_data(self) -> pd.DataFrame:
        try:
            df = pd.read_csv(self.csv_file_path, delimiter="|", encoding="utf-8")
            required_columns = ["date", "speakers", "title", "subtitle", "contents"]
            for col in required_columns:
                if col not in df.columns:
                    raise ValueError(f"Missing column: {col}")
            logger.info(f"Loaded {len(df)} speeches from CSV")
            return df
        except Exception as e:
            logger.error(f"Failed to load CSV: {e}")
            raise

    def clean_text(self, text: str) -> str:
        if pd.isna(text) or not isinstance(text, str):
            return ""
        text = re.sub(r"\s+", " ", text.strip())
        text = re.sub(r"<[^>]+>", "", text)
        text = re.sub(r"[^\w\s\.,;:!?\-\(\)\"\'€%]", "", text)
        text = text.replace("â€™", "'").replace("â€œ", '"').replace("â€", '"')
        text = text.replace('â€"', "–").replace('â€"', "—")
        text = re.sub(r"([.!?]){2,}", r"\1", text)
        return text.strip()

    def parse_date(self, date_str: str) -> str:
        for fmt in ["%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%Y/%m/%d"]:
            try:
                return datetime.strptime(str(date_str).strip(), fmt).strftime(
                    "%Y-%m-%d"
                )
            except ValueError:
                continue
        return str(date_str).strip()

    def clean_speakers(self, speakers_str: str) -> List[str]:
        if pd.isna(speakers_str):
            return []
        speakers = [s.strip() for s in str(speakers_str).split(",")]
        return [self.clean_text(s) for s in speakers if s]

    def extract_speech_type(self, content: str) -> str:
        content = str(content).upper()
        if "SPEECH" in content:
            return "SPEECH"
        if "OPENING REMARKS" in content:
            return "OPENING_REMARKS"
        if "PRESS CONFERENCE" in content:
            return "PRESS_CONFERENCE"
        if "INTERVIEW" in content:
            return "INTERVIEW"
        return "OTHER"

    def extract_location(self, content: str) -> str:
        patterns = [
            r"at the (.+?),",
            r"in (.+?),",
            r"at (.+?) on",
            r"in (.+?) on",
            r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*),\s+\d{1,2}\s+\w+\s+\d{4}",
        ]
        for pattern in patterns:
            match = re.search(pattern, str(content))
            if match:
                location = match.group(1).strip()
                return re.sub(r"^(the\s+)", "", location, flags=re.IGNORECASE)[:100]
        return ""

    def clean_main_content(self, content: str) -> str:
        if pd.isna(content):
            return ""
        content = str(content)
        start_patterns = [
            r"(Ladies and gentlemen,)",
            r"(Thank you.*?)",
            r"(Good morning.*?)",
        ]
        for pattern in start_patterns:
            match = re.search(pattern, content, re.IGNORECASE | re.DOTALL)
            if match:
                content = content[match.start() :]
                break
        content = re.sub(r"\[\d+\].*$", "", content, flags=re.DOTALL)
        content = re.sub(r"^\s*\d+\s+.*$", "", content, flags=re.MULTILINE)
        content = self.clean_text(content)
        sentences = content.split(".")
        return ". ".join([s.strip() for s in sentences if len(s.strip()) > 20])

    def chunk_text(
        self, text: str, chunk_size: int = 1000, overlap: int = 200
    ) -> List[str]:
        """Split text into overlapping chunks, preferring to end near a sentence boundary."""
        if not text:
            return []
        text = text.strip()
        n = len(text)
        if n <= chunk_size:
            return [text]

        chunks = []
        start = 0
        while start < n:
            end = min(start + chunk_size, n)

            # Try to end on a sentence boundary within a lookahead window
            if end < n:
                lookahead = text[start : end + 200]  # small window to find a period
                last_dot = lookahead.rfind(".")
                # Accept the boundary if it's in the latter 40% of the chunk, to avoid tiny tails
                if last_dot != -1 and last_dot > int(0.6 * len(lookahead)):
                    end = start + last_dot + 1  # include the dot

            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)

            if end >= n:
                break
            start = max(0, end - overlap)

        return chunks

    def process_speeches(self) -> Tuple[List[Dict], List[str]]:
        df = self.load_csv_data()
        all_chunks, all_metadata = [], []

        for idx, row in df.iterrows():
            try:
                # --- Core metadata ---
                date_str = self.parse_date(row["date"])
                speakers = self.clean_speakers(row["speakers"])
                title = self.clean_text(row["title"])
                subtitle = self.clean_text(row["subtitle"])
                speech_type = self.extract_speech_type(row["contents"])
                location = self.extract_location(row["contents"])

                # --- Speech body ---
                main_text = self.clean_main_content(row["contents"])
                if not main_text or len(main_text) < 100:
                    continue

                # --- Larger chunks ---
                chunks = self.chunk_text(main_text, chunk_size=2000, overlap=250)

                # --- Create stable speech_id ---
                speech_id = f"{date_str}_{title.replace(' ', '_')}_{idx}"

                # --- Build metadata rows ---
                for i, ch in enumerate(chunks):
                    meta = {
                        "speech_id": speech_id,
                        "chunk_index": i,
                        "total_chunks": len(chunks),
                        "title": title,
                        "subtitle": subtitle,
                        "speakers": speakers,
                        "speaker": speakers[0] if speakers else "Unknown",
                        "date": date_str,
                        "speech_type": speech_type,
                        "location": location,
                        "chunk_text": ch,
                        "chunk_length": len(ch),
                    }
                    all_metadata.append(meta)
                    all_chunks.append(ch)

            except Exception as e:
                logger.warning(f"Error on row {idx}: {e}")
                continue

        logger.info(f"Generated {len(all_chunks)} chunks from {len(df)} speeches")
        return all_metadata, all_chunks

    def save_processed_data(self, metadata_list: List[Dict], chunks_list: List[str]):
        txt_file = os.path.join(self.output_dir, "ecb_speeches_chunks.txt")
        json_file = os.path.join(self.output_dir, "ecb_speeches_metadata.json")
        stats_file = os.path.join(self.output_dir, "processing_stats.json")
        pickle_file = os.path.join(self.output_dir, "ecb_speeches_chunks.plk")

        with open(txt_file, "w", encoding="utf-8") as f:
            for chunk in chunks_list:
                f.write(chunk + "\n---CHUNK_SEPARATOR---\n")

        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(metadata_list, f, indent=2, ensure_ascii=False)

        with open(pickle_file, "wb") as pf:
            pickle.dump(
                [
                    {
                        "content": chunk,
                        "speaker": (
                            meta["speakers"][0] if meta["speakers"] else "Unknown"
                        ),
                        "date": meta["date"],
                    }
                    for chunk, meta in zip(chunks_list, metadata_list)
                ],
                pf,
            )

        stats = {
            "total_chunks": len(chunks_list),
            "total_speeches": len(set(m["speech_id"] for m in metadata_list)),
            "average_chunk_length": sum(len(c) for c in chunks_list) / len(chunks_list),
            "speakers": list(set(s for m in metadata_list for s in m["speakers"])),
            "date_range": {
                "earliest": min(m["date"] for m in metadata_list),
                "latest": max(m["date"] for m in metadata_list),
            },
            "speech_types": list(set(m["speech_type"] for m in metadata_list)),
            "processing_date": datetime.now().isoformat(),
        }

        with open(stats_file, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)

        logger.info("Data saved: .txt, .json, .plk, and stats.json")

    def run_preprocessing(self):
        logger.info("Starting ECB preprocessing...")
        metadata_list, chunks_list = self.process_speeches()
        if not chunks_list:
            logger.error("No chunks were generated.")
            return
        self.save_processed_data(metadata_list, chunks_list)
        logger.info("Preprocessing completed!")


def main():
    csv_file = "data/all_ECB_speeches_cleaned.csv"
    output_dir = "processed_ecb_data"
    preprocessor = ECBSpeechPreprocessor(csv_file, output_dir)
    preprocessor.run_preprocessing()


if __name__ == "__main__":
    main()
