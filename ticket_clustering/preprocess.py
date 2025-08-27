import re
import unicodedata
from typing import Dict, List, Tuple

import pandas as pd

# basic stop word list - German/English mix, excluding negations
default_stop_words = {
    "und", "oder", "aber", "the", "a", "an", "is", "ist", "im", "in", "am",
    "mit", "auf", "zu", "für", "von", "der", "die", "das", "den", "des",
}


def normalize_text(text: str, ignore_words: list = None) -> str:
    """Normalize text by lowercasing, unicode NFKC and removing punctuation. Removes ignore_words if provided."""
    text = unicodedata.normalize("NFKC", text)
    if ignore_words:
        for word in ignore_words:
            # Remove all occurrences, case-insensitive, even as substring
            text = re.sub(rf"{re.escape(word)}", " ", text, flags=re.IGNORECASE)
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = [t for t in text.split() if t not in default_stop_words]
    return " ".join(tokens)


def preprocess_dataframe(df: pd.DataFrame, domain_terms: list = None, ignore_words: list = None) -> Dict[str, Tuple[int, str]]:
    """Return mapping normalized_text -> (occurrences, original_text). Removes ignore_words if provided."""
    domain_terms = domain_terms or []
    ignore_words = ignore_words or []
    mapping: Dict[str, Tuple[int, str]] = {}
    debug_lines = []
    for _, row in df.iterrows():
        original = str(row["text"])
        occ = int(row.get("occurrences", 1))
        norm = normalize_text(original, ignore_words=ignore_words)
        debug_lines.append(f"ORIGINAL: {original}\nNORMALIZED: {norm}\n")
        # ensure domain terms preserved
        for term in domain_terms:
            if term.lower() in original.lower() and term.lower() not in norm:
                norm = f"{norm} {term.lower()}".strip()
        if norm in mapping:
            mapping[norm] = (mapping[norm][0] + occ, mapping[norm][1])
        else:
            mapping[norm] = (occ, original)
    # Write debug output
    with open("debug.txt", "w", encoding="utf-8") as dbg:
        dbg.writelines(debug_lines)
    return mapping
