import html
import re
from dataclasses import dataclass

import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


WHITESPACE_RE = re.compile(r"\s+")
URL_RE = re.compile(r"https?://\S+|www\.\S+")
EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b")
NON_ALPHA_RE = re.compile(r"[^a-z\s]")
REDACTION_TOKEN_RE = re.compile(r"^x{2,}$")


@dataclass(frozen=True)
class TextColumns:
    source_text: str = "Consumer complaint narrative"
    target: str = "Product"
    raw_text: str = "text_raw"
    transformer_text: str = "text_transformer"
    ml_text: str = "text_ml_clean"
    label: str = "target"


def basic_text(value: object) -> str:
    """Convert a value to readable text and normalize whitespace only."""
    if pd.isna(value):
        return ""
    text = html.unescape(str(value))
    return WHITESPACE_RE.sub(" ", text).strip()


def clean_for_transformer(value: object) -> str:
    """Light clean for transformers: keep casing, punctuation, and complaint wording."""
    return basic_text(value)


def is_redaction_token(token: str) -> bool:
    """Return True for CFPB privacy placeholders such as xx, xxxx, or xxxxx."""
    return bool(REDACTION_TOKEN_RE.fullmatch(token))


def clean_for_classical_ml(value: object, remove_stopwords: bool = True) -> str:
    """Normalize text for TF-IDF and baseline classical ML models."""
    text = basic_text(value).lower()
    text = URL_RE.sub(" ", text)
    text = EMAIL_RE.sub(" ", text)
    text = NON_ALPHA_RE.sub(" ", text)
    text = WHITESPACE_RE.sub(" ", text).strip()

    normalized_text = text

    if remove_stopwords:
        tokens = [
            token
            for token in text.split()
            if token not in ENGLISH_STOP_WORDS and not is_redaction_token(token)
        ]
        text = " ".join(tokens)

    return text or "empty_text"


def add_text_features(df: pd.DataFrame, columns: TextColumns | None = None) -> pd.DataFrame:
    """Add the text and length columns used by later phases."""
    columns = columns or TextColumns()
    output = df.copy()

    output[columns.raw_text] = output[columns.source_text].map(basic_text)
    output[columns.transformer_text] = output[columns.source_text].map(clean_for_transformer)
    output[columns.ml_text] = output[columns.source_text].map(clean_for_classical_ml)
    output[columns.label] = output[columns.target]

    output["char_count"] = output[columns.raw_text].str.len()
    output["word_count"] = output[columns.raw_text].str.split().str.len()
    output["approx_token_count"] = (output["word_count"] * 1.3).round().astype("int64")
    output["is_duplicate_text"] = output.duplicated(subset=[columns.raw_text], keep=False)

    return output
