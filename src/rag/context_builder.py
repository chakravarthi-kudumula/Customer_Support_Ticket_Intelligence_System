from __future__ import annotations

import re
from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class ContextConfig:
    max_snippet_chars: int = 700
    min_similarity: float = 0.35


def clean_text(value: object) -> str:
    text = "" if pd.isna(value) else str(value)
    return re.sub(r"\s+", " ", text).strip()


def snippet_text(value: object, max_chars: int) -> str:
    text = clean_text(value)
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit(" ", 1)[0].strip() + "..."


def confidence_label(top_score: float) -> str:
    if top_score >= 0.70:
        return "high"
    if top_score >= 0.55:
        return "medium"
    if top_score >= 0.40:
        return "low"
    return "weak"


def build_context(results: pd.DataFrame, config: ContextConfig = ContextConfig()) -> list[dict]:
    context = []
    for _, row in results.iterrows():
        similarity = float(row.get("similarity", 0.0))
        if similarity < config.min_similarity:
            continue
        context.append(
            {
                "rank": int(row.get("rank", len(context) + 1)),
                "similarity": round(similarity, 4),
                "complaint_id": str(row.get("Complaint ID", "")),
                "product": clean_text(row.get("Product", "")),
                "issue": clean_text(row.get("Issue", "")),
                "company": clean_text(row.get("Company", "")),
                "state": clean_text(row.get("State", "")),
                "snippet": snippet_text(row.get("text_transformer", ""), config.max_snippet_chars),
            }
        )
    return context


def summarize_context(context: list[dict]) -> dict:
    if not context:
        return {
            "top_score": 0.0,
            "confidence": "weak",
            "products": [],
            "issues": [],
            "companies": [],
            "complaint_ids": [],
        }

    frame = pd.DataFrame(context)
    top_score = float(frame["similarity"].max())
    return {
        "top_score": round(top_score, 4),
        "confidence": confidence_label(top_score),
        "products": frame["product"].value_counts().head(3).index.tolist(),
        "issues": frame["issue"].value_counts().head(3).index.tolist(),
        "companies": frame["company"].value_counts().head(3).index.tolist(),
        "complaint_ids": frame["complaint_id"].tolist(),
    }
