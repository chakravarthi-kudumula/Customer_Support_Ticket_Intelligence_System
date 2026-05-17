from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class MetadataFilters(BaseModel):
    product: str | None = None
    company: str | None = None
    issue: str | None = None
    state: str | None = None


class HealthResponse(BaseModel):
    status: str
    service: str
    version: str


class MetadataResponse(BaseModel):
    project: str
    dataset_rows: int | None
    embedding_rows: int | None
    faiss_rows: int | None
    products: list[str]
    supported_filters: list[str]
    endpoints: list[str]


class ClassifyRequest(BaseModel):
    text: str = Field(..., min_length=1)


class ClassifyResponse(BaseModel):
    predicted_product: str
    confidence: float | None = None
    model_name: str
    class_scores: list[dict[str, Any]] = []


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = Field(default=5, ge=1, le=25)
    fetch_k: int = Field(default=100, ge=1, le=500)
    filters: MetadataFilters = Field(default_factory=MetadataFilters)


class SearchResult(BaseModel):
    rank: int
    similarity: float
    complaint_id: str
    product: str
    issue: str
    company: str
    state: str | None = None
    company_response: str | None = None
    timely_response: str | None = None
    snippet: str


class SearchResponse(BaseModel):
    query: str
    top_k: int
    retrieved_count: int
    filters: dict[str, Any]
    results: list[SearchResult]


class RagRequest(SearchRequest):
    max_snippet_chars: int = Field(default=700, ge=100, le=2000)
    min_similarity: float = Field(default=0.35, ge=0.0, le=1.0)


class RagResponse(BaseModel):
    query: str
    answer: str
    confidence: str
    top_score: float
    citations: list[str]
    retrieved_count: int
    filters: dict[str, Any]
    context: list[SearchResult]


class SummarizeRequest(BaseModel):
    text: str | None = None
    complaint_id: str | None = None
    max_summary_tokens: int = Field(default=110, ge=30, le=220)
    min_summary_tokens: int = Field(default=35, ge=5, le=120)


class SummarizeResponse(BaseModel):
    summary: str
    input_word_count: int
    summary_word_count: int
    compression_ratio: float
    was_truncated: bool
    model_name: str
    complaint_id: str | None = None


class AnalyzeRequest(BaseModel):
    text: str = Field(..., min_length=1)
    top_k: int = Field(default=5, ge=1, le=25)
    filters: MetadataFilters = Field(default_factory=MetadataFilters)
    include_summary: bool = False


class AnalyzeResponse(BaseModel):
    classification: ClassifyResponse
    rag: RagResponse
    summary: SummarizeResponse | None = None
