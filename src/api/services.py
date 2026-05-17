from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from src.api.schemas import (
    AnalyzeRequest,
    AnalyzeResponse,
    ClassifyResponse,
    MetadataFilters,
    MetadataResponse,
    RagResponse,
    SearchResponse,
    SearchResult,
    SummarizeResponse,
)
from src.preprocessing.text_preprocessor import clean_for_classical_ml
from src.rag.context_builder import ContextConfig
from src.rag.retrieval_assistant import RetrievalFilters, answer_query
from src.retrieval.faiss_search import DEFAULT_MANIFEST, search
from src.summarization.complaint_summarizer import (
    DEFAULT_INPUT_PATH,
    ComplaintSummarizer,
    SummaryConfig,
    load_complaint_by_id,
)
from src.utils.config import project_path


CLASSIFIER_MODEL_PATH = project_path('artifacts/models/tfidf_logistic_regression.joblib')
BASELINE_RESULTS_PATH = project_path('artifacts/reports/baseline_model_results.csv')
CLEAN_DATA_PATH = project_path('data/processed/cfpb_sample_90k_clean.csv')
EMBEDDING_MANIFEST_PATH = project_path('artifacts/embeddings/latest_embedding_manifest.json')
FAISS_MANIFEST_PATH = project_path('artifacts/vector_indexes/latest_faiss_manifest.json')


def to_retrieval_filters(filters: MetadataFilters) -> RetrievalFilters:
    return RetrievalFilters(
        product=filters.product,
        company=filters.company,
        issue=filters.issue,
        state=filters.state,
    )


def compact_context_item(item: dict) -> SearchResult:
    return SearchResult(
        rank=int(item.get('rank', 0)),
        similarity=float(item.get('similarity', 0.0)),
        complaint_id=str(item.get('complaint_id', '')),
        product=str(item.get('product', '')),
        issue=str(item.get('issue', '')),
        company=str(item.get('company', '')),
        state=item.get('state') or None,
        company_response=item.get('company_response') or None,
        timely_response=item.get('timely_response') or None,
        snippet=str(item.get('snippet', '')),
    )


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding='utf-8'))


@lru_cache(maxsize=1)
def load_classifier():
    if not CLASSIFIER_MODEL_PATH.exists():
        raise FileNotFoundError(f'Classifier model not found: {CLASSIFIER_MODEL_PATH}')
    return joblib.load(CLASSIFIER_MODEL_PATH)


@lru_cache(maxsize=1)
def load_baseline_model_name() -> str:
    if not BASELINE_RESULTS_PATH.exists():
        return CLASSIFIER_MODEL_PATH.stem
    results = pd.read_csv(BASELINE_RESULTS_PATH)
    if results.empty or 'model_name' not in results.columns:
        return CLASSIFIER_MODEL_PATH.stem
    return str(results.iloc[0]['model_name'])


@lru_cache(maxsize=1)
def load_metadata_frame() -> pd.DataFrame:
    if not CLEAN_DATA_PATH.exists():
        return pd.DataFrame()
    columns = ['Product', 'Issue', 'Company', 'State', 'Complaint ID']
    return pd.read_csv(CLEAN_DATA_PATH, usecols=lambda col: col in columns, low_memory=False)


@lru_cache(maxsize=4)
def get_summarizer(max_summary_tokens: int, min_summary_tokens: int) -> ComplaintSummarizer:
    return ComplaintSummarizer(
        SummaryConfig(
            max_summary_tokens=max_summary_tokens,
            min_summary_tokens=min_summary_tokens,
            local_files_only=True,
        )
    )


def classify_text(text: str) -> ClassifyResponse:
    model = load_classifier()
    cleaned = clean_for_classical_ml(text)
    prediction = str(model.predict([cleaned])[0])
    scores: list[dict[str, Any]] = []
    confidence = None

    if hasattr(model, 'predict_proba'):
        probabilities = model.predict_proba([cleaned])[0]
        classes = [str(label) for label in model.classes_]
        pairs = sorted(zip(classes, probabilities), key=lambda pair: pair[1], reverse=True)
        scores = [
            {'product': product, 'score': round(float(score), 4)}
            for product, score in pairs[:6]
        ]
        confidence = scores[0]['score'] if scores else None

    return ClassifyResponse(
        predicted_product=prediction,
        confidence=confidence,
        model_name=load_baseline_model_name(),
        class_scores=scores,
    )


def search_complaints(query: str, top_k: int, fetch_k: int, filters: MetadataFilters) -> SearchResponse:
    result = answer_query(
        query=query,
        top_k=top_k,
        filters=to_retrieval_filters(filters),
        fetch_k=fetch_k,
    )
    return SearchResponse(
        query=query,
        top_k=top_k,
        retrieved_count=result['retrieved_count'],
        filters=result['filters'],
        results=[compact_context_item(item) for item in result['context']],
    )


def rag_answer(
    query: str,
    top_k: int,
    fetch_k: int,
    filters: MetadataFilters,
    max_snippet_chars: int,
    min_similarity: float,
) -> RagResponse:
    result = answer_query(
        query=query,
        top_k=top_k,
        filters=to_retrieval_filters(filters),
        fetch_k=fetch_k,
        context_config=ContextConfig(
            max_snippet_chars=max_snippet_chars,
            min_similarity=min_similarity,
        ),
    )
    return RagResponse(
        query=query,
        answer=result['answer'],
        confidence=result['confidence'],
        top_score=float(result['top_score']),
        citations=result['citations'],
        retrieved_count=result['retrieved_count'],
        filters=result['filters'],
        context=[compact_context_item(item) for item in result['context']],
    )


def summarize_text_or_complaint(
    text: str | None,
    complaint_id: str | None,
    max_summary_tokens: int,
    min_summary_tokens: int,
) -> SummarizeResponse:
    if not text and not complaint_id:
        raise ValueError('Provide either text or complaint_id.')

    source_text = text
    resolved_complaint_id = complaint_id
    if complaint_id:
        row = load_complaint_by_id(DEFAULT_INPUT_PATH, complaint_id, 'text_transformer')
        source_text = str(row['text_transformer'])
        resolved_complaint_id = str(row.get('Complaint ID', complaint_id))

    summarizer = get_summarizer(max_summary_tokens, min_summary_tokens)
    result = summarizer.summarize(str(source_text))
    return SummarizeResponse(
        summary=result['summary'],
        input_word_count=int(result['input_word_count']),
        summary_word_count=int(result['summary_word_count']),
        compression_ratio=float(result['compression_ratio']),
        was_truncated=bool(result['was_truncated']),
        model_name=str(result['model_name']),
        complaint_id=resolved_complaint_id,
    )


def analyze_text(request: AnalyzeRequest) -> AnalyzeResponse:
    classification = classify_text(request.text)
    filters = request.filters
    if not filters.product:
        filters = MetadataFilters(
            product=classification.predicted_product,
            company=filters.company,
            issue=filters.issue,
            state=filters.state,
        )
    rag = rag_answer(
        query=request.text,
        top_k=request.top_k,
        fetch_k=100,
        filters=filters,
        max_snippet_chars=700,
        min_similarity=0.35,
    )
    summary = None
    if request.include_summary:
        summary = summarize_text_or_complaint(request.text, None, 110, 35)
    return AnalyzeResponse(classification=classification, rag=rag, summary=summary)


def metadata_response() -> MetadataResponse:
    frame = load_metadata_frame()
    products = []
    dataset_rows = None
    if not frame.empty:
        dataset_rows = int(len(frame))
        products = sorted(str(value) for value in frame['Product'].dropna().unique())

    embedding_manifest = read_json(EMBEDDING_MANIFEST_PATH)
    faiss_manifest = read_json(FAISS_MANIFEST_PATH)
    return MetadataResponse(
        project='Customer Support Ticket Intelligence System',
        dataset_rows=dataset_rows,
        embedding_rows=embedding_manifest.get('rows'),
        faiss_rows=faiss_manifest.get('rows'),
        products=products,
        supported_filters=['product', 'company', 'issue', 'state'],
        endpoints=['/health', '/metadata', '/classify', '/search', '/rag', '/summarize', '/analyze'],
    )
