from __future__ import annotations

import os

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src.api.schemas import (
    AnalyzeRequest,
    AnalyzeResponse,
    ClassifyRequest,
    ClassifyResponse,
    HealthResponse,
    MetadataResponse,
    RagRequest,
    RagResponse,
    SearchRequest,
    SearchResponse,
    SummarizeRequest,
    SummarizeResponse,
)
from src.api.services import (
    analyze_text,
    classify_text,
    metadata_response,
    rag_answer,
    search_complaints,
    summarize_text_or_complaint,
)


app = FastAPI(
    title='Customer Support Ticket Intelligence API',
    version='0.1.0',
    description='APIs for complaint classification, semantic search, retrieval-grounded answers, and summarization.',
)

allowed_origins = [origin.strip() for origin in os.getenv('API_CORS_ORIGINS', '*').split(',') if origin.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)


@app.get('/')
def root() -> dict[str, str]:
    return {'service': 'ticket-intelligence-api', 'docs': '/docs', 'health': '/health'}


@app.get('/health', response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status='ok', service='ticket-intelligence-api', version='0.1.0')


@app.get('/metadata', response_model=MetadataResponse)
def metadata() -> MetadataResponse:
    return metadata_response()


@app.post('/classify', response_model=ClassifyResponse)
def classify(request: ClassifyRequest) -> ClassifyResponse:
    try:
        return classify_text(request.text)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post('/search', response_model=SearchResponse)
def search_endpoint(request: SearchRequest) -> SearchResponse:
    try:
        return search_complaints(
            query=request.query,
            top_k=request.top_k,
            fetch_k=request.fetch_k,
            filters=request.filters,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post('/rag', response_model=RagResponse)
def rag_endpoint(request: RagRequest) -> RagResponse:
    try:
        return rag_answer(
            query=request.query,
            top_k=request.top_k,
            fetch_k=request.fetch_k,
            filters=request.filters,
            max_snippet_chars=request.max_snippet_chars,
            min_similarity=request.min_similarity,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post('/summarize', response_model=SummarizeResponse)
def summarize(request: SummarizeRequest) -> SummarizeResponse:
    try:
        return summarize_text_or_complaint(
            text=request.text,
            complaint_id=request.complaint_id,
            max_summary_tokens=request.max_summary_tokens,
            min_summary_tokens=request.min_summary_tokens,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post('/analyze', response_model=AnalyzeResponse)
def analyze(request: AnalyzeRequest) -> AnalyzeResponse:
    try:
        return analyze_text(request)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
