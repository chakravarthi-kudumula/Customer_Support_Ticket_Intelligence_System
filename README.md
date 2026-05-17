# Customer Support Ticket Intelligence System

NLP project for working with Consumer Financial Protection Bureau complaint data.

## Project Vision

The goal is to build this step by step into a system that can:

- classify customer complaints
- retrieve semantically similar complaints
- summarize complaint narratives
- support intelligent search
- optionally support a RAG-style assistant

## Phase Roadmap

| Phase | Focus |
| --- | --- |
| 0 | Environment + Dataset |
| 1 | NLP EDA + Preprocessing |
| 2 | Baseline ML Models |
| 3 | Transformer Fine-Tuning |
| 4 | Embeddings + Semantic Search |
| 5 | Vector Database System |
| 6 | Summarization |
| 7 | RAG-style Retrieval |
| 8 | FastAPI Deployment |
| 9 | Docker + Productionization |
| 10 | MLflow + Experiment Tracking |
| 11 | Advanced Improvements |

## Setup

Create and activate the virtual environment:

```bash
python3 -m venv ticket
source ticket/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Project Structure

```text
.
|-- data/
|   |-- raw/
|   `-- processed/
|-- notebooks/
|-- src/
|   |-- data/
|   |-- preprocessing/
|   |-- models/
|   |-- embeddings/
|   |-- retrieval/
|   |-- api/
|   `-- utils/
|-- app/
|-- configs/
|-- tests/
|-- mlruns/
|-- requirements.txt
`-- README.md
```

## Dataset Placement

Put the downloaded CFPB CSV file here:

```text
data/raw/
```

The default config looks for:

```text
data/raw/complaints.csv
```

## Balanced Sample

To create a balanced 90k-row sample with 15k complaint narratives per product:

```bash
source ticket/bin/activate
python -m src.data.create_balanced_sample
```

The sample is saved here:

```text
data/processed/cfpb_sample_90k.csv
```

## Phase 1: NLP EDA + Preprocessing

To create the cleaned dataset with text columns for transformers and baseline ML:

```bash
source ticket/bin/activate
python -m src.preprocessing.build_preprocessed_dataset
```

To create the EDA report and charts:

```bash
python -m src.data.phase1_eda_report
```

Notebook:

```text
notebooks/01_nlp_eda_preprocessing.ipynb
```

Phase 1 files:

```text
data/processed/cfpb_sample_90k_clean.csv
artifacts/reports/phase1_eda_summary.md
artifacts/figures/
```

## Phase Pipelines

Phase scripts live here:

```text
src/pipelines/
```

Phase 0 sampling:

```bash
python -m src.pipelines.phase0_create_sample
```

Phase 1 preprocessing and EDA:

```bash
python -m src.pipelines.phase1_preprocess_and_eda
```

Phase 2 script:

```text
src/pipelines/phase2_train_baselines.py
```

## Phase 2: Baseline Models

To train and evaluate the baseline NLP models:

```bash
source ticket/bin/activate
python -m src.pipelines.phase2_train_baselines
```

Phase 2 files:

```text
artifacts/models/*.joblib
artifacts/reports/baseline_model_results.csv
artifacts/reports/baseline_classification_report.md
artifacts/reports/baseline_confusion_matrix.csv
artifacts/reports/baseline_error_examples.csv
artifacts/figures/baseline_confusion_matrix.png
notebooks/02_baseline_models.ipynb
```

## Phase 3: Transformer Fine-Tuning

Fine-tune DistilBERT for complaint classification. On this machine, MPS can be used when the run has normal local access.

Pilot run used for the current saved results:

```bash
source ticket/bin/activate
python -m src.pipelines.phase3_finetune_transformer --max-samples-per-class 1000 --epochs 1 --train-batch-size 8 --eval-batch-size 16
```

BERT pilot run, with smaller batches because BERT is larger:

```bash
python -m src.pipelines.phase3_finetune_transformer --model-name bert-base-uncased --max-samples-per-class 1000 --epochs 1 --train-batch-size 4 --eval-batch-size 8
```

Full 90k-row DistilBERT run for a fairer comparison with the baseline models:

```bash
python -m src.pipelines.phase3_finetune_transformer --full-dataset --epochs 2 --train-batch-size 8 --eval-batch-size 16
```

Phase 3 files:

```text
artifacts/models/distilbert_complaint_classifier/
artifacts/reports/transformer_model_results.csv
artifacts/reports/transformer_classification_report.md
artifacts/reports/transformer_confusion_matrix.csv
artifacts/reports/transformer_error_examples.csv
artifacts/reports/baseline_vs_transformer.csv
artifacts/reports/phase3_next_steps.md
artifacts/figures/transformer_confusion_matrix.png
notebooks/03_transformer_finetuning.ipynb
```

## Phase 4: Embeddings + Semantic Search

To build SBERT embeddings for complaint search:

```bash
source ticket/bin/activate
python -m src.pipelines.phase4_build_embeddings --full-dataset --batch-size 64
```

Search with the NumPy version:

```bash
python -m src.retrieval.semantic_search --query "My account was charged twice" --top-k 5
```

Phase 4 files:

```text
artifacts/embeddings/complaint_embeddings_all_minilm_l6_v2_full.npy
artifacts/embeddings/complaint_metadata_all_minilm_l6_v2_full.csv
artifacts/embeddings/latest_embedding_manifest.json
artifacts/reports/semantic_search_examples.md
notebooks/04_embeddings_semantic_search.ipynb
```

## Phase 5: Vector Database System

To build the FAISS index for the full 90k SBERT embeddings and compare it with NumPy search:

```bash
source ticket/bin/activate
python -m src.pipelines.phase5_build_vector_index
```

Search with FAISS:

```bash
python -m src.retrieval.faiss_search --query "My account was charged twice" --top-k 5
```

Optional: build an approximate HNSW index for larger datasets:

```bash
python -m src.retrieval.faiss_vector_store --index-type hnsw_ip
```

Phase 5 files:

```text
artifacts/vector_indexes/faiss_flat_ip_all_minilm_l6_v2_full.index
artifacts/vector_indexes/latest_faiss_manifest.json
artifacts/reports/phase5_vector_index_benchmark.csv
artifacts/reports/phase5_vector_index_benchmark.md
notebooks/05_vector_database_faiss.ipynb
```

## Phase 6: Summarization

To summarize long complaint narratives with a pretrained seq2seq model:

```bash
source ticket/bin/activate
python -m src.pipelines.phase6_generate_summaries --samples 12 --min-words 300 --local-files-only
```

Summarize one complaint by CFPB complaint ID:

```bash
python -m src.summarization.complaint_summarizer --complaint-id 7887539 --local-files-only
```

Summarize custom text:

```bash
python -m src.summarization.complaint_summarizer --text "My bank charged me twice and has not refunded the duplicate charge." --local-files-only
```

Phase 6 files:

```text
artifacts/summarization/models/
artifacts/reports/phase6_summary_examples.csv
artifacts/reports/phase6_summarization_report.md
notebooks/06_complaint_summarization.ipynb
```

## Phase 7: RAG-Style Retrieval

This phase builds a retrieval-grounded assistant without a paid LLM. It retrieves similar complaints, builds a small context, adds confidence, and cites complaint IDs.

Run the example queries:

```bash
source ticket/bin/activate
python -m src.pipelines.phase7_rag_examples
```

Ask one question:

```bash
python -m src.rag.retrieval_assistant --query "My bank charged me twice for the same transaction" --top-k 5
```

Use metadata filters when the product, company, issue, or state is already known:

```bash
python -m src.rag.retrieval_assistant --query "My account was charged twice" --product "Checking or savings account" --top-k 5
python -m src.rag.retrieval_assistant --query "Debt collector keeps calling me" --company "ENCORE CAPITAL GROUP INC." --top-k 5
```

Phase 7 files:

```text
src/rag/context_builder.py
src/rag/retrieval_assistant.py
artifacts/reports/phase7_rag_examples.csv
artifacts/reports/phase7_rag_answers.md
artifacts/reports/phase7_rag_report.md
notebooks/07_rag_retrieval_system.ipynb
```

## Phase 8: FastAPI Deployment

Run the API locally:

```bash
source ticket/bin/activate
uvicorn src.api.main:app --reload --host 127.0.0.1 --port 8000
```

Open the API docs:

```text
http://127.0.0.1:8000/docs
```

Main endpoints:

```text
GET  /health
GET  /metadata
POST /classify
POST /search
POST /rag
POST /summarize
POST /analyze
```

Example request:

```bash
curl -X POST http://127.0.0.1:8000/rag \
  -H "Content-Type: application/json" \
  -d '{"query":"My account was charged twice","top_k":3,"filters":{"product":"Checking or savings account"}}'
```

Phase 8 files:

```text
src/api/main.py
src/api/schemas.py
src/api/services.py
notebooks/08_fastapi_deployment.ipynb
```
