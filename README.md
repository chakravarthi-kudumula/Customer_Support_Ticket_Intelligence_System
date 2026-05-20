# Customer Support Ticket Intelligence System

Enterprise-grade Deep Learning & NLP intelligence system for Consumer Financial Protection Bureau complaint data.

## Project Vision

This project will evolve through staged phases into a system that can:

- classify customer complaints
- retrieve semantically similar complaints
- summarize complaint narratives
- support intelligent search
- Powerd a RAG-style assistant

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

## Project Workflow

```mermaid
flowchart TD
    A["Raw CFPB Complaints CSV<br/>~1.1 GB"] --> B["Balanced Sampling<br/>15k rows per product<br/>6 products = 90k rows"]

    B --> C["Phase 1: NLP EDA + Preprocessing"]
    C --> C1["EDA<br/>product distribution<br/>issue distribution<br/>text length<br/>duplicates<br/>missing narratives"]
    C --> C2["Preprocessing<br/>transformer-safe text<br/>classical ML cleaned text<br/>CFPB redaction handling"]

    C2 --> D["Processed Dataset<br/>cfpb_sample_90k_clean.csv"]

    D --> E["Phase 2: Baseline ML Classification"]
    E --> E1["TF-IDF / Bag of Words"]
    E1 --> E2["Logistic Regression<br/>Linear SVM<br/>Dummy Baseline"]
    E2 --> E3["Metrics<br/>accuracy<br/>precision<br/>recall<br/>macro F1<br/>weighted F1"]
    E3 --> E4["Best Baseline Model<br/>TF-IDF + Logistic Regression"]

    D --> F["Phase 3: Transformer Fine-Tuning"]
    F --> F1["Tokenizer<br/>DistilBERT / BERT"]
    F1 --> F2["Transformer Encoder"]
    F2 --> F3["Classification Head"]
    F3 --> F4["Complaint Product Prediction"]
    F4 --> F5["Transformer Metrics + Reports"]

    D --> G["Phase 4: Embeddings + Semantic Search"]
    G --> G1["Sentence Transformer<br/>all-MiniLM-L6-v2"]
    G1 --> G2["384-dim Complaint Embeddings"]
    G2 --> G3["NumPy Cosine Similarity Search"]

    G2 --> H["Phase 5: Vector Database"]
    H --> H1["FAISS Index<br/>IndexFlatIP"]
    H1 --> H2["Vector Search<br/>Query Embedding -> Top-K Similar Complaints"]
    H2 --> H3["Retrieval Benchmark<br/>NumPy vs FAISS<br/>latency + overlap"]

    D --> I["Phase 6: Summarization"]
    I --> I1["Seq2Seq Transformer<br/>DistilBART"]
    I1 --> I2["Long Complaint Narrative"]
    I2 --> I3["Executive Summary"]

    H2 --> J["Phase 7: Retrieval-Grounded Assistant"]
    J --> J1["User Query"]
    J1 --> J2["SBERT Query Embedding"]
    J2 --> J3["FAISS Retrieval"]
    J3 --> J4["Metadata Filters<br/>product<br/>company<br/>issue<br/>state"]
    J4 --> J5["Context Builder"]
    J5 --> J6["Extractive RAG-Style Answer<br/>citations<br/>confidence<br/>similar complaints<br/>company outcomes"]

    E4 --> K["Phase 8: FastAPI Service"]
    F4 --> K
    H2 --> K
    I3 --> K
    J6 --> K

    K --> K1["API Endpoints<br/>GET /health<br/>GET /metadata<br/>POST /classify<br/>POST /search<br/>POST /rag<br/>POST /summarize<br/>POST /analyze"]

    K1 --> L["Streamlit UI"]
    L --> L1["Analyze"]
    L --> L2["Retrieval Answer"]
    L --> L3["Search"]
    L --> L4["Classify"]
    L --> L5["Summarize"]

    K --> M["Phase 9: Dockerization"]
    L --> M
    M --> M1["Dockerfile.api<br/>FastAPI backend"]
    M --> M2["Dockerfile.ui<br/>Streamlit frontend"]
    M --> M3["docker-compose.yml<br/>API + UI"]
    M --> M4["Runtime Volume Mounts<br/>./artifacts -> /app/artifacts<br/>./data -> /app/data"]
    M --> M5["CPU-only PyTorch<br/>reduced API image size"]

    E3 --> N["Phase 10: MLflow Tracking"]
    F5 --> N
    H3 --> N
    I3 --> N
    J6 --> N
    N --> N1["MLflow SQLite Backend<br/>mlflow.db"]
    N --> N2["Artifact Store<br/>mlruns/"]
    N --> N3["Tracked Items<br/>params<br/>metrics<br/>model artifacts<br/>reports"]
```

## System Architecture

```mermaid
flowchart LR
    subgraph DataLayer["Data + Artifact Layer"]
        RAW["Raw CFPB CSV<br/>data/raw/complaints.csv"]
        CLEAN["Processed 90k Dataset<br/>data/processed/cfpb_sample_90k_clean.csv"]
        MODELS["Model Artifacts<br/>artifacts/models/"]
        EMB["SBERT Embeddings<br/>artifacts/embeddings/"]
        FAISSIDX["FAISS Index<br/>artifacts/vector_indexes/"]
        REPORTS["Reports + Figures<br/>artifacts/reports/<br/>artifacts/figures/"]
    end

    subgraph OfflineLayer["Offline Training + Batch Pipelines"]
        SAMPLE["Balanced Sampling<br/>15k rows x 6 products"]
        PREP["EDA + Preprocessing<br/>text_ml_clean<br/>text_transformer"]
        BASELINE["Baseline ML<br/>TF-IDF / BoW<br/>Logistic Regression / Linear SVM"]
        TRANSFORMER["Transformer Classifier<br/>DistilBERT / BERT<br/>PyTorch + Hugging Face"]
        EMBEDPIPE["Embedding Builder<br/>Sentence Transformers<br/>all-MiniLM-L6-v2"]
        INDEXPIPE["Vector Index Builder<br/>FAISS IndexFlatIP"]
        SUMPIPE["Summarization Setup<br/>DistilBART"]
    end

    subgraph IntelligenceLayer["Runtime Intelligence Layer"]
        CLASSIFY["Classification<br/>Complaint -> Product"]
        SEARCH["Semantic Search<br/>Query -> Similar Complaints"]
        RAG["Retrieval-Grounded Answer<br/>context + citations + outcomes"]
        SUMMARIZE["Summarization<br/>Long Complaint -> Executive Summary"]
    end

    subgraph ApiLayer["FastAPI Service Layer"]
        API["FastAPI App<br/>src/api/main.py"]
        HEALTH["GET /health"]
        META["GET /metadata"]
        EPCLASSIFY["POST /classify"]
        EPSEARCH["POST /search"]
        EPRAG["POST /rag"]
        EPSUM["POST /summarize"]
        EPANALYZE["POST /analyze"]
    end

    subgraph UiLayer["Application UI"]
        UI["Streamlit App<br/>app/streamlit_app.py"]
        USER["Support Analyst / User"]
    end

    subgraph DeploymentLayer["Deployment Layer"]
        DOCKERAPI["Dockerfile.api<br/>FastAPI backend"]
        DOCKERUI["Dockerfile.ui<br/>Streamlit frontend"]
        COMPOSE["docker-compose.yml<br/>api + ui"]
        VOLUMES["Mounted Runtime Volumes<br/>./data -> /app/data<br/>./artifacts -> /app/artifacts"]
    end

    subgraph TrackingLayer["Experiment Tracking"]
        MLFLOW["MLflow<br/>sqlite:///mlflow.db"]
        MLRUNS["Artifact Store<br/>mlruns/"]
    end

    RAW --> SAMPLE
    SAMPLE --> CLEAN
    CLEAN --> PREP
    PREP --> REPORTS

    CLEAN --> BASELINE
    BASELINE --> MODELS
    BASELINE --> REPORTS

    CLEAN --> TRANSFORMER
    TRANSFORMER --> MODELS
    TRANSFORMER --> REPORTS

    CLEAN --> EMBEDPIPE
    EMBEDPIPE --> EMB

    EMB --> INDEXPIPE
    INDEXPIPE --> FAISSIDX
    INDEXPIPE --> REPORTS

    CLEAN --> SUMPIPE
    SUMPIPE --> REPORTS

    MODELS --> CLASSIFY
    EMB --> SEARCH
    FAISSIDX --> SEARCH
    SEARCH --> RAG
    CLEAN --> SUMMARIZE

    CLASSIFY --> API
    SEARCH --> API
    RAG --> API
    SUMMARIZE --> API

    API --> HEALTH
    API --> META
    API --> EPCLASSIFY
    API --> EPSEARCH
    API --> EPRAG
    API --> EPSUM
    API --> EPANALYZE

    USER --> UI
    UI --> API

    DOCKERAPI --> COMPOSE
    DOCKERUI --> COMPOSE
    VOLUMES --> COMPOSE
    COMPOSE --> API
    COMPOSE --> UI

    REPORTS --> MLFLOW
    MODELS --> MLFLOW
    MLFLOW --> MLRUNS
```


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

Place the downloaded CFPB CSV file in:

```text
data/raw/
```

The default config expects:

```text
data/raw/complaints.csv
```

## Balanced Sample

Create a random balanced 90k-row sample with 15k complaint narratives per target product:

```bash
source ticket/bin/activate
python -m src.data.create_balanced_sample
```

The sampled file is written to:

```text
data/processed/cfpb_sample_90k.csv
```

## Phase 1: NLP EDA + Preprocessing

Generate the cleaned dataset with both transformer-safe and classical-ML text columns:

```bash
source ticket/bin/activate
python -m src.preprocessing.build_preprocessed_dataset
```

Generate the EDA report and figures:

```bash
python -m src.data.phase1_eda_report
```

Open the notebook:

```text
notebooks/01_nlp_eda_preprocessing.ipynb
```

Phase 1 outputs:

```text
data/processed/cfpb_sample_90k_clean.csv
artifacts/reports/phase1_eda_summary.md
artifacts/figures/
```

## Phase Pipelines

End-to-end phase workflows live in:

```text
src/pipelines/
```

Run Phase 0 balanced sampling:

```bash
python -m src.pipelines.phase0_create_sample
```

Run Phase 1 preprocessing and EDA:

```bash
python -m src.pipelines.phase1_preprocess_and_eda
```

Phase 2 has a pipeline scaffold ready at:

```text
src/pipelines/phase2_train_baselines.py
```

## Phase 2: Baseline Models

Train and evaluate traditional NLP classifiers:

```bash
source ticket/bin/activate
python -m src.pipelines.phase2_train_baselines
```

Phase 2 outputs:

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

Fine-tune DistilBERT for complaint classification. On this machine, MPS acceleration is available when running outside the restricted sandbox.

Pilot run used for the current artifacts:

```bash
source ticket/bin/activate
python -m src.pipelines.phase3_finetune_transformer --max-samples-per-class 1000 --epochs 1 --train-batch-size 8 --eval-batch-size 16
```

BERT pilot comparison, using smaller batches because BERT is larger:

```bash
python -m src.pipelines.phase3_finetune_transformer --model-name bert-base-uncased --max-samples-per-class 1000 --epochs 1 --train-batch-size 4 --eval-batch-size 8
```

Full 90k-row DistilBERT run when you want the proper baseline comparison:

```bash
python -m src.pipelines.phase3_finetune_transformer --full-dataset --epochs 2 --train-batch-size 8 --eval-batch-size 16
```

Phase 3 outputs:

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

Build SBERT embeddings for semantic complaint retrieval:

```bash
source ticket/bin/activate
python -m src.pipelines.phase4_build_embeddings --full-dataset --batch-size 64
```

Run a semantic search query with the NumPy baseline retriever:

```bash
python -m src.retrieval.semantic_search --query "My account was charged twice" --top-k 5
```

Phase 4 outputs:

```text
artifacts/embeddings/complaint_embeddings_all_minilm_l6_v2_full.npy
artifacts/embeddings/complaint_metadata_all_minilm_l6_v2_full.csv
artifacts/embeddings/latest_embedding_manifest.json
artifacts/reports/semantic_search_examples.md
notebooks/04_embeddings_semantic_search.ipynb
```

## Phase 5: Vector Database System

Build a FAISS vector index over the full 90k SBERT embeddings and benchmark it against the NumPy baseline:

```bash
source ticket/bin/activate
python -m src.pipelines.phase5_build_vector_index
```

Run semantic search through FAISS:

```bash
python -m src.retrieval.faiss_search --query "My account was charged twice" --top-k 5
```

Build an approximate HNSW index when scaling beyond the current 90k rows:

```bash
python -m src.retrieval.faiss_vector_store --index-type hnsw_ip
```

Phase 5 outputs:

```text
artifacts/vector_indexes/faiss_flat_ip_all_minilm_l6_v2_full.index
artifacts/vector_indexes/latest_faiss_manifest.json
artifacts/reports/phase5_vector_index_benchmark.csv
artifacts/reports/phase5_vector_index_benchmark.md
notebooks/05_vector_database_faiss.ipynb
```

## Phase 6: Summarization

Generate executive summaries for long complaint narratives with a pretrained sequence-to-sequence transformer:

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

Phase 6 outputs:

```text
artifacts/summarization/models/
artifacts/reports/phase6_summary_examples.csv
artifacts/reports/phase6_summarization_report.md
notebooks/06_complaint_summarization.ipynb
```

## Phase 7: RAG-Style Retrieval

Generate retrieval-grounded answers without a paid LLM. The system retrieves similar complaints, builds context, adds confidence, and cites complaint IDs.

Run the example queries:

```bash
source ticket/bin/activate
python -m src.pipelines.phase7_rag_examples
```

Ask one question:

```bash
python -m src.rag.retrieval_assistant --query "My bank charged me twice for the same transaction" --top-k 5
```

Use metadata filters when product, company, issue, or state is already known:

```bash
python -m src.rag.retrieval_assistant --query "My account was charged twice" --product "Checking or savings account" --top-k 5
python -m src.rag.retrieval_assistant --query "Debt collector keeps calling me" --company "ENCORE CAPITAL GROUP INC." --top-k 5
```

Phase 7 outputs:

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

Phase 8 outputs:

```text
src/api/main.py
src/api/schemas.py
src/api/services.py
notebooks/08_fastapi_deployment.ipynb
```

## Demo UI

![Streamlit demo](docs/images/streamlit-demo.png)

Start the FastAPI backend:

```bash
source ticket/bin/activate
uvicorn src.api.main:app --host 127.0.0.1 --port 8000
```

In another terminal, start the Streamlit UI:

```bash
source ticket/bin/activate
streamlit run app/streamlit_app.py --server.address 127.0.0.1 --server.port 8501 --server.headless true --browser.gatherUsageStats false
```

Open:

```text
http://127.0.0.1:8501
```

The UI calls the API and supports classification, semantic search, retrieval-grounded answers, summarization, metadata filters, and combined analysis.

## Phase 9: Dockerization

Build and run the API and UI with Docker Compose:

```bash
docker compose up --build
```

Open:

```text
API docs: http://127.0.0.1:8000/docs
UI:       http://127.0.0.1:8501
```

The containers keep large local files out of the image. `docker-compose.yml` mounts these folders at runtime:

```text
./artifacts -> /app/artifacts
./data      -> /app/data
```

The API image uses CPU-only PyTorch so Docker does not pull unnecessary CUDA packages.

Phase 9 outputs:

```text
Dockerfile.api
Dockerfile.ui
docker-compose.yml
.dockerignore
docker/requirements-api.txt
docker/requirements-ui.txt
```

Useful Docker commands:

```bash
docker compose ps
docker compose logs api --tail=100
docker compose down
```

## Phase 10: MLflow + Experiment Tracking

Log the completed experiments into MLflow using the reports and artifacts already created in earlier phases:

```bash
source ticket/bin/activate
python -m src.pipelines.phase10_track_experiments
```

Open the MLflow UI:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000
```

Then open:

```text
http://127.0.0.1:5000
```

Phase 10 tracks:

```text
baseline model metrics and artifacts
transformer classifier metrics and artifacts
FAISS retrieval benchmark metrics
summarization sample metrics
retrieval-context examples
```

Phase 10 outputs:

```text
src/tracking/mlflow_utils.py
src/pipelines/phase10_track_experiments.py
notebooks/10_mlflow_tracking.ipynb
mlflow.db
mlruns/
```
