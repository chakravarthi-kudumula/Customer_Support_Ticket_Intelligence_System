# Docker Deployment

Run the API and Streamlit UI with Docker Compose.

```bash
docker compose up --build
```

Open:

```text
API docs: http://127.0.0.1:8000/docs
UI:       http://127.0.0.1:8501
```

The images do not copy the large CFPB data or model artifacts. Compose mounts them at runtime:

```text
./artifacts -> /app/artifacts
./data      -> /app/data
```

Useful commands:

```bash
docker compose ps
docker compose logs api --tail=100
docker compose down
```

Verified endpoints:

```text
GET  /health
GET  /metadata
POST /rag
```
