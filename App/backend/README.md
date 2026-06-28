# LLM2Seq Backend

FastAPI backend for the LLM2Seq demo app.

The backend loads the Qwen-based LLM2Seq checkpoint from Hugging Face, exposes health/model-info endpoints, and serves summarization through either standard autoregressive decoding or verified MTP decoding.

## Endpoints

```text
GET  /api/health
GET  /api/model-info
GET  /api/random-sample
POST /api/summarize
```

## Local Run

From the repository root:

```bash
cd App/backend
python3 -m pip install -r requirements.txt
export HF_TOKEN=your_huggingface_token
uvicorn main:app --host 0.0.0.0 --port 8000
```

The backend reads:

```text
App/backend/config.yaml
```

The first startup can take time because the encoder and checkpoints are downloaded from Hugging Face.

## Docker

From the repository root:

```bash
docker build -f deploy/docker/backend.Dockerfile -t llm2seq-backend:local .
docker run --rm -p 8000:8000 -e HF_TOKEN=$HF_TOKEN llm2seq-backend:local
```

For the full backend/frontend demo, use `deploy/docker/docker-compose.yml`.
