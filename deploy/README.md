# LLM2Seq Deployment

This folder contains deployment files for the LLM2Seq demo app.

The demo has two services:

- backend: FastAPI service that loads the Qwen-based LLM2Seq checkpoint and exposes `/api/*`.
- frontend: React static app served by Nginx. Nginx proxies `/api/*` to the backend.

## Docker Compose

From the repository root:

```bash
export HF_TOKEN=your_huggingface_token
docker compose up --build
```

Open:

```text
http://localhost:5173
```

The backend API is also exposed at:

```text
http://localhost:8000
```

The first startup can be slow because the backend downloads the encoder and checkpoint files from Hugging Face. The downloaded files are cached in the `hf-cache` Docker volume.

If you deploy on a GPU host, add the GPU runtime setting supported by your Docker installation, for example `gpus: all` under the backend service in `docker-compose.yml`.

## Kubernetes

The Kubernetes manifests are in:

```text
deploy/k8s/
```

Before applying them, build and push images:

```bash
docker build -f App/backend/Dockerfile -t YOUR_REGISTRY/llm2seq-backend:latest .
docker build -f App/frontend/Dockerfile -t YOUR_REGISTRY/llm2seq-frontend:latest .
docker push YOUR_REGISTRY/llm2seq-backend:latest
docker push YOUR_REGISTRY/llm2seq-frontend:latest
```

Then replace the placeholder image names in:

```text
deploy/k8s/backend.yaml
deploy/k8s/frontend.yaml
```

Create the Hugging Face token secret:

```bash
kubectl create namespace llm2seq
kubectl -n llm2seq create secret generic hf-token --from-literal=HF_TOKEN=your_huggingface_token
```

Apply the manifests:

```bash
kubectl apply -f deploy/k8s/
```

For a quick local check without Ingress:

```bash
kubectl -n llm2seq port-forward service/frontend 5173:80
```

Then open:

```text
http://localhost:5173
```

## Notes

- The backend loads the model during application startup, so readiness can take several minutes on a cold node.
- For GPU deployment, install the NVIDIA device plugin and add a GPU resource request to the backend container.
- The provided manifests are a deployable template, not a production security baseline.
