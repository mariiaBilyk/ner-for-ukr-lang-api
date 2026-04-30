# Ukrainian NER Microservice

A production-grade Named Entity Recognition API for Ukrainian text, powered by local LLMs via Ollama. Recognises 13 entity types with an intelligent retry loop, versioned prompt management, and pluggable inference backends.

---

## What it does

Send a POST request with Ukrainian text, get back a structured list of named entities:

```bash
curl -X POST http://localhost:8000/ner \
  -H "Content-Type: application/json" \
  -d '{"text": "Петро Порошенко з Києва зустрівся з директором SoftServe 10 квітня."}'
```

```json
{
  "entities": [
    {"label": "PERS", "text": "Петро Порошенко"},
    {"label": "LOC",  "text": "Києва"},
    {"label": "ORG",  "text": "SoftServe"},
    {"label": "DATE", "text": "10 квітня"}
  ]
}
```

Supported labels: `ART` `DATE` `DOC` `JOB` `LOC` `MISC` `MON` `ORG` `PCT` `PERIOD` `PERS` `QUANT` `TIME`

---

## Why this project stands out

This project implement best practices of architecture - not "tutorial style", but production style.
It was developed as learning project and had to be open for experiments, but experiments are fun, if they are easy to be made. 
And next design desitions make it happen:
- **Pluggable inference backends** — switching from local Ollama to HuggingFace or Azure ML is a one-line config change, no code change.
- **Two-loop extraction architecture** — a structural retry loop (NERAgent) and a semantic self-critique loop (SelfCritiqueExtractionStrategy) are fully independent and composable. The agent never knows whether critique is happening inside the strategy. So it easy to experiment with promnt engineering. 
- **Versioned prompt management** — prompts are versioned artifacts (YAML front-matter, semver) loaded from disk. A/B testing two prompts requires no code change.
- **Config-driven everything** — strategy, model, max attempts, prompt version — all driven by environment variables. We can deploy the same image with different behaviour and compare their behaviour for a long periond with no code change.
- **Structured JSON logs** — every request produces a machine-readable log line with request ID, latency, label counts, and inference metadata. Ready for ELK, Azure Monitor, or any log aggregator. 

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│  HTTP (FastAPI)                                  api/   │
│  POST /ner  ·  GET /health  ·  GET /metrics             │
└────────────────────────┬────────────────────────────────┘
                         │ Depends()
┌────────────────────────▼────────────────────────────────┐
│  Application                               application/ │
│                                                         │
│  NerService ──► NERAgent ──► ExtractionStrategy         │
│                  (retry)        (simple | self_critique) │
└────────────────────────┬────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────┐
│  Infrastructure                         infrastructure/ │
│                                                         │
│  LLMClient ──► InferenceFactory ──► OllamaBackend       │
│  PromptRegistry (YAML front-matter, semver)             │
│  Settings (pydantic-settings, .env)                     │
│  configure_logging (structlog JSON)                     │
└─────────────────────────────────────────────────────────┘
```

### Config-driven behaviour

Every behavioural parameter lives in `Settings` (pydantic-settings), read from environment variables or `.env`:

```
EXTRACTION_STRATEGY=self_critique   # switch strategy
OLLAMA_MODEL=llama3.2               # switch model
NER_AGENT_MAX_ATTEMPTS=5            # more retries
PROMPT_NAME=ner                     # which prompt
PROMPT_VERSION=2.0.0                # pin version
```

The same Docker image runs in any mode. No code change, no rebuild.

---

## Project structure

```
src/
├── api/                   HTTP layer — routers only
├── application/           Use cases — NerService, NERAgent, ExtractionStrategy
├── models/                Pydantic schemas — NerRequest, NerEntity, NerLabel, HealthStatus
├── infrastructure/        I/O — LLM backends, config, logging
│   └── inference/         InferenceBackend protocol + OllamaBackend + InferenceFactory
├── prompts/               PromptRegistry
├── evaluation/            F1 scorer
└── utils/                 parse_entities, normalize_entities, decode

prompts/                   Versioned prompt files (YAML front-matter)
tests/                     pytest suite mirroring src/ structure
deploy/
├── docker/                docker-compose.yml for local dev stack
└── scripts/               deploy-azure.sh — one-time infra setup
.github/workflows/         ci-cd.yml — test → build → deploy pipeline
```

---

## To start this project you need

### Prerequisites

| Tool | Purpose | Install |
|---|---|---|
| Python 3.11+ | Runtime | `brew install python` |
| Ollama | Local LLM inference | [ollama.com](https://ollama.com) |
| Docker + Docker Compose | Containerised dev stack | [docker.com](https://docker.com) |
| Azure CLI *(deploy only)* | Provision Azure resources | `brew install azure-cli` |

### Option A — Docker Compose (recommended)

```bash
# 1. Clone and configure
git clone <repo>
cd nlp-hw-1
cp .env.example .env          # edit OLLAMA_MODEL if needed

# 2. Start the full stack (Ollama + model download + API)
docker compose -f deploy/docker/docker-compose.yml up --build
```

The first run downloads the Qwen 2.5 7B model (~4.7 GB). Subsequent starts use the cached volume.

The API is available at `http://localhost:8000`.

### Option B — Native Python

```bash
# 1. Start Ollama and pull the model
ollama pull qwen2.5:7b-instruct-q4_K_M

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure
cp .env.example .env

# 4. Run
PYTHONPATH=src uvicorn main:app --reload --port 8000
```

### Run tests

```bash
pip install pytest pytest-asyncio
PYTHONPATH=src pytest tests/ -v
```

### Deploy to Azure Container Apps

```bash
# 1. One-time infra setup (creates resource group, Container App env)
az login
bash deploy/scripts/deploy-azure.sh

# 2. Push image to Docker Hub (CI/CD does this automatically on merge to main)
docker build -t yourdockerhubuser/ner-api:latest .
docker push yourdockerhubuser/ner-api:latest
```

Add these secrets to your GitHub repository (Settings → Secrets → Actions):

| Secret | Value |
|---|---|
| `DOCKERHUB_USERNAME` | your Docker Hub username |
| `DOCKERHUB_TOKEN` | Docker Hub access token |
| `AZURE_CREDENTIALS` | service principal JSON (printed by deploy-azure.sh) |

After that, every push to `main` runs tests, builds the image, and deploys automatically.

---

## API reference

| Endpoint | Method | Description |
|---|---|---|
| `/ner` | `POST` | Extract named entities from text |
| `/health` | `GET` | Check API and Ollama reachability |
| `/metrics` | `GET` | Inference provider metadata |

### POST /ner

```json
// Request
{"text": "Петро Порошенко з Києва."}

// Response
{"entities": [{"label": "PERS", "text": "Петро Порошенко"}, {"label": "LOC", "text": "Києва"}]}
```

---

## Configuration reference

| Variable | Default | Description |
|---|---|---|
| `INFERENCE_BACKEND` | `ollama` | Backend: `ollama` \| `huggingface` \| `azure_ml` |
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama server URL |
| `OLLAMA_MODEL` | `qwen2.5:7b-instruct-q4_K_M` | Model to use |
| `PROMPT_NAME` | `ner` | Prompt name from registry |
| `PROMPT_VERSION` | *(empty = latest)* | Pin a specific semver |
| `EXTRACTION_STRATEGY` | `simple` | `simple` \| `self_critique` |
| `NER_AGENT_MAX_ATTEMPTS` | `3` | Max structural retry attempts |
