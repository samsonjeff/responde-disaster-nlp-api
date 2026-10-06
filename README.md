---
title: Responde NLP Classifier
emoji: 🚨
colorFrom: blue
colorTo: green
sdk: docker
app_port: 8000
pinned: false
---

# Responde NLP Classifier

[![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://www.docker.com/)
[![Caddy](https://img.shields.io/badge/Caddy-22B573?style=for-the-badge&logo=caddy&logoColor=white)](https://caddyserver.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](LICENSE)

FastAPI inference microservice for the **Responde** Incident & Disaster Management System in **Talisay, Batangas**.
Fine-tuned on a localized Tagalog disaster dataset using `jcblaise/roberta-tagalog-base` for multi-task emergency classification.

---

## Production Endpoint

The active production inference server is hosted on a containerized DigitalOcean Droplet behind Caddy reverse proxy:

- **Base URL**: `https://responde-ml.duckdns.org`
- **Interactive Swagger Docs**: `https://responde-ml.duckdns.org/docs`

---

## Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | Service status, health metadata, and API links |
| `GET` | `/health` | Liveness healthcheck (`{"status":"ok","model_loaded":true}`) |
| `GET` | `/ready` | Readiness check (returns `200` if model loaded, `503` if still loading) |
| `POST` | `/classify` | Classifies a Tagalog disaster message |
| `GET` | `/docs` | Interactive Swagger UI documentation |

---

### `/classify` Request Body

```json
{
  "text": "may sunog po dito sa brgy banga kailangan ng bumbero agad!"
}
```

### `/classify` Response Example

```json
{
  "intent": "EMERGENCY_REPORT",
  "urgency": "CRITICAL",
  "incident_type": "fire",
  "intent_confidence": 0.9821,
  "urgency_confidence": 0.8743,
  "incident_type_confidence": 0.9612,
  "needs_review": false,
  "low_confidence_fields": [],
  "barangay": "Banga",
  "contact_numbers": []
}
```

---

## Architecture & Hosting

The microservice runs in an optimized CPU-only PyTorch container:

1. **Docker Container**:
   - Base image: `python:3.11-slim`
   - CPU-only PyTorch wheels (saving ~2GB image footprint)
   - Bound locally to `127.0.0.1:8000`
2. **Reverse Proxy & SSL**:
   - Caddy reverse-proxies incoming HTTPS traffic to `127.0.0.1:8000`
   - Automatic TLS certificate issuance and renewal via Let's Encrypt
3. **Memory & Swap Shield**:
   - Host swap space allocated as a safety ceiling against burst loads and OOM events

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `PORT` | `8000` | Port on which FastAPI / Uvicorn binds |
| `CORS_ORIGINS` | `*` | Comma-separated list of allowed origins |
| `MODEL_DIR` | `./model` | Directory path containing model weights and tokenizer configs |
| `HF_MODEL_REPO` | *(optional)* | Hugging Face repository to pull weights from if not present locally |
| `MODEL_DOWNLOAD_URL` | *(optional)* | Remote HTTPS URL to download weights from if not present locally |

---

## Running with Docker

### Build the Image
```bash
docker build -t responde-ml .
```

### Run Container
```bash
docker run -d \
  --name responde-ml \
  --restart unless-stopped \
  -p 127.0.0.1:8000:8000 \
  responde-ml
```

---

## Local Development (Without Docker)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Start development server
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

---

## License

This project is licensed under the [MIT License](LICENSE).