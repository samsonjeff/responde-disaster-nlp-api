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

## Model Benchmark & Evaluation Proof (v3 — October 30, 2026)

The v3 model (`v3-OCT302026`) was trained and evaluated following manual data relabeling and conflict resolution across Talisay, Batangas disaster reports. `CASUALTY_REPORT` was merged into `EMERGENCY_REPORT` to eliminate task ambiguity.

The model was evaluated against **103 held-out test samples** (stratified across all disaster categories and intents with zero data leakage):

### Summary Metrics

| Task | Number of Classes | Accuracy | Macro F1 | Weighted F1 | Avg Confidence |
|---|:---:|:---:|:---:|:---:|:---:|
| **Incident Type** | 7 | **89.3%** | **89.6%** | **89.2%** | 75.5% |
| **Intent** | 6 | **76.7%** | **79.5%** | **76.8%** | 77.8% |
| **Urgency** | 4 | **72.8%** | **65.8%** | **72.9%** | 75.5% |

---

### Detailed Task Breakdown

#### 1. Incident Type Classification (`89.6%` Macro F1)
| Class | Precision | Recall | F1-Score | Support |
|---|:---:|:---:|:---:|:---:|
| `fire` | **1.000** | **1.000** | **1.000** | 8 |
| `landslide` | 0.889 | **1.000** | **0.941** | 8 |
| `none` | 0.907 | 0.886 | **0.897** | 44 |
| `volcanic_eruption` | 0.875 | 0.875 | **0.875** | 8 |
| `flood` | 0.909 | 0.833 | **0.870** | 12 |
| `typhoon` | 0.765 | **1.000** | **0.867** | 13 |
| `earthquake` | **1.000** | 0.700 | **0.824** | 10 |

#### 2. Intent Classification (`79.5%` Macro F1)
| Class | Precision | Recall | F1-Score | Support |
|---|:---:|:---:|:---:|:---:|
| `FEEDBACK_OR_THANKS` | **1.000** | **1.000** | **1.000** | 4 |
| `STATUS_INQUIRY` | **1.000** | 0.818 | **0.900** | 11 |
| `EMERGENCY_REPORT` | 0.800 | 0.833 | **0.816** | 48 |
| `CASUAL_OR_GREETING` | 0.800 | 0.800 | **0.800** | 5 |
| `RESOURCE_REQUEST` | 0.706 | 0.571 | **0.632** | 21 |
| `OTHER` | 0.556 | 0.714 | **0.625** | 14 |

#### 3. Urgency Classification (`72.9%` Weighted F1)
| Class | Precision | Recall | F1-Score | Support |
|---|:---:|:---:|:---:|:---:|
| `CRITICAL` | **0.923** | 0.667 | **0.774** | 18 |
| `HIGH` | 0.761 | 0.795 | **0.778** | 44 |
| `LOW` | 0.714 | 0.781 | **0.746** | 32 |
| `MEDIUM` | 0.333 | 0.333 | **0.333** | 9 |

---

### Real-World Test Inference Proof (Batangas Tagalog)

| Sample Input Message | Predicted Intent | Predicted Urgency | Predicted Incident |
|---|---|---|---|
| *"Tulong po! Baha na hanggang bubong dito sa Barangay Aya, may matanda at bata kaming kasama!"* | `EMERGENCY_REPORT` (87.6%) | `HIGH` (54.4%) | `flood` (85.5%) |
| *"Kailangan po namin ng food supply at malinis na tubig dine sa evacuation center ng Brgy Tranca"* | `RESOURCE_REQUEST` (93.7%) | `MEDIUM` (80.0%) | `none` (77.2%) |
| *"May malaking sunog po dine sa Purok 3 ng Brgy Tumaway! Mabilis kumakalat ang apoy!"* | `EMERGENCY_REPORT` (83.8%) | `CRITICAL` (68.2%) | `fire` (84.4%) |
| *"Signal No. 3 na po ba sa Talisay? Passable pa po ba ang kalsada sa Sampaloc?"* | `STATUS_INQUIRY` (88.3%) | `LOW` (65.8%) | `typhoon` (48.4%) |
| *"Salamat po sa MDRRMO at sa mga linemen na nag-ayos ng kuryente namin dito sa Pob 7"* | `FEEDBACK_OR_THANKS` (89.0%) | `LOW` (91.3%) | `none` (86.5%) |
| *"opo tama po yan"* | `OTHER` (87.7%) | `LOW` (94.8%) | `none` (93.7%) |
| *"May naipit po sa gumuhong lupa dito sa Brgy Caloocan kailangan ng rescue agad!"* | `EMERGENCY_REPORT` (90.3%) | `CRITICAL` (88.4%) | `landslide` (68.4%) |
| *"Sobrang kapal na po ng ashfall at amoy asupre dine sa Brgy San Guillermo, hirap huminga ang mga bata"* | `EMERGENCY_REPORT` (75.6%) | `HIGH` (68.8%) | `volcanic_eruption` (87.2%) |

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