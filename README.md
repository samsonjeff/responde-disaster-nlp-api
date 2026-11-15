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
  "intent_confidence": 0.7975,
  "urgency_confidence": 0.7188,
  "incident_type_confidence": 0.8223,
  "needs_review": false,
  "low_confidence_fields": [],
  "barangay": "Banga",
  "contact_numbers": []
}
```

---

## Model Benchmark & Evaluation Proof (v5 — November 2026)

The **v5 model** (`Version5_nlp_model`) expands incident classification to **8 classes** (introducing `general` alongside specific natural disasters and `none`), while retaining **6 intent classes** and **4 urgency classes**.

The model was evaluated against **103 held-out test samples** (`datasets/test.jsonl`, stratified across all disaster categories and intents with zero data leakage).

### Benchmark Status: PASSED (F1-Score Evaluation)

All multi-task classification heads successfully **pass** the production benchmark threshold ($\ge 70.0\%$ Macro F1):

| Classification Task | Production Benchmark | Model Result (Macro F1) | Weighted F1 | Status |
|---|:---:|:---:|:---:|:---:|
| **Incident Type** (8 classes) | $\ge 70.0\%$ | **82.9%** | **77.5%** | **PASSED** |
| **Intent** (6 classes) | $\ge 70.0\%$ | **80.4%** | **78.5%** | **PASSED** |
| **Urgency** (4 classes) | $\ge 70.0\%$ | **71.1%** | **76.7%** | **PASSED** |
| **Combined Overall Average** | $\ge 70.0\%$ | **78.1%** | **77.6%** | **PASSED** |

> **Key Observations & Improvements:**
> - **Intent Head**: Surpasses the benchmark at **80.4%** Macro F1 (improved from v3's 79.5%).
> - **Urgency Head**: Surpasses the benchmark at **71.1%** Macro F1 / **76.7%** Weighted F1 (improved from v3's 65.8%).
> - **Incident Type Head**: Achieves **82.9%** Macro F1 across 8 classes, with exceptional scores on critical emergency categories (`fire`: 100%, `landslide`: 94.1%, `typhoon`: 88.9%, `volcanic_eruption`: 88.9%, `earthquake`: 82.4%, `flood`: 81.8%).

---

### Overall Multi-Task Averages (Combined Across All 3 Heads)

| Metric | Intent (6 classes) | Urgency (4 classes) | Incident Type (8 classes) | **Overall Average** |
|---|:---:|:---:|:---:|:---:|
| **Accuracy** | **78.6%** (81/103) | **76.7%** (79/103) | **78.6%** (81/103) | **78.0%** |
| **Macro F1-Score** | **80.4%** | **71.1%** | **82.9%** | **78.1%** |
| **Macro Precision** | 81.0% | 71.4% | 85.1% | **79.2%** |
| **Macro Recall** | 80.5% | 70.8% | 84.2% | **78.5%** |
| **Weighted F1-Score** | 78.5% | 76.7% | 77.5% | **77.6%** |
| **Mean Confidence (Correct)** | 82.1% | 78.2% | 73.9% | **78.1%** |
| **Mean Confidence (Incorrect)** | 61.2% | 70.6% | 63.7% | **65.2%** |

---

### Detailed Task Breakdown

#### 1. Incident Type Classification (`82.9%` Macro F1 | `78.6%` Accuracy)
| Class | Precision | Recall | F1-Score | Support |
|---|:---:|:---:|:---:|:---:|
| `fire` | **1.000** | **1.000** | **1.000** | 8 |
| `landslide` | 0.889 | **1.000** | **0.941** | 8 |
| `typhoon` | 0.857 | 0.923 | **0.889** | 13 |
| `volcanic_eruption` | 0.800 | **1.000** | **0.889** | 8 |
| `earthquake` | **1.000** | 0.700 | **0.824** | 10 |
| `flood` | 0.900 | 0.750 | **0.818** | 12 |
| `none` | 0.594 | 0.950 | **0.731** | 20 |
| `general` | 0.769 | 0.417 | **0.541** | 24 |

#### 2. Intent Classification (`80.4%` Macro F1 | `78.6%` Accuracy)
| Class | Precision | Recall | F1-Score | Support |
|---|:---:|:---:|:---:|:---:|
| `FEEDBACK_OR_THANKS` | **1.000** | **1.000** | **1.000** | 4 |
| `STATUS_INQUIRY` | 0.900 | 0.818 | **0.857** | 11 |
| `EMERGENCY_REPORT` | 0.837 | 0.854 | **0.845** | 48 |
| `CASUAL_OR_GREETING` | 0.800 | 0.800 | **0.800** | 5 |
| `OTHER` | 0.611 | 0.786 | **0.688** | 14 |
| `RESOURCE_REQUEST` | 0.706 | 0.571 | **0.632** | 21 |

#### 3. Urgency Classification (`71.1%` Macro F1 | `76.7%` Accuracy)
| Class | Precision | Recall | F1-Score | Support |
|---|:---:|:---:|:---:|:---:|
| `CRITICAL` | **0.824** | 0.778 | **0.800** | 18 |
| `LOW` | 0.788 | 0.812 | **0.800** | 32 |
| `HIGH` | 0.795 | 0.795 | **0.795** | 44 |
| `MEDIUM` | 0.444 | 0.444 | **0.444** | 9 |

---

### Real-World Test Inference Proof (Batangas Tagalog)

| Sample Input Message | Predicted Intent | Predicted Urgency | Predicted Incident |
|---|---|---|---|
| *"Tulong po! Baha na hanggang bubong dito sa Barangay Aya, may matanda at bata kaming kasama!"* | `EMERGENCY_REPORT` (85.5%) | `CRITICAL` (80.0%) | `flood` (81.1%) |
| *"Kailangan po namin ng food supply at malinis na tubig dine sa evacuation center ng Brgy Tranca"* | `RESOURCE_REQUEST` (94.4%) | `MEDIUM` (86.1%) | `none` (75.6%) |
| *"May malaking sunog po dine sa Purok 3 ng Brgy Tumaway! Mabilis kumakalat ang apoy!"* | `EMERGENCY_REPORT` (78.5%) | `CRITICAL` (71.8%) | `fire` (82.7%) |
| *"Signal No. 3 na po ba sa Talisay? Passable pa po ba ang kalsada sa Sampaloc?"* | `STATUS_INQUIRY` (90.9%) | `LOW` (65.1%) | `typhoon` (29.5%) |
| *"Salamat po sa MDRRMO at sa mga linemen na nag-ayos ng kuryente namin dito sa Pob 7"* | `FEEDBACK_OR_THANKS` (89.7%) | `LOW` (91.9%) | `none` (85.4%) |
| *"opo tama po yan"* | `OTHER` (87.1%) | `LOW` (96.9%) | `none` (90.4%) |
| *"May naipit po sa gumuhong lupa dito sa Brgy Caloocan kailangan ng rescue agad!"* | `EMERGENCY_REPORT` (87.0%) | `CRITICAL` (80.9%) | `landslide` (74.4%) |
| *"Sobrang kapal na po ng ashfall at amoy asupre dine sa Brgy San Guillermo, hirap huminga ang mga bata"* | `EMERGENCY_REPORT` (73.0%) | `HIGH` (77.7%) | `volcanic_eruption` (84.7%) |

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

## Pre-trained Model Weights

Because model weight files (`model.safetensors`, ~416 MB) exceed GitHub's standard file size limits, the trained weights are excluded from Git and hosted publicly on Hugging Face:

**Hugging Face Model Repository**: [`samsonjeff/responde-tagalog-disaster-triage`](https://huggingface.co/samsonjeff/responde-tagalog-disaster-triage)

### Downloading the Weights for Local Run / Self-Hosting

You can pull the weights into the `./model` folder using the Hugging Face CLI:

```bash
# Download model repository into the model/ folder
hf download samsonjeff/responde-tagalog-disaster-triage --local-dir ./model
```

Alternatively, set the environment variable:
```bash
HF_MODEL_REPO=samsonjeff/responde-tagalog-disaster-triage
```
The server will automatically pull the weights on startup if `./model` is empty.

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
