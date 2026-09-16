<<<<<<< HEAD
# disaster-nlp-transformer
=======
﻿---
title: Responde NLP Classifier
emoji: 🤖
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

# Responde NLP Classifier

FastAPI inference server for the **Responde** disaster-reporting system.
Fine-tuned on a Tagalog disaster-response dataset using `jcblaise/roberta-tagalog-base`.

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Returns `{"status":"ok","model_loaded":true/false}` |
| `POST` | `/classify` | Classifies a Tagalog disaster message |
| `GET` | `/docs` | Interactive Swagger UI |

### `/classify` request body

```json
{ "text": "may sunog dito sa brgy banga!" }
```

### `/classify` response

```json
{
  "intent": "EMERGENCY_REPORT",
  "urgency": "CRITICAL",
  "incident_type": "fire",
  "intent_confidence": 0.9821,
  "urgency_confidence": 0.8743,
  "incident_type_confidence": 0.9612,
  "barangay": "Banga",
  "contact_numbers": []
}
```

## Model Weights

Model weights (`model/`) are **not committed to this repo** (gitignored).

To deploy with weights, either:
1. Upload via the HF web UI (Files tab)
2. Use `git-lfs` to commit the `model/` directory

## Local Development

```bash
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 7860 --reload
```
>>>>>>> master
