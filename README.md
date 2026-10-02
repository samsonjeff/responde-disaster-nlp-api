---
title: Responde NLP Classifier
emoji: 🚨
colorFrom: blue
colorTo: green
sdk: docker
app_port: 8080
pinned: false
---

# Responde NLP Classifier

FastAPI inference server for the **Responde** disaster-reporting system.
Fine-tuned on a Tagalog disaster-response dataset using `jcblaise/roberta-tagalog-base`.

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Service status, health metadata, and API links |
| `GET` | `/health` | Liveness healthcheck (`{"status":"ok","model_loaded":true}`) |
| `GET` | `/ready` | Readiness check (returns `200` if model loaded, `503` if still loading) |
| `POST` | `/classify` | Classifies a Tagalog disaster message |
| `GET` | `/docs` | Interactive Swagger UI documentation |

### `/classify` Request Body

```json
{
  "text": "may sunog dito sa brgy banga!"
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

## Deploying to Railway

This repository is pre-configured with `railway.json`, an optimized CPU-only PyTorch `Dockerfile`, `.dockerignore`, and `.railwayignore`.

### Option A: Deploy via Railway CLI (Recommended for Local Weights)

Since `model/model.safetensors` (~436 MB) is stored locally and gitignored from GitHub, using the Railway CLI uploads your local workspace directly to Railway's builder:

1. **Login to Railway**:
   ```bash
   npx @railway/cli login
   ```
   *(Or `npx @railway/cli login --browserless` if running on a remote/headless terminal)*

2. **Link or Create a Project**:
   ```bash
   npx @railway/cli init
   # Or link to an existing project:
   npx @railway/cli link
   ```

3. **Deploy**:
   ```bash
   npx @railway/cli up
   ```
   *`.railwayignore` ensures your local `model/` folder is uploaded while ignoring git history and training artifacts.*

4. **Generate a Public Domain**:
   ```bash
   npx @railway/cli domain
   ```

---

### Option B: Deploy via GitHub Repository Integration

If you connect the GitHub repository directly to Railway:

1. Connect the GitHub repo (`samsonjeff/nlp-transformer-api`) in the Railway dashboard.
2. Railway will automatically detect `railway.json` and build using `Dockerfile`.
3. Provide the model weights using **one** of the following methods:
   - **Environment Variable `HF_MODEL_REPO`**: Set to your Hugging Face repo (e.g. `your-user/responde-tagalog-roberta`).
   - **Environment Variable `MODEL_DOWNLOAD_URL`**: Direct HTTPS download link for `model.safetensors` or zip/tar archive (e.g. from AWS S3, Cloudflare R2, Google Cloud Storage, or GitHub Release).
   - **Railway Volume**: Attach a volume at `/app/model` and upload weights via `npx @railway/cli service files upload`.

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | `8080` | Port on which FastAPI / Uvicorn binds (Railway sets this dynamically) |
| `CORS_ORIGINS` | `*` | Comma-separated list of allowed origins (e.g. `https://responde.ph,http://localhost:3000`) |
| `MODEL_DIR` | `./model` | Directory path containing model weights and tokenizers |
| `HF_MODEL_REPO` | *(optional)* | Hugging Face repository to pull weights from if not present locally |
| `MODEL_DOWNLOAD_URL` | *(optional)* | Remote URL to download weights from if not present locally |

---

## Local Development

```bash
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 8080 --reload
```