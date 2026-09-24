"""
app.py

FastAPI inference server for the trained multi-task RoBERTa-Tagalog model.
Standalone entry point for Hugging Face Spaces (Docker SDK) deployment.

Endpoints:
    GET  /health       -> { "status": "ok", "model_loaded": true }
    POST /classify     -> { "intent", "urgency", "incident_type",
                           "intent_confidence", "urgency_confidence",
                           "incident_type_confidence",
                           "barangay", "contact_numbers" }

Local usage:
    uvicorn app:app --host 0.0.0.0 --port 7860 --reload

Hugging Face Spaces (Docker):
    Automatically started via Dockerfile CMD.

Model: jcblaise/roberta-tagalog-base (fine-tuned)
Weights: place trained weights in ./model/ directory
         (commit via git-lfs or upload via HF Hub CLI)
"""

import json
import os
import re
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Optional

import torch
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoConfig, AutoTokenizer

# Both app.py and train.py live at the repo root.
from train import (
    MultiTaskRoberta,
    INTENT_LABELS,
    URGENCY_LABELS,
    INCIDENT_LABELS,
    ID2INTENT,
    ID2URGENCY,
    ID2INCIDENT,
)

MODEL_DIR = Path(__file__).resolve().parent / "model"

BARANGAYS = [
    "Aya", "Balas", "Banga", "Buco", "Caloocan", "Leynes", "Miranda",
    "Poblacion Barangay 1", "Poblacion Barangay 2", "Poblacion Barangay 3",
    "Poblacion Barangay 4", "Poblacion Barangay 5", "Poblacion Barangay 6",
    "Poblacion Barangay 7", "Poblacion Barangay 8",
    "Quiling", "Sampaloc", "San Guillermo", "Santa Maria", "Tranca", "Tumaway",
]

BARANGAY_ALIASES = {
    "pob 1": "Poblacion Barangay 1", "poblacion 1": "Poblacion Barangay 1", "brgy 1": "Poblacion Barangay 1",
    "pob 2": "Poblacion Barangay 2", "poblacion 2": "Poblacion Barangay 2", "brgy 2": "Poblacion Barangay 2",
    "pob 3": "Poblacion Barangay 3", "poblacion 3": "Poblacion Barangay 3", "brgy 3": "Poblacion Barangay 3",
    "pob 4": "Poblacion Barangay 4", "poblacion 4": "Poblacion Barangay 4", "brgy 4": "Poblacion Barangay 4",
    "pob 5": "Poblacion Barangay 5", "poblacion 5": "Poblacion Barangay 5", "brgy 5": "Poblacion Barangay 5",
    "pob 6": "Poblacion Barangay 6", "poblacion 6": "Poblacion Barangay 6", "brgy 6": "Poblacion Barangay 6",
    "pob 7": "Poblacion Barangay 7", "poblacion 7": "Poblacion Barangay 7", "brgy 7": "Poblacion Barangay 7",
    "pob 8": "Poblacion Barangay 8", "poblacion 8": "Poblacion Barangay 8", "brgy 8": "Poblacion Barangay 8",
    "sta. maria": "Santa Maria", "sta maria": "Santa Maria",
    "san guillermo": "San Guillermo",
}

CONTACT_REGEX = re.compile(
    r'(?:'
    r'(?:\+63|0)\s*9\d{2}[\s\-]?\d{3}[\s\-]?\d{4}'
    r'|09\d{9}'
    r'|\+639\d{9}'
    r'|\d{3}[\s\-]?\d{3}[\s\-]?\d{4,5}'
    r')',
    re.IGNORECASE
)


def detect_barangay(text: str) -> Optional[str]:
    """Detect a Talisay barangay name from free text."""
    if not text:
        return None
    lower = text.lower()
    for alias, canonical in sorted(BARANGAY_ALIASES.items(), key=lambda x: -len(x[0])):
        if alias in lower:
            return canonical
    for brgy in sorted(BARANGAYS, key=lambda x: -len(x)):
        if brgy.lower() in lower:
            return brgy
    return None


def extract_contacts(text: str) -> List[str]:
    """Extract Philippine phone numbers from text."""
    if not text:
        return []
    matches = CONTACT_REGEX.findall(text)
    return [re.sub(r'[\s\-]', '', m) for m in matches]


_model       = None
_tokenizer   = None
_device      = None
_model_ok    = False
_id2intent   = ID2INTENT
_id2urgency  = ID2URGENCY
_id2incident = ID2INCIDENT


def load_model():
    global _model, _tokenizer, _device, _model_ok, _id2intent, _id2urgency, _id2incident

    if not (MODEL_DIR / "config.json").exists():
        print(
            f"WARNING: Model weights not found at {MODEL_DIR}.\n"
            "   The server will start but /classify will return 503 until weights are loaded.\n"
            "   Upload trained weights to the model/ directory via git-lfs or HF Hub CLI."
        )
        return

    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Loading model on {_device}...")

    label_map_file = MODEL_DIR / "label_maps.json"
    if label_map_file.exists():
        try:
            with open(label_map_file, encoding="utf-8") as f:
                lm = json.load(f)
                if "id2intent" in lm:
                    _id2intent = {int(k): v for k, v in lm["id2intent"].items()}
                elif "intent" in lm and "id2label" in lm["intent"]:
                    _id2intent = {int(k): v for k, v in lm["intent"]["id2label"].items()}

                if "id2urgency" in lm:
                    _id2urgency = {int(k): v for k, v in lm["id2urgency"].items()}
                elif "urgency" in lm and "id2label" in lm["urgency"]:
                    _id2urgency = {int(k): v for k, v in lm["urgency"]["id2label"].items()}

                if "id2incident" in lm:
                    _id2incident = {int(k): v for k, v in lm["id2incident"].items()}
                elif "incident_type" in lm and "id2label" in lm["incident_type"]:
                    _id2incident = {int(k): v for k, v in lm["incident_type"]["id2label"].items()}

                print(f"  Loaded label_maps.json ({len(_id2intent)} intents, {len(_id2urgency)} urgencies, {len(_id2incident)} incidents).")
        except Exception as e:
            print(f"  Could not parse label_maps.json ({e}), using default label maps.")

    _tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    config     = AutoConfig.from_pretrained(MODEL_DIR)
    _model     = MultiTaskRoberta.from_pretrained(
        MODEL_DIR, config=config,
        n_intent   = len(_id2intent),
        n_urgency  = len(_id2urgency),
        n_incident = len(_id2incident),
        ignore_mismatched_sizes=True,
    )
    _model.to(_device)
    _model.eval()
    _model_ok = True
    print(f"  Model ready on {_device}.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("=" * 64)
    print("  Responde NLP Inference Server")
    print("=" * 64)
    load_model()
    yield
    print("  Server shutting down.")


app = FastAPI(
    title="Responde NLP Classifier",
    description="Multi-task intent / urgency / incident-type classifier for Talisay Batangas disaster messages.",
    version="1.0.0",
    lifespan=lifespan,
)


# Empirically calibrated confidence thresholds from test set evaluation.
# Predictions below these cutoffs trigger human review (needs_review = True).
CONFIDENCE_THRESHOLDS = {
    "intent":        0.42,   # Below 0.40 is coin-flip (50% acc); 0.42 captures ambiguous pleas/complaints
    "urgency":       0.50,   # Below 0.50 is low reliability (57% acc); avoids over-flagging 4-class head
    "incident_type": 0.45,   # Captures 69.2% of incident errors; clean set reaches 94.7% accuracy
}


class ClassifyRequest(BaseModel):
    text: str


class ClassifyResponse(BaseModel):
    intent:                   str
    urgency:                  str
    incident_type:            str
    intent_confidence:        float
    urgency_confidence:       float
    incident_type_confidence: float
    needs_review:             bool = False
    low_confidence_fields:    List[str] = []
    barangay:                 Optional[str] = None
    contact_numbers:          List[str] = []


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": _model_ok}


@app.post("/classify", response_model=ClassifyResponse)
def classify(req: ClassifyRequest):
    if not _model_ok:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Upload weights to model/ and restart the server.",
        )

    text = (req.text or "").strip()
    if not text:
        raise HTTPException(status_code=422, detail="text must not be empty.")

    enc = _tokenizer(
        text,
        max_length=128,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    input_ids      = enc["input_ids"].to(_device)
    attention_mask = enc["attention_mask"].to(_device)

    with torch.no_grad():
        _, li, lu, linc = _model(input_ids, attention_mask)

    pi   = F.softmax(li,   dim=1).squeeze(0)
    pu   = F.softmax(lu,   dim=1).squeeze(0)
    pinc = F.softmax(linc, dim=1).squeeze(0)

    intent_id   = pi.argmax().item()
    urgency_id  = pu.argmax().item()
    incident_id = pinc.argmax().item()

    i_conf   = round(pi[intent_id].item(), 4)
    u_conf   = round(pu[urgency_id].item(), 4)
    inc_conf = round(pinc[incident_id].item(), 4)

    # Evaluate per-head confidence against calibrated thresholds
    low_confidence_fields = []
    if i_conf < CONFIDENCE_THRESHOLDS["intent"]:
        low_confidence_fields.append("intent")
    if u_conf < CONFIDENCE_THRESHOLDS["urgency"]:
        low_confidence_fields.append("urgency")
    if inc_conf < CONFIDENCE_THRESHOLDS["incident_type"]:
        low_confidence_fields.append("incident_type")

    needs_review = len(low_confidence_fields) > 0

    barangay = detect_barangay(text)
    contacts = extract_contacts(text)

    return ClassifyResponse(
        intent                   = _id2intent[intent_id],
        urgency                  = _id2urgency[urgency_id],
        incident_type            = _id2incident[incident_id],
        intent_confidence        = i_conf,
        urgency_confidence       = u_conf,
        incident_type_confidence = inc_conf,
        needs_review             = needs_review,
        low_confidence_fields    = low_confidence_fields,
        barangay                 = barangay,
        contact_numbers          = contacts,
    )
