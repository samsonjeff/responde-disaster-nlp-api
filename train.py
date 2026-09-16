"""
train.py

Fine-tunes jcblaise/roberta-tagalog on the Responde disaster-response dataset.
Uses a single shared RoBERTa encoder with three independent classification heads:
    - intent       (7 classes)
    - urgency      (4 classes)
    - incident_type (7 classes)

The combined loss is the sum of three class-weighted cross-entropy losses.
Best checkpoint (by macro-averaged val F1 across all 3 tasks) is saved to ml/model/.

Usage:
    python train.py
    python train.py --epochs 12 --batch-size 8 --lr 3e-5
    python train.py --dry-run   # tokenise only, no training â€” quick sanity check
"""

import argparse
import json
import os
import sys
import warnings
from pathlib import Path

# Force UTF-8 output on Windows (avoids cp1252 UnicodeEncodeError)
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import torch
import torch.nn as nn
from collections import Counter
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoTokenizer,
    RobertaModel,
    RobertaPreTrainedModel,
    get_linear_schedule_with_warmup,
)
from sklearn.metrics import f1_score, classification_report

warnings.filterwarnings("ignore", category=FutureWarning)

# â”€â”€ Paths â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

ROOT      = Path(__file__).resolve().parent
DATASETS   = ROOT / "datasets"
MODEL_DIR = ROOT / "model"
BASE_MODEL = "jcblaise/roberta-tagalog-base"

TRAIN_FILE = DATASETS / "train.jsonl"
VAL_FILE   = DATASETS / "val.jsonl"

# â”€â”€ Label maps â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

INTENT_LABELS = sorted([
    "CASUAL_OR_GREETING",
    "CASUALTY_REPORT",
    "EMERGENCY_REPORT",
    "FEEDBACK_OR_THANKS",
    "OTHER",
    "RESOURCE_REQUEST",
    "STATUS_INQUIRY",
])

URGENCY_LABELS = sorted(["CRITICAL", "HIGH", "LOW", "MEDIUM"])

INCIDENT_LABELS = sorted([
    "earthquake",
    "fire",
    "flood",
    "landslide",
    "none",
    "typhoon",
    "volcanic_eruption",
])

INTENT2ID   = {l: i for i, l in enumerate(INTENT_LABELS)}
URGENCY2ID  = {l: i for i, l in enumerate(URGENCY_LABELS)}
INCIDENT2ID = {l: i for i, l in enumerate(INCIDENT_LABELS)}

ID2INTENT   = {i: l for l, i in INTENT2ID.items()}
ID2URGENCY  = {i: l for l, i in URGENCY2ID.items()}
ID2INCIDENT = {i: l for l, i in INCIDENT2ID.items()}


def compute_class_weights(dataset, field, label2id, device):
    """Compute inverse-frequency class weights for balanced loss."""
    counts = Counter()
    for sample in dataset.samples:
        counts[sample[field]] += 1
    n_classes = len(label2id)
    total = sum(counts.values())
    weights = torch.ones(n_classes, device=device)
    for cls_id in range(n_classes):
        count = counts.get(cls_id, 0)
        if count > 0:
            weights[cls_id] = total / (n_classes * count)
    return weights

# â”€â”€ Dataset â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def load_jsonl(path: Path):
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return records


class RespondeDataset(Dataset):
    def __init__(self, records, tokenizer, max_length=128):
        self.samples = []
        skipped = 0
        for rec in records:
            text = (rec.get("text") or "").strip()
            if not text:
                skipped += 1
                continue
            intent   = rec.get("intent", "OTHER")
            urgency  = rec.get("urgency", "LOW")
            incident = rec.get("incident_type", "none")

            # Gracefully handle unknown labels
            if intent   not in INTENT2ID:   intent   = "OTHER"
            if urgency  not in URGENCY2ID:  urgency  = "LOW"
            if incident not in INCIDENT2ID: incident = "none"

            self.samples.append({
                "text":      text,
                "intent":    INTENT2ID[intent],
                "urgency":   URGENCY2ID[urgency],
                "incident":  INCIDENT2ID[incident],
            })

        self.tokenizer  = tokenizer
        self.max_length = max_length
        if skipped:
            print(f"  [Dataset] Skipped {skipped} rows with empty text.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        enc = self.tokenizer(
            sample["text"],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "intent_label":   torch.tensor(sample["intent"],   dtype=torch.long),
            "urgency_label":  torch.tensor(sample["urgency"],  dtype=torch.long),
            "incident_label": torch.tensor(sample["incident"], dtype=torch.long),
        }

# â”€â”€ Model â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def _make_head(in_features, n_classes, dropout_p=0.2):
    """Two-layer classification head with ReLU activation."""
    return nn.Sequential(
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(dropout_p),
        nn.Linear(256, n_classes),
    )


class MultiTaskRoberta(RobertaPreTrainedModel):
    """
    RoBERTa encoder with three independent 2-layer classification heads.
    The [CLS] token representation is fed to each head.
    Uses class-weighted cross-entropy loss with label smoothing.
    """

    def __init__(self, config, n_intent, n_urgency, n_incident):
        super().__init__(config)
        self.roberta = RobertaModel(config, add_pooling_layer=False)

        hidden = config.hidden_size
        dropout_p = getattr(config, "classifier_dropout", None) or 0.1

        self.dropout = nn.Dropout(dropout_p)
        self.intent_head   = _make_head(hidden, n_intent)
        self.urgency_head  = _make_head(hidden, n_urgency)
        self.incident_head = _make_head(hidden, n_incident)

        # Placeholders for class weights â€” set after model init
        self.intent_weights   = None
        self.urgency_weights  = None
        self.incident_weights = None
        self.label_smoothing  = 0.1

        self.post_init()

    def set_class_weights(self, intent_w, urgency_w, incident_w):
        """Register class weight tensors for weighted loss computation."""
        self.intent_weights   = intent_w
        self.urgency_weights  = urgency_w
        self.incident_weights = incident_w

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        cls = self.dropout(outputs.last_hidden_state[:, 0, :])   # [CLS] token

        logits_intent   = self.intent_head(cls)
        logits_urgency  = self.urgency_head(cls)
        logits_incident = self.incident_head(cls)

        loss = None
        if labels is not None:
            ls = self.label_smoothing
            loss_i = nn.CrossEntropyLoss(weight=self.intent_weights,   label_smoothing=ls)
            loss_u = nn.CrossEntropyLoss(weight=self.urgency_weights,  label_smoothing=ls)
            loss_c = nn.CrossEntropyLoss(weight=self.incident_weights, label_smoothing=ls)
            loss = (
                loss_i(logits_intent,   labels["intent"])   +
                loss_u(logits_urgency,  labels["urgency"])  +
                loss_c(logits_incident, labels["incident"])
            )

        return loss, logits_intent, logits_urgency, logits_incident

# â”€â”€ Evaluation helper â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_intent, all_urgency, all_incident = [], [], []
    pred_intent, pred_urgency, pred_incident = [], [], []
    total_loss = 0.0
    n_batches  = 0

    for batch in loader:
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = {
            "intent":   batch["intent_label"].to(device),
            "urgency":  batch["urgency_label"].to(device),
            "incident": batch["incident_label"].to(device),
        }

        loss, li, lu, linc = model(input_ids, attention_mask, labels=labels)
        total_loss += loss.item()
        n_batches  += 1

        all_intent.extend(labels["intent"].cpu().tolist())
        all_urgency.extend(labels["urgency"].cpu().tolist())
        all_incident.extend(labels["incident"].cpu().tolist())

        pred_intent.extend(li.argmax(dim=1).cpu().tolist())
        pred_urgency.extend(lu.argmax(dim=1).cpu().tolist())
        pred_incident.extend(linc.argmax(dim=1).cpu().tolist())

    f1_i = f1_score(all_intent,   pred_intent,   average="macro", zero_division=0)
    f1_u = f1_score(all_urgency,  pred_urgency,  average="macro", zero_division=0)
    f1_c = f1_score(all_incident, pred_incident, average="macro", zero_division=0)
    avg_f1 = (f1_i + f1_u + f1_c) / 3

    return {
        "loss":          total_loss / max(n_batches, 1),
        "f1_intent":     f1_i,
        "f1_urgency":    f1_u,
        "f1_incident":   f1_c,
        "avg_f1":        avg_f1,
    }

# â”€â”€ Training loop â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n  Device : {device}")
    print(f"  Model  : {BASE_MODEL}")
    print(f"  Epochs : {args.epochs}  |  Batch : {args.batch_size}  |  LR : {args.lr}\n")

    # â”€â”€ Load data â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    print("  Loading datasetsâ€¦")
    train_records = load_jsonl(TRAIN_FILE)
    val_records   = load_jsonl(VAL_FILE)
    print(f"    train : {len(train_records)} records")
    print(f"    val   : {len(val_records)} records")

    # â”€â”€ Tokenizer â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    print("\n  Loading tokenizerâ€¦")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    train_ds = RespondeDataset(train_records, tokenizer, max_length=args.max_len)
    val_ds   = RespondeDataset(val_records,   tokenizer, max_length=args.max_len)

    print(f"    train samples : {len(train_ds)}")
    print(f"    val   samples : {len(val_ds)}")

    if args.dry_run:
        print("\n  âœ…  Dry-run complete â€” tokenisation OK. Exiting without training.\n")
        return

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=0)

    # â”€â”€ Class weights â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    print("\n  Computing class weights (inverse-frequency)â€¦")
    w_intent   = compute_class_weights(train_ds, "intent",   INTENT2ID,   device)
    w_urgency  = compute_class_weights(train_ds, "urgency",  URGENCY2ID,  device)
    w_incident = compute_class_weights(train_ds, "incident", INCIDENT2ID, device)
    print(f"    intent weights:   {[f'{v:.2f}' for v in w_intent.tolist()]}")
    print(f"    urgency weights:  {[f'{v:.2f}' for v in w_urgency.tolist()]}")
    print(f"    incident weights: {[f'{v:.2f}' for v in w_incident.tolist()]}")

    # â”€â”€ Model â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    print("\n  Loading base model from HuggingFaceâ€¦")
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(BASE_MODEL)
    model  = MultiTaskRoberta.from_pretrained(
        BASE_MODEL, config=config,
        n_intent   = len(INTENT_LABELS),
        n_urgency  = len(URGENCY_LABELS),
        n_incident = len(INCIDENT_LABELS),
        ignore_mismatched_sizes=True,
    )
    model.to(device)
    model.set_class_weights(w_intent, w_urgency, w_incident)

    # â”€â”€ Optimiser + scheduler â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = len(train_loader) * args.epochs
    scheduler   = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, total_steps // 7),   # ~15% warmup
        num_training_steps=total_steps,
    )

    # â”€â”€ Training â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    print("\n" + "=" * 64)
    print("  Training â€” jcblaise/roberta-tagalog")
    print("=" * 64 + "\n")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    best_f1    = -1.0
    best_epoch = -1

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_batches  = 0

        for step, batch in enumerate(train_loader, 1):
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = {
                "intent":   batch["intent_label"].to(device),
                "urgency":  batch["urgency_label"].to(device),
                "incident": batch["incident_label"].to(device),
            }

            optimizer.zero_grad()
            loss, *_ = model(input_ids, attention_mask, labels=labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            n_batches  += 1

            if step % 10 == 0 or step == len(train_loader):
                print(f"  Epoch {epoch}/{args.epochs}  step {step}/{len(train_loader)}"
                      f"  loss={epoch_loss/n_batches:.4f}", end="\r")

        # â”€â”€ Val â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        metrics = evaluate(model, val_loader, device)
        print(f"\n  Epoch {epoch}  train_loss={epoch_loss/n_batches:.4f}"
              f"  val_loss={metrics['loss']:.4f}"
              f"  F1[intent={metrics['f1_intent']:.3f}"
              f"  urgency={metrics['f1_urgency']:.3f}"
              f"  incident={metrics['f1_incident']:.3f}"
              f"  avg={metrics['avg_f1']:.3f}]")

        if metrics["avg_f1"] > best_f1:
            best_f1    = metrics["avg_f1"]
            best_epoch = epoch
            # Save model + tokenizer + label maps
            model.save_pretrained(MODEL_DIR)
            tokenizer.save_pretrained(MODEL_DIR)
            label_maps = {
                "intent":        {"id2label": ID2INTENT,   "label2id": INTENT2ID},
                "urgency":       {"id2label": ID2URGENCY,  "label2id": URGENCY2ID},
                "incident_type": {"id2label": ID2INCIDENT, "label2id": INCIDENT2ID},
            }
            with open(MODEL_DIR / "label_maps.json", "w") as f:
                json.dump(label_maps, f, indent=2)
            print(f"  ðŸ’¾  Saved best checkpoint (epoch {epoch}, avg_F1={best_f1:.3f})")

    print(f"\n  âœ…  Training complete. Best epoch: {best_epoch}  avg_F1: {best_f1:.3f}")
    print(f"  Model saved to: {MODEL_DIR}\n")


# â”€â”€ Entry point â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")

    if not TRAIN_FILE.exists():
        print(f"âŒ  {TRAIN_FILE} not found. Run `npm run split:dataset` first.")
        sys.exit(1)
    if not VAL_FILE.exists():
        print(f"âŒ  {VAL_FILE} not found. Run `npm run split:dataset` first.")
        sys.exit(1)

    parser = argparse.ArgumentParser(description="Fine-tune jcblaise/roberta-tagalog")
    parser.add_argument("--epochs",     type=int,   default=12,   help="Number of training epochs")
    parser.add_argument("--batch-size", type=int,   default=8,    help="Batch size")
    parser.add_argument("--lr",         type=float, default=3e-5, help="Learning rate")
    parser.add_argument("--max-len",    type=int,   default=128,  help="Max token length")
    parser.add_argument("--dry-run",    action="store_true",      help="Tokenise only, no training")
    args = parser.parse_args()

    print("=" * 64)
    print("  Responde NLP â€” RoBERTa-Tagalog Multi-Task Fine-Tuner")
    print("=" * 64)

    train(args)

