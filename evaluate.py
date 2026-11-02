"""
evaluate.py - Standalone evaluation with Confusion Matrix diagnostics
for Responde Multi-Task RoBERTa model against test.jsonl.
"""

import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoTokenizer
from sklearn.metrics import classification_report, confusion_matrix

ROOT      = Path(__file__).resolve().parent
DATASETS  = ROOT / "datasets"
MODEL_DIR = ROOT / "model"
TEST_FILE = DATASETS / "test.jsonl"


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


def print_confusion_matrix(y_true, y_pred, labels, title):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(labels))))
    print(f"\n{'=' * 75}")
    print(f"CONFUSION MATRIX: {title}")
    print(f"{'=' * 75}")

    max_len = max(len(l) for l in labels)
    col_w = max(6, max_len)

    header = "True \\ Pred".ljust(max_len + 2) + "".join(f"{l[:col_w]:>{col_w + 2}}" for l in labels)
    print(header)
    print("-" * len(header))

    for i, row in enumerate(cm):
        row_str = f"{labels[i]:<{max_len + 2}}" + "".join(f"{val:>{col_w + 2}}" for val in row)
        print(row_str)
    print()

    # Off-diagonal misclassifications
    confusions = []
    for i in range(len(labels)):
        for j in range(len(labels)):
            if i != j and cm[i][j] > 0:
                confusions.append((cm[i][j], labels[i], labels[j]))

    confusions.sort(key=lambda x: x[0], reverse=True)
    if confusions:
        print("  ⚠️  Top Misclassifications (True -> Predicted):")
        for count, true_l, pred_l in confusions[:6]:
            print(f"     • {count}x: {true_l} was predicted as {pred_l}")
    print()


def print_calibration_analysis(y_true, y_pred, confidences, title):
    total = len(y_true)
    correct_confs = [c for t, p, c in zip(y_true, y_pred, confidences) if t == p]
    incorrect_confs = [c for t, p, c in zip(y_true, y_pred, confidences) if t != p]
    total_errors = len(incorrect_confs)

    mean_correct = sum(correct_confs) / len(correct_confs) if correct_confs else 0.0
    mean_incorrect = sum(incorrect_confs) / len(incorrect_confs) if incorrect_confs else 0.0

    print(f"\n{'=' * 75}")
    print(f"CONFIDENCE CALIBRATION & THRESHOLD ANALYSIS: {title}")
    print(f"{'=' * 75}")
    print(f"  • Total Samples: {total}")
    print(f"  • Overall Accuracy: {len(correct_confs) / total:.1%} ({len(correct_confs)}/{total})")
    print(f"  • Mean Confidence (When CORRECT):   {mean_correct:.3f}")
    print(f"  • Mean Confidence (When INCORRECT): {mean_incorrect:.3f}")
    print(f"  • Confidence Gap (Correct - Wrong): {mean_correct - mean_incorrect:+.3f}")
    print()

    # 1. Binned accuracy table
    bins = [(0.0, 0.40), (0.40, 0.50), (0.50, 0.60), (0.60, 0.70), (0.70, 0.80), (0.80, 0.90), (0.90, 1.01)]
    print(f"  {'Confidence Bin':<18} | {'Total':<6} | {'Correct':<8} | {'Errors':<7} | {'Accuracy':<10}")
    print(f"  {'-' * 18}-+-{'-' * 6}-+-{'-' * 8}-+-{'-' * 7}-+-{'-' * 10}")
    for low, high in bins:
        bin_items = [(t, p, c) for t, p, c in zip(y_true, y_pred, confidences) if (low <= c < high) or (high > 1.0 and c >= low)]
        count = len(bin_items)
        if count == 0:
            continue
        correct = sum(1 for t, p, c in bin_items if t == p)
        errors = count - correct
        acc = (correct / count) * 100
        bin_label = f"[{low:.2f} - {min(high, 1.0):.2f})" if high <= 1.0 else f"[{low:.2f} - 1.00]"
        print(f"  {bin_label:<18} | {count:<6} | {correct:<8} | {errors:<7} | {acc:>8.1f}%")
    print()

    # 2. Threshold simulation (what happens if needs_review threshold is set to T?)
    print(f"  {'Threshold (T)':<18} | {'Flagged':<8} | {'Errors Caught':<15} | {'False Alarms':<14} | {'Clean Set Acc':<14}")
    print(f"  {'-' * 18}-+-{'-' * 8}-+-{'-' * 15}-+-{'-' * 14}-+-{'-' * 14}")
    for thresh in [0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]:
        flagged = [(t, p, c) for t, p, c in zip(y_true, y_pred, confidences) if c < thresh]
        unflagged = [(t, p, c) for t, p, c in zip(y_true, y_pred, confidences) if c >= thresh]
        n_flagged = len(flagged)
        errors_caught = sum(1 for t, p, c in flagged if t != p)
        false_alarms = sum(1 for t, p, c in flagged if t == p)
        unflagged_correct = sum(1 for t, p, c in unflagged if t == p)
        clean_acc = (unflagged_correct / len(unflagged) * 100) if unflagged else 100.0
        err_recall = (errors_caught / total_errors * 100) if total_errors else 0.0
        print(f"  < {thresh:<16.2f} | {n_flagged:<8} | {errors_caught:>2}/{total_errors:<2} ({err_recall:>4.1f}%) | {false_alarms:>2} correct    | {clean_acc:>12.1f}%")
    print()


def main():
    print("=" * 75)
    print("  Responde NLP — Multi-Task Test Evaluation & Confusion Diagnostics")
    print("=" * 75 + "\n")

    if not TEST_FILE.exists():
        print(f"❌ {TEST_FILE} not found.")
        sys.exit(1)

    if not (MODEL_DIR / "config.json").exists():
        print(f"❌ No saved model found at {MODEL_DIR}.")
        print("   Run `python train.py` first or place trained weights in model/.")
        sys.exit(1)

    from train import (
        MultiTaskRoberta,
        RespondeDataset,
        INTENT_LABELS, URGENCY_LABELS, INCIDENT_LABELS,
        ID2INTENT, ID2URGENCY, ID2INCIDENT,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    intent_labels = INTENT_LABELS
    urgency_labels = URGENCY_LABELS
    incident_labels = INCIDENT_LABELS

    label_map_file = MODEL_DIR / "label_maps.json"
    if label_map_file.exists():
        try:
            with open(label_map_file, encoding="utf-8") as f:
                lm = json.load(f)
                if "intent" in lm and "id2label" in lm["intent"]:
                    intent_labels = [lm["intent"]["id2label"][str(i)] for i in range(len(lm["intent"]["id2label"]))]
                elif "id2intent" in lm:
                    intent_labels = [lm["id2intent"][str(i)] for i in range(len(lm["id2intent"]))]

                if "urgency" in lm and "id2label" in lm["urgency"]:
                    urgency_labels = [lm["urgency"]["id2label"][str(i)] for i in range(len(lm["urgency"]["id2label"]))]
                elif "id2urgency" in lm:
                    urgency_labels = [lm["id2urgency"][str(i)] for i in range(len(lm["id2urgency"]))]

                if "incident_type" in lm and "id2label" in lm["incident_type"]:
                    incident_labels = [lm["incident_type"]["id2label"][str(i)] for i in range(len(lm["incident_type"]["id2label"]))]
                elif "id2incident" in lm:
                    incident_labels = [lm["id2incident"][str(i)] for i in range(len(lm["id2incident"]))]

                print(f"Loaded label_maps.json: {len(intent_labels)} intents, {len(urgency_labels)} urgencies, {len(incident_labels)} incident types.")
        except Exception as e:
            print(f"Warning: could not parse label_maps.json ({e}), using default labels.")

    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    config    = AutoConfig.from_pretrained(MODEL_DIR)
    model     = MultiTaskRoberta.from_pretrained(
        MODEL_DIR, config=config,
        n_intent   = len(intent_labels),
        n_urgency  = len(urgency_labels),
        n_incident = len(incident_labels),
        ignore_mismatched_sizes=True,
    ).to(device)
    model.eval()

    print(f"Loading test set from {TEST_FILE}...")
    test_records = load_jsonl(TEST_FILE)
    print(f"Total test samples: {len(test_records)}\n")

    test_ds     = RespondeDataset(test_records, tokenizer, max_length=128)
    test_loader = DataLoader(test_ds, batch_size=16, shuffle=False)

    all_intent, all_urgency, all_incident = [], [], []
    pred_intent, pred_urgency, pred_incident = [], [], []
    conf_intent, conf_urgency, conf_incident = [], [], []

    with torch.no_grad():
        for batch in test_loader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = {
                "intent":   batch["intent_label"].to(device),
                "urgency":  batch["urgency_label"].to(device),
                "incident": batch["incident_label"].to(device),
            }
            _, li, lu, linc = model(input_ids, attention_mask, labels=labels)

            prob_i   = torch.softmax(li, dim=-1)
            prob_u   = torch.softmax(lu, dim=-1)
            prob_inc = torch.softmax(linc, dim=-1)

            c_i, p_i     = prob_i.max(dim=-1)
            c_u, p_u     = prob_u.max(dim=-1)
            c_inc, p_inc = prob_inc.max(dim=-1)

            all_intent.extend(labels["intent"].cpu().tolist())
            all_urgency.extend(labels["urgency"].cpu().tolist())
            all_incident.extend(labels["incident"].cpu().tolist())

            pred_intent.extend(p_i.cpu().tolist())
            pred_urgency.extend(p_u.cpu().tolist())
            pred_incident.extend(p_inc.cpu().tolist())

            conf_intent.extend(c_i.cpu().tolist())
            conf_urgency.extend(c_u.cpu().tolist())
            conf_incident.extend(c_inc.cpu().tolist())

    tasks = [
        ("INTENT CLASSIFICATION", all_intent, pred_intent, conf_intent, intent_labels),
        ("URGENCY CLASSIFICATION", all_urgency, pred_urgency, conf_urgency, urgency_labels),
        ("INCIDENT TYPE CLASSIFICATION", all_incident, pred_incident, conf_incident, incident_labels),
    ]

    # 1. Classification Reports
    for title, y_true, y_pred, _, labels in tasks:
        print("=" * 75)
        print(f"{title} REPORT:")
        print("=" * 75)
        print(classification_report(y_true, y_pred, labels=list(range(len(labels))), target_names=labels, zero_division=0))

    # 2. Confusion Matrices
    for title, y_true, y_pred, _, labels in tasks:
        print_confusion_matrix(y_true, y_pred, labels, title)

    # 3. Confidence Calibration & Threshold Simulation
    for title, y_true, y_pred, confs, _ in tasks:
        print_calibration_analysis(y_true, y_pred, confs, title)

    print("✅ Evaluation, confusion matrix, & calibration diagnostics complete.")


if __name__ == "__main__":
    main()
