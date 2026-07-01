#!/usr/bin/env python3
"""
probe.py
~~~~~~~~
Linear probing on top of frozen model embeddings.

Two tasks:
  category  — 4-class classification (letters/records/religious/other)
              loss: weighted cross-entropy
              metrics: accuracy, macro-F1, per-class F1, confusion matrix

  date      — distribution prediction over 9 bins (50-year)
              loss: KL divergence (as in Aeneas, Assael et al. 2025)
              metrics: mean/median distance to ground-truth interval (years)

Outputs (RESULT_DIR = outputs/classification/results).
Per {model, text_field} (text_field ∈ {target, masked}):
  {model}_category_preds_{field}.csv    — category test predictions
  {model}_category_cm_{field}.png       — category confusion matrix
  {model}_category_{field}_probe.pth    — trained category probe weights
  {model}_date_preds_{field}.csv        — date test predictions
  {model}_date_{field}_probe.pth        — trained date probe weights
  tsne_{model}_{field}.pdf / .png       — t-SNE of test embeddings
Once per run:
  probe_results.json                    — all metrics (every model × field)

Usage:
  python probe.py
  python probe.py --models ModernBERT
  python probe.py --task category
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader, TensorDataset

from config_probe import (
    DATA_DIR, EMBED_DIR, RESULT_DIR, MODELS,
    CATEGORY_LABELS, CATEGORY_TO_IDX,
    PROBE_LR, PROBE_EPOCHS, PROBE_BATCH, PROBE_HIDDEN,
    PROBE_DROPOUT, PROBE_SEED,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from data.date_bins import N_BINS, predicted_year, date_distance, BIN_LABELS, BIN_MIDPOINTS

torch.manual_seed(PROBE_SEED)
np.random.seed(PROBE_SEED)
RESULT_DIR.mkdir(parents=True, exist_ok=True)

CAT_COLORS = ["#4c72b0", "#dd8452", "#55a868", "#c44e52"]


# ── Data loading ──────────────────────────────────────────────────────────────

def load_split_records(split: str) -> list[dict]:
    with open(DATA_DIR / f"{split}.jsonl", encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def load_embeddings(model_name: str, split: str,
                    text_field: str = "target") -> np.ndarray:
    return np.load(EMBED_DIR / f"{model_name}_{split}_{text_field}.npy")


# ── Probe architectures ───────────────────────────────────────────────────────

class LinearProbe(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Dropout(dropout),
            nn.Linear(in_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class MLPProbe(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int,
                 dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)


def build_probe(in_dim: int, out_dim: int) -> nn.Module:
    if PROBE_HIDDEN is None:
        return LinearProbe(in_dim, out_dim, PROBE_DROPOUT)
    return MLPProbe(in_dim, PROBE_HIDDEN, out_dim, PROBE_DROPOUT)


# ── Confusion matrix ──────────────────────────────────────────────────────────

def plot_confusion_matrix(cm: np.ndarray, labels: list[str],
                           title: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    plt.colorbar(im, ax=ax)

    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title, fontsize=11)

    # Annotate cells
    thresh = cm.max() / 2
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, str(cm[i, j]),
                    ha="center", va="center", fontsize=9,
                    color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"    Confusion matrix → {path}")


# ── t-SNE ─────────────────────────────────────────────────────────────────────

def plot_tsne(embs: np.ndarray, records: list[dict],
              model_name: str, path: Path) -> None:
    """Two subplots: coloured by category and by date bin."""

    # Compute t-SNE
    print(f"    Computing t-SNE ({len(embs)} points)...")
    tsne = TSNE(n_components=2, perplexity=30, random_state=PROBE_SEED,
                max_iter=1000)
    coords = tsne.fit_transform(embs)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f"{model_name} predictions", fontsize=13)

    # ── Left: colour by category ──────────────────────────────────────────
    ax = axes[0]
    ax.set_title("Category")
    for i, cat in enumerate(CATEGORY_LABELS):
        mask = np.array([r.get("category_mapped") == cat for r in records])
        ax.scatter(coords[mask, 0], coords[mask, 1],
                   c=CAT_COLORS[i], label=cat, s=12, alpha=0.7)
    # grey for None
    none_mask = np.array([r.get("category_mapped") is None for r in records])
    if none_mask.any():
        ax.scatter(coords[none_mask, 0], coords[none_mask, 1],
                   c="grey", label="—", s=8, alpha=0.4)
    ax.legend(fontsize=8, markerscale=2)
    ax.set_xticks([])
    ax.set_yticks([])

    # ── Right: colour by date bin (midpoint year) ─────────────────────────
    ax = axes[1]
    ax.set_title("Date (midpoint year)")

    date_colors = []
    for r in records:
        dt = r.get("date_target")
        if dt is not None:
            mid = float(np.dot(dt, BIN_MIDPOINTS))
        else:
            mid = float("nan")
        date_colors.append(mid)
    date_colors = np.array(date_colors)

    valid = ~np.isnan(date_colors)
    sc = ax.scatter(coords[valid, 0], coords[valid, 1],
                    c=date_colors[valid], cmap="plasma",
                    vmin=BIN_MIDPOINTS[0], vmax=BIN_MIDPOINTS[-1],
                    s=12, alpha=0.7)
    if (~valid).any():
        ax.scatter(coords[~valid, 0], coords[~valid, 1],
                   c="grey", s=8, alpha=0.4)
    plt.colorbar(sc, ax=ax, label="Year")
    ax.set_xticks([])
    ax.set_yticks([])

    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.savefig(path.with_suffix('.pdf'))
    plt.close()
    print(f"    t-SNE → {path}")


# ── Category task ─────────────────────────────────────────────────────────────

def prepare_category(records: list[dict], embs: np.ndarray):
    idx, labels = [], []
    for i, r in enumerate(records):
        cat = r.get("category_mapped")
        if cat is not None and cat in CATEGORY_TO_IDX:
            idx.append(i)
            labels.append(CATEGORY_TO_IDX[cat])
    return embs[idx], np.array(labels), idx


def compute_class_weights(labels: np.ndarray,
                           n_classes: int) -> torch.Tensor:
    counts = np.bincount(labels, minlength=n_classes).astype(float)
    counts = np.maximum(counts, 1)
    weights = counts.sum() / (n_classes * counts)
    return torch.tensor(weights, dtype=torch.float)


def predict_category(probe: nn.Module, embs: torch.Tensor,
                     device: torch.device) -> np.ndarray:
    probe.eval()
    with torch.no_grad():
        return probe(embs.to(device)).argmax(dim=-1).cpu().numpy()


def eval_category(preds: np.ndarray, labels: np.ndarray) -> dict:
    acc = float((preds == labels).mean())
    f1s = []
    for c in range(len(CATEGORY_LABELS)):
        tp = int(((preds == c) & (labels == c)).sum())
        fp = int(((preds == c) & (labels != c)).sum())
        fn = int(((preds != c) & (labels == c)).sum())
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        f1s.append(f1)
    return {
        "accuracy":  round(acc, 4),
        "macro_f1":  round(float(np.mean(f1s)), 4),
        "per_class": {CATEGORY_LABELS[i]: round(f1s[i], 4)
                      for i in range(len(CATEGORY_LABELS))},
    }


def run_category(model_name: str, device: torch.device,
                 text_field: str = "target") -> dict:
    print(f"\n  [{model_name}] Category probing...")

    train_rec = load_split_records("train")
    val_rec   = load_split_records("val")
    test_rec  = load_split_records("test")

    train_emb_np, train_labels, _ = prepare_category(
        train_rec, load_embeddings(model_name, "train", text_field))
    val_emb_np,   val_labels,   _ = prepare_category(
        val_rec,   load_embeddings(model_name, "val",   text_field))
    test_emb_np,  test_labels, test_idx = prepare_category(
        test_rec,  load_embeddings(model_name, "test",  text_field))
    test_rec_filtered = [test_rec[i] for i in test_idx]

    in_dim = train_emb_np.shape[1]
    probe  = build_probe(in_dim, len(CATEGORY_LABELS)).to(device)
    weights = compute_class_weights(train_labels, len(CATEGORY_LABELS)).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(probe.parameters(), lr=PROBE_LR)

    train_emb = torch.tensor(train_emb_np, dtype=torch.float32)
    train_lbl = torch.tensor(train_labels, dtype=torch.long)
    val_emb   = torch.tensor(val_emb_np,   dtype=torch.float32)
    val_lbl   = torch.tensor(val_labels,   dtype=torch.long)
    test_emb  = torch.tensor(test_emb_np,  dtype=torch.float32)

    loader = DataLoader(
        TensorDataset(train_emb, train_lbl),
        batch_size=PROBE_BATCH, shuffle=True,
    )

    best_val_f1, best_state = -1.0, None
    for epoch in range(PROBE_EPOCHS):
        probe.train()
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = criterion(probe(xb.to(device)), yb.to(device))
            loss.backward()
            optimizer.step()

        val_preds = predict_category(probe, val_emb, device)
        val_m = eval_category(val_preds, val_labels)
        if val_m["macro_f1"] > best_val_f1:
            best_val_f1 = val_m["macro_f1"]
            best_state  = {k: v.clone() for k, v in probe.state_dict().items()}

        if (epoch + 1) % 10 == 0:
            print(f"    epoch {epoch+1:3d}  val_f1={val_m['macro_f1']:.4f}")

    probe.load_state_dict(best_state)
    torch.save(best_state, RESULT_DIR / f"{model_name}_category_{text_field}_probe.pth")
    test_preds = predict_category(probe, test_emb, device)
    test_m = eval_category(test_preds, test_labels)
    print(f"  Test  acc={test_m['accuracy']:.4f}  macro_f1={test_m['macro_f1']:.4f}")

    # Confusion matrix
    n = len(CATEGORY_LABELS)
    cm = np.zeros((n, n), dtype=int)
    for t, p in zip(test_labels, test_preds):
        cm[t, p] += 1
    plot_confusion_matrix(
        cm, CATEGORY_LABELS,
        title=f"{model_name} — category [{text_field}]",
        path=RESULT_DIR / f"{model_name}_category_cm_{text_field}.png",
    )

    # Save predictions CSV (text_field in the name so the target/masked runs
    # do not overwrite each other).
    df = pd.DataFrame({
        "number":     [r.get("number", "") for r in test_rec_filtered],
        "true":       [CATEGORY_LABELS[l] for l in test_labels],
        "pred":       [CATEGORY_LABELS[p] for p in test_preds],
        "correct":    test_preds == test_labels,
    })
    df.to_csv(RESULT_DIR / f"{model_name}_category_preds_{text_field}.csv",
              index=False, encoding="utf-8-sig")

    return test_m


# ── Date task ─────────────────────────────────────────────────────────────────

def prepare_date(records: list[dict], embs: np.ndarray):
    idx, targets, date_strs = [], [], []
    for i, r in enumerate(records):
        dt = r.get("date_target")
        ds = r.get("date")
        if dt is not None and ds:
            idx.append(i)
            targets.append(dt)
            date_strs.append(ds)
    return embs[idx], np.array(targets, dtype=np.float32), date_strs


def eval_date(probs: np.ndarray, date_strs: list[str]) -> dict:
    distances = []
    pred_years = []
    for pred_dist, ds in zip(probs, date_strs):
        d = date_distance(pred_dist, ds)
        if d is not None:
            distances.append(d)
            pred_years.append(predicted_year(pred_dist))
    return {
        "mean_dist":   round(float(np.mean(distances)), 2),
        "median_dist": round(float(np.median(distances)), 2),
        "n":           len(distances),
    }, pred_years


def run_date(model_name: str, device: torch.device,
             text_field: str = "target") -> dict:
    print(f"\n  [{model_name}] Date probing...")

    train_rec = load_split_records("train")
    val_rec   = load_split_records("val")
    test_rec  = load_split_records("test")

    train_emb_np, train_targets, _       = prepare_date(
        train_rec, load_embeddings(model_name, "train", text_field))
    val_emb_np,   val_targets,   val_ds  = prepare_date(
        val_rec,   load_embeddings(model_name, "val",   text_field))
    test_emb_np,  test_targets,  test_ds = prepare_date(
        test_rec,  load_embeddings(model_name, "test",  text_field))

    in_dim = train_emb_np.shape[1]
    probe  = build_probe(in_dim, N_BINS).to(device)
    optimizer = torch.optim.Adam(probe.parameters(), lr=PROBE_LR)

    train_emb = torch.tensor(train_emb_np,  dtype=torch.float32)
    train_tgt = torch.tensor(train_targets, dtype=torch.float32)
    val_emb   = torch.tensor(val_emb_np,    dtype=torch.float32)
    test_emb  = torch.tensor(test_emb_np,   dtype=torch.float32)

    loader = DataLoader(
        TensorDataset(train_emb, train_tgt),
        batch_size=PROBE_BATCH, shuffle=True,
    )

    best_val_dist, best_state = float("inf"), None
    for epoch in range(PROBE_EPOCHS):
        probe.train()
        for xb, yb in loader:
            optimizer.zero_grad()
            log_probs = F.log_softmax(probe(xb.to(device)), dim=-1)
            tgt = yb.to(device).clamp(min=1e-8)
            tgt = tgt / tgt.sum(dim=-1, keepdim=True)
            loss = F.kl_div(log_probs, tgt, reduction="batchmean")
            loss.backward()
            optimizer.step()

        probe.eval()
        with torch.no_grad():
            val_probs = F.softmax(probe(val_emb.to(device)), dim=-1).cpu().numpy()
        val_m, _ = eval_date(val_probs, val_ds)
        if val_m["mean_dist"] < best_val_dist:
            best_val_dist = val_m["mean_dist"]
            best_state = {k: v.clone() for k, v in probe.state_dict().items()}

        if (epoch + 1) % 10 == 0:
            print(f"    epoch {epoch+1:3d}  val_mean_dist={val_m['mean_dist']:.1f} yr")

    probe.load_state_dict(best_state)
    torch.save(best_state, RESULT_DIR / f"{model_name}_date_{text_field}_probe.pth")
    probe.eval()
    with torch.no_grad():
        test_probs = F.softmax(
            probe(test_emb.to(device)), dim=-1).cpu().numpy()

    test_m, pred_years = eval_date(test_probs, test_ds)
    print(f"  Test  mean_dist={test_m['mean_dist']:.1f} yr  "
          f"median_dist={test_m['median_dist']:.1f} yr")

    # Save predictions CSV (text_field in the name so target/masked runs
    # do not overwrite each other).
    pd.DataFrame({
        "date_str":  test_ds,
        "pred_year": [round(y, 1) for y in pred_years],
        "distance":  [round(date_distance(p, ds) or 0, 1)
                      for p, ds in zip(test_probs, test_ds)],
    }).to_csv(RESULT_DIR / f"{model_name}_date_preds_{text_field}.csv",
              index=False, encoding="utf-8-sig")

    return test_m


# ── t-SNE (test set) ──────────────────────────────────────────────────────────

def run_tsne(model_name: str, text_field: str = "target") -> None:
    test_rec = load_split_records("test")
    test_emb = load_embeddings(model_name, "test", text_field)
    plot_tsne(
        test_emb, test_rec, f"{model_name} [{text_field}]",
        path=RESULT_DIR / f"tsne_{model_name}_{text_field}.pdf",
    )


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", default=list(MODELS.keys()),
                   choices=list(MODELS.keys()))
    p.add_argument("--task", choices=["category", "date", "both"],
                   default="both")
    p.add_argument("--text_fields", nargs="+",
                   default=["target", "masked"],
                   choices=["target", "masked", "original"])
    p.add_argument("--no_tsne", action="store_true",
                   help="Skip t-SNE (faster)")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    all_results = {}

    for model_name in args.models:
        all_results[model_name] = {}

        for text_field in args.text_fields:
            key = text_field
            all_results[model_name][key] = {}

            if args.task in ("category", "both"):
                all_results[model_name][key]["category"] = run_category(
                    model_name, device, text_field)

            if args.task in ("date", "both"):
                all_results[model_name][key]["date"] = run_date(
                    model_name, device, text_field)

            if not args.no_tsne:
                run_tsne(model_name, text_field)

    # Save JSON results
    out = RESULT_DIR / "probe_results.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\nResults → {out}")

    # Print summary table
    print("\n" + "=" * 80)
    print(f"{'Model':15s}  {'Field':8s}  {'Cat Acc':>8s}  {'Cat F1':>8s}  "
          f"{'Date mean':>10s}  {'Date med':>9s}")
    print("-" * 80)
    for m, fields in all_results.items():
        for field, res in fields.items():
            cat = res.get("category", {})
            dat = res.get("date", {})
            print(f"{m:15s}  {field:8s}  "
                  f"{cat.get('accuracy', '-'):>8}  "
                  f"{cat.get('macro_f1', '-'):>8}  "
                  f"{dat.get('mean_dist', '-'):>10}  "
                  f"{dat.get('median_dist', '-'):>9}")


if __name__ == "__main__":
    main()