#!/usr/bin/env python3
"""
tfidf_classification_baseline.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Non-neural baselines for the two auxiliary classification tasks, to
contextualize the probing results from frozen encoder embeddings.

For both tasks the baseline replaces encoder embeddings with character
n-gram TF-IDF features (char_wb, 2–5) and a linear classifier, evaluated on
exactly the same train/val/test splits as the probes (produced by
prepare_probe_data.py). Character n-grams are used rather than word n-grams
because the texts are fragmentary and orthographically non-standard, with few
repeated whole words — the same motivation cited for n-gram methods on
historical Slavic data in the literature.

Tasks:
  * Category — 4-class (letters/records/religious/other). TF-IDF → multinomial
    logistic regression with balanced class weights. Metrics: accuracy and
    macro-F1, identical to probe.py (eval_category).
  * Date — TF-IDF → 9-class logistic regression. Because logistic regression
    does not accept soft targets, the classifier is trained on the dominant
    date bin (argmax of date_target), unlike the probe, which is trained with
    KL divergence on the full soft distribution. At inference the predicted
    class distribution Q(l) is taken as the date distribution and scored with
    the same distance-to-interval metric as the probe (date_bins.date_distance),
    so the two are compared on an identical metric even though their training
    targets differ.

Hyperparameters (logistic-regression C) are tuned on the validation split,
mirroring the probes' val-based checkpoint selection: by macro-F1 for category
and by mean distance-to-interval for date.

Trivial lower bounds are also reported:
  * Category — majority class.
  * Date — constant prediction at the corpus mean target distribution.

Both inputs ("target" and "masked" text) are evaluated, matching the probe's
two text-field settings.

Usage:
    python tfidf_classification_baseline.py \
        --data_dir ../outputs/classification/data \
        --output_dir ../outputs/baseline
"""

import argparse
import json
import logging
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.date_bins import N_BINS, date_distance

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)s  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

_HERE = Path(__file__).parent.parent

CATEGORY_LABELS = ["letters", "records", "religious", "other"]
CAT_TO_IDX = {c: i for i, c in enumerate(CATEGORY_LABELS)}

# char n-gram TF-IDF settings
NGRAM_RANGE = (2, 5)
ANALYZER    = "char_wb"

# Logistic-regression inverse-regularization grid, tuned on the validation split
# (mirroring how the probes select their checkpoint on val).
C_GRID = [0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0]


# ── I/O ────────────────────────────────────────────────────────────────────────

def load_split(data_dir: Path, split: str) -> list[dict]:
    path = data_dir / f"{split}.jsonl"
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def get_text(rec: dict, field: str) -> str:
    """field in {'target','masked'}; fall back to target."""
    return rec.get(field) or rec.get("target") or ""


# ── Category task ──────────────────────────────────────────────────────────────

def eval_category(preds: np.ndarray, labels: np.ndarray) -> dict:
    """Identical metric definition to probe.py eval_category."""
    acc = float((preds == labels).mean())
    f1s = []
    for c in range(len(CATEGORY_LABELS)):
        tp = int(((preds == c) & (labels == c)).sum())
        fp = int(((preds == c) & (labels != c)).sum())
        fn = int(((preds != c) & (labels == c)).sum())
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec  = tp / (tp + fn) if (tp + fn) else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        f1s.append(f1)
    return {
        "accuracy": round(acc, 4),
        "macro_f1": round(float(np.mean(f1s)), 4),
        "per_class": {CATEGORY_LABELS[i]: round(f1s[i], 4)
                      for i in range(len(CATEGORY_LABELS))},
    }


def prepare_category(records: list[dict], field: str):
    texts, labels, idx = [], [], []
    for i, r in enumerate(records):
        cat = r.get("category_mapped")
        if cat in CAT_TO_IDX:
            texts.append(get_text(r, field))
            labels.append(CAT_TO_IDX[cat])
            idx.append(i)
    return texts, np.array(labels), idx


def run_category(train, val, test, field: str) -> dict:
    tr_t, tr_y, _ = prepare_category(train, field)
    va_t, va_y, _ = prepare_category(val,   field)
    te_t, te_y, _ = prepare_category(test,  field)

    vec = TfidfVectorizer(analyzer=ANALYZER, ngram_range=NGRAM_RANGE)
    Xtr = vec.fit_transform(tr_t)
    Xva = vec.transform(va_t)
    Xte = vec.transform(te_t)

    # Select C on the validation split by macro-F1 (as the probes select their
    # checkpoint on val). The chosen model is then evaluated once on the test set.
    best_C, best_val = None, -1.0
    for C in C_GRID:
        clf = LogisticRegression(max_iter=2000, class_weight="balanced",
                                 C=C, random_state=42)
        clf.fit(Xtr, tr_y)
        va_f1 = eval_category(clf.predict(Xva), va_y)["macro_f1"]
        if va_f1 > best_val:
            best_val, best_C = va_f1, C

    clf = LogisticRegression(max_iter=2000, class_weight="balanced",
                             C=best_C, random_state=42)
    clf.fit(Xtr, tr_y)
    te_pred = clf.predict(Xte)
    m = eval_category(te_pred, te_y)

    # trivial majority baseline
    maj = np.bincount(tr_y, minlength=len(CATEGORY_LABELS)).argmax()
    maj_m = eval_category(np.full_like(te_y, maj), te_y)

    log.info("  [category/%s] C=%.2g (val_f1=%.4f) | TF-IDF acc=%.4f macro_f1=%.4f "
             "| majority acc=%.4f macro_f1=%.4f",
             field, best_C, best_val, m["accuracy"], m["macro_f1"],
             maj_m["accuracy"], maj_m["macro_f1"])
    return {"tfidf": m, "majority": maj_m, "best_C": best_C}


# ── Date task ──────────────────────────────────────────────────────────────────

def eval_date(probs: np.ndarray, date_strs: list[str]) -> dict:
    """Identical to probe.py eval_date: distance-to-interval in years."""
    distances = []
    for pred_dist, ds in zip(probs, date_strs):
        d = date_distance(pred_dist, ds)
        if d is not None:
            distances.append(d)
    return {
        "mean_dist":   round(float(np.mean(distances)), 2),
        "median_dist": round(float(np.median(distances)), 2),
        "n":           len(distances),
    }


def prepare_date(records: list[dict], field: str):
    """Returns texts, hard bin labels (argmax of date_target), soft targets,
    date strings. Only records with a usable date_target are kept."""
    texts, hard, soft, dstr = [], [], [], []
    for r in records:
        dt = r.get("date_target")
        ds = r.get("date")
        if dt is not None and ds:
            texts.append(get_text(r, field))
            arr = np.asarray(dt, dtype=np.float32)
            hard.append(int(arr.argmax()))      # dominant bin for training labels
            soft.append(arr)
            dstr.append(ds)
    return texts, np.array(hard), np.array(soft, dtype=np.float32), dstr


def run_date(train, val, test, field: str) -> dict:
    tr_t, tr_hard, tr_soft, _   = prepare_date(train, field)
    va_t, va_hard, _, va_ds     = prepare_date(val,   field)
    te_t, _, _, te_ds           = prepare_date(test,  field)

    vec = TfidfVectorizer(analyzer=ANALYZER, ngram_range=NGRAM_RANGE)
    Xtr = vec.fit_transform(tr_t)
    Xva = vec.transform(va_t)
    Xte = vec.transform(te_t)

    def fit_predict_Q(C, X):
        """Fit a 9-class LR on dominant-bin labels and return the full 9-bin
        distribution Q(l) for the rows of X (expanding over absent bins)."""
        clf = LogisticRegression(max_iter=2000, class_weight="balanced",
                                 C=C, random_state=42)
        clf.fit(Xtr, tr_hard)
        proba = clf.predict_proba(X)
        Q = np.zeros((X.shape[0], N_BINS), dtype=np.float32)
        for j, cls in enumerate(clf.classes_):
            Q[:, cls] = proba[:, j]
        return Q / Q.sum(axis=1, keepdims=True).clip(min=1e-9)

    # Select C on validation by mean distance-to-interval (lower is better),
    # mirroring the probes' val-based model selection for date.
    best_C, best_val = None, float("inf")
    for C in C_GRID:
        Qva = fit_predict_Q(C, Xva)
        va_mean = eval_date(Qva, va_ds)["mean_dist"]
        if va_mean < best_val:
            best_val, best_C = va_mean, C

    Q = fit_predict_Q(best_C, Xte)
    m = eval_date(Q, te_ds)

    # trivial baseline: constant = mean train soft target, broadcast to all
    const = tr_soft.mean(axis=0, keepdims=True).repeat(len(te_ds), axis=0)
    const_m = eval_date(const, te_ds)

    log.info("  [date/%s] C=%.2g (val_mean=%.2f) | TF-IDF mean=%.2f median=%.2f (n=%d) "
             "| const mean=%.2f median=%.2f",
             field, best_C, best_val, m["mean_dist"], m["median_dist"], m["n"],
             const_m["mean_dist"], const_m["median_dist"])
    return {"tfidf": m, "constant": const_m, "best_C": best_C}


# ── Main ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir",
                   default=str(_HERE / "outputs/classification/data"), type=Path,
                   help="Folder with train.jsonl/val.jsonl/test.jsonl from prepare_probe_data.py")
    p.add_argument("--output_dir",
                   default=str(_HERE / "outputs/baseline"), type=Path)
    p.add_argument("--text_fields", nargs="+", default=["target", "masked"],
                   choices=["target", "masked"])
    return p.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train = load_split(args.data_dir, "train")
    val   = load_split(args.data_dir, "val")
    test  = load_split(args.data_dir, "test")
    log.info("train=%d  val=%d  test=%d", len(train), len(val), len(test))

    rows = []
    for field in args.text_fields:
        log.info("\n=== text field: %s ===", field)
        cat = run_category(train, val, test, field)
        dat = run_date(train, val, test, field)
        rows.append({
            "text_field": field,
            "cat_best_C":         cat["best_C"],
            "cat_tfidf_acc":      cat["tfidf"]["accuracy"],
            "cat_tfidf_macrof1":  cat["tfidf"]["macro_f1"],
            "cat_major_acc":      cat["majority"]["accuracy"],
            "cat_major_macrof1":  cat["majority"]["macro_f1"],
            "date_best_C":        dat["best_C"],
            "date_tfidf_mean":    dat["tfidf"]["mean_dist"],
            "date_tfidf_median":  dat["tfidf"]["median_dist"],
            "date_const_mean":    dat["constant"]["mean_dist"],
            "date_const_median":  dat["constant"]["median_dist"],
        })
        # per-class F1 detail
        log.info("    category per-class F1: %s", cat["tfidf"]["per_class"])

    df = pd.DataFrame(rows)
    out = args.output_dir / "classification_baseline_summary.csv"
    df.to_csv(out, index=False)
    log.info("\n%s", df.to_string(index=False))
    log.info("Summary → %s", out)


if __name__ == "__main__":
    main()