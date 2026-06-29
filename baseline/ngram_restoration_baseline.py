#!/usr/bin/env python3
"""
ngram_restoration_baseline.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Non-neural baselines for character-level lacuna restoration, to contextualize
the fine-tuned encoder results.

Two baselines:
  * Char n-gram (bidirectional) — a forward + backward character language model
    with interpolated absolute-discounting smoothing and backoff. The score of a
    candidate character c at a masked position is
        P_fwd(c | left_context) * P_bwd(c | right_context),
    i.e. the masked position is filled using context from both sides, mirroring
    the bidirectional MLM setting.
  * Unigram (frequency prior) — absolute lower bound; always predicts by global
    character frequency, ignoring context.

The output space matches the encoder character-level evaluation: Cyrillic
letters, the three corpus punctuation marks, and the inter-word space. Metrics
(Hit@1 / Hit@5 / CER) are computed exactly as in finetune_all_models_v3.py, so
the numbers drop straight into the same tables.

Evaluation mirrors the encoder pipeline:
  * Test A — random_mask applied to test_a.txt (same masking scheme/seed).
  * Test B — real editorial reconstructions from test_b.jsonl.
Each masked position is predicted independently (non-autoregressive), exactly
as the encoders are evaluated.

Self-contained: pure Python + numpy, no KenLM / external LM dependency, so it is
fully reproducible inside the project environment. Trains on train.txt in
seconds (character-level counts over a small corpus).

Usage:
    python ngram_restoration_baseline.py \
        --data_dir ../data/splits \
        --output_dir ../outputs/baseline \
        --max_order 5
"""

import argparse
import json
import logging
import re
from collections import defaultdict
from pathlib import Path

import Levenshtein
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)s  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

K_VALUES  = (1, 3, 5)
GAP_TOKEN = "[GAP]"
GAP_LEN   = len(GAP_TOKEN)
_HERE     = Path(__file__).parent.parent
BOS       = "\x02"   # sentinel for sequence start (never a real target)
EOS       = "\x03"   # sentinel for sequence end


# ── Text utilities (identical semantics to finetune_all_models_v3.py) ───────────

_GAP_RE   = re.compile(r"\[GAP\]")
_SPACE_RE = re.compile(r" {2,}")


def _is_cyrillic(ch: str) -> bool:
    return "\u0400" <= ch <= "\u052F" or "\uA640" <= ch <= "\uA69F"


_ALLOWED_PUNCT = {"·", ":", "+"}


def _is_allowed_char(ch: str) -> bool:
    """Single-character prediction target (no space)."""
    return _is_cyrillic(ch) or ch in _ALLOWED_PUNCT


def _is_maskable_char(ch: str) -> bool:
    """Masking target on Test A: Cyrillic + punctuation + space."""
    return _is_cyrillic(ch) or ch in _ALLOWED_PUNCT or ch == " "


def strip_gaps(text: str) -> str:
    return _SPACE_RE.sub(" ", _GAP_RE.sub(" ", text)).strip()


def masked_input_to_dashes(text: str) -> str:
    def _repl(m):
        return "-" * m.group(0).count("[MASK]")
    return re.sub(r"(\[MASK\])+", _repl, text)


def random_mask(text: str, *, mask_prob=0.08, span_p=0.35,
                rng: np.random.Generator) -> str:
    """Identical to the encoder pipeline so Test A masking matches exactly."""
    out, i = [], 0
    while i < len(text):
        if text[i:i + GAP_LEN] == GAP_TOKEN:
            out.append(GAP_TOKEN)
            i += GAP_LEN
        elif rng.random() < mask_prob:
            span = int(rng.geometric(span_p))
            j = i
            while (j < len(text) and (j - i) < span
                   and text[j] != "["
                   and _is_maskable_char(text[j])):
                j += 1
            out.append("-" * max(1, j - i))
            i = max(i + 1, j)
        else:
            out.append(text[i])
            i += 1
    return "".join(out)


# ── I/O ────────────────────────────────────────────────────────────────────────

def load_txt(path: Path) -> list[str]:
    return [l.strip() for l in path.read_text("utf-8").splitlines() if l.strip()]


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text("utf-8").splitlines() if l.strip()]


# ── Character n-gram LM with absolute discounting + backoff ─────────────────────

class CharNGram:
    """One-directional character n-gram model with interpolated absolute
    discounting (Ney et al., 1994).

    Trained on a list of strings. Provides prob(next_char | context) with
    interpolation/backoff down to the unigram distribution. The same class is
    used for the forward direction (predict char given left context) and, on
    reversed strings, for the backward direction (predict char given right
    context). A fixed discount D is subtracted from each non-zero count and the
    freed probability mass is interpolated with the lower-order model.
    """

    def __init__(self, max_order: int = 5, discount: float = 0.75):
        self.n = max_order
        self.D = discount
        # counts[k] : dict context(str len k) -> Counter(next_char -> count)
        self.counts = [defaultdict(lambda: defaultdict(int)) for _ in range(self.n)]
        self.vocab: set[str] = set()
        self.unigram: dict[str, float] = {}

    def _ngrams(self, seq: str):
        seq = BOS * (self.n - 1) + seq + EOS
        for i in range(self.n - 1, len(seq)):
            for k in range(self.n):           # order k+1 : context len k
                ctx = seq[i - k:i]
                nxt = seq[i]
                yield k, ctx, nxt

    def fit(self, texts: list[str]):
        for t in texts:
            for k, ctx, nxt in self._ngrams(t):
                self.counts[k][ctx][nxt] += 1
                if nxt not in (BOS, EOS):
                    self.vocab.add(nxt)
        # unigram distribution over the real vocabulary
        uni = self.counts[0][""]
        tot = sum(v for c, v in uni.items() if c not in (BOS, EOS))
        self.unigram = {c: v / tot for c, v in uni.items()
                        if c not in (BOS, EOS) and tot > 0}
        log.info("  CharNGram fitted: vocab=%d  max_order=%d", len(self.vocab), self.n)

    def _prob_order(self, k: int, ctx: str, ch: str) -> float:
        """Interpolated absolute-discounting probability at order k+1
        (context length k), backing off to order k when ctx is unseen."""
        if k == 0:
            return self.unigram.get(ch, 1e-8)
        table = self.counts[k].get(ctx)
        if not table:
            return self._prob_order(k - 1, ctx[1:], ch)
        total = sum(table.values())
        cnt   = table.get(ch, 0)
        n_types = len(table)
        disc  = self.D
        higher = max(cnt - disc, 0.0) / total if total else 0.0
        lam    = (disc * n_types) / total if total else 1.0
        lower  = self._prob_order(k - 1, ctx[1:], ch)
        return higher + lam * lower

    def prob(self, ctx: str, ch: str) -> float:
        ctx = ctx[-(self.n - 1):] if self.n > 1 else ""
        return self._prob_order(len(ctx), ctx, ch)


class BiCharNGram:
    """Bidirectional char n-gram: forward + backward models, multiplied."""

    def __init__(self, max_order: int = 5, discount: float = 0.75):
        self.fwd = CharNGram(max_order, discount)
        self.bwd = CharNGram(max_order, discount)
        self.candidates: list[str] = []

    def fit(self, texts: list[str]):
        clean = [strip_gaps(t) for t in texts]
        log.info("Fitting forward model...")
        self.fwd.fit(clean)
        log.info("Fitting backward model...")
        self.bwd.fit([t[::-1] for t in clean])
        # candidate output space = observed allowed chars (+ space if seen)
        self.candidates = sorted(
            c for c in self.fwd.vocab if _is_maskable_char(c)
        )
        log.info("  candidate output space: %d chars", len(self.candidates))

    def predict_topk(self, left: str, right: str, k: int) -> list[str]:
        """Score each candidate by P_fwd(c|left)*P_bwd(c|reversed right)."""
        right_rev = right[::-1]
        scores = []
        for c in self.candidates:
            pf = self.fwd.prob(left, c)
            pb = self.bwd.prob(right_rev, c)
            scores.append((pf * pb, c))
        scores.sort(reverse=True)
        return [c for _, c in scores[:k]]


class UnigramBaseline:
    """Absolute lower bound: predict by global frequency, no context."""

    def __init__(self):
        self.ranked: list[str] = []

    def fit(self, texts: list[str]):
        from collections import Counter
        cnt = Counter()
        for t in texts:
            for ch in strip_gaps(t):
                if _is_maskable_char(ch):
                    cnt[ch] += 1
        self.ranked = [c for c, _ in cnt.most_common()]

    def predict_topk(self, left: str, right: str, k: int) -> list[str]:
        return self.ranked[:k]


# ── Evaluation (mirrors eval_gaps in the encoder script) ────────────────────────

def eval_gaps(model, rows: list[dict], *, k_values=K_VALUES,
              context_window: int = 20,
              report_path: Path | None = None) -> dict:
    """rows: [{"text_eval": str with '-' at masked positions, "text_target": str}]"""
    hit         = {f"hit@{k}": 0 for k in k_values}
    total       = 0
    report_rows = [] if report_path else None
    doc_preds: dict[int, dict[int, str]] = {}

    for doc_idx, row in enumerate(rows):
        ev, tgt = row["text_eval"], row["text_target"]
        n = min(len(ev), len(tgt))
        gaps = [i for i in range(n) if ev[i] == "-"]
        if not gaps:
            continue
        doc_preds[doc_idx] = {}
        for pos in gaps:
            # context taken from the (masked) eval string, both sides
            left  = ev[max(0, pos - context_window):pos]
            right = ev[pos + 1:pos + 1 + context_window]
            # contexts must not contain dashes/gap markers leaking the answer;
            # dashes are fine as ordinary unseen chars but we strip them so the
            # n-gram sees only real characters on each side.
            left  = left.replace("-", "").replace(GAP_TOKEN, "")
            right = right.replace("-", "").replace(GAP_TOKEN, "")

            top = model.predict_topk(left, right, max(k_values))
            if not top:
                continue
            true_char = tgt[pos]
            pred_char = top[0]
            doc_preds[doc_idx][pos] = pred_char
            total += 1
            for k in k_values:
                if true_char in top[:k]:
                    hit[f"hit@{k}"] += 1

            if report_rows is not None:
                true_rank = next((r + 1 for r, c in enumerate(top)
                                  if c == true_char), None)
                report_rows.append({
                    "doc_idx": doc_idx, "position": pos,
                    "true_char": true_char, "pred_char": pred_char,
                    "match": pred_char == true_char,
                    "true_rank": true_rank,
                    "top5": "|".join(top[:5]),
                })

    # CER per document (same definition as the encoder script)
    cer_sum, n_docs = 0.0, 0
    for doc_idx, row in enumerate(rows):
        preds = doc_preds.get(doc_idx, {})
        if not preds:
            continue
        tgt = row["text_target"]
        n = min(len(row["text_eval"]), len(tgt))
        order = sorted(p for p in preds if p < n)
        true_str = "".join(tgt[p] for p in order)
        pred_str = "".join(preds[p] for p in order)
        if true_str:
            cer_sum += Levenshtein.distance(pred_str, true_str) / len(true_str)
            n_docs  += 1

    if report_rows is not None:
        pd.DataFrame(report_rows).to_csv(report_path, index=False,
                                         encoding="utf-8-sig")
        log.info("  report → %s", report_path)

    return {
        "total_gaps": total,
        "cer": round(cer_sum / n_docs, 4) if n_docs else 0.0,
        **{k: round(v / total, 4) if total else 0.0 for k, v in hit.items()},
    }


def build_test_a_rows(texts: list[str], *, seed=42,
                      mask_prob=0.08, span_p=0.35) -> list[dict]:
    rng = np.random.default_rng(seed)
    return [{"text_eval":   random_mask(strip_gaps(t), mask_prob=mask_prob,
                                        span_p=span_p, rng=rng),
             "text_target": strip_gaps(t)}
            for t in texts]


def build_test_b_rows(records: list[dict]) -> list[dict]:
    return [{"text_eval":   masked_input_to_dashes(r["masked_input"]),
             "text_target": r["target"]}
            for r in records]


# ── Main ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir",   default=str(_HERE / "data/splits"), type=Path)
    p.add_argument("--output_dir", default=str(_HERE / "outputs/baseline"), type=Path)
    p.add_argument("--max_order",  default=5, type=int)
    p.add_argument("--discount",   default=0.75, type=float)
    p.add_argument("--seed",       default=42, type=int)
    p.add_argument("--mask_prob",  default=0.08, type=float)
    p.add_argument("--span_p",     default=0.35, type=float)
    return p.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train = load_txt(args.data_dir / "train.txt")
    # Test A uses the ModernBERT (unsplit) text track, matching the char eval.
    test_a = load_txt(args.data_dir / "test_a.txt")
    test_b = load_jsonl(args.data_dir / "test_b.jsonl")

    # Validation split (eval.txt) is used to select max_order without touching
    # the test sets. It is masked with the same random_mask scheme as Test A but
    # with a different seed, so val and Test A never coincide.
    val_path = args.data_dir / "eval.txt"
    val_texts = load_txt(val_path) if val_path.exists() else []
    log.info("train=%d  val=%d  test_a=%d  test_b=%d",
             len(train), len(val_texts), len(test_a), len(test_b))

    rows_val = (build_test_a_rows(val_texts, seed=args.seed + 1,
                                  mask_prob=args.mask_prob, span_p=args.span_p)
                if val_texts else [])
    rows_a = build_test_a_rows(test_a, seed=args.seed,
                               mask_prob=args.mask_prob, span_p=args.span_p)
    rows_b = build_test_b_rows(test_b)

    summary = []

    # ── Char n-gram (bidirectional) ─────────────────────────────────────────────
    log.info("\n=== Char n-gram (bidirectional), order=%d ===", args.max_order)
    bi = BiCharNGram(max_order=args.max_order, discount=args.discount)
    bi.fit(train)
    v = eval_gaps(bi, rows_val) if rows_val else {}
    a = eval_gaps(bi, rows_a, report_path=args.output_dir / "ngram_test_a.csv")
    b = eval_gaps(bi, rows_b, report_path=args.output_dir / "ngram_test_b.csv")
    if v:
        log.info("VAL    (order=%d): hit@1=%.4f hit@5=%.4f cer=%.4f  <-- model selection",
                 args.max_order, v.get("hit@1", 0), v.get("hit@5", 0), v.get("cer", 0))
    log.info("Test A: %s", a)
    log.info("Test B: %s", b)
    summary.append({"model": f"CharNgram-bi(n={args.max_order})",
                    "val_hit@1": v.get("hit@1", 0.0),
                    "val_cer":   v.get("cer",   0.0),
                    **{f"a_{k}": v2 for k, v2 in a.items()},
                    **{f"b_{k}": v2 for k, v2 in b.items()}})

    # ── Unigram lower bound ─────────────────────────────────────────────────────
    log.info("\n=== Unigram (frequency prior) ===")
    uni = UnigramBaseline()
    uni.fit(train)
    a = eval_gaps(uni, rows_a)
    b = eval_gaps(uni, rows_b)
    log.info("Test A: %s", a)
    log.info("Test B: %s", b)
    summary.append({"model": "Unigram",
                    **{f"a_{k}": v for k, v in a.items()},
                    **{f"b_{k}": v for k, v in b.items()}})

    df = pd.DataFrame(summary)
    out = args.output_dir / "restoration_baseline_summary.csv"
    df.to_csv(out, index=False)
    log.info("\n%s", df.to_string(index=False))
    log.info("Summary → %s", out)


if __name__ == "__main__":
    main()