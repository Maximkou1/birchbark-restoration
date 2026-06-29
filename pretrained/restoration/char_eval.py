#!/usr/bin/env python3
"""
char_eval.py
~~~~~~~~~~~~
Shared character-level evaluation utilities for the birchbark restoration
pipeline. Imported by both:
  - finetune_char.py     (uses evaluate_test_a / evaluate_test_b in the
                          per-epoch callback for checkpoint selection)
  - eval_char_report.py  (uses them with report_path to dump per-position CSVs)

Character-level prediction constrains the model's output to single characters
of the target space (Cyrillic letters, the three retained punctuation marks
· : +, and the inter-word space). WordPiece tokenizers have no single-token
space, so spaces stay unreachable for them — by design.
"""

import logging
import re
from pathlib import Path

import Levenshtein
import numpy as np
import pandas as pd
import torch

log = logging.getLogger(__name__)

K_VALUES  = (1, 3, 5)
GAP_TOKEN = "[GAP]"
GAP_LEN   = len(GAP_TOKEN)

_GAP_RE   = re.compile(r"\[GAP\]")
_SPACE_RE = re.compile(r" {2,}")

# Punctuation kept in the corpus (see normalization scheme): middle dot, colon, cross.
_ALLOWED_PUNCT = {"·", ":", "+"}


# ── Character predicates ───────────────────────────────────────────────────────

def _is_cyrillic(ch: str) -> bool:
    return "\u0400" <= ch <= "\u052F" or "\uA640" <= ch <= "\uA69F"


def _is_allowed_char(ch: str) -> bool:
    """Allowed as a single-character prediction target (no space)."""
    return _is_cyrillic(ch) or ch in _ALLOWED_PUNCT


def _is_maskable_char(ch: str) -> bool:
    """Allowed as a masking target on Test A (Cyrillic + punctuation + space)."""
    return _is_cyrillic(ch) or ch in _ALLOWED_PUNCT or ch == " "


def _decoded_single_char(tokenizer, tid: int) -> str | None:
    """Return the single allowed character a token decodes to, else None.

    Whitespace is handled BEFORE .strip(), since .strip() would erase it.
    Inter-word space is only representable for BPE tokenizers (e.g. ModernBERT);
    WordPiece (mBERT, BERTislav) has no single-token space, so it stays
    unreachable for those models — by design.
    """
    raw = tokenizer.decode([tid], skip_special_tokens=True,
                           clean_up_tokenization_spaces=False)
    if raw in (" ", "\u2581", "\u0120"):
        return " "
    s = raw.replace("\u2581", "").replace("\u0120", "").strip()
    if len(s) == 1 and _is_allowed_char(s):
        return s
    return None


# ── Text utilities ─────────────────────────────────────────────────────────────

def strip_gaps(text: str) -> str:
    return _SPACE_RE.sub(" ", _GAP_RE.sub(" ", text)).strip()


def masked_input_to_dashes(text: str) -> str:
    def _repl(m):
        return "-" * m.group(0).count("[MASK]")
    return re.sub(r"(\[MASK\])+", _repl, text)


def random_mask(text: str, *, mask_prob=0.08, span_p=0.35,
                rng: np.random.Generator) -> str:
    out, i = [], 0
    while i < len(text):
        if text[i:i + GAP_LEN] == GAP_TOKEN:
            out.append(GAP_TOKEN)
            i += GAP_LEN
        elif rng.random() < mask_prob:
            span = int(rng.geometric(span_p))
            j = i
            while (j < len(text) and (j - i) < span
                   and text[j] != "["          # do not run into [GAP]
                   and _is_maskable_char(text[j])):
                j += 1
            out.append("-" * max(1, j - i))
            i = max(i + 1, j)
        else:
            out.append(text[i])
            i += 1
    return "".join(out)


# ── Logits processor: constrain to single-character output ─────────────────────

class SingleCharCyrillicProcessor:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self._cache: dict = {}

    def _build_mask(self, vocab_size: int) -> torch.Tensor:
        mask = torch.zeros(vocab_size, dtype=torch.bool)
        for tid in range(vocab_size):
            if _decoded_single_char(self.tokenizer, tid) is not None:
                mask[tid] = True
        return mask

    def _get_mask(self, vocab_size: int, device) -> torch.Tensor:
        if vocab_size not in self._cache:
            self._cache[vocab_size] = self._build_mask(vocab_size)
        return self._cache[vocab_size].to(device)

    def __call__(self, input_ids: torch.Tensor,
                 scores: torch.Tensor) -> torch.Tensor:
        return scores.masked_fill(
            ~self._get_mask(scores.shape[-1], scores.device), float("-inf"))


# ── Evaluation core ────────────────────────────────────────────────────────────

def _decode_top(tokenizer, top_ids: list[int]) -> list[str]:
    chars = []
    for tid in top_ids:
        c = _decoded_single_char(tokenizer, tid)
        if c is not None:
            chars.append(c)
    return chars


def eval_gaps(model, tokenizer, rows: list[dict], *,
              k_values=K_VALUES,
              batch_size: int = 64,
              max_eval_gaps: int | None = None,
              report_path: Path | None = None,
              context_window: int = 20) -> dict:

    proc   = SingleCharCyrillicProcessor(tokenizer)
    device = next(model.parameters()).device
    mask_t = tokenizer.mask_token

    all_items = []
    doc_gaps  = {}

    for doc_idx, row in enumerate(rows):
        ev, tgt = row["text_eval"], row["text_target"]
        n    = min(len(ev), len(tgt))
        gaps = [i for i in range(n) if ev[i] == "-"]
        if not gaps:
            continue
        doc_gaps[doc_idx] = []
        for pos in gaps:
            prompt = ev[:pos] + mask_t + ev[pos + 1:]
            cs = max(0, pos - context_window)
            ce = min(len(tgt), pos + context_window + 1)
            all_items.append({
                "doc_idx": doc_idx,
                "pos":     pos,
                "true":    tgt[pos],
                "prompt":  prompt,
                "context": tgt[cs:ce],
            })
            doc_gaps[doc_idx].append(pos)

    if max_eval_gaps and len(all_items) > max_eval_gaps:
        rng_sample = np.random.default_rng(42)
        idx = rng_sample.choice(len(all_items), max_eval_gaps, replace=False)
        all_items = [all_items[i] for i in sorted(idx)]
        log.info("  eval limited to %d positions", max_eval_gaps)

    total       = len(all_items)
    hit         = {f"hit@{k}": 0 for k in k_values}
    report_rows = [] if report_path else None
    doc_preds: dict[int, dict[int, str]] = {d: {} for d in doc_gaps}

    model.eval()
    for batch_start in range(0, total, batch_size):
        batch   = all_items[batch_start: batch_start + batch_size]
        prompts = [item["prompt"] for item in batch]

        enc = tokenizer(prompts, return_tensors="pt", padding=True,
                        truncation=True, max_length=512)
        enc = {k: v.to(device) for k, v in enc.items()}

        with torch.no_grad():
            logits_all = model(**enc).logits

        for i, item in enumerate(batch):
            midx = (enc["input_ids"][i] == tokenizer.mask_token_id
                    ).nonzero(as_tuple=True)[0]
            if not len(midx):
                continue

            logits    = logits_all[i:i+1, int(midx[0]), :]
            logits    = proc(enc["input_ids"][i:i+1], logits)
            top_ids   = logits[0].topk(max(k_values)).indices.tolist()
            top_chars = _decode_top(tokenizer, top_ids)
            if not top_chars:
                continue

            true_char = item["true"]
            pred_char = top_chars[0]
            doc_preds[item["doc_idx"]][item["pos"]] = pred_char

            for k in k_values:
                if true_char in top_chars[:k]:
                    hit[f"hit@{k}"] += 1

            if report_rows is not None:
                true_rank = next((r + 1 for r, c in enumerate(top_chars)
                                  if c == true_char), None)
                report_rows.append({
                    "doc_idx":   item["doc_idx"],
                    "position":  item["pos"],
                    "context":   item["context"],
                    "true_char": true_char,
                    "pred_char": pred_char,
                    "match":     pred_char == true_char,
                    "true_rank": true_rank,
                    "top5":      "|".join(top_chars[:5]),
                })

        if (batch_start // batch_size) % 20 == 0:
            done = min(batch_start + batch_size, total)
            log.info("  eval %d/%d gaps (%.0f%%)", done, total,
                     100 * done / total if total else 0)

    cer_sum, n_docs = 0.0, 0
    for doc_idx, row in enumerate(rows):
        preds = doc_preds.get(doc_idx, {})
        if not preds:
            continue
        ev, tgt = row["text_eval"], row["text_target"]
        n = min(len(ev), len(tgt))
        true_str = "".join(tgt[p] for p in sorted(preds) if p < n)
        pred_str = "".join(preds[p] for p in sorted(preds) if p < n)
        if true_str:
            cer_sum += Levenshtein.distance(pred_str, true_str) / len(true_str)
            n_docs  += 1

    if report_rows is not None:
        pd.DataFrame(report_rows).to_csv(report_path, index=False,
                                         encoding="utf-8-sig")
        log.info("  report saved → %s", report_path)

    return {
        "total_gaps": total,
        "cer":        round(cer_sum / n_docs, 4) if n_docs else 0.0,
        **{k: round(v / total, 4) if total else 0.0 for k, v in hit.items()},
    }


def evaluate_test_a(model, tokenizer, texts: list[str], *,
                    seed=42, mask_prob=0.08, span_p=0.35,
                    max_eval_gaps: int | None = None,
                    report_path: Path | None = None) -> dict:
    rng  = np.random.default_rng(seed)
    rows = [{"text_eval":   random_mask(strip_gaps(t), mask_prob=mask_prob,
                                        span_p=span_p, rng=rng),
             "text_target": strip_gaps(t)}
            for t in texts]
    return eval_gaps(model, tokenizer, rows,
                     max_eval_gaps=max_eval_gaps, report_path=report_path)


def evaluate_test_b(model, tokenizer, records: list[dict], *,
                    max_eval_gaps: int | None = None,
                    report_path: Path | None = None) -> dict:
    rows = [{"text_eval":   masked_input_to_dashes(r["masked_input"]),
             "text_target": r["target"]}
            for r in records]
    return eval_gaps(model, tokenizer, rows,
                     max_eval_gaps=max_eval_gaps, report_path=report_path)
