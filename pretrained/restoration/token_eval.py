#!/usr/bin/env python3
"""
token_eval.py
~~~~~~~~~~~~~
Shared token-level evaluation utilities for the birchbark restoration pipeline.
Imported by both:
  - finetune_tokens.py     (per-epoch callback: evaluate_test_a_tokens /
                            evaluate_test_b_tokens_hit for checkpoint selection)
  - eval_tokens_report.py  (eval_report_test_b to dump per-position CSVs)

Token-level prediction lets each model predict over its native subword
vocabulary, without the single-character constraint of the char-level mode.
"""

import logging
import re

import Levenshtein
import numpy as np
import torch

log = logging.getLogger(__name__)

K_VALUES  = (1, 5, 20)
GAP_TOKEN = "[GAP]"

_GAP_RE   = re.compile(r"\[GAP\]")
_SPACE_RE = re.compile(r" {2,}")


def strip_gaps(text: str) -> str:
    return _SPACE_RE.sub(" ", _GAP_RE.sub(" ", text)).strip()


def decode_tok(tokenizer, tid: int) -> str:
    """Decode a single token id to its surface string (BPE space markers
    surfaced as a normal space so they are not lost downstream)."""
    raw = tokenizer.decode([tid], skip_special_tokens=True,
                           clean_up_tokenization_spaces=False)
    return raw.replace("\u2581", " ").replace("\u0120", " ")


# ── Test A: token-level evaluation (random masking) ───────────────────────────

def evaluate_test_a_tokens(model, tokenizer, texts: list[str], *,
                            k_values=K_VALUES, seed: int = 42,
                            mask_prob: float = 0.08,
                            max_eval_gaps: int | None = None,
                            batch_size: int = 32,
                            normalizer=None) -> dict:
    rng    = np.random.default_rng(seed)
    device = next(model.parameters()).device
    SPECIAL = {tokenizer.cls_token_id, tokenizer.sep_token_id,
                tokenizer.pad_token_id, tokenizer.mask_token_id}
    norm = normalizer or (lambda t: t)

    all_items = []
    for text in texts:
        enc = tokenizer(norm(strip_gaps(text)), truncation=True,
                        max_length=512, return_tensors=None)
        ids  = enc["input_ids"]
        attn = enc["attention_mask"]
        for i, tid in enumerate(ids):
            if tid in SPECIAL:
                continue
            if rng.random() < mask_prob:
                all_items.append({"pos": i, "true_id": tid,
                                  "input_ids": ids, "attn": attn})

    if max_eval_gaps and len(all_items) > max_eval_gaps:
        idx = np.random.default_rng(42).choice(
            len(all_items), max_eval_gaps, replace=False)
        all_items = [all_items[i] for i in sorted(idx)]

    return _score_items(model, tokenizer, all_items,
                        k_values=k_values, batch_size=batch_size)


# ── Test B: per-position token prediction (metrics only) ──────────────────────

def evaluate_test_b_tokens_hit(model, tokenizer, records: list[dict], *,
                                k_values=K_VALUES,
                                max_eval_gaps: int | None = None,
                                batch_size: int = 32) -> dict:
    all_items = []
    for rec in records:
        for mi, ti in zip(rec["mask_indices"], rec["target_token_ids"]):
            all_items.append({"input_ids": rec["input_ids"],
                              "attn": rec["attention_mask"],
                              "pos": mi, "true_id": ti})

    if max_eval_gaps and len(all_items) > max_eval_gaps:
        idx = np.random.default_rng(42).choice(
            len(all_items), max_eval_gaps, replace=False)
        all_items = [all_items[i] for i in sorted(idx)]

    return _score_items(model, tokenizer, all_items,
                        k_values=k_values, batch_size=batch_size)


# ── Shared scoring core (Hit@K + token CER) ───────────────────────────────────

def _score_items(model, tokenizer, all_items, *, k_values, batch_size) -> dict:
    device = next(model.parameters()).device
    total = len(all_items)
    if total == 0:
        return {"total_gaps": 0, "tok_cer": 0.0,
                **{f"tok_hit@{k}": 0.0 for k in k_values}}

    hit = {f"tok_hit@{k}": 0 for k in k_values}
    cer_sum, n_cer = 0.0, 0
    model.eval()

    for bs in range(0, total, batch_size):
        batch   = all_items[bs: bs + batch_size]
        max_len = max(len(it["input_ids"]) for it in batch)
        pad_id  = tokenizer.pad_token_id
        b_ids   = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
        b_attn  = torch.zeros((len(batch), max_len), dtype=torch.long)

        for bi, it in enumerate(batch):
            masked = list(it["input_ids"])
            masked[it["pos"]] = tokenizer.mask_token_id
            b_ids[bi, :len(masked)]      = torch.tensor(masked)
            b_attn[bi, :len(it["attn"])] = torch.tensor(it["attn"])

        with torch.no_grad():
            logits = model(input_ids=b_ids.to(device),
                           attention_mask=b_attn.to(device)).logits

        for bi, it in enumerate(batch):
            top_ids = logits[bi, it["pos"]].topk(max(k_values)).indices.tolist()
            true_id = it["true_id"]
            pred_id = top_ids[0]

            for k in k_values:
                if true_id in top_ids[:k]:
                    hit[f"tok_hit@{k}"] += 1

            true_s = tokenizer.decode([true_id], skip_special_tokens=True,
                                      clean_up_tokenization_spaces=False).strip()
            pred_s = tokenizer.decode([pred_id], skip_special_tokens=True,
                                      clean_up_tokenization_spaces=False).strip()
            if true_s:
                cer_sum += Levenshtein.distance(pred_s, true_s) / len(true_s)
                n_cer   += 1

    return {
        "total_gaps": total,
        "tok_cer": round(cer_sum / n_cer, 4) if n_cer else 0.0,
        **{k: round(v / total, 4) for k, v in hit.items()},
    }


# ── Report generation (eval-only, per-position CSV) ───────────────────────────

def eval_report_test_b(model, tokenizer, records, *,
                       k_values=K_VALUES, batch_size=32,
                       report_path) -> dict:
    import pandas as pd
    device = next(model.parameters()).device

    items = []
    for doc_idx, rec in enumerate(records):
        for mi, ti in zip(rec["mask_indices"], rec["target_token_ids"]):
            items.append({"doc_idx": doc_idx, "input_ids": rec["input_ids"],
                          "attn": rec["attention_mask"], "pos": mi, "true_id": ti})

    total = len(items)
    hit = {f"tok_hit@{k}": 0 for k in k_values}
    rows = []
    model.eval()

    for bs in range(0, total, batch_size):
        batch = items[bs: bs + batch_size]
        max_len = max(len(it["input_ids"]) for it in batch)
        pad_id = tokenizer.pad_token_id
        b_ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
        b_attn = torch.zeros((len(batch), max_len), dtype=torch.long)
        for bi, it in enumerate(batch):
            masked = list(it["input_ids"])
            masked[it["pos"]] = tokenizer.mask_token_id
            b_ids[bi, :len(masked)] = torch.tensor(masked)
            b_attn[bi, :len(it["attn"])] = torch.tensor(it["attn"])

        with torch.no_grad():
            logits = model(input_ids=b_ids.to(device),
                           attention_mask=b_attn.to(device)).logits

        for bi, it in enumerate(batch):
            top_ids = logits[bi, it["pos"]].topk(max(k_values)).indices.tolist()
            true_id = it["true_id"]
            pred_id = top_ids[0]
            for k in k_values:
                if true_id in top_ids[:k]:
                    hit[f"tok_hit@{k}"] += 1

            true_tok = decode_tok(tokenizer, true_id)
            pred_tok = decode_tok(tokenizer, pred_id)
            top5 = [decode_tok(tokenizer, t) for t in top_ids[:5]]
            true_rank = next((r + 1 for r, t in enumerate(top_ids)
                              if t == true_id), None)

            rows.append({
                "doc_idx":      it["doc_idx"],
                "mask_index":   it["pos"],
                "true_token":   true_tok,
                "pred_token":   pred_tok,
                "true_id":      true_id,
                "pred_id":      pred_id,
                "match":        pred_id == true_id,
                "true_rank":    true_rank,
                "top5":         "|".join(top5),
                "n_chars_true": len(true_tok.strip()),
            })

        if (bs // batch_size) % 20 == 0:
            log.info("  %d/%d", min(bs + batch_size, total), total)

    pd.DataFrame(rows).to_csv(report_path, index=False, encoding="utf-8-sig")
    metrics = {"total_gaps": total,
               **{k: round(v / total, 4) if total else 0.0 for k, v in hit.items()}}
    log.info("  report → %s", report_path)
    log.info("  metrics: %s", metrics)
    return metrics
