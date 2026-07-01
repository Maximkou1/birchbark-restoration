#!/usr/bin/env python3
"""
evaluate_model.py
~~~~~~~~~~~~~~~~~
Unified RoFormer evaluation.

Metrics:
  eval / test_a  — PPL + token Hit@K + token CER   (collator masking)
  test_b         — PPL + token Hit@K + token CER    (bracket masking, forward pass)
                       + span Hit@K + span macro-CER (bracket masking, beam search)
"""

import json
import math
import re
from pathlib import Path

import Levenshtein
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, RoFormerForMaskedLM

from collator import RoFormerPhysicalDegradationCollator
from beam_search import beam_search, Beam

_HERE = Path(__file__).resolve().parent              # from_scratch/RoFormerBPE/
_ROOT = _HERE.parent.parent                          # repo root

SPAN_PATTERN = re.compile(
    r"\(([^)]+)\)|\[(?!(?:GAP|MASK|PAD|UNK|CLS|SEP)\]|CTX_)([^\]]+)\]"
)
K_VALUES = (1, 5, 20)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _decode_token(tokenizer, token_id: int) -> str:
    return tokenizer.decode(
        [token_id], skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    ).strip()


def _token_cer(pred_id: int, true_id: int, tokenizer) -> float:
    pred_str = _decode_token(tokenizer, pred_id)
    true_str = _decode_token(tokenizer, true_id)
    if not true_str:
        return 0.0
    return Levenshtein.distance(pred_str, true_str) / len(true_str)


def _extract_spans(orig_text: str, target_text: str):
    spans, orig_idx = [], 0
    for m in SPAN_PATTERN.finditer(orig_text):
        span_text = m.group(1) or m.group(2)
        start = target_text.find(span_text, orig_idx)
        if start != -1:
            spans.append((start, start + len(span_text), span_text))
            orig_idx = start + len(span_text)
    return spans


def _mask_span(input_ids, offsets, span_start, span_end, mask_token_id):
    masked = input_ids.clone()
    mask_indices = []
    for idx, (tok_start, tok_end) in enumerate(offsets):
        if tok_start == tok_end:
            continue
        if max(tok_start, span_start) < min(tok_end, span_end):
            masked[idx] = mask_token_id
            mask_indices.append(idx)
    return masked, mask_indices


def _get_gap_id(tokenizer) -> int | None:
    if "[GAP]" in tokenizer.get_vocab():
        return tokenizer.convert_tokens_to_ids("[GAP]")
    return None


def _mask_gap(logits_1d: torch.Tensor, gap_id: int | None) -> torch.Tensor:
    """Return cloned logits with [GAP] token set to -inf."""
    if gap_id is None:
        return logits_1d
    logits_1d = logits_1d.clone()
    logits_1d[gap_id] = float("-inf")
    return logits_1d


# ── Collator-based evaluation (eval / test_a) ─────────────────────────────────

def evaluate_with_collator(
    model,
    tokenizer,
    dataset,
    collator,
    device,
    *,
    batch_size: int = 8,
    seed: int = 42,
    output_path: Path | None = None,
) -> dict:
    torch.manual_seed(seed)
    loader = DataLoader(
        dataset, batch_size=batch_size,
        collate_fn=collator, shuffle=False,
    )

    gap_id = _get_gap_id(tokenizer)

    total_loss = 0.0
    n_batches  = 0
    hit        = {k: 0 for k in K_VALUES}
    n_masked   = 0
    cer_total  = 0.0
    report_rows: list[dict] = []

    model.eval()
    with torch.no_grad():
        for batch in tqdm(loader, desc="  eval"):
            batch  = {k: v.to(device) for k, v in batch.items()}
            output = model(**batch)
            total_loss += output.loss.item()
            n_batches  += 1

            labels = batch["labels"]
            logits = output.logits

            for b in range(labels.shape[0]):
                for pos in (labels[b] != -100).nonzero(as_tuple=True)[0].tolist():
                    true_id    = labels[b, pos].item()
                    pos_logits = _mask_gap(logits[b, pos], gap_id)
                    top_ids    = pos_logits.topk(max(K_VALUES)).indices.tolist()

                    for k in K_VALUES:
                        if true_id in top_ids[:k]:
                            hit[k] += 1

                    pred_id   = top_ids[0]
                    cer       = _token_cer(pred_id, true_id, tokenizer)
                    cer_total += cer
                    n_masked  += 1

                    true_rank = next(
                        (r + 1 for r, tid in enumerate(top_ids) if tid == true_id),
                        None,
                    )
                    report_rows.append({
                        "seq_pos":   pos,
                        "true":      _decode_token(tokenizer, true_id),
                        "pred":      _decode_token(tokenizer, pred_id),
                        "match":     pred_id == true_id,
                        "cer":       round(cer, 4),
                        "true_rank": true_rank,
                        "top5":      "|".join(_decode_token(tokenizer, t) for t in top_ids[:5]),
                        "hit@1":     true_id in top_ids[:1],
                        "hit@5":     true_id in top_ids[:5],
                        "hit@20":    true_id in top_ids[:20],
                    })

    ppl = math.exp(total_loss / n_batches) if n_batches else float("inf")

    if output_path and report_rows:
        pd.DataFrame(report_rows).to_csv(output_path, index=False, encoding="utf-8-sig")
        print(f"  Report ({len(report_rows)} rows) → {output_path}")

    return {
        "ppl":       round(ppl, 4),
        "hit@1":     round(hit[1]  / n_masked, 4) if n_masked else 0.0,
        "hit@5":     round(hit[5]  / n_masked, 4) if n_masked else 0.0,
        "hit@20":    round(hit[20] / n_masked, 4) if n_masked else 0.0,
        "token_cer": round(cer_total / n_masked, 4) if n_masked else 0.0,
        "n_masked":  n_masked,
    }


# ── Bracket-based evaluation (test_b) ────────────────────────────────────────

def evaluate_test_b(
    model,
    tokenizer,
    records: list[dict],
    device,
    *,
    beam_width:   int = 20,
    max_span_len: int = 10,
    output_path:  Path | None = None,
) -> dict:
    gap_id = _get_gap_id(tokenizer)
    banned = [gap_id] if gap_id is not None and gap_id < model.config.vocab_size else []

    total_loss  = 0.0
    n_ppl_items = 0
    tok_hit     = {k: 0 for k in K_VALUES}
    tok_cer     = 0.0
    n_tok       = 0

    cer_by_len:  dict[int, list[float]]     = {}
    hits_by_len: dict[int, dict[int, list]] = {k: {} for k in [1, 5, 20]}

    report_rows: list[dict] = []

    model.eval()
    with torch.no_grad():
        for record in tqdm(records, desc="  test_b"):
            orig_text   = record["original"]
            target_text = record["target"]

            encoded   = tokenizer(
                target_text,
                return_offsets_mapping=True,
                return_tensors="pt",
                return_token_type_ids=False,
                truncation=True,
                max_length=512,
            )
            input_ids = encoded["input_ids"][0].clone()
            offsets   = encoded["offset_mapping"][0]

            spans = _extract_spans(orig_text, target_text)
            if not spans:
                continue

            # ── PPL: mask all spans at once ───────────────────────────────
            full_masked = input_ids.clone()
            labels_ppl  = torch.full_like(input_ids, -100)
            any_masked  = False

            for span_start, span_end, _ in spans:
                _, midxs = _mask_span(
                    input_ids, offsets, span_start, span_end,
                    tokenizer.mask_token_id,
                )
                for idx in midxs:
                    full_masked[idx] = tokenizer.mask_token_id
                    tid = input_ids[idx].item()
                    labels_ppl[idx] = tid if tid < model.config.vocab_size else -100
                    any_masked = True

            if any_masked:
                out = model(
                    full_masked.unsqueeze(0).to(device),
                    labels=labels_ppl.unsqueeze(0).to(device),
                )
                if out.loss is not None and not torch.isnan(out.loss):
                    total_loss  += out.loss.item()
                    n_ppl_items += 1

            # ── Token + Span metrics per span ─────────────────────────────
            for span_start, span_end, true_span in spans:
                span_len = min(len(true_span), max_span_len)

                masked_ids, mask_indices = _mask_span(
                    input_ids, offsets, span_start, span_end,
                    tokenizer.mask_token_id,
                )
                if not mask_indices:
                    continue

                out    = model(masked_ids.unsqueeze(0).to(device))
                logits = out.logits[0]

                span_tok_preds = []
                span_tok_trues = []

                for idx in mask_indices:
                    true_id = input_ids[idx].item()
                    if true_id >= model.config.vocab_size:
                        continue
                    pos_logits = _mask_gap(logits[idx], gap_id)
                    top_ids    = pos_logits.topk(max(K_VALUES)).indices.tolist()
                    pred_id    = top_ids[0]

                    for k in K_VALUES:
                        if true_id in top_ids[:k]:
                            tok_hit[k] += 1

                    cer = _token_cer(pred_id, true_id, tokenizer)
                    tok_cer += cer
                    n_tok   += 1

                    span_tok_preds.append(pred_id)
                    span_tok_trues.append(true_id)

                beams = beam_search(
                    masked_ids.to(device), model, tokenizer,
                    beam_width=beam_width,
                    banned_token_ids=banned,
                )

                pred_spans = []
                for beam in beams:
                    pred_ids  = [beam.input_ids[i].item() for i in mask_indices]
                    pred_text = tokenizer.decode(
                        pred_ids, skip_special_tokens=True,
                        clean_up_tokenization_spaces=False,
                    ).strip()
                    pred_spans.append(pred_text)

                top1_span = pred_spans[0] if pred_spans else ""
                span_cer  = Levenshtein.distance(top1_span, true_span) / max(len(true_span), 1)

                cer_by_len.setdefault(span_len, []).append(span_cer)
                for k in [1, 5, 20]:
                    hits_by_len[k].setdefault(span_len, []).append(
                        int(true_span in pred_spans[:k])
                    )

                tok_pred_str = tokenizer.decode(
                    span_tok_preds, skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                ).strip()
                tok_true_str = tokenizer.decode(
                    span_tok_trues, skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                ).strip()

                report_rows.append({
                    "true_span":      true_span,
                    "tok_pred":       tok_pred_str,
                    "span_pred_top1": top1_span,
                    "span_len":       len(true_span),
                    "tok_cer":        round(
                        Levenshtein.distance(tok_pred_str, true_span) / max(len(true_span), 1), 4
                    ),
                    "span_cer":       round(span_cer, 4),
                    "span_hit@1":     true_span in pred_spans[:1],
                    "span_hit@5":     true_span in pred_spans[:5],
                    "span_hit@20":    true_span in pred_spans[:20],
                    "top3_spans":     " | ".join(pred_spans[:3]),
                })

    ppl = math.exp(total_loss / n_ppl_items) if n_ppl_items else float("inf")

    lengths    = sorted(cer_by_len.keys())
    macro_cer  = (sum(np.mean(cer_by_len[l]) for l in lengths) / len(lengths)
                  if lengths else 0.0)
    macro_hits = {
        k: (sum(np.mean(hits_by_len[k].get(l, [0])) for l in lengths) / len(lengths)
            if lengths else 0.0)
        for k in [1, 5, 20]
    }

    if output_path and report_rows:
        pd.DataFrame(report_rows).to_csv(output_path, index=False, encoding="utf-8-sig")
        print(f"  Report ({len(report_rows)} rows) → {output_path}")

    return {
        "ppl":            round(ppl, 4),
        "tok_hit@1":      round(tok_hit[1]  / n_tok, 4) if n_tok else 0.0,
        "tok_hit@5":      round(tok_hit[5]  / n_tok, 4) if n_tok else 0.0,
        "tok_hit@20":     round(tok_hit[20] / n_tok, 4) if n_tok else 0.0,
        "tok_cer":        round(tok_cer / n_tok, 4) if n_tok else 0.0,
        "n_tok":          n_tok,
        "span_hit@1":     round(macro_hits[1],  4),
        "span_hit@5":     round(macro_hits[5],  4),
        "span_hit@20":    round(macro_hits[20], 4),
        "span_macro_cer": round(macro_cer, 4),
        "n_spans":        len(report_rows),
    }


def print_metrics(name: str, m: dict):
    print(f"\n  ── {name} ──")
    if "ppl"          in m: print(f"    PPL:           {m['ppl']:.4f}")
    if "hit@1"        in m: print(f"    Hit@1 (tok):   {m['hit@1']:.4f}")
    if "tok_hit@1"    in m: print(f"    Hit@1 (tok):   {m['tok_hit@1']:.4f}")
    if "hit@5"        in m: print(f"    Hit@5 (tok):   {m['hit@5']:.4f}")
    if "tok_hit@5"    in m: print(f"    Hit@5 (tok):   {m['tok_hit@5']:.4f}")
    if "hit@20"       in m: print(f"    Hit@20 (tok):  {m['hit@20']:.4f}")
    if "tok_hit@20"   in m: print(f"    Hit@20 (tok):  {m['tok_hit@20']:.4f}")
    if "token_cer"    in m: print(f"    CER (tok):     {m['token_cer']:.4f}")
    if "tok_cer"      in m: print(f"    CER (tok):     {m['tok_cer']:.4f}")
    if "span_hit@1"   in m:
        print(f"    Hit@1 (span):  {m['span_hit@1']:.4f}")
        print(f"    Hit@5 (span):  {m['span_hit@5']:.4f}")
        print(f"    Hit@20 (span): {m['span_hit@20']:.4f}")
        print(f"    CER (span):    {m['span_macro_cer']:.4f}")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    from datasets import load_from_disk

    parser = argparse.ArgumentParser()
    parser.add_argument("--model",      default=str(_HERE / "outputs/checkpoints/checkpoint-16020"))
    parser.add_argument("--tokenizer",  default=str(_HERE / "tokenizer"))
    parser.add_argument("--dataset",    default=str(_HERE / "dataset"))
    parser.add_argument("--test_b",     default=str(_ROOT / "data/splits/test_b.jsonl"))
    parser.add_argument("--out_dir",    default=str(_HERE / "outputs"))
    parser.add_argument("--beam_width", default=20, type=int)
    parser.add_argument("--batch_size", default=8,  type=int)
    args = parser.parse_args()

    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device:    {device}")
    print(f"Model:     {args.model}")
    print(f"Tokenizer: {args.tokenizer}")

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    tokenizer.add_special_tokens({"additional_special_tokens": ["[GAP]"]})
    model = RoFormerForMaskedLM.from_pretrained(args.model).to(device)
    model.eval()

    collator = RoFormerPhysicalDegradationCollator(
        tokenizer=tokenizer, mlm_prob=0.08, max_span=3,
        edge_prob=0.1, add_random_gaps=False,
    )

    dataset = load_from_disk(args.dataset)
    summary = {}

    if "test_a" in dataset:
        print("\nEvaluating test_a...")
        m = evaluate_with_collator(
            model, tokenizer, dataset["test_a"], collator, device,
            batch_size=args.batch_size,
            output_path=out_dir / "report_test_a.csv",
        )
        print_metrics("test_a", m)
        summary["test_a"] = m

    print("\nEvaluating test_b...")
    records = []
    with open(args.test_b, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                records.append({"original": r["original"], "target": r["target"]})

    m = evaluate_test_b(
        model, tokenizer, records, device,
        beam_width=args.beam_width,
        output_path=out_dir / "report_test_b.csv",
    )
    print_metrics("test_b", m)
    summary["test_b"] = m

    summary_path = out_dir / "eval_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n  Summary → {summary_path}")