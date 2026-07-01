#!/usr/bin/env python3
"""
evaluate_model.py
~~~~~~~~~~~~~~~~
Unified evaluation for DualBertForMaskedLM (char + word tokenization).

Since dual uses character-level tokenization (1 char = 1 token).

eval / test_a : PPL + char Hit@K + char CER  (collator masking)
test_b        : PPL + char Hit@K + char CER  (bracket masking, forward pass)
                    + span Hit@K + span macro-CER  (bracket masking, beam search)
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

from align_dual import load_vocab, align_char_to_word, SPECIAL_RE
from beam_search import beam_search_dual, DualBeam

_HERE = Path(__file__).resolve().parent              # from_scratch/DualEmbLM/
_ROOT = _HERE.parent.parent                          # repo root

SPAN_PATTERN = re.compile(
    r"\(([^)]+)\)|\[(?!(?:GAP|MASK|PAD|UNK|CLS|SEP)\]|CTX_)([^\]]+)\]"
)
K_VALUES = (1, 5, 20)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _topk_no_special(logits_1d: torch.Tensor, k: int,
                     unk_char_id: int | None = None) -> list[int]:
    """Top-k predictions with [UNK] excluded."""
    if unk_char_id is not None:
        logits_1d = logits_1d.clone()
        logits_1d[unk_char_id] = float("-inf")
    return logits_1d.topk(k).indices.tolist()


def _extract_spans(orig_text: str, target_text: str):
    spans, orig_idx = [], 0
    for m in SPAN_PATTERN.finditer(orig_text):
        span_text = m.group(1) or m.group(2)
        start = target_text.find(span_text, orig_idx)
        if start != -1:
            spans.append((start, start + len(span_text), span_text))
            orig_idx = start + len(span_text)
    return spans


def _build_char_to_token_map(text: str) -> dict[int, int]:
    token_pos = 1
    char_idx  = 0
    mapping   = {}

    for part in SPECIAL_RE.split(text.strip()):
        if not part:
            continue
        if SPECIAL_RE.fullmatch(part):
            for i in range(len(part)):
                mapping[char_idx + i] = token_pos
            char_idx  += len(part)
            token_pos += 1
        else:
            for ch in part:
                mapping[char_idx] = token_pos
                char_idx  += 1
                token_pos += 1

    return mapping


def _mask_span_dual(target_text, orig_text, char_vocab, word_vocab,
                    span_start, span_end, max_len=256):
    mask_id  = char_vocab["[MASK]"]
    unk_word = word_vocab["[UNK_WORD]"]

    enc = align_char_to_word(
        target_text, char_vocab, word_vocab,
        max_len=max_len, add_cls_sep=True,
    )
    c2t = _build_char_to_token_map(target_text)

    mask_indices  = []
    true_char_ids = []
    seen          = set()

    for ci in range(span_start, span_end):
        tp = c2t.get(ci)
        if tp is None or tp >= max_len or tp in seen:
            continue
        orig_id = enc["input_ids"][tp]
        if orig_id in (char_vocab.get("[CLS]", -1),
                       char_vocab.get("[SEP]", -1),
                       char_vocab.get("[PAD]", -1)):
            continue
        mask_indices.append(tp)
        true_char_ids.append(orig_id)
        seen.add(tp)

    if not mask_indices:
        return None, [], []

    masked_enc = dict(enc)
    masked_enc["input_ids"] = list(enc["input_ids"])
    masked_enc["word_ids"]  = list(enc["word_ids"])
    for tp in mask_indices:
        masked_enc["input_ids"][tp] = mask_id
        masked_enc["word_ids"][tp]  = unk_word

    return masked_enc, mask_indices, true_char_ids


# ── Collator-based evaluation (eval / test_a) ─────────────────────────────────

def evaluate_with_collator_dual(
    model,
    id_to_char: dict[int, str],
    dataset,
    collator,
    device,
    *,
    batch_size: int = 32,
    seed: int = 42,
    unk_char_id: int | None = None,
    output_path: Path | None = None,
) -> dict:
    torch.manual_seed(seed)
    loader = DataLoader(
        dataset, batch_size=batch_size,
        collate_fn=collator, shuffle=False,
    )

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
                    true_id = labels[b, pos].item()
                    top_ids = _topk_no_special(
                        logits[b, pos], max(K_VALUES), unk_char_id=unk_char_id)
                    pred_id = top_ids[0]

                    for k in K_VALUES:
                        if true_id in top_ids[:k]:
                            hit[k] += 1

                    true_ch = id_to_char.get(true_id, "")
                    pred_ch = id_to_char.get(pred_id, "")
                    cer = (Levenshtein.distance(pred_ch, true_ch) / max(len(true_ch), 1)
                           if true_ch else 0.0)
                    cer_total += cer
                    n_masked  += 1

                    true_rank = next(
                        (r + 1 for r, tid in enumerate(top_ids) if tid == true_id),
                        None,
                    )
                    report_rows.append({
                        "seq_pos":   pos,
                        "true":      true_ch,
                        "pred":      pred_ch,
                        "match":     pred_id == true_id,
                        "cer":       round(cer, 4),
                        "true_rank": true_rank,
                        "top5":      "|".join(id_to_char.get(t, "") for t in top_ids[:5]),
                        "hit@1":     true_id in top_ids[:1],
                        "hit@5":     true_id in top_ids[:5],
                        "hit@20":    true_id in top_ids[:20],
                    })

    ppl = math.exp(total_loss / n_batches) if n_batches else float("inf")

    if output_path and report_rows:
        pd.DataFrame(report_rows).to_csv(output_path, index=False, encoding="utf-8-sig")
        print(f"  Report ({len(report_rows)} rows) → {output_path}")

    return {
        "ppl":      round(ppl, 4),
        "hit@1":    round(hit[1]  / n_masked, 4) if n_masked else 0.0,
        "hit@5":    round(hit[5]  / n_masked, 4) if n_masked else 0.0,
        "hit@20":   round(hit[20] / n_masked, 4) if n_masked else 0.0,
        "char_cer": round(cer_total / n_masked, 4) if n_masked else 0.0,
        "n_masked": n_masked,
    }


# ── Bracket-based evaluation (test_b) ────────────────────────────────────────

def evaluate_test_b_dual(
    model,
    char_vocab: dict[str, int],
    word_vocab: dict[str, int],
    records: list[dict],
    device,
    *,
    beam_width:   int = 20,
    max_span_len: int = 10,
    max_len:      int = 256,
    unk_char_id:  int | None = None,
    output_path:  Path | None = None,
) -> dict:
    id_to_char = {v: k for k, v in char_vocab.items()}
    mask_id    = char_vocab["[MASK]"]
    unk_word   = word_vocab["[UNK_WORD]"]

    special_ids = {v for k, v in char_vocab.items()
                   if k.startswith("[") and k.endswith("]")}
    allowed_ids = [v for k, v in char_vocab.items()
                   if v not in special_ids]

    total_loss  = 0.0
    n_ppl_items = 0
    hit         = {k: 0 for k in K_VALUES}
    n_char      = 0
    cer_char    = 0.0

    cer_by_len:  dict[int, list[float]]     = {}
    hits_by_len: dict[int, dict[int, list]] = {k: {} for k in [1, 5, 20]}
    report_rows: list[dict] = []

    model.eval()
    with torch.no_grad():
        for record in tqdm(records, desc="  test_b"):
            orig_text   = record["original"]
            target_text = record["target"]

            spans = _extract_spans(orig_text, target_text)
            if not spans:
                continue

            # ── PPL ───────────────────────────────────────────────────────
            enc_full = align_char_to_word(
                target_text, char_vocab, word_vocab,
                max_len=max_len, add_cls_sep=True,
            )
            c2t        = _build_char_to_token_map(target_text)
            inp_full   = list(enc_full["input_ids"])
            wrd_full   = list(enc_full["word_ids"])
            labels_ppl = [-100] * max_len
            any_masked = False

            for span_start, span_end, _ in spans:
                for ci in range(span_start, span_end):
                    tp = c2t.get(ci)
                    if tp is None or tp >= max_len:
                        continue
                    orig_id = enc_full["input_ids"][tp]
                    if orig_id in special_ids:
                        continue
                    labels_ppl[tp] = orig_id
                    inp_full[tp]   = mask_id
                    wrd_full[tp]   = unk_word
                    any_masked     = True

            if any_masked:
                inp_t  = torch.tensor([inp_full], dtype=torch.long).to(device)
                wrd_t  = torch.tensor([wrd_full], dtype=torch.long).to(device)
                lbl_t  = torch.tensor([labels_ppl], dtype=torch.long).to(device)
                attn_t = torch.tensor(
                    [enc_full["attention_mask"]], dtype=torch.long).to(device)
                out = model(input_ids=inp_t, word_ids=wrd_t,
                            attention_mask=attn_t, labels=lbl_t)
                if out.loss is not None and not torch.isnan(out.loss):
                    total_loss  += out.loss.item()
                    n_ppl_items += 1

            # ── Per-span ──────────────────────────────────────────────────
            for span_start, span_end, true_span in spans:
                span_len = min(len(true_span), max_span_len)

                masked_enc, mask_indices, true_char_ids = _mask_span_dual(
                    target_text, orig_text, char_vocab, word_vocab,
                    span_start, span_end, max_len=max_len,
                )
                if not mask_indices:
                    continue

                inp_t  = torch.tensor([masked_enc["input_ids"]], dtype=torch.long).to(device)
                wrd_t  = torch.tensor([masked_enc["word_ids"]], dtype=torch.long).to(device)
                attn_t = torch.tensor([masked_enc["attention_mask"]], dtype=torch.long).to(device)

                out    = model(input_ids=inp_t, word_ids=wrd_t, attention_mask=attn_t)
                logits = out.logits[0]

                # Char-level metrics
                span_pred_ids = []
                for idx, true_id in zip(mask_indices, true_char_ids):
                    top_ids = _topk_no_special(
                        logits[idx], max(K_VALUES), unk_char_id=unk_char_id)
                    pred_id = top_ids[0]
                    span_pred_ids.append(pred_id)

                    for k in K_VALUES:
                        if true_id in top_ids[:k]:
                            hit[k] += 1

                    true_ch = id_to_char.get(true_id, "")
                    pred_ch = id_to_char.get(pred_id, "")
                    c = (Levenshtein.distance(pred_ch, true_ch) / max(len(true_ch), 1)
                         if true_ch else 0.0)
                    cer_char += c
                    n_char   += 1

                # Span-level metrics (beam search)
                beams = beam_search_dual(
                    torch.tensor(masked_enc["input_ids"], dtype=torch.long).to(device),
                    torch.tensor(masked_enc["word_ids"],  dtype=torch.long).to(device),
                    torch.tensor(masked_enc["attention_mask"], dtype=torch.long).to(device),
                    model, char_vocab,
                    beam_width=beam_width,
                    allowed_char_ids=allowed_ids,
                )

                pred_spans = []
                for beam in beams:
                    pred_chars = [id_to_char.get(beam.input_ids[i].item(), "")
                                  for i in mask_indices]
                    pred_spans.append("".join(pred_chars))

                top1_span = pred_spans[0] if pred_spans else ""
                span_cer  = Levenshtein.distance(top1_span, true_span) / max(len(true_span), 1)

                cer_by_len.setdefault(span_len, []).append(span_cer)
                for k in [1, 5, 20]:
                    hits_by_len[k].setdefault(span_len, []).append(
                        int(true_span in pred_spans[:k])
                    )

                tok_pred_str = "".join(id_to_char.get(i, "") for i in span_pred_ids)
                report_rows.append({
                    "true_span":      true_span,
                    "char_pred":      tok_pred_str,
                    "span_pred_top1": top1_span,
                    "span_len":       len(true_span),
                    "char_cer":       round(
                        Levenshtein.distance(tok_pred_str, true_span) / max(len(true_span), 1), 4),
                    "span_cer":       round(span_cer, 4),
                    "span_hit@1":     true_span in pred_spans[:1],
                    "span_hit@5":     true_span in pred_spans[:5],
                    "span_hit@20":    true_span in pred_spans[:20],
                    "top3_spans":     " | ".join(pred_spans[:3]),
                })

    ppl     = math.exp(total_loss / n_ppl_items) if n_ppl_items else float("inf")
    lengths = sorted(cer_by_len.keys())

    macro_span_cer = (sum(np.mean(cer_by_len[l]) for l in lengths) / len(lengths)
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
        "char_hit@1":     round(hit[1]  / n_char, 4) if n_char else 0.0,
        "char_hit@5":     round(hit[5]  / n_char, 4) if n_char else 0.0,
        "char_hit@20":    round(hit[20] / n_char, 4) if n_char else 0.0,
        "char_cer":       round(cer_char / n_char, 4) if n_char else 0.0,
        "n_chars":        n_char,
        "span_hit@1":     round(macro_hits[1],  4),
        "span_hit@5":     round(macro_hits[5],  4),
        "span_hit@20":    round(macro_hits[20], 4),
        "span_macro_cer": round(macro_span_cer, 4),
        "n_spans":        len(report_rows),
    }


def print_metrics(name: str, m: dict):
    print(f"\n  ── {name} ──")
    if "ppl"          in m: print(f"    PPL:             {m['ppl']:.4f}")
    if "hit@1"        in m: print(f"    Hit@1 (char):    {m['hit@1']:.4f}")
    if "char_hit@1"   in m: print(f"    Hit@1 (char):    {m['char_hit@1']:.4f}")
    if "hit@5"        in m: print(f"    Hit@5 (char):    {m['hit@5']:.4f}")
    if "char_hit@5"   in m: print(f"    Hit@5 (char):    {m['char_hit@5']:.4f}")
    if "hit@20"       in m: print(f"    Hit@20 (char):   {m['hit@20']:.4f}")
    if "char_hit@20"  in m: print(f"    Hit@20 (char):   {m['char_hit@20']:.4f}")
    if "char_cer"     in m: print(f"    CER (char):      {m['char_cer']:.4f}")
    if "span_hit@1"   in m:
        print(f"    Hit@1 (span):    {m['span_hit@1']:.4f}")
        print(f"    Hit@5 (span):    {m['span_hit@5']:.4f}")
        print(f"    Hit@20 (span):   {m['span_hit@20']:.4f}")
        print(f"    CER (span):      {m['span_macro_cer']:.4f}")

# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    from datasets import load_from_disk

    from collator import DualPhysicalDegradationCollator
    from build_char_tokenizer import SPECIAL_TOKENS
    from model import DualBertForMaskedLM

    parser = argparse.ArgumentParser()
    parser.add_argument("--model",
                        default=str(_ROOT / "outputs/from_scratch/DualEmbLM/final_model"))
    parser.add_argument("--char_vocab",
                        default=str(_HERE / "char_tokenizer/char_vocab.json"))
    parser.add_argument("--word_vocab",
                        default=str(_HERE / "word_vocab.json"))
    parser.add_argument("--dataset",
                        default=str(_ROOT / "outputs/from_scratch/DualEmbLM/dataset"))
    parser.add_argument("--test_b",
                        default=str(_ROOT / "data/splits/test_b.jsonl"))
    parser.add_argument("--out_dir",
                        default=str(_ROOT / "outputs/from_scratch/DualEmbLM"))
    parser.add_argument("--beam_width", default=20, type=int)
    parser.add_argument("--batch_size", default=32, type=int)
    args = parser.parse_args()

    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device: {device}")
    print(f"Model:  {args.model}")

    char_vocab = load_vocab(args.char_vocab)
    word_vocab = load_vocab(args.word_vocab)
    id_to_char = {v: k for k, v in char_vocab.items()}
    unk_char_id = char_vocab.get("[UNK]")

    model = DualBertForMaskedLM.from_pretrained(args.model).to(device)
    model.eval()

    special_ids = [char_vocab[t] for t in SPECIAL_TOKENS if t in char_vocab]
    collator = DualPhysicalDegradationCollator(
        mask_token_id=char_vocab["[MASK]"],
        pad_token_id=char_vocab["[PAD]"],
        unk_word_id=word_vocab["[UNK_WORD]"],
        unk_char_id=unk_char_id,
        vocab_char_size=len(char_vocab),
        special_token_ids=special_ids,
        mlm_prob=0.08,
        max_span=3,
        edge_prob=0.1,
        add_random_gaps=False,
    )

    dataset = load_from_disk(args.dataset)
    summary = {}

    if "test_a" in dataset:
        print("\nEvaluating test_a...")
        m = evaluate_with_collator_dual(
            model, id_to_char, dataset["test_a"], collator, device,
            batch_size=args.batch_size, unk_char_id=unk_char_id,
            output_path=out_dir / "report_test_a.csv",
        )
        print_metrics("test_a", m)
        summary["test_a"] = m

    print("\nEvaluating test_b (beam search)...")
    records = []
    with open(args.test_b, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                records.append({"original": r["original"], "target": r["target"]})

    m = evaluate_test_b_dual(
        model, char_vocab, word_vocab, records, device,
        beam_width=args.beam_width, unk_char_id=unk_char_id,
        output_path=out_dir / "report_test_b.csv",
    )
    print_metrics("test_b", m)
    summary["test_b"] = m

    summary_path = out_dir / "eval_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n  Summary → {summary_path}")