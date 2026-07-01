"""
beam_search.py
~~~~~~~~~~~~~~
Confidence-ordered (easy-first) beam search for DualBertForMaskedLM,
in the style of Ithaca/Aeneas: at each step the highest-confidence masked
position is filled first, and the context is updated before the next step.

The model takes two tensors — input_ids (characters) and word_ids (words).
When a masked position is filled, its word_id is set to [UNK_WORD], since the
word is not yet known at prediction time.

Used by evaluate_model.py for span-level Test B evaluation.
"""

import math
from dataclasses import dataclass, field
from pathlib import Path

import Levenshtein
import numpy as np
import pandas as pd
import torch

_HERE = Path(__file__).resolve().parent

@dataclass
class DualBeam:
    input_ids: torch.Tensor   # [seq_len] — символьные id
    word_ids:  torch.Tensor   # [seq_len] — словарные id
    log_prob:  float = 0.0
    filled:    list[tuple] = field(default_factory=list)  # (pos, char_id)

def beam_search_dual(
    input_ids:  torch.Tensor,      # [seq_len], на нужном device
    word_ids:   torch.Tensor,      # [seq_len]
    attention_mask: torch.Tensor,  # [seq_len]
    model,
    char_vocab:  dict[str, int],
    *,
    beam_width:       int = 20,
    temperature:      float = 1.0,
    allowed_char_ids: list[int] | None = None,  # допустимые символы
    banned_char_ids:  list[int] | None = None,  # запрещённые (GAP и т.п.)
) -> list[DualBeam]:
    """
    Confidence-ordered beam search для DualBertForMaskedLM.
    Возвращает beam_width бимов, отсортированных по убыванию log-probability.
    """
    device   = input_ids.device
    mask_id  = char_vocab["[MASK]"]
    unk_word = 1  # [UNK_WORD] — позиция в word_vocab

    allowed = set(allowed_char_ids) if allowed_char_ids else None
    banned  = set(banned_char_ids  or [])
    if "[GAP]" in char_vocab:
        banned.add(char_vocab["[GAP]"])

    beams: list[DualBeam] = [DualBeam(
        input_ids=input_ids.clone(),
        word_ids=word_ids.clone(),
    )]

    n_masks = (input_ids == mask_id).sum().item()

    with torch.no_grad():
        for _ in range(n_masks):
            candidates: list[DualBeam] = []

            for beam in beams:
                mask_positions = (beam.input_ids == mask_id).nonzero(
                    as_tuple=True)[0].tolist()
                if not mask_positions:
                    candidates.append(beam)
                    continue

                # Forward pass
                out = model(
                    input_ids=beam.input_ids.unsqueeze(0),
                    word_ids=beam.word_ids.unsqueeze(0),
                    attention_mask=attention_mask.unsqueeze(0),
                )
                logits = out.logits[0]  # [seq_len, vocab_char]

                # Находим самую уверенную позицию
                best_pos, best_conf = None, -1.0
                for pos in mask_positions:
                    pos_logits = logits[pos] / max(temperature, 1e-6)
                    if banned:
                        pos_logits = pos_logits.clone()
                        for tid in banned:
                            pos_logits[tid] = float("-inf")
                    if allowed:
                        mask_t = torch.full_like(pos_logits, float("-inf"))
                        for tid in allowed:
                            mask_t[tid] = pos_logits[tid]
                        pos_logits = mask_t
                    max_prob = pos_logits.softmax(dim=-1).max().item()
                    if max_prob > best_conf:
                        best_conf = max_prob
                        best_pos  = pos

                # Расширяем лучшую позицию
                pos_logits = logits[best_pos] / max(temperature, 1e-6)
                if banned:
                    pos_logits = pos_logits.clone()
                    for tid in banned:
                        pos_logits[tid] = float("-inf")
                if allowed:
                    mask_t = torch.full_like(pos_logits, float("-inf"))
                    for tid in allowed:
                        mask_t[tid] = pos_logits[tid]
                    pos_logits = mask_t

                probs = pos_logits.softmax(dim=-1)
                top_probs, top_ids = probs.topk(beam_width)

                for prob, char_id in zip(top_probs.tolist(), top_ids.tolist()):
                    if prob <= 0:
                        continue
                    new_input = beam.input_ids.clone()
                    new_word  = beam.word_ids.clone()
                    new_input[best_pos] = char_id
                    new_word[best_pos]  = unk_word   # слово неизвестно
                    candidates.append(DualBeam(
                        input_ids=new_input,
                        word_ids=new_word,
                        log_prob=beam.log_prob + math.log(prob + 1e-12),
                        filled=beam.filled + [(best_pos, char_id)],
                    ))

            beams = sorted(candidates, key=lambda b: b.log_prob, reverse=True)
            beams = beams[:beam_width]

    return beams

def decode_char_ids(char_ids: list[int], id_to_char: dict[int, str]) -> str:
    return "".join(id_to_char.get(cid, "") for cid in char_ids)

# ── Span-level Hit@K + macro-CER evaluation ───────────────────────────────────

import json
import re

SPAN_PATTERN = re.compile(
    r"\(([^)]+)\)|\[(?!(?:GAP|MASK|PAD|UNK|CLS|SEP)\]|CTX_)([^\]]+)\]"
)

def evaluate_test_b_dual(
    model,
    char_vocab: dict[str, int],
    word_vocab: dict[str, int],
    test_b_dataset,
    device,
    *,
    beam_width:    int = 20,
    max_span_len:  int = 10,
    output_path:   Path | None = None,
) -> dict:
    """
    Span-level Hit@1/5/20 и macro-CER для DualBertForMaskedLM.
    Метрики усредняются по длинам лакун (как в Ithaca/Aeneas).
    """
    print(f"\n--- Test B Dual (beam={beam_width}) ---")

    id_to_char = {v: k for k, v in char_vocab.items()}
    mask_id    = char_vocab["[MASK]"]
    allowed    = [
        v for k, v in char_vocab.items()
        if len(k) == 1 and k not in ("[PAD]", "[UNK]", "[CLS]",
                                      "[SEP]", "[MASK]", "[GAP]")
    ]

    cer_by_len:  dict[int, list] = {}
    hits_by_len: dict[int, dict] = {k: {} for k in [1, 5, 20]}
    report_rows: list[dict] = []

    model.eval()
    for record in test_b_dataset:
        orig_text   = record.get("original", "")
        target_text = record.get("target_text", "")
        if not orig_text or not target_text:
            continue

        # input_ids уже содержат [MASK] на позициях лакун (из encode_test_b)
        inp  = torch.tensor(record["input_ids"],   device=device)
        wrd  = torch.tensor(record["word_ids"],    device=device)
        attn = torch.tensor(record["attention_mask"], device=device)
        lbl  = torch.tensor(record["labels"],      device=device)

        # Находим маскированные позиции (те у которых label != -100)
        mask_positions = (lbl != -100).nonzero(as_tuple=True)[0].tolist()
        if not mask_positions:
            continue

        # Истинные символы
        true_chars = [id_to_char.get(lbl[p].item(), "") for p in mask_positions]
        true_span  = "".join(true_chars)
        span_len   = min(len(true_span), max_span_len)
        if not true_span:
            continue

        # Beam search
        beams = beam_search_dual(
            inp, wrd, attn, model, char_vocab,
            beam_width=beam_width,
            allowed_char_ids=allowed,
        )

        # Декодируем предсказанные символы на позициях лакуны
        pred_spans = []
        for beam in beams:
            pred_chars = [id_to_char.get(beam.input_ids[p].item(), "")
                          for p in mask_positions]
            pred_spans.append("".join(pred_chars))

        top1_pred = pred_spans[0] if pred_spans else ""
        cer = Levenshtein.distance(top1_pred, true_span) / max(len(true_span), 1)

        cer_by_len.setdefault(span_len, []).append(cer)
        for k in [1, 5, 20]:
            hits_by_len[k].setdefault(span_len, []).append(
                int(true_span in pred_spans[:k])
            )

        report_rows.append({
            "true_span":  true_span,
            "pred_top1":  top1_pred,
            "span_len":   len(true_span),
            "cer":        round(cer, 4),
            "hit@1":      true_span in pred_spans[:1],
            "hit@5":      true_span in pred_spans[:5],
            "hit@20":     true_span in pred_spans[:20],
            "top3_preds": " | ".join(pred_spans[:3]),
        })

    if not cer_by_len:
        return {"total_spans": 0, "macro_cer": 0.0,
                "hit@1": 0.0, "hit@5": 0.0, "hit@20": 0.0}

    lengths    = sorted(cer_by_len.keys())
    macro_cer  = sum(np.mean(cer_by_len[l]) for l in lengths) / len(lengths)
    macro_hits = {k: sum(np.mean(hits_by_len[k].get(l, [0])) for l in lengths) / len(lengths)
                  for k in [1, 5, 20]}

    metrics = {
        "total_spans": len(report_rows),
        "macro_cer":   round(macro_cer,      4),
        "hit@1":       round(macro_hits[1],  4),
        "hit@5":       round(macro_hits[5],  4),
        "hit@20":      round(macro_hits[20], 4),
    }

    print(f"  Spans:     {metrics['total_spans']}")
    print(f"  Macro-CER: {metrics['macro_cer']:.4f}")
    print(f"  Hit@1:     {metrics['hit@1']:.4f}")
    print(f"  Hit@5:     {metrics['hit@5']:.4f}")
    print(f"  Hit@20:    {metrics['hit@20']:.4f}")

    if output_path:
        pd.DataFrame(report_rows).to_csv(output_path, index=False, encoding="utf-8-sig")
        print(f"  Report → {output_path}")

    return metrics