#!/usr/bin/env python3
"""
prepare_test_b_tokens.py
~~~~~~~~~~~~~~~~~~~~~~~~
Builds per-model token-level test_b files from the existing test_b.jsonl.

For each model:
  1. Takes `original` field (text with brackets marking lacunae)
  2. Tokenizes with the model's tokenizer
  3. Finds tokens that overlap with bracket spans
  4. Masks those tokens (extends to full token boundaries)
  5. Saves input_ids, attention_mask, mask_indices, target_token_ids, target_str

Output (read by finetune_tokens.py / zeroshot.py):
  data/splits/test_b_tokens_bertislav.jsonl
  data/splits/test_b_tokens_mbert.jsonl
  data/splits/test_b_tokens_modernbert.jsonl

Run this (and prenormalize.py) before the token-level experiments.

Usage:
  python prepare_test_b_tokens.py
  python prepare_test_b_tokens.py --models BERTislav mBERT
"""

import argparse
import json
import logging
import re
import sys
from pathlib import Path

from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)s  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

# ── Normalization ──────────────────────────────────────────────────────────────
try:
    from normalize import norm_bertislav, norm_mbert
except ImportError:
    log.warning("normalize.py not found — normalization will be skipped.")
    def norm_bertislav(t): return t
    def norm_mbert(t): return t

NORMALIZERS = {
    "BERTislav":  norm_bertislav,
    "mBERT":      norm_mbert,
    "ModernBERT": lambda t: t,
}

MODELS = {
    "BERTislav":  "npedrazzini/BERTislav",
    "mBERT":      "google-bert/bert-base-multilingual-cased",
    "ModernBERT": "answerdotai/ModernBERT-base",
}

_HERE = Path(__file__).resolve().parents[2]   # repo root

# Matches editorial brackets but NOT special tokens like [GAP], [MASK] etc.
ROUND_PAT  = re.compile(r"\(([^)]+)\)")
SQUARE_PAT = re.compile(r"\[(?!(?:GAP|MASK|PAD|UNK|CLS|SEP|CTX_[A-Z_]+)\])([^\]]+)\]")


def find_lacuna_spans(original: str) -> list[tuple[int, int, str]]:
    """
    Returns list of (start, end, text) for each lacuna in `original`.
    Coordinates are character offsets in the de-bracketed string (target).
    """
    spans = []
    result_offset = 0

    i = 0
    while i < len(original):
        # Round bracket
        m = ROUND_PAT.match(original, i)
        if m:
            inner = m.group(1)
            spans.append((result_offset, result_offset + len(inner), inner))
            result_offset += len(inner)
            i = m.end()
            continue
        # Square bracket (non-special)
        m = SQUARE_PAT.match(original, i)
        if m:
            inner = m.group(1)
            spans.append((result_offset, result_offset + len(inner), inner))
            result_offset += len(inner)
            i = m.end()
            continue
        # Regular character
        result_offset += 1
        i += 1

    return spans


def get_target_text(original: str) -> str:
    """Remove brackets to get the plain target string."""
    text = ROUND_PAT.sub(r"\1", original)
    text = SQUARE_PAT.sub(r"\1", text)
    return text


def build_token_record(original: str, tokenizer, max_length: int = 512) -> dict | None:
    """
    Tokenizes the target text and identifies which tokens to mask
    based on lacuna char spans.

    Returns None if no maskable tokens found.
    """
    target = get_target_text(original)
    lacuna_spans = find_lacuna_spans(original)
    if not lacuna_spans:
        return None

    # Tokenize with offset mapping so we can align chars to tokens
    enc = tokenizer(
        target,
        return_offsets_mapping=True,
        truncation=True,
        max_length=max_length,
        return_tensors=None,
    )

    input_ids      = enc["input_ids"]
    attention_mask = enc["attention_mask"]
    offsets        = enc["offset_mapping"]   # list of (char_start, char_end) per token

    # Find which token positions overlap with any lacuna span
    mask_indices: list[int] = []
    for tok_idx, (tok_start, tok_end) in enumerate(offsets):
        if tok_start == tok_end:      # special tokens ([CLS], [SEP], etc.)
            continue
        for lac_start, lac_end, _ in lacuna_spans:
            # overlap condition: token and lacuna share at least one char
            if tok_start < lac_end and tok_end > lac_start:
                mask_indices.append(tok_idx)
                break

    if not mask_indices:
        return None

    # Target token ids for masked positions (ground truth)
    target_token_ids = [input_ids[i] for i in mask_indices]

    # Build masked input_ids
    masked_ids = list(input_ids)
    for i in mask_indices:
        masked_ids[i] = tokenizer.mask_token_id

    # Decode target tokens back to string for CER computation
    target_str = tokenizer.decode(
        target_token_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    ).strip()

    return {
        "original":         original,
        "target":           target,
        "target_str":       target_str,
        "input_ids":        masked_ids,
        "attention_mask":   attention_mask,
        "mask_indices":     mask_indices,
        "target_token_ids": target_token_ids,
        "lacuna_spans":     [(s, e, t) for s, e, t in lacuna_spans],
    }


def prepare_model(model_key: str, hf_name: str,
                  records: list[dict],
                  output_path: Path,
                  max_length: int = 512) -> None:
    log.info("Loading tokeniser: %s (%s)", model_key, hf_name)
    tokenizer  = AutoTokenizer.from_pretrained(hf_name)
    normalizer = NORMALIZERS[model_key]

    out_records = []
    skipped = 0

    for i, rec in enumerate(records):
        original = rec.get("original", "")
        if not original:
            skipped += 1
            continue

        # Normalize the full original text (including bracket contents).
        # Lacuna spans are re-computed after normalization so offsets stay valid.
        # [GAP] is protected from lowercasing by using a placeholder.
        original = original.replace("[GAP]", "GAP")
        original = normalizer(original)
        original = original.replace("GAP", "[GAP]").replace("gap", "[GAP]")

        result = build_token_record(original, tokenizer, max_length)
        if result is None:
            skipped += 1
            continue

        # Carry over useful fields from original record
        result["tag"]  = rec.get("tag", "")
        result["n_masked_chars"] = rec.get("n_masked_chars", 0)
        out_records.append(result)

        if (i + 1) % 200 == 0:
            log.info("  %d / %d processed", i + 1, len(records))

    with output_path.open("w", encoding="utf-8") as f:
        for rec in out_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    log.info(
        "%s → %d records written, %d skipped → %s",
        model_key, len(out_records), skipped, output_path,
    )


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir",   default=str(_HERE / "data/splits"), type=Path)
    p.add_argument("--output_dir", default=str(_HERE / "data/splits"), type=Path)
    p.add_argument("--max_length", default=512, type=int)
    p.add_argument("--models", nargs="+",
                   default=list(MODELS.keys()),
                   choices=list(MODELS.keys()))
    return p.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    src = args.data_dir / "test_b.jsonl"
    log.info("Reading %s", src)
    records = [json.loads(l) for l in src.read_text("utf-8").splitlines() if l.strip()]
    log.info("%d records loaded", len(records))

    for model_key in args.models:
        hf_name = MODELS[model_key]
        out_path = args.output_dir / f"test_b_tokens_{model_key.lower()}.jsonl"
        prepare_model(model_key, hf_name, records, out_path, args.max_length)


if __name__ == "__main__":
    main()
