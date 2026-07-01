#!/usr/bin/env python3
"""
prepare_datasets.py
~~~~~~~~~~~~~~~~~~~
Prepares RoFormer dataset from split files.

Splits:
  train   → packed blocks for training
  eval    → packed blocks for validation during training
  test_a  → packed blocks for final collator evaluation
  test_b  → raw strings for dynamic bracket masking
"""

import argparse
import json
from pathlib import Path

from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer

_HERE = Path(__file__).resolve().parent              # from_scratch/RoFormerBPE/
_ROOT = _HERE.parent.parent                          # repo root


def read_txt_lines(path: Path, limit: int = 0) -> list[str]:
    out = []
    if not path.exists():
        print(f"Warning: {path} not found.")
        return []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(line)
                if limit and len(out) >= limit:
                    break
    return out


def read_jsonl(path: Path, limit: int = 0) -> list[dict]:
    out = []
    if not path.exists():
        print(f"Warning: {path} not found.")
        return []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
            if limit and len(out) >= limit:
                break
    return out


def encode_lines(lines: list[str], tokenizer) -> dict:
    enc = tokenizer(
        lines,
        truncation=False,
        padding=False,
        add_special_tokens=True,
        return_token_type_ids=False,
    )
    return {k: v for k, v in enc.items() if k in ("input_ids", "attention_mask")}


def pad_or_chunk(examples: dict, block_size: int, pad_token_id: int) -> dict:
    """
    Per-document chunking: no cross-document context leakage.
    Short docs → padded to block_size.
    Long docs  → split into non-overlapping chunks within document boundary.
    """
    result = {"input_ids": [], "attention_mask": []}
    for ids, attn in zip(examples["input_ids"], examples["attention_mask"]):
        for start in range(0, max(1, len(ids)), block_size):
            chunk      = ids[start:start + block_size]
            chunk_attn = attn[start:start + block_size]
            pad        = block_size - len(chunk)
            result["input_ids"].append(chunk + [pad_token_id] * pad)
            result["attention_mask"].append(chunk_attn + [0] * pad)
    return result


def build_packed_ds(lines: list[str], tokenizer, block_size: int,
                    desc: str) -> Dataset:
    if not lines:
        return Dataset.from_dict({"input_ids": [], "attention_mask": []})
    raw = encode_lines(lines, tokenizer)
    pad_id = tokenizer.pad_token_id
    return Dataset.from_dict(raw).map(
        lambda x: pad_or_chunk(x, block_size, pad_id),
        batched=True,
        desc=desc,
        remove_columns=["input_ids", "attention_mask"],
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits_dir",
                        default=str(_ROOT / "data/splits"))
    parser.add_argument("--tokenizer_path",
                        default=str(_HERE / "tokenizer"))
    parser.add_argument("--out_dir",
                        default=str(_ROOT / "outputs/from_scratch/RoFormerBPE/dataset"))
    parser.add_argument("--max_len",  type=int, default=256)
    parser.add_argument("--limit",    type=int, default=0,
                        help="Debug: limit lines per split (0 = no limit)")
    args = parser.parse_args()

    print(f"Loading tokenizer from {args.tokenizer_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    tokenizer.add_special_tokens({"additional_special_tokens": ["[GAP]"]})
    print(f"Vocab size: {len(tokenizer):,}")

    splits_dir = Path(args.splits_dir)
    lim = args.limit

    # ── Train ─────────────────────────────────────────────────────────────────
    print("Processing TRAIN...")
    train_lines = read_txt_lines(splits_dir / "train.txt", limit=lim)
    train_ds = build_packed_ds(
        train_lines, tokenizer, args.max_len,
        f"Chunking train (max_len={args.max_len})",
    )

    # ── Eval ──────────────────────────────────────────────────────────────────
    print("Processing EVAL...")
    eval_lines = read_txt_lines(splits_dir / "eval.txt", limit=lim)
    eval_ds = build_packed_ds(
        eval_lines, tokenizer, args.max_len,
        f"Chunking eval (max_len={args.max_len})",
    )

    # ── Test A ────────────────────────────────────────────────────────────────
    print("Processing TEST_A...")
    test_a_lines = read_txt_lines(splits_dir / "test_a.txt", limit=lim)
    test_a_ds = build_packed_ds(
        test_a_lines, tokenizer, args.max_len,
        f"Chunking test_a (max_len={args.max_len})",
    )

    # ── Test B (raw strings for dynamic bracket masking) ──────────────────────
    print("Processing TEST_B (raw strings)...")
    test_b_records = read_jsonl(splits_dir / "test_b.jsonl", limit=lim)
    test_b_list = [
        {
            "original":   r.get("original", ""),
            "target":     r.get("target", ""),
            "tag":        r.get("tag", ""),
        }
        for r in test_b_records
        if r.get("original") and r.get("target")
    ]
    test_b_ds = Dataset.from_list(test_b_list)

    # ── Save ──────────────────────────────────────────────────────────────────
    ds = DatasetDict({
        "train":  train_ds,
        "eval":   eval_ds,
        "test_a": test_a_ds,
        "test_b": test_b_ds,
    })
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ds.save_to_disk(str(out_dir))

    print(f"\nDataset saved to: {out_dir}")
    print(f"  train:  {len(train_ds):,} blocks")
    print(f"  eval:   {len(eval_ds):,} blocks")
    print(f"  test_a: {len(test_a_ds):,} blocks")
    print(f"  test_b: {len(test_b_ds):,} records")


if __name__ == "__main__":
    main()