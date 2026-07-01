#!/usr/bin/env python3
"""
prepare_datasets.py
~~~~~~~~~~~~~~~~~~~~~~~~
Prepares DualBERT dataset from split files.

Splits:
  train   → encoded with char + word alignment
  eval    → same encoding, for validation
  test_a  → same encoding, for final collator evaluation
  test_b  → raw strings (original + target) for dynamic bracket masking
"""

import argparse
import json
from pathlib import Path

from datasets import Dataset, DatasetDict

from align_dual import load_vocab, align_char_to_word

_HERE = Path(__file__).resolve().parent              # from_scratch/DualEmbLM/
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


def encode_lines(lines: list[str], char_vocab, word_vocab, max_len: int) -> list[dict]:
    return [
        align_char_to_word(text, char_vocab, word_vocab,
                           max_len=max_len, add_cls_sep=True)
        for text in lines
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits_dir",
                        default=str(_ROOT / "data/splits"))
    parser.add_argument("--char_vocab_path",
                        default=str(_HERE / "char_tokenizer/char_vocab.json"))
    parser.add_argument("--word_vocab_path",
                        default=str(_HERE / "word_vocab.json"))
    parser.add_argument("--out_dir",
                        default=str(_ROOT / "outputs/from_scratch/DualEmbLM/dataset"))
    parser.add_argument("--max_len", type=int, default=256)
    parser.add_argument("--limit",   type=int, default=0)
    args = parser.parse_args()

    char_vocab = load_vocab(args.char_vocab_path)
    word_vocab = load_vocab(args.word_vocab_path)
    splits_dir = Path(args.splits_dir)
    lim        = args.limit

    print(f"char vocab size: {len(char_vocab)}")
    print(f"word vocab size: {len(word_vocab)}")

    # ── Train ─────────────────────────────────────────────────────────────────
    print("Processing TRAIN...")
    train_lines = read_txt_lines(splits_dir / "train.txt", limit=lim)
    train_ds = Dataset.from_list(
        encode_lines(train_lines, char_vocab, word_vocab, args.max_len)
    )

    # ── Eval ──────────────────────────────────────────────────────────────────
    print("Processing EVAL...")
    eval_lines = read_txt_lines(splits_dir / "eval.txt", limit=lim)
    eval_ds = Dataset.from_list(
        encode_lines(eval_lines, char_vocab, word_vocab, args.max_len)
    )

    # ── Test A ────────────────────────────────────────────────────────────────
    print("Processing TEST_A...")
    test_a_lines = read_txt_lines(splits_dir / "test_a.txt", limit=lim)
    test_a_ds = Dataset.from_list(
        encode_lines(test_a_lines, char_vocab, word_vocab, args.max_len)
    )

    # ── Test B (raw strings for dynamic bracket masking) ──────────────────────
    print("Processing TEST_B (raw strings)...")
    test_b_records = read_jsonl(splits_dir / "test_b.jsonl", limit=lim)
    test_b_list = [
        {
            "original": r.get("original", ""),
            "target":   r.get("target", ""),
            "tag":      r.get("tag", ""),
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
    print(f"  train:  {len(train_ds):,}")
    print(f"  eval:   {len(eval_ds):,}")
    print(f"  test_a: {len(test_a_ds):,}")
    print(f"  test_b: {len(test_b_ds):,}")


if __name__ == "__main__":
    main()