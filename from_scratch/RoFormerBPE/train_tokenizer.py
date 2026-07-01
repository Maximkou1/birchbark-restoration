#!/usr/bin/env python3
"""
train_tokenizer.py
~~~~~~~~~~~~~~~~~~
Trains a Byte-Level BPE tokenizer on Ancient Russian text.
Produces a HuggingFace-compatible tokenizer saved to roformer/tokenizer/.

Run from roformer/ directory:
    python train_tokenizer.py
    python train_tokenizer.py --vocab_size 30000
"""

import argparse
import os
from pathlib import Path

from tokenizers import ByteLevelBPETokenizer
from transformers import PreTrainedTokenizerFast

_HERE = Path(__file__).resolve().parent              # from_scratch/RoFormerBPE/
_ROOT = _HERE.parent.parent                          # repo root


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--corpus",     default=str(_ROOT / "data/splits/train.txt"))
    p.add_argument("--save_dir",   default=str(_HERE / "tokenizer"))
    p.add_argument("--vocab_size", default=50_000, type=int)
    p.add_argument("--min_freq",   default=2,      type=int)
    return p.parse_args()


def main():
    args = parse_args()

    if not os.path.exists(args.corpus):
        print(f"Error: corpus not found at {args.corpus}")
        return

    print(f"Training BPE tokenizer on {args.corpus}  vocab_size={args.vocab_size}")

    special_tokens = [
        "<s>", "<pad>", "</s>", "<unk>", "<mask>",
        "[GAP]",   # the only special token for lacunae
    ]

    bpe = ByteLevelBPETokenizer()
    bpe.train(
        files=[args.corpus],
        vocab_size=args.vocab_size,
        min_frequency=args.min_freq,
        show_progress=True,
        special_tokens=special_tokens,
    )

    os.makedirs(args.save_dir, exist_ok=True)
    tok_json = os.path.join(args.save_dir, "tokenizer.json")
    bpe.save(tok_json)

    fast = PreTrainedTokenizerFast(
        tokenizer_file=tok_json,
        max_len=512,
        bos_token="<s>",
        eos_token="</s>",
        sep_token="</s>",
        cls_token="<s>",
        unk_token="<unk>",
        pad_token="<pad>",
        mask_token="<mask>",
    )
    fast.add_special_tokens({"additional_special_tokens": ["[GAP]"]})
    fast.save_pretrained(args.save_dir)

    print(f"\nTokenizer saved → {args.save_dir}/")
    print(f"Vocab size: {fast.vocab_size:,}")

    print("\n--- Validation ---")
    tests = [
        "поклонъ · ѿ · бориса · [GAP] · настасии",
        "· ꙅ҃ · десѧ · коуно ·",
    ]
    for text in tests:
        tokens = fast.tokenize(text)
        decoded = fast.decode(fast.encode(text))
        print(f"  in:  {text}")
        print(f"  tok: {tokens}")
        print(f"  out: {decoded}\n")


if __name__ == "__main__":
    main()