#!/usr/bin/env python3
"""
embed.py
~~~~~~~~
Extracts embeddings from each model for all records in a split.

For each model and split (train/val/test), saves:
  outputs/classification/embeddings/{model_name}_{split}.npy   — float32 [N, D]
  outputs/classification/embeddings/{model_name}_{split}_ids.json — list of record numbers

Usage:
  python embed.py                        # all models, all splits
  python embed.py --models ModernBERT    # one model
  python embed.py --splits train val     # specific splits
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

from config_probe import (
    DATA_DIR, EMBED_DIR, MODELS, BIRCH_ROOT,
)

sys.path.insert(0, str(BIRCH_ROOT / "from_scratch/RoFormerBPE"))
sys.path.insert(0, str(BIRCH_ROOT / "from_scratch/DualEmbLM"))

EMBED_DIR.mkdir(parents=True, exist_ok=True)


# ── Loaders ───────────────────────────────────────────────────────────────────

def load_split(split: str) -> list[dict]:
    path = DATA_DIR / f"{split}.jsonl"
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


# ── Embedding functions ───────────────────────────────────────────────────────

def pool(hidden: torch.Tensor, attention_mask: torch.Tensor,
         mode: str) -> torch.Tensor:
    """Pool hidden states: 'cls' or 'mean'."""
    if mode == "cls":
        return hidden[:, 0, :]
    else:  # mean over non-padding positions
        mask = attention_mask.unsqueeze(-1).float()
        return (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)


@torch.no_grad()
def embed_hf_mlm(records: list[dict], cfg: dict,
                 device: torch.device, batch_size: int = 32) -> np.ndarray:
    from transformers import AutoModelForMaskedLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(cfg["path"])
    model = AutoModelForMaskedLM.from_pretrained(
        cfg["path"], output_hidden_states=True).to(device)
    model.eval()

    all_embs = []
    texts = [r[cfg.get("text_field", "target")] for r in records]

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        enc = tokenizer(
            batch_texts, return_tensors="pt", padding=True,
            truncation=True, max_length=512,
        ).to(device)
        out = model(**enc)
        hidden = out.hidden_states[-1]   # last layer [B, T, D]
        emb = pool(hidden, enc["attention_mask"], cfg["pooling"])
        all_embs.append(emb.cpu().float().numpy())

    return np.concatenate(all_embs, axis=0)


@torch.no_grad()
def embed_roformer(records: list[dict], cfg: dict,
                   device: torch.device, batch_size: int = 32) -> np.ndarray:
    from transformers import AutoTokenizer, RoFormerModel

    tokenizer = AutoTokenizer.from_pretrained(cfg["tokenizer"])
    tokenizer.add_special_tokens({"additional_special_tokens": ["[GAP]"]})
    model = RoFormerModel.from_pretrained(cfg["path"]).to(device)
    model.eval()

    all_embs = []
    texts = [r[cfg.get("text_field", "target")] for r in records]

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        enc = tokenizer(
            batch_texts, return_tensors="pt", padding=True,
            truncation=True, max_length=512,
            return_token_type_ids=False,
        ).to(device)
        out = model(**enc)
        hidden = out.last_hidden_state   # [B, T, D]
        emb = pool(hidden, enc["attention_mask"], cfg["pooling"])
        all_embs.append(emb.cpu().float().numpy())

    return np.concatenate(all_embs, axis=0)


@torch.no_grad()
def embed_dualemb(records: list[dict], cfg: dict,
                  device: torch.device, batch_size: int = 32) -> np.ndarray:
    from align_dual import load_vocab, align_char_to_word
    from model import DualBertForMaskedLM

    char_vocab = load_vocab(cfg["char_vocab"])
    word_vocab = load_vocab(cfg["word_vocab"])
    model = DualBertForMaskedLM.from_pretrained(cfg["path"]).to(device)
    model.eval()

    max_len = model.config.max_position_embeddings
    all_embs = []

    texts = [r[cfg.get("text_field", "target")] for r in records]
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        encoded = [
            align_char_to_word(t, char_vocab, word_vocab,
                                max_len=max_len, add_cls_sep=True)
            for t in batch_texts
        ]
        input_ids = torch.tensor(
            [e["input_ids"] for e in encoded], dtype=torch.long).to(device)
        word_ids = torch.tensor(
            [e["word_ids"] for e in encoded], dtype=torch.long).to(device)
        attention_mask = torch.tensor(
            [e["attention_mask"] for e in encoded], dtype=torch.long).to(device)

        # Get encoder hidden states via dual_embeddings + encoder
        emb_in = model.dual_embeddings(input_ids, word_ids)
        ext_mask = model.get_extended_attention_mask(
            attention_mask, input_ids.shape, device)
        enc_out = model.encoder(
            emb_in, attention_mask=ext_mask,
            head_mask=[None] * model.config.num_hidden_layers,
            return_dict=True,
        )
        hidden = enc_out.last_hidden_state   # [B, T, D]
        emb = pool(hidden, attention_mask, cfg["pooling"])
        all_embs.append(emb.cpu().float().numpy())

    return np.concatenate(all_embs, axis=0)


# ── Main ──────────────────────────────────────────────────────────────────────

EMBEDDERS = {
    "hf_mlm":   embed_hf_mlm,
    "roformer": embed_roformer,
    "dualemb":  embed_dualemb,
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", default=list(MODELS.keys()),
                   choices=list(MODELS.keys()))
    p.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    p.add_argument("--text_fields", nargs="+", default=["target", "masked"],
                   choices=["target", "masked", "original"])
    p.add_argument("--batch_size", type=int, default=32)
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    for split in args.splits:
        records = load_split(split)
        ids = [r.get("number", str(i)) for i, r in enumerate(records)]

        for model_name in args.models:
            cfg = MODELS[model_name]

            for text_field in args.text_fields:
                out_emb = EMBED_DIR / f"{model_name}_{split}_{text_field}.npy"
                out_ids = EMBED_DIR / f"{model_name}_{split}_{text_field}_ids.json"

                if out_emb.exists():
                    print(f"  {model_name}/{split}/{text_field}: already exists, skipping")
                    continue

                print(f"\n  Embedding {model_name} / {split} / {text_field} "
                      f"({len(records)} records)...")
                embedder = EMBEDDERS[cfg["type"]]
                embs = embedder(records, {**cfg, "text_field": text_field},
                                device, batch_size=args.batch_size)

                np.save(out_emb, embs)
                with open(out_ids, "w") as f:
                    json.dump(ids, f)

                print(f"  Saved {embs.shape} → {out_emb}")

            # Free GPU memory between models
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()