#!/usr/bin/env python3
"""
eval_tokens_report.py
~~~~~~~~~~~~~~~~~~~~~~
Eval-only: load already fine-tuned TOKEN-level checkpoints and write a
per-position prediction report for Test B, WITHOUT retraining.

For each model it loads MLM/outputs_tokens/<Model>/best_by_val and the
matching data/splits/test_b_tokens_<suffix>.jsonl, then writes
MLM/outputs_tokens/<Model>/report_test_b_tokens.csv

Report columns:
  doc_idx, mask_index, true_token, pred_token, true_id, pred_id,
  match, true_rank, top5, n_chars_true

Evaluation/decoding logic is shared with finetune_tokens.py via token_eval.py.

Usage:
  python eval_tokens_report.py
  python eval_tokens_report.py --models ModernBERT mBERT
"""

import argparse
import json
import logging
from pathlib import Path

import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

from token_eval import eval_report_test_b

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)s  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

MODELS = {
    "BERTislav":  {"suffix": "bertislav"},
    "mBERT":      {"suffix": "mbert"},
    "ModernBERT": {"suffix": "modernbert"},
}

_ROOT = Path(__file__).resolve().parents[2]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir",   default=str(_ROOT / "data/splits"), type=Path)
    p.add_argument("--output_dir", default=str(_ROOT / "MLM/outputs_tokens"), type=Path)
    p.add_argument("--batch_size", default=32, type=int)
    p.add_argument("--models", nargs="+", default=list(MODELS.keys()),
                   choices=list(MODELS.keys()))
    return p.parse_args()


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text("utf-8").splitlines() if l.strip()]


def main():
    args = parse_args()
    for model_key in args.models:
        suffix = MODELS[model_key]["suffix"]
        best_dir = args.output_dir / model_key / "best_by_val"
        tb_path = args.data_dir / f"test_b_tokens_{suffix}.jsonl"

        if not best_dir.exists():
            log.error("missing checkpoint: %s", best_dir); continue
        if not tb_path.exists():
            log.error("missing test_b: %s", tb_path); continue

        log.info("=== %s ===", model_key)
        tok = AutoTokenizer.from_pretrained(best_dir)
        model = AutoModelForMaskedLM.from_pretrained(best_dir).to(
            "cuda" if torch.cuda.is_available() else "cpu")
        records = load_jsonl(tb_path)
        eval_report_test_b(
            model, tok, records, batch_size=args.batch_size,
            report_path=args.output_dir / model_key / "report_test_b_tokens.csv")
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
