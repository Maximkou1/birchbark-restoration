#!/usr/bin/env python3
"""
eval_char_report.py
~~~~~~~~~~~~~~~~~~~~
Eval-only: load already fine-tuned CHARACTER-level checkpoints and write
per-position prediction reports for Test A and Test B, WITHOUT retraining.

For each model it loads outputs/finetune_char/<Model>/best_by_val and the matching
data/splits/<model>/{test_a.txt,test_b.jsonl}, then writes
  outputs/finetune_char/<Model>/report_test_a.csv
  outputs/finetune_char/<Model>/report_test_b.csv

Evaluation/decoding logic is shared with finetune_char.py via char_eval.py.

Usage:
  python eval_char_report.py
  python eval_char_report.py --models ModernBERT mBERT
"""

import argparse
import json
import logging
from pathlib import Path

import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

from char_eval import evaluate_test_a, evaluate_test_b

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)s  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

MODELS = {
    "BERTislav":  {"data_dir": "bertislav"},
    "mBERT":      {"data_dir": "mbert"},
    "ModernBERT": {"data_dir": None},
}

_HERE = Path(__file__).resolve().parents[2]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir",   default=str(_HERE / "data/splits"), type=Path)
    p.add_argument("--output_dir", default=str(_HERE / "outputs/finetune_char"), type=Path)
    p.add_argument("--seed",       default=42, type=int)
    p.add_argument("--max_gaps",   default=0,  type=int,
                   help="Max positions per report (0 = no limit).")
    p.add_argument("--models", nargs="+", default=list(MODELS.keys()),
                   choices=list(MODELS.keys()))
    return p.parse_args()


def load_txt(path: Path) -> list[str]:
    return [l.strip() for l in path.read_text("utf-8").splitlines() if l.strip()]


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text("utf-8").splitlines() if l.strip()]


def main():
    args = parse_args()
    max_gaps = args.max_gaps if args.max_gaps > 0 else None

    for model_key in args.models:
        subdir   = MODELS[model_key]["data_dir"]
        data_dir = args.data_dir / subdir if subdir else args.data_dir
        best_dir = args.output_dir / model_key / "best_by_val"

        if not best_dir.exists():
            log.error("missing checkpoint: %s", best_dir); continue
        if not data_dir.exists():
            log.error("missing data dir: %s", data_dir); continue

        log.info("=== %s ===", model_key)
        tok   = AutoTokenizer.from_pretrained(best_dir)
        model = AutoModelForMaskedLM.from_pretrained(best_dir).to(
            "cuda" if torch.cuda.is_available() else "cpu")

        test_a = load_txt(data_dir / "test_a.txt")
        test_b = load_jsonl(data_dir / "test_b.jsonl")

        out = args.output_dir / model_key
        evaluate_test_a(model, tok, test_a, seed=args.seed,
                        max_eval_gaps=max_gaps,
                        report_path=out / "report_test_a.csv")
        evaluate_test_b(model, tok, test_b,
                        max_eval_gaps=max_gaps,
                        report_path=out / "report_test_b.csv")

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()