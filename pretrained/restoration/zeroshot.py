#!/usr/bin/env python3
"""
zeroshot.py
~~~~~~~~~~~
Zero-shot evaluation of BERTislav / mBERT / ModernBERT, in both character-level
and token-level modes, on Test A (artificial gaps) and Test B (real editorial
reconstructions). No fine-tuning: each pretrained encoder is loaded and scored
as-is.

Evaluation logic is shared with the fine-tuning scripts via char_eval.py and
token_eval.py, so zero-shot and fine-tuned numbers are produced by identical
code.

Data layout (same as finetune_char / finetune_tokens):
  - char mode reads the model-specific normalized split written by
    prenormalize.py:  data/splits/<model>/{test_a.txt, test_b.jsonl}
    (ModernBERT uses the un-normalized data/splits/ directly).
  - token mode reads the raw data/splits/test_a.txt and applies the model's
    normalizer online, plus the pre-tokenized
    data/splits/test_b_tokens_<suffix>.jsonl.

Outputs (per model) under outputs/zeroshot/<Model>/:
  report_test_a_char.csv, report_test_b_char.csv
and a run-level outputs/zeroshot/zeroshot_summary.csv.

Run prenormalize.py and prepare_test_b_tokens.py first.

Usage:
  python zeroshot.py
  python zeroshot.py --models ModernBERT mBERT
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

from char_eval import evaluate_test_a, evaluate_test_b
from token_eval import evaluate_test_a_tokens, evaluate_test_b_tokens_hit, K_VALUES

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
try:
    from normalize import norm_bertislav, norm_mbert
except ImportError:
    logging.getLogger(__name__).warning("normalize.py not found — skipping.")
    def norm_bertislav(t): return t
    def norm_mbert(t): return t

NORMALIZERS = {
    "BERTislav":  norm_bertislav,
    "mBERT":      norm_mbert,
    "ModernBERT": lambda t: t,
}

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)s  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

MODELS = {
    "BERTislav": {
        "hf_name":  "npedrazzini/BERTislav",
        "data_dir": "bertislav",        # normalized split (char mode)
        "suffix":   "bertislav",        # test_b_tokens_<suffix>.jsonl (token mode)
    },
    "mBERT": {
        "hf_name":  "google-bert/bert-base-multilingual-cased",
        "data_dir": "mbert",
        "suffix":   "mbert",
    },
    "ModernBERT": {
        "hf_name":  "answerdotai/ModernBERT-base",
        "data_dir": None,               # uses un-normalized data/splits/
        "suffix":   "modernbert",
    },
}

_HERE = Path(__file__).resolve().parents[2]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir",      default=str(_HERE / "data/splits"), type=Path)
    p.add_argument("--output_dir",    default=str(_HERE / "outputs/zeroshot"), type=Path)
    p.add_argument("--seed",          default=42,   type=int)
    p.add_argument("--mask_prob",     default=0.08, type=float)
    p.add_argument("--span_p",        default=0.35, type=float)
    p.add_argument("--max_eval_gaps", default=5000, type=int,
                   help="Max positions per eval (0 = no limit).")
    p.add_argument("--models", nargs="+", default=list(MODELS.keys()),
                   choices=list(MODELS.keys()))
    return p.parse_args()


def load_txt(path: Path) -> list[str]:
    return [l.strip() for l in path.read_text("utf-8").splitlines() if l.strip()]


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text("utf-8").splitlines() if l.strip()]


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    max_eval_gaps = args.max_eval_gaps if args.max_eval_gaps > 0 else None

    summary_rows = []
    for model_key in args.models:
        cfg      = MODELS[model_key]
        hf_name  = cfg["hf_name"]
        subdir   = cfg["data_dir"]
        suffix   = cfg["suffix"]
        # char mode: model-specific normalized split (ModernBERT = root split)
        char_dir = args.data_dir / subdir if subdir else args.data_dir

        log.info("\n%s\n  %s  (%s)  [zero-shot]\n%s",
                 "=" * 60, model_key, hf_name, "=" * 60)

        if not char_dir.exists():
            log.error("missing %s — run prenormalize.py", char_dir); continue

        # ── inputs ────────────────────────────────────────────────────────────
        test_a_char = load_txt(char_dir / "test_a.txt")          # normalized
        test_b_char = load_jsonl(char_dir / "test_b.jsonl")      # normalized
        test_a_raw  = load_txt(args.data_dir / "test_a.txt")     # raw (token mode)
        tb_tok_path = args.data_dir / f"test_b_tokens_{suffix}.jsonl"
        test_b_tok  = load_jsonl(tb_tok_path) if tb_tok_path.exists() else []
        if not test_b_tok:
            log.warning("missing %s — token Test B skipped", tb_tok_path)

        out_dir = args.output_dir / model_key
        out_dir.mkdir(parents=True, exist_ok=True)

        tokenizer = AutoTokenizer.from_pretrained(hf_name)
        model = AutoModelForMaskedLM.from_pretrained(hf_name).to(
            "cuda" if torch.cuda.is_available() else "cpu")
        model.eval()
        normalizer = NORMALIZERS.get(model_key, lambda t: t)

        # ── character-level (shared char_eval; data already normalized) ───────
        log.info("--- char-level ---")
        a_char = evaluate_test_a(model, tokenizer, test_a_char,
                                 seed=args.seed, mask_prob=args.mask_prob,
                                 span_p=args.span_p, max_eval_gaps=max_eval_gaps,
                                 report_path=out_dir / "report_test_a_char.csv")
        b_char = evaluate_test_b(model, tokenizer, test_b_char,
                                 max_eval_gaps=max_eval_gaps,
                                 report_path=out_dir / "report_test_b_char.csv")
        log.info("Test A (char): %s", a_char)
        log.info("Test B (char): %s", b_char)

        # ── token-level (shared token_eval; normalize raw Test A online) ──────
        log.info("--- token-level ---")
        a_tok = evaluate_test_a_tokens(model, tokenizer, test_a_raw,
                                       seed=args.seed, mask_prob=args.mask_prob,
                                       max_eval_gaps=max_eval_gaps,
                                       normalizer=normalizer)
        b_tok = (evaluate_test_b_tokens_hit(model, tokenizer, test_b_tok,
                                            max_eval_gaps=max_eval_gaps)
                 if test_b_tok else
                 {"total_gaps": 0, "tok_cer": 0.0,
                  **{f"tok_hit@{k}": 0.0 for k in K_VALUES}})
        log.info("Test A (tok): %s", a_tok)
        log.info("Test B (tok): %s", b_tok)

        summary_rows.append({
            "model": model_key,
            "a_char_hit@1": a_char.get("hit@1", 0),
            "a_char_hit@5": a_char.get("hit@5", 0),
            "a_char_cer":   a_char.get("cer", 0),
            "b_char_hit@1": b_char.get("hit@1", 0),
            "b_char_hit@5": b_char.get("hit@5", 0),
            "b_char_cer":   b_char.get("cer", 0),
            "a_tok_hit@1":  a_tok.get("tok_hit@1", 0),
            "a_tok_hit@5":  a_tok.get("tok_hit@5", 0),
            "a_tok_cer":    a_tok.get("tok_cer", 0),
            "b_tok_hit@1":  b_tok.get("tok_hit@1", 0),
            "b_tok_hit@5":  b_tok.get("tok_hit@5", 0),
            "b_tok_cer":    b_tok.get("tok_cer", 0),
        })

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    df = pd.DataFrame(summary_rows)
    out = args.output_dir / "zeroshot_summary.csv"
    df.to_csv(out, index=False)
    log.info("\n%s", df.to_string(index=False))
    log.info("Summary → %s", out)


if __name__ == "__main__":
    main()