#!/usr/bin/env python3
"""
finetune_tokens.py
~~~~~~~~~~~~~~~~~~
Fine-tune BERTislav / mBERT / ModernBERT in TOKEN-level mode (full-vocabulary
MLM prediction). Per-epoch token Hit@K / CER / PPL are written to
epoch_log.csv; the checkpoint with the best validation token Hit@1 is retained.

Evaluation logic lives in token_eval.py (shared with eval_tokens_report.py).

Dataset layout:
  data/splits/train.txt
  data/splits/eval.txt
  data/splits/test_a.txt
  data/splits/test_b_tokens_<suffix>.jsonl   (built by prepare_test_b_tokens.py)
"""

import argparse
import json
import logging
import math
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch
from datasets import Dataset
from transformers import (AutoModelForMaskedLM, AutoTokenizer,
                          DataCollatorForLanguageModeling,
                          Trainer, TrainingArguments, TrainerCallback)

from token_eval import (GAP_TOKEN,
                        evaluate_test_a_tokens, evaluate_test_b_tokens_hit,
                        eval_report_test_b)

# normalize.py lives one level up, in pretrained/ (shared with classification).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Normalization (mBERT / BERTislav need it; ModernBERT does not).
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
    "BERTislav":  {"hf_name": "npedrazzini/BERTislav",        "suffix": "bertislav"},
    "mBERT":      {"hf_name": "google-bert/bert-base-multilingual-cased",
                   "suffix": "mbert"},
    "ModernBERT": {"hf_name": "answerdotai/ModernBERT-base",  "suffix": "modernbert"},
}

_ROOT = Path(__file__).resolve().parents[2]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir",      default=str(_ROOT / "data/splits"), type=Path)
    p.add_argument("--output_dir",    default=str(_ROOT / "outputs/finetune_tokens"), type=Path)
    p.add_argument("--epochs",        default=30,   type=int)
    p.add_argument("--batch_size",    default=64,   type=int)
    p.add_argument("--lr",            default=5e-5, type=float)
    p.add_argument("--max_length",    default=256,  type=int)
    p.add_argument("--mlm_prob",      default=0.08, type=float)
    p.add_argument("--seed",          default=42,   type=int)
    p.add_argument("--max_eval_gaps", default=5000, type=int)
    p.add_argument("--patience",      default=3,    type=int)
    p.add_argument("--models", nargs="+",
                   default=list(MODELS.keys()), choices=list(MODELS.keys()))
    return p.parse_args()


def load_txt(path: Path) -> list[str]:
    return [l.strip() for l in path.read_text("utf-8").splitlines() if l.strip()]


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text("utf-8").splitlines() if l.strip()]


def build_train_dataset(texts, tokenizer, max_length, model_key="ModernBERT") -> Dataset:
    ds = Dataset.from_dict({"text": texts})
    normalizer = NORMALIZERS.get(model_key, lambda t: t)

    def _tok(batch):
        result = {"input_ids": [], "attention_mask": []}
        for text in batch["text"]:
            enc  = tokenizer(normalizer(text), add_special_tokens=True,
                             truncation=False, return_attention_mask=True)
            ids, attn = enc["input_ids"], enc["attention_mask"]
            for start in range(0, max(1, len(ids)), max_length):
                chunk = ids[start:start + max_length]
                cattn = attn[start:start + max_length]
                pad   = max_length - len(chunk)
                result["input_ids"].append(chunk + [tokenizer.pad_token_id] * pad)
                result["attention_mask"].append(cattn + [0] * pad)
        return result

    return ds.map(_tok, batched=True, remove_columns=["text"])


class EvalCallback(TrainerCallback):
    """After each epoch: token Hit@K/CER on val/test_a/test_b, append to
    epoch_log.csv, keep best checkpoint by val token Hit@1."""

    def __init__(self, tokenizer, test_a_texts, test_b_records, best_dir,
                 log_path, max_eval_gaps, model_key, run_id,
                 patience=3, val_texts=None):
        self.tokenizer         = tokenizer
        self.a_texts           = test_a_texts
        self.b_records         = test_b_records
        self.val_texts         = val_texts or []
        self.best_dir          = best_dir
        self.log_path          = log_path
        self.max_eval_gaps     = max_eval_gaps
        self.model_key         = model_key
        self.run_id            = run_id
        self.best_hit1         = -1.0
        self.best_epoch        = 0
        self.epoch_metrics     = []
        self.epochs_no_improve = 0
        self.patience          = patience
        self.normalizer        = NORMALIZERS.get(model_key, lambda t: t)

    def on_evaluate(self, args, state, control, model=None, metrics=None, **kwargs):
        eval_loss  = (metrics or {}).get("eval_loss", float("nan"))
        ppl        = math.exp(eval_loss) if math.isfinite(eval_loss) else float("nan")
        train_loss = next(
            (e["loss"] for e in reversed(state.log_history)
             if "loss" in e and "eval_loss" not in e), float("nan"))

        a = evaluate_test_a_tokens(model, self.tokenizer, self.a_texts,
                                   max_eval_gaps=self.max_eval_gaps,
                                   normalizer=self.normalizer)
        b = evaluate_test_b_tokens_hit(model, self.tokenizer, self.b_records,
                                       max_eval_gaps=self.max_eval_gaps)
        val = evaluate_test_a_tokens(model, self.tokenizer, self.val_texts,
                                     max_eval_gaps=self.max_eval_gaps,
                                     normalizer=self.normalizer) if self.val_texts else {}
        val_hit1 = val.get("tok_hit@1", 0.0)

        row = {
            "run_id": self.run_id, "model": self.model_key,
            "epoch": int(state.epoch),
            "train_loss": round(train_loss, 4), "eval_loss": round(eval_loss, 4),
            "ppl": round(ppl, 2),
            **{f"a_{k}": v for k, v in a.items()},
            **{f"b_{k}": v for k, v in b.items()},
            "val_tok_hit@1": val_hit1,
        }
        self.epoch_metrics.append(row)

        row_df = pd.DataFrame([row])
        if self.log_path.exists():
            row_df.to_csv(self.log_path, mode="a", header=False, index=False)
        else:
            row_df.to_csv(self.log_path, index=False)

        log.info("epoch=%d  ppl=%.2f  val_tok_hit@1=%.4f  a_tok_hit@1=%.4f  "
                 "b_tok_hit@1=%.4f  b_tok_cer=%.4f", row["epoch"], ppl, val_hit1,
                 a.get("tok_hit@1", 0), b.get("tok_hit@1", 0), b.get("tok_cer", 0))

        if val_hit1 > self.best_hit1:
            self.best_hit1, self.best_epoch = val_hit1, row["epoch"]
            self.epochs_no_improve = 0
            model.save_pretrained(self.best_dir)
            self.tokenizer.save_pretrained(self.best_dir)
            log.info("↑ new best (val_tok_hit@1=%.4f) epoch=%d → %s",
                     self.best_hit1, self.best_epoch, self.best_dir)
        else:
            self.epochs_no_improve += 1
            if self.epochs_no_improve >= self.patience:
                log.info("Early stopping: no improvement for %d epochs", self.patience)
                control.should_training_stop = True
        return control


def finetune(model_name, train_texts, eval_texts, test_a_texts, test_b_records, *,
             output_dir, log_path, epochs, lr, batch_size, max_length, mlm_prob,
             max_eval_gaps, model_key, run_id, patience=3):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model     = AutoModelForMaskedLM.from_pretrained(model_name).to("cuda")
    tokenizer.add_special_tokens({"additional_special_tokens": [GAP_TOKEN]})
    model.resize_token_embeddings(len(tokenizer))

    train_ds = build_train_dataset(train_texts, tokenizer, max_length, model_key=model_key)
    eval_ds  = build_train_dataset(eval_texts,  tokenizer, max_length, model_key=model_key)
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True,
                                               mlm_probability=mlm_prob)
    best_dir = output_dir / "best_by_val"
    callback = EvalCallback(tokenizer, test_a_texts, test_b_records, best_dir,
                            log_path=log_path, max_eval_gaps=max_eval_gaps,
                            model_key=model_key, run_id=run_id, patience=patience,
                            val_texts=eval_texts)

    t_args = TrainingArguments(
        output_dir=str(output_dir), num_train_epochs=epochs, learning_rate=lr,
        per_device_train_batch_size=batch_size, per_device_eval_batch_size=batch_size,
        evaluation_strategy="epoch", save_strategy="epoch",
        load_best_model_at_end=False, logging_steps=50, report_to=[],
        warmup_ratio=0.05, max_grad_norm=1.0)
    Trainer(model=model, args=t_args, train_dataset=train_ds, eval_dataset=eval_ds,
            data_collator=collator, callbacks=[callback]).train()
    return model, tokenizer, callback


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    max_eval_gaps = args.max_eval_gaps if args.max_eval_gaps > 0 else None
    run_id   = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = args.output_dir / "epoch_log.csv"

    raw_train = load_txt(args.data_dir / "train.txt")
    raw_eval  = load_txt(args.data_dir / "eval.txt")
    test_a    = load_txt(args.data_dir / "test_a.txt")
    log.info("train=%d  eval=%d  test_a=%d", len(raw_train), len(raw_eval), len(test_a))

    summary_rows = []
    for model_key in args.models:
        cfg = MODELS[model_key]
        out_dir = args.output_dir / model_key
        tb_path = args.data_dir / f"test_b_tokens_{cfg['suffix']}.jsonl"
        if not tb_path.exists():
            log.error("missing %s — run prepare_test_b_tokens.py first", tb_path)
            continue

        test_b = load_jsonl(tb_path)
        log.info("=== %s  test_b=%d ===", model_key, len(test_b))

        model, tokenizer, cb = finetune(
            cfg["hf_name"], raw_train, raw_eval, test_a, test_b,
            output_dir=out_dir, log_path=log_path,
            epochs=args.epochs, lr=args.lr, batch_size=args.batch_size,
            max_length=args.max_length, mlm_prob=args.mlm_prob,
            max_eval_gaps=max_eval_gaps, model_key=model_key, run_id=run_id,
            patience=args.patience)

        del model
        torch.cuda.empty_cache()

        # Final per-position Test B report on the best checkpoint (same logic as
        # eval_tokens_report.py, which can regenerate it without retraining).
        best_dir = out_dir / "best_by_val"
        log.info("Writing report from best checkpoint → %s", best_dir)
        best_model = AutoModelForMaskedLM.from_pretrained(best_dir).to("cuda")
        eval_report_test_b(best_model, tokenizer, test_b,
                           report_path=out_dir / "report_test_b_tokens.csv")
        del best_model
        torch.cuda.empty_cache()

        best_row = next((r for r in cb.epoch_metrics
                         if r["epoch"] == cb.best_epoch), {})
        summary_rows.append({
            "model": model_key, "best_epoch": cb.best_epoch,
            "ppl": best_row.get("ppl", float("nan")),
            "val_tok_hit@1": best_row.get("val_tok_hit@1", 0),
            **{k: best_row.get(k, 0) for k in
               ("a_tok_hit@1", "a_tok_hit@5", "a_tok_cer", "a_total_gaps",
                "b_tok_hit@1", "b_tok_hit@5", "b_tok_cer", "b_total_gaps")},
        })

    df = pd.DataFrame(summary_rows)
    df.to_csv(args.output_dir / "comparison_summary_tokens.csv", index=False)
    log.info("\n%s", df.to_string(index=False))


if __name__ == "__main__":
    main()