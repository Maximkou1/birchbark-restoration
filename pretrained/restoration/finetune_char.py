#!/usr/bin/env python3
"""
finetune_char.py
~~~~~~~~~~~~~~~~
Fine-tune BERTislav / mBERT / ModernBERT on pre-split data and evaluate in
CHARACTER-level mode. Per-epoch metrics (Hit@1/3/5, CER, PPL) are written to
epoch_log.csv; the checkpoint with the best validation Hit@1 is retained.

Evaluation/decoding logic lives in char_eval.py (shared with
eval_char_report.py).

Dataset layout:
  data/splits/train.txt
  data/splits/eval.txt
  data/splits/<model>/test_a.txt
  data/splits/<model>/test_b.jsonl
"""

import argparse
import json
import logging
import math
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch
from datasets import Dataset
from transformers import (AutoModelForMaskedLM, AutoTokenizer,
                          DataCollatorForLanguageModeling,
                          Trainer, TrainingArguments, TrainerCallback)

from char_eval import (GAP_TOKEN, _ALLOWED_PUNCT,
                       SingleCharCyrillicProcessor, _decoded_single_char,
                       evaluate_test_a, evaluate_test_b)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)s  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

MODELS = {
    "BERTislav":  {"hf_name": "npedrazzini/BERTislav",
                   "data_dir": "bertislav"},
    "mBERT":      {"hf_name": "google-bert/bert-base-multilingual-cased",
                   "data_dir": "mbert"},
    "ModernBERT": {"hf_name": "answerdotai/ModernBERT-base",
                   "data_dir": None},
}

_HERE = Path(__file__).resolve().parents[2]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir",       default=str(_HERE / "data/splits"), type=Path)
    p.add_argument("--output_dir",     default=str(_HERE / "outputs/finetune_char"), type=Path)
    p.add_argument("--epochs",         default=30,   type=int)
    p.add_argument("--patience",       default=3,    type=int)
    p.add_argument("--batch_size",     default=64,   type=int)
    p.add_argument("--lr",             default=5e-5, type=float)
    p.add_argument("--max_length",     default=256,  type=int)
    p.add_argument("--mlm_prob",       default=0.08, type=float)
    p.add_argument("--seed",           default=42,   type=int)
    p.add_argument("--max_eval_gaps",  default=5000, type=int)
    p.add_argument("--max_final_gaps", default=0,    type=int,
                   help="Max positions for the final report eval (0 = no limit).")
    p.add_argument("--models",         nargs="+",
                   default=list(MODELS.keys()), choices=list(MODELS.keys()))
    return p.parse_args()


def load_txt(path: Path) -> list[str]:
    return [l.strip() for l in path.read_text("utf-8").splitlines() if l.strip()]


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text("utf-8").splitlines() if l.strip()]


def build_train_dataset(texts: list[str], tokenizer, max_length: int) -> Dataset:
    ds = Dataset.from_dict({"text": texts})

    def _tok(batch):
        result = {"input_ids": [], "attention_mask": []}
        for text in batch["text"]:
            enc  = tokenizer(text, add_special_tokens=True,
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
    """After each epoch: compute char Hit@K/CER on val/test_a/test_b, append a
    row to epoch_log.csv, and keep the best checkpoint by val Hit@1."""

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

    def on_evaluate(self, args, state, control, model=None, metrics=None, **kwargs):
        eval_loss  = (metrics or {}).get("eval_loss", float("nan"))
        ppl        = math.exp(eval_loss) if math.isfinite(eval_loss) else float("nan")
        train_loss = next(
            (e["loss"] for e in reversed(state.log_history)
             if "loss" in e and "eval_loss" not in e), float("nan"))

        a = evaluate_test_a(model, self.tokenizer, self.a_texts,
                            max_eval_gaps=self.max_eval_gaps)
        b = evaluate_test_b(model, self.tokenizer, self.b_records,
                            max_eval_gaps=self.max_eval_gaps)
        val = evaluate_test_a(model, self.tokenizer, self.val_texts,
                              max_eval_gaps=self.max_eval_gaps) if self.val_texts else {}
        val_hit1 = val.get("hit@1", 0.0)

        row = {
            "run_id": self.run_id, "model": self.model_key,
            "epoch": int(state.epoch),
            "train_loss": round(train_loss, 4), "eval_loss": round(eval_loss, 4),
            "ppl": round(ppl, 2),
            **{f"a_{k}": v for k, v in a.items()},
            **{f"b_{k}": v for k, v in b.items()},
            "val_hit@1": val_hit1,
        }
        self.epoch_metrics.append(row)

        row_df = pd.DataFrame([row])
        if self.log_path.exists():
            row_df.to_csv(self.log_path, mode="a", header=False, index=False)
        else:
            row_df.to_csv(self.log_path, index=False)

        log.info("epoch=%d  ppl=%.2f  val_hit@1=%.4f  a_hit@1=%.4f  "
                 "b_hit@1=%.4f  b_cer=%.4f", row["epoch"], ppl, val_hit1,
                 a.get("hit@1", 0), b.get("hit@1", 0), b.get("cer", 0))

        if val_hit1 > self.best_hit1:
            self.best_hit1, self.best_epoch = val_hit1, row["epoch"]
            self.epochs_no_improve = 0
            model.save_pretrained(self.best_dir)
            self.tokenizer.save_pretrained(self.best_dir)
            log.info("↑ new best (val_hit@1=%.4f) epoch=%d → %s",
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

    # Sanity check: reachable single-character output space for this model.
    _proc = SingleCharCyrillicProcessor(tokenizer)
    _m    = _proc._get_mask(model.config.vocab_size, "cpu")
    _allowed_ids = _m.nonzero().flatten().tolist()
    _has_space = any(_decoded_single_char(tokenizer, t) == " " for t in _allowed_ids)
    _puncts = sorted({_decoded_single_char(tokenizer, t) for t in _allowed_ids}
                     & _ALLOWED_PUNCT)
    log.info("char output space: %d tokens | space_reachable=%s | punct=%s",
             len(_allowed_ids), _has_space, _puncts)

    train_ds = build_train_dataset(train_texts, tokenizer, max_length)
    eval_ds  = build_train_dataset(eval_texts,  tokenizer, max_length)
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
    log.info("train=%d  eval=%d", len(raw_train), len(raw_eval))

    summary_rows = []
    for model_key in args.models:
        cfg      = MODELS[model_key]
        subdir   = cfg["data_dir"]
        out_dir  = args.output_dir / model_key
        data_dir = args.data_dir / subdir if subdir else args.data_dir

        if not data_dir.exists():
            log.error("missing data dir %s — run prenormalize.py", data_dir)
            continue

        test_a_texts = load_txt(data_dir / "test_a.txt")
        test_b_texts = load_jsonl(data_dir / "test_b.jsonl")
        log.info("=== %s  test_a=%d  test_b=%d ===",
                 model_key, len(test_a_texts), len(test_b_texts))

        model, tokenizer, cb = finetune(
            cfg["hf_name"], raw_train, raw_eval, test_a_texts, test_b_texts,
            output_dir=out_dir, log_path=log_path,
            epochs=args.epochs, lr=args.lr, batch_size=args.batch_size,
            max_length=args.max_length, mlm_prob=args.mlm_prob,
            max_eval_gaps=max_eval_gaps, model_key=model_key, run_id=run_id,
            patience=args.patience)

        del model
        torch.cuda.empty_cache()

        # Final per-position report on the best checkpoint (same logic as
        # eval_char_report.py, which can regenerate these without retraining).
        best_dir = out_dir / "best_by_val"
        log.info("Writing reports from best checkpoint → %s", best_dir)
        best_model = AutoModelForMaskedLM.from_pretrained(best_dir).to("cuda")
        max_final = args.max_final_gaps if args.max_final_gaps > 0 else None
        evaluate_test_a(best_model, tokenizer, test_a_texts, seed=args.seed,
                        max_eval_gaps=max_final,
                        report_path=out_dir / "report_test_a.csv")
        evaluate_test_b(best_model, tokenizer, test_b_texts,
                        max_eval_gaps=max_final,
                        report_path=out_dir / "report_test_b.csv")
        del best_model
        torch.cuda.empty_cache()

        best_row = next((r for r in cb.epoch_metrics
                         if r["epoch"] == cb.best_epoch), {})
        summary_rows.append({
            "model": model_key, "best_epoch": cb.best_epoch,
            "ppl": best_row.get("ppl", float("nan")),
            "val_hit@1": best_row.get("val_hit@1", 0),
            **{k: best_row.get(k, 0) for k in
               ("a_hit@1", "a_hit@3", "a_hit@5", "a_cer", "a_total_gaps",
                "b_hit@1", "b_hit@3", "b_hit@5", "b_cer", "b_total_gaps")},
        })

    df = pd.DataFrame(summary_rows)
    df.to_csv(args.output_dir / "comparison_summary.csv", index=False)
    log.info("\n%s", df.to_string(index=False))


if __name__ == "__main__":
    main()