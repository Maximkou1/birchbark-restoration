#!/usr/bin/env python3
"""
train.py
~~~~~~~~
Training script for DualBertForMaskedLM (char + word tokenization).

Splits:
  train   → training
  eval    → validation during training (PPL + top-k accuracy per epoch,
            full metrics per epoch via EpochEvalCallback)
  test_a  → final evaluation via collator
  test_b  → final evaluation via beam search (bracket masking)
"""

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import load_from_disk
from transformers import (Trainer, TrainingArguments, TrainerCallback,
                          EarlyStoppingCallback)

from align_dual import load_vocab
from build_char_tokenizer import SPECIAL_TOKENS
from collator import DualPhysicalDegradationCollator
from config import DualBertConfig
from evaluate_model import (evaluate_with_collator_dual,
                            evaluate_test_b_dual, print_metrics)
from model import DualBertForMaskedLM

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
log = logging.getLogger(__name__)

_HERE = Path(__file__).resolve().parent              # from_scratch/DualEmbLM/
_ROOT = _HERE.parent.parent                          # repo root


def save_json(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    log.info(f"Saved: {path}")


def preprocess_logits_for_metrics(logits, labels):
    if isinstance(logits, tuple):
        logits = logits[0]
    # Mask out [UNK] so it never appears in top-k predictions.
    # Its id is passed in via preprocess_logits_for_metrics._unk_id (set in main).
    unk_id = getattr(preprocess_logits_for_metrics, "_unk_id", None)
    if unk_id is not None:
        logits[:, :, unk_id] = float("-inf")
    return torch.topk(logits, k=5, dim=-1).indices


def compute_metrics(eval_preds):
    """Fast token-level top-k accuracy for training monitoring."""
    preds, labels = eval_preds
    mask   = labels != -100
    labels = labels[mask]
    preds  = preds[mask]
    if labels.size == 0:
        return {"top1_accuracy": 0.0, "top3_accuracy": 0.0, "top5_accuracy": 0.0}
    return {
        "top1_accuracy": float(np.mean(preds[:, 0] == labels)),
        "top3_accuracy": float(np.mean(np.any(preds[:, :3] == labels[:, None], axis=1))),
        "top5_accuracy": float(np.mean(np.any(preds[:, :5] == labels[:, None], axis=1))),
    }


class LoggingTrainer(Trainer):
    def __init__(self, *args, log_path: Path | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.log_path    = log_path
        self.log_history = []

    def log(self, logs, start_time=None):
        super().log(logs, start_time)
        self.log_history.append({"timestamp": datetime.now().isoformat(), **logs})
        if self.log_path and len(self.log_history) % 10 == 0:
            with open(self.log_path, "w", encoding="utf-8") as f:
                json.dump(self.log_history, f, ensure_ascii=False, indent=2)


class EpochEvalCallback(TrainerCallback):
    """Full metrics (PPL + char Hit@K + CER) on eval after each epoch."""

    def __init__(self, model, id_to_char, eval_dataset, collator,
                 device, output_dir: Path):
        self.model      = model
        self.id_to_char = id_to_char
        self.eval_ds    = eval_dataset
        self.collator   = collator
        self.device     = device
        self.output_dir = output_dir
        self.rows: list[dict] = []

    def on_epoch_end(self, args, state, control, **kwargs):
        epoch = int(state.epoch)
        log.info(f"Epoch {epoch} — computing full eval metrics...")

        prev_gaps = self.collator.add_random_gaps
        self.collator.add_random_gaps = False

        m = evaluate_with_collator_dual(
            self.model, self.id_to_char,
            self.eval_ds, self.collator, self.device,
        )

        self.collator.add_random_gaps = prev_gaps
        row = {"epoch": epoch, **m}
        self.rows.append(row)

        csv_path = self.output_dir / "epoch_metrics.csv"
        pd.DataFrame(self.rows).to_csv(csv_path, index=False)

        log.info(
            "Epoch %d — PPL=%.4f  Hit@1=%.4f  Hit@5=%.4f  CER=%.4f",
            epoch, m["ppl"], m["hit@1"], m["hit@5"], m["char_cer"],
        )
        return control


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_dir",     default=str(_ROOT / "outputs/from_scratch/DualEmbLM/dataset"))
    p.add_argument("--char_vocab_path", default=str(_HERE / "char_tokenizer/char_vocab.json"))
    p.add_argument("--word_vocab_path", default=str(_HERE / "word_vocab.json"))
    p.add_argument("--output_dir",      default=str(_ROOT / "outputs/from_scratch/DualEmbLM"))
    p.add_argument("--test_b_path",     default=str(_ROOT / "data/splits/test_b.jsonl"))
    p.add_argument("--epochs",          type=int,   default=30)
    p.add_argument("--train_bs",        type=int,   default=64)
    p.add_argument("--eval_bs",         type=int,   default=64)
    p.add_argument("--grad_accum",      type=int,   default=2)
    p.add_argument("--lr",              type=float, default=5e-4)
    p.add_argument("--warmup_steps",    type=int,   default=1000)
    p.add_argument("--beam_width",      type=int,   default=20)
    p.add_argument("--fp16",            action="store_true", default=True)
    p.add_argument("--seed",            type=int,   default=42)
    p.add_argument("--patience",        type=int,   default=3,
                   help="Early stopping: epochs without improvement.")
    return p.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")

    # ── Vocabularies ──────────────────────────────────────────────────────────
    char_vocab = load_vocab(args.char_vocab_path)
    word_vocab = load_vocab(args.word_vocab_path)
    id_to_char = {v: k for k, v in char_vocab.items()}

    log.info(f"char vocab: {len(char_vocab)}  word vocab: {len(word_vocab)}")

    # Pass [UNK] id to preprocess_logits_for_metrics so it can mask it out.
    unk_id = char_vocab.get("[UNK]")
    preprocess_logits_for_metrics._unk_id = unk_id
    log.info(f"[UNK] id = {unk_id} — will be masked from predictions")

    # ── Dataset ───────────────────────────────────────────────────────────────
    dataset = load_from_disk(args.dataset_dir)
    log.info(f"train: {len(dataset['train']):,}  eval: {len(dataset['eval']):,}  "
             f"test_a: {len(dataset['test_a']):,}  test_b: {len(dataset['test_b']):,}")

    # ── Model ─────────────────────────────────────────────────────────────────
    special_ids = [char_vocab[t] for t in SPECIAL_TOKENS if t in char_vocab]
    gap_id      = char_vocab.get("[GAP]")

    config = DualBertConfig(
        vocab_char_size=len(char_vocab),
        vocab_word_size=len(word_vocab),
        word_char_emb_dim=192,
        hidden_size=512,
        num_hidden_layers=6,
        num_attention_heads=8,
        intermediate_size=2048,
        max_position_embeddings=256,
        pad_token_id=char_vocab["[PAD]"],
    )
    model = DualBertForMaskedLM(config)
    n_params = sum(p.numel() for p in model.parameters())
    log.info(f"Parameters: {n_params:,}")

    # ── Collator ──────────────────────────────────────────────────────────────
    collator = DualPhysicalDegradationCollator(
        mask_token_id=char_vocab["[MASK]"],
        pad_token_id=char_vocab["[PAD]"],
        unk_word_id=word_vocab["[UNK_WORD]"],
        unk_char_id=char_vocab.get("[UNK]"),
        vocab_char_size=len(char_vocab),
        special_token_ids=special_ids,
        mlm_prob=0.08,
        max_span=3,
        edge_prob=0.1,
        add_random_gaps=True,
        gap_token_id=gap_id,
        gap_prob=0.05,
    )

    # ── Training ──────────────────────────────────────────────────────────────
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path  = output_dir / f"training_log_{timestamp}.json"

    epoch_cb = EpochEvalCallback(
        model=model,
        id_to_char=id_to_char,
        eval_dataset=dataset["eval"],
        collator=collator,
        device=device,
        output_dir=output_dir,
    )

    training_args = TrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.train_bs,
        per_device_eval_batch_size=args.eval_bs,
        gradient_accumulation_steps=args.grad_accum,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=3,
        logging_steps=50,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=args.warmup_steps,
        weight_decay=0.01,
        fp16=args.fp16,
        dataloader_num_workers=4,
        report_to=[],
        load_best_model_at_end=True,
        metric_for_best_model="top1_accuracy",
        greater_is_better=True,
        remove_unused_columns=False,
        max_grad_norm=1.0,
        seed=args.seed,
    )

    trainer = LoggingTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["eval"],
        data_collator=collator,
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        log_path=log_path,
        callbacks=[epoch_cb,
                   EarlyStoppingCallback(early_stopping_patience=args.patience)],
    )

    log.info("Starting training...")
    trainer.train()

    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(trainer.log_history, f, ensure_ascii=False, indent=2)

    # ── Final evaluation ──────────────────────────────────────────────────────
    collator.add_random_gaps = False
    summary = {}

    log.info("Final evaluation: test_a...")
    m_a = evaluate_with_collator_dual(
        model, id_to_char, dataset["test_a"], collator, device,
        output_path=output_dir / f"report_test_a_{timestamp}.csv",
    )
    print_metrics("test_a", m_a)
    save_json(m_a, output_dir / f"metrics_test_a_{timestamp}.json")
    summary["test_a"] = m_a

    log.info("Final evaluation: test_b (beam search)...")
    records = []
    with open(args.test_b_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                records.append({"original": r["original"], "target": r["target"]})

    m_b = evaluate_test_b_dual(
        model, char_vocab, word_vocab, records, device,
        beam_width=args.beam_width,
        output_path=output_dir / f"report_test_b_{timestamp}.csv",
    )
    print_metrics("test_b", m_b)
    save_json(m_b, output_dir / f"metrics_test_b_{timestamp}.json")
    summary["test_b"] = m_b

    save_json(summary, output_dir / f"eval_summary_{timestamp}.json")

    # ── Save model ────────────────────────────────────────────────────────────
    trainer.save_model(str(output_dir / "final_model"))
    log.info(f"Model saved → {output_dir / 'final_model'}")

    import shutil
    shutil.copy(args.char_vocab_path, output_dir / "final_model/char_vocab.json")
    shutil.copy(args.word_vocab_path, output_dir / "final_model/word_vocab.json")


if __name__ == "__main__":
    main()