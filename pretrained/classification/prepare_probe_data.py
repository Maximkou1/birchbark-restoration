#!/usr/bin/env python3
"""
prepare_probe_data.py
~~~~~~~~~~~~~~~~~~~~~
Reads birchbark_classes.jsonl and creates train/val/test splits for
both classification tasks (category and date).

Single split for both tasks — same documents in each fold.

Filters:
  - Records without date_target are excluded from date task
  - Records with category_mapped=None are excluded from category task
  - Both tasks share the same split indices (stratified by category)

Output:
  class_prediction/data/train.jsonl
  class_prediction/data/val.jsonl
  class_prediction/data/test.jsonl
"""

import json
import random
from collections import defaultdict
from pathlib import Path

from config_probe import (
    CLASSES_JSONL, DATA_DIR, SPLIT_RATIOS, SPLIT_SEED,
    CATEGORY_LABELS,
)

random.seed(SPLIT_SEED)


def load_records(path: Path) -> list[dict]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def stratified_split(
    records: list[dict],
    ratios: tuple[float, float, float],
) -> tuple[list[dict], list[dict], list[dict]]:
    """
    Stratified split by category_mapped.
    Records without category_mapped participate in split but are not
    stratified (placed proportionally in each fold).
    """
    train_r, val_r, test_r = ratios

    # Group by category
    by_cat: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        cat = r.get("category_mapped") or "_none"
        by_cat[cat].append(r)

    train, val, test = [], [], []
    for cat, recs in by_cat.items():
        random.shuffle(recs)
        n = len(recs)
        n_test = max(1, round(n * test_r))
        n_val  = max(1, round(n * val_r))
        n_train = n - n_val - n_test

        train.extend(recs[:n_train])
        val.extend(recs[n_train:n_train + n_val])
        test.extend(recs[n_train + n_val:])

    random.shuffle(train)
    random.shuffle(val)
    random.shuffle(test)
    return train, val, test


def print_stats(name: str, records: list[dict]) -> None:
    cat_counts = defaultdict(int)
    date_ok = 0
    for r in records:
        cat_counts[r.get("category_mapped") or "None"] += 1
        if r.get("date_target") is not None:
            date_ok += 1
    print(f"\n  {name} ({len(records)} records, {date_ok} with date_target):")
    for cat in CATEGORY_LABELS + ["None"]:
        n = cat_counts.get(cat, 0)
        if n:
            print(f"    {cat:12s}: {n}")


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    records = load_records(CLASSES_JSONL)
    print(f"Loaded {len(records):,} records from {CLASSES_JSONL}")

    train, val, test = stratified_split(records, SPLIT_RATIOS)

    print("\nSplit statistics:")
    print_stats("train", train)
    print_stats("val",   val)
    print_stats("test",  test)

    for name, split in [("train", train), ("val", val), ("test", test)]:
        out = DATA_DIR / f"{name}.jsonl"
        with open(out, "w", encoding="utf-8") as f:
            for r in split:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"\nSaved {len(split):,} → {out}")


if __name__ == "__main__":
    main()