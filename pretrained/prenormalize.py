#!/usr/bin/env python3
"""
prenormalize.py
~~~~~~~~~~~~~~~
Нормализует train/eval/test_a/test_b один раз для BERTislav и mBERT,
сохраняет в отдельные папки.

Структура вывода:
  data/splits/bertislav/train.txt
  data/splits/bertislav/eval.txt
  data/splits/bertislav/test_a.txt
  data/splits/bertislav/test_b.jsonl
  data/splits/mbert/  (то же самое)

ModernBERT использует оригинальные данные без нормализации.

Запуск:
  python prenormalize.py
  python prenormalize.py --data_dir /path/to/splits
"""

import argparse
import json
import re
from pathlib import Path

from normalize import norm_bertislav, norm_mbert

MODELS = {
    "bertislav": norm_bertislav,
    "mbert":     norm_mbert,
}

_HERE = Path(__file__).resolve().parents[1]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default=str(_HERE / "data/splits"), type=Path)
    return p.parse_args()


def apply_norm(text: str, norm_fn) -> str:
    """Нормализует текст, защищая служебные токены."""
    protected = {}

    def _protect(m):
        key = f"\x00{len(protected)}\x00"
        protected[key] = m.group(0)
        return key

    text = re.sub(r"\[(?:GAP|MASK|CTX_[A-Z_]+)\]", _protect, text)
    text = norm_fn(text)
    for key, val in protected.items():
        text = text.replace(key, val)
    return text


def normalize_txt(src: Path, dst: Path, norm_fn):
    lines = [l.strip() for l in src.read_text("utf-8").splitlines() if l.strip()]
    normalized = [apply_norm(l, norm_fn) for l in lines]
    dst.write_text("\n".join(normalized) + "\n", encoding="utf-8")
    print(f"  {src.name}: {len(lines)} строк → {dst}")


def normalize_jsonl(src: Path, dst: Path, norm_fn):
    records = [json.loads(l) for l in src.read_text("utf-8").splitlines() if l.strip()]
    normalized = []
    for r in records:
        normalized.append({
            **r,
            "masked_input": apply_norm(r["masked_input"], norm_fn),
            "target":       apply_norm(r["target"],       norm_fn),
        })
    dst.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in normalized) + "\n",
        encoding="utf-8",
    )
    print(f"  {src.name}: {len(records)} записей → {dst}")


def main():
    args = parse_args()
    src_dir = args.data_dir

    for model_key, norm_fn in MODELS.items():
        print(f"\n{'='*50}")
        print(f"  {model_key}  ({norm_fn.__name__})")
        print('='*50)

        dst_dir = src_dir / model_key
        dst_dir.mkdir(parents=True, exist_ok=True)

        for fname in ("train.txt", "eval.txt", "test_a.txt"):
            src = src_dir / fname
            if not src.exists():
                print(f"  ⚠️  {fname} не найден, пропускаю")
                continue
            normalize_txt(src, dst_dir / fname, norm_fn)

        src_b = src_dir / "test_b.jsonl"
        if src_b.exists():
            normalize_jsonl(src_b, dst_dir / "test_b.jsonl", norm_fn)
        else:
            print(f"  ⚠️  test_b.jsonl is not found, skipping")

    print("\n✅ Ready.")
    print(f"\nStructure:")
    print(f"  {src_dir}/bertislav/  ← BERTislav")
    print(f"  {src_dir}/mbert/      ← mBERT")
    print(f"  {src_dir}/            ← ModernBERT (original, no normalization)")


if __name__ == "__main__":
    main()