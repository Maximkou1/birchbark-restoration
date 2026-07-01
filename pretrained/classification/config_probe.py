"""
config_probe.py
~~~~~~~~~~~~~~~
Paths and settings for the classification probing experiments.
"""

from pathlib import Path

# ── Root ──────────────────────────────────────────────────────────────────────

BIRCH_ROOT = Path(__file__).resolve().parents[2]
CLASS_DIR  = BIRCH_ROOT / "outputs" / "classification"
DATA_DIR   = CLASS_DIR  / "data"
EMBED_DIR  = CLASS_DIR  / "embeddings"
RESULT_DIR = CLASS_DIR  / "results"

# ── Data ──────────────────────────────────────────────────────────────────────

CLASSES_JSONL = BIRCH_ROOT / "data" / "birchbark_classes.jsonl"

# Train / val / test split ratios
SPLIT_RATIOS = (0.70, 0.15, 0.15)
SPLIT_SEED   = 42

# ── Category task ─────────────────────────────────────────────────────────────

CATEGORY_LABELS = ["letters", "records", "religious", "other"]
CATEGORY_TO_IDX = {c: i for i, c in enumerate(CATEGORY_LABELS)}

# Class weights for weighted cross-entropy (inverse frequency, approx)
# letters~696, records~330, religious~45, other~171
CATEGORY_WEIGHTS = [1.0, 2.1, 15.5, 4.1]

# ── Models ────────────────────────────────────────────────────────────────────

# Each entry: display name → dict with type and path
# type: "hf_mlm" | "roformer" | "dualemb"
MODELS = {
    # ── Char fine-tuned (existing) ─────────────────────────────────────────
    "BERTislav": {
        "type":    "hf_mlm",
        "path":    str(BIRCH_ROOT / "outputs/finetune_char/BERTislav/best_by_val"),
        "pooling": "cls",
    },
    "mBERT": {
        "type":    "hf_mlm",
        "path":    str(BIRCH_ROOT / "outputs/finetune_char/mBERT/best_by_val"),
        "pooling": "cls",
    },
    "ModernBERT": {
        "type":    "hf_mlm",
        "path":    str(BIRCH_ROOT / "outputs/finetune_char/ModernBERT/best_by_val"),
        "pooling": "cls",
    },
    # ── Token fine-tuned (new) ─────────────────────────────────────────────
    "BERTislav_tok": {
        "type":    "hf_mlm",
        "path":    str(BIRCH_ROOT / "outputs/finetune_tokens/BERTislav/best_by_val"),
        "pooling": "cls",
    },
    "mBERT_tok": {
        "type":    "hf_mlm",
        "path":    str(BIRCH_ROOT / "outputs/finetune_tokens/mBERT/best_by_val"),
        "pooling": "cls",
    },
    "ModernBERT_tok": {
        "type":    "hf_mlm",
        "path":    str(BIRCH_ROOT / "outputs/finetune_tokens/ModernBERT/best_by_val"),
        "pooling": "cls",
    },
    "RoFormer": {
        "type": "roformer",
        "path": str(BIRCH_ROOT / "outputs/from_scratch/RoFormerBPE/final_model"),
        "tokenizer": str(BIRCH_ROOT / "from_scratch/RoFormerBPE/tokenizer"),
        "pooling": "mean",
    },
    "DualEmb": {
        "type": "dualemb",
        "path": str(BIRCH_ROOT / "outputs/from_scratch/DualEmbLM/final_model"),
        "char_vocab": str(BIRCH_ROOT / "from_scratch/DualEmbLM/char_tokenizer/char_vocab.json"),
        "word_vocab": str(BIRCH_ROOT / "from_scratch/DualEmbLM/word_vocab.json"),
        "pooling": "mean",
    },
}

# ── Probing classifier ────────────────────────────────────────────────────────

PROBE_LR        = 1e-3
PROBE_EPOCHS    = 50
PROBE_BATCH     = 64
PROBE_HIDDEN    = None   # None = linear probe; int = one hidden layer MLP
PROBE_DROPOUT   = 0.1
PROBE_SEED      = 42