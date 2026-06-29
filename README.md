# Birchbark Restoration

Code for automatic restoration of lost fragments in Old Novgorodian
birchbark manuscripts, and for two auxiliary classification probes (genre and
dating).

The core experiment compares three pretrained BERT-style encoders — [**mBERT**](https://huggingface.co/google-bert/bert-base-multilingual-cased),
[**BERTislav**](https://huggingface.co/npedrazzini/BERTislav), and [**ModernBERT**](https://huggingface.co/answerdotai/ModernBERT-base) — at filling masked positions in fragmentary
Old East Slavic text, evaluated in two prediction modes:

- **character-level**, where the model's output is constrained to a single
  character of the target alphabet, and
- **token-level**, where the model predicts over its native subword vocabulary.

Each encoder is evaluated zero-shot and after fine-tuning, on two test
tracks: Test A (artificial gaps) and Test B (real
editorial reconstructions). A character n-gram
restoration model and a TF-IDF classifier are used as baselines.

Сharacter- and token-level restoration on real
editorial reconstructions, Hit@1 (%):

| Model | char, zero-shot | char, fine-tuned | token, fine-tuned |
|-------|:---:|:---:|:---:|
| mBERT | 4.98 | 6.40 | 30.08 |
| BERTislav | 4.74 | 4.20 | 20.22 |
| ModernBERT | **7.38** | **16.54** | **30.72** |
| Char *n*-gram (n=5) | — | 15.01 | — |

## Repository layout

```
birchbark-restoration/
├── data/
│   └── date_bins.py            # dating-bin schema (shared by probe + baseline)
├── pretrained/                 # experiments with pretrained encoders
│   ├── normalize.py            # model-specific character normalization
│   ├── prenormalize.py         # apply normalization to the splits, once
│   ├── restoration/            # the masked-restoration task
│   │   ├── char_eval.py        #   shared char-level eval core
│   │   ├── token_eval.py       #   shared token-level eval core
│   │   ├── finetune_char.py    #   fine-tune + report (character mode)
│   │   ├── finetune_tokens.py  #   fine-tune + report (token mode)
│   │   ├── eval_char_report.py #   regenerate char reports without retraining
│   │   ├── eval_tokens_report.py # regenerate token reports without retraining
│   │   ├── zeroshot.py         #   zero-shot evaluation (both modes)
│   │   ├── prepare_test_b_tokens.py  # build per-model token-level Test B
│   │   └── plot_finetune_curves.py
│   └── classification/         # genre / dating probes on frozen embeddings
│       ├── config_probe.py     #   paths, model registry, hyperparameters
│       ├── prepare_probe_data.py
│       ├── embed.py            #   extract frozen embeddings
│       └── probe.py            #   train + evaluate the linear probes
├── baseline/                   # non-neural reference systems
│   ├── ngram_restoration_baseline.py
│   └── tfidf_classification_baseline.py
└── outputs/                    # all results land here (created at run time)
    ├── finetune_char/
    ├── finetune_tokens/
    ├── zeroshot/
    ├── baseline/
    └── classification/
```

## Installation

Install the dependencies:

```bash
pip install -r requirements.txt
```

## Data

The restoration scripts expect pre-split corpus files under `data/splits/`:

```
data/splits/train.txt          # one document per line
data/splits/eval.txt           # validation
data/splits/test_a.txt         # Test A source (random gaps applied at eval time)
data/splits/test_b.jsonl       # Test B: {"masked_input": ..., "target": ...}
```

The classification probes additionally expect `data/birchbark_classes.jsonl`
(records with `category_mapped`, `date`, and `date_target` fields).

### Corpus sources
 
The fine-tuning corpus is assembled from the preprocessed sources below.
 
| Source | Language | Word tokens | Link |
|--------|----------|-------------|------|
| Birchbark manuscripts | Old Novgorodian (mostly) | 19,045 | [gramoty.ru](https://gramoty.ru) |
| Epigraphy | Old Church Slavonic (mostly) | 7,095 | [epigraphica.ru](https://epigraphica.ru) |
| DIACU | Old Church Slavonic; Church Slavonic (Old Russian, Middle Bulgarian, Serbian, Resava recensions); Middle Russian | 1,588,323 | [ACL Anthology](https://aclanthology.org/2025.bsnlp-1.12/) |
| TOROT | Old Russian; Church Slavonic | 603,047 | [torottreebank.github.io](https://torottreebank.github.io) |
| Bible (Ponomar) | Church Slavonic | 682,430 | [GitHub](https://github.com/typiconman/ponomar/tree/master/Ponomar/languages/cu/bible/elis) |
| Byliny | Old Russian (11th–17th c.) | 42,412 | [rusneb.ru](https://rusneb.ru/catalog/000199_000009_003636356/) |
| Pushkin House | Old Russian | 430,103 | [lib2.pushkinskijdom.ru](https://lib2.pushkinskijdom.ru) |
| Military Statute (Part 2) | Old Russian | 49,787 | [rusneb.ru](https://rusneb.ru/catalog/000199_000009_004093983/) |
| NKRYA (historical) | Old Russian (11th–18th c.), Old Novgorodian | 327,315 | [ruscorpora.ru](https://ruscorpora.ru) |
 
The sources carry differing licenses, so we cannot redistribute the assembled
training corpus. We are able to share only the birchbark and epigraphic
material underlying Test B and the classification dataset. For every other source, the table above links to
the original.

## Running the experiments

All scripts take sensible defaults and write under `outputs/`. Run each from
its own directory.

### 1. Prepare model-specific data

mBERT and BERTislav need their text normalized to their vocabularies;
ModernBERT uses the original text. This step writes
`data/splits/{mbert,bertislav}/`:

```bash
cd pretrained
python prenormalize.py
```

The token-level experiments also need per-model tokenized Test B files
(`data/splits/test_b_tokens_<model>.jsonl`):

```bash
cd restoration
python prepare_test_b_tokens.py    # → data/splits/test_b_tokens_<model>.jsonl
```

### 2. Restoration — zero-shot

```bash
cd pretrained/restoration
python zeroshot.py             # → outputs/zeroshot/<Model>/
```

### 3. Restoration — fine-tuning

Character mode and token mode are separate scripts. Each fine-tunes all three
encoders, keeps the checkpoint with the best validation Hit@1, and writes a
per-position prediction report at the end.

```bash
cd pretrained/restoration

python finetune_char.py        # → outputs/finetune_char/<Model>/
python finetune_tokens.py      # → outputs/finetune_tokens/<Model>/
```

Flags: `--models ModernBERT` (one model), `--epochs N`, `--batch_size N`.

### 4. Regenerating reports (optional)

To rebuild the per-position CSV reports from an existing checkpoint without
retraining:

```bash
cd pretrained/restoration
python eval_char_report.py     # reads outputs/finetune_char/<Model>/best_by_val
python eval_tokens_report.py   # reads outputs/finetune_tokens/<Model>/best_by_val
```

### 5. Non-neural baselines

```bash
cd baseline
python ngram_restoration_baseline.py        # → outputs/baseline/
python tfidf_classification_baseline.py     # → outputs/baseline/
```

The *n*-gram baseline mirrors the encoders' character-level evaluation exactly
(same masking scheme and seed on Test A). The TF-IDF baseline uses the same
classification splits as the probes.

### 6. Classification probes

Probes read the fine-tuned encoders from `outputs/finetune_char/` and
`outputs/finetune_tokens/`, so run the fine-tuning step first.

```bash
cd pretrained/classification

python prepare_probe_data.py   # → outputs/classification/data/{train,val,test}.jsonl
python embed.py                # → outputs/classification/embeddings/
python probe.py                # → outputs/classification/results/
```

`config_probe.py` holds the model registry, paths, and probe hyperparameters.

### 7. Plots

```bash
cd pretrained/restoration
python plot_finetune_curves.py   # validation Hit@1 curves (char + token)
```

## Outputs

| Directory | Produced by | Contents |
|---|---|---|
| `outputs/finetune_char/` | `finetune_char.py` | checkpoints, `report_test_a/b.csv`, `epoch_log.csv` |
| `outputs/finetune_tokens/` | `finetune_tokens.py` | checkpoints, `report_test_b_tokens.csv`, `epoch_log.csv` |
| `outputs/zeroshot/` | `zeroshot.py` | per-model zero-shot reports + `zeroshot_summary.csv` |
| `outputs/baseline/` | baselines | `restoration_baseline_summary.csv`, `classification_baseline_summary.csv`, report CSVs |
| `outputs/classification/` | probe pipeline | `data/`, `embeddings/`, `results/` (preds, confusion matrices, t-SNE, `probe_results.json`) |

## Acknowledgements

Special thanks to Oleksandr Sychov and Yurii Mikhalchevskyi, who assembled most of the fine-tuning corpus and provided invaluable discussions. The from-scratch model experiments were developed in collaboration with them; see [restoring-ancient-russian-texts](https://github.com/ukolchuga/restoring-ancient-russian-texts).