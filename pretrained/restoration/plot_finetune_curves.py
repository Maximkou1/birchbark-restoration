#!/usr/bin/env python3
"""
plot_finetune_curves.py
~~~~~~~~~~~~~~~~~~~~~~~
Plots validation Hit@1 curves for fine-tuned models (char and token modes).

The val metric is stored as the last comma-separated value in each row
(appended beyond the standard column count):
  - char log:  last value = val_hit@1 (from SingleCharCyrillicProcessor eval)
  - token log: last value = val_tok_hit@1

Usage:
    python plot_finetune_curves.py \
        --char_log epoch_log.csv \
        --tok_log  epoch_log_tokens.csv \
        --output   finetune_val_curves.pdf
"""

import argparse
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def read_log(path: str) -> list[dict]:
    """Read CSV with variable number of columns per row."""
    rows = []
    with open(path, encoding='utf-8') as f:
        header = f.readline().strip().split(',')
        n_cols = len(header)
        for line in f:
            parts = line.strip().split(',')
            if len(parts) < n_cols:
                continue
            row = dict(zip(header, parts[:n_cols]))
            # Last column beyond the header = val metric
            if len(parts) > n_cols:
                row['val_metric'] = float(parts[-1])
            else:
                row['val_metric'] = None
            rows.append(row)
    return rows


def latest_run(rows: list[dict]) -> list[dict]:
    """Keep only the most recent run_id per model."""
    by_model = defaultdict(list)
    for r in rows:
        by_model[r['model']].append(r)
    result = []
    for model, rs in by_model.items():
        latest_id = max(r['run_id'] for r in rs)
        result.extend(r for r in rs if r['run_id'] == latest_id)
    return result


def plot_val_curves(
    char_log: str,
    tok_log:  str,
    output:   str,
) -> None:
    char_rows = latest_run(read_log(char_log))
    tok_rows  = latest_run(read_log(tok_log))

    colors = {
        'mBERT':      '#2980B9',
        'BERTislav':  '#C0392B',
        'ModernBERT': '#27AE60',
    }
    dashes = {
        'mBERT':      (1, 0),
        'BERTislav':  (1, 0),
        'ModernBERT': (1, 0),
    }
    models = ['mBERT', 'BERTislav', 'ModernBERT']

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    fig.patch.set_facecolor('white')

    for ax, rows, ylabel, title, show_legend in [
        (axes[0], char_rows,
         'Val Hit@1 (char, %)',
         'Character-level fine-tuning\n(validation Hit@1)', False),
        (axes[1], tok_rows,
         'Val Hit@1 (token, %)',
         'Token-level fine-tuning\n(validation Hit@1)', True),
    ]:
        for model in models:
            sub = sorted(
                [r for r in rows if r['model'] == model],
                key=lambda x: int(x['epoch'])
            )
            ep  = [int(r['epoch'])   for r in sub if r['val_metric'] is not None]
            val = [r['val_metric'] * 100 for r in sub if r['val_metric'] is not None]
            if not ep:
                continue
            ax.plot(ep, val,
                    color=colors[model],
                    label=model,
                    linewidth=1.8,
                    dashes=dashes[model],
                    marker='o',
                    markersize=2.5)

        ax.set_title(title, fontsize=10, pad=6)
        ax.set_xlabel('Epoch', fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.tick_params(labelsize=8)
        if show_legend:
            ax.legend(fontsize=8, framealpha=0.7)
        ax.grid(True, alpha=0.25, linestyle=':')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))

    plt.tight_layout(pad=1.5)
    plt.savefig(output, bbox_inches='tight')
    # Also save PNG alongside
    png_path = output.replace('.pdf', '.png')
    plt.savefig(png_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output}")
    print(f"Saved: {png_path}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--char_log', default='outputs/finetune_char/epoch_log.csv')
    p.add_argument('--tok_log',  default='outputs/finetune_tokens/epoch_log.csv')
    p.add_argument('--output',   default='outputs/finetune_val_curves.pdf')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    plot_val_curves(args.char_log, args.tok_log, args.output)