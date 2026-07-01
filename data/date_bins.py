"""
date_bins.py
~~~~~~~~~~~~
Aeneas-style date binning for birchbark letters.

50-year bins covering 1050–1500:
  bin 0: 1050–1099
  bin 1: 1100–1149
  ...
  bin 8: 1450–1499  (edge bin, few records)

Ground truth: uniform distribution over bins overlapping [date_min, date_max].
Loss:         KL divergence (computed in probe.py via torch.nn.functional.kl_div,
              following Aeneas, Assael et al. 2025).
Prediction:   weighted mean of predicted distribution.
Metric:       distance from predicted mean to [date_min, date_max].
"""

import re
import numpy as np

# ── Bin definitions ───────────────────────────────────────────────────────────

BIN_START  = 1050
BIN_SIZE   = 50
N_BINS     = 9       # 1050–1499  (last bin 1450–1499)

BINS = [(BIN_START + i * BIN_SIZE, BIN_START + (i + 1) * BIN_SIZE - 1)
        for i in range(N_BINS)]

BIN_MIDPOINTS = np.array([(lo + hi) / 2 for lo, hi in BINS])

# Bin labels for display
BIN_LABELS = [f"{lo}–{hi}" for lo, hi in BINS]


# ── Date parsing ──────────────────────────────────────────────────────────────

_YEAR_RE = re.compile(r"\d{4}")

def parse_date_interval(date_str: str) -> tuple[int, int] | None:
    """
    Extract (year_min, year_max) from strings like:
      '1140‒1160'
      '1140‒1160 (с вероятным смещением назад)'
      '1025‒1050 (с вероятным смещением вперёд)'

    Returns None if unparseable.
    'Вероятное смещение' is ignored — the nominal interval is used as-is.
    """
    years = [int(y) for y in _YEAR_RE.findall(date_str)]
    if len(years) < 2:
        return None
    return min(years), max(years)


# ── Target distribution ───────────────────────────────────────────────────────

def make_date_target(date_str: str) -> np.ndarray | None:
    """
    Returns a uniform probability distribution over the bins that overlap
    [year_min, year_max], following Aeneas' binarized-bin approach.

    Returns None if date is unparseable or outside 1050–1499.
    """
    interval = parse_date_interval(date_str)
    if interval is None:
        return None

    year_min, year_max = interval
    weights = np.zeros(N_BINS, dtype=np.float32)

    for i, (bin_lo, bin_hi) in enumerate(BINS):
        # +1 to treat both boundaries as inclusive
        overlap = max(0, min(year_max, bin_hi) - max(year_min, bin_lo) + 1)
        weights[i] = overlap

    # If no overlap at all (entirely outside range), skip
    if weights.sum() == 0:
        return None

    return weights / weights.sum()


# ── Prediction → year ─────────────────────────────────────────────────────────

def predicted_year(dist: np.ndarray) -> float:
    """Weighted mean of the predicted distribution (in years)."""
    return float(np.dot(dist, BIN_MIDPOINTS))


# ── Metric ────────────────────────────────────────────────────────────────────

def date_distance(pred_dist: np.ndarray, date_str: str) -> float | None:
    """
    Distance (years) from predicted mean to ground-truth interval.
    Returns 0 if predicted mean falls inside the interval.
    Follows Aeneas' metric definition.
    """
    interval = parse_date_interval(date_str)
    if interval is None:
        return None

    gt_min, gt_max = interval
    pred_avg = predicted_year(pred_dist)

    if gt_min <= pred_avg <= gt_max:
        return 0.0
    elif pred_avg > gt_max:
        return pred_avg - gt_max
    else:
        return gt_min - pred_avg


# if __name__ == "__main__":
#     test_cases = [
#         "1140‒1160",
#         "1140‒1160 (с вероятным смещением назад)",
#         "1025‒1050",
#         "1380‒1400",
#         "1430‒1450",
#         "1450‒1500",
#     ]
#
#     print(f"Bins ({N_BINS} total, {BIN_SIZE}-year):")
#     for i, (lo, hi) in enumerate(BINS):
#         print(f"  [{i}] {lo}–{hi}  midpoint={BIN_MIDPOINTS[i]:.0f}")
#
#     print()
#     for ds in test_cases:
#         interval = parse_date_interval(ds)
#         target   = make_date_target(ds)
#         if target is None:
#             print(f"  {ds!r:40s}  → SKIP")
#             continue
#         pred_y   = predicted_year(target)
#         dist_str = " ".join(f"{v:.2f}" for v in target)
#         print(f"  {ds!r:45s}  interval={interval}  "
#               f"mean={pred_y:.1f}  dist=[{dist_str}]")