"""
beam_search.py
~~~~~~~~~~~~~~
Confidence-ordered (easy-first) beam search for RoFormer, in the style of
Ithaca (Assael et al. 2022).

Algorithm per step:
  1. Forward pass — get logits for ALL remaining [MASK] positions at once.
  2. Pick the position with the highest model confidence.
  3. Expand only that position: take its top-k tokens.
  4. Prune to top-k beams by cumulative log-probability.
  5. Repeat until no masks remain.

Used by evaluate_model.py for span-level Test B evaluation.
"""

from pathlib import Path
import math
from dataclasses import dataclass, field

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase


@dataclass
class Beam:
    input_ids: torch.Tensor          # [seq_len]
    log_prob:  float = 0.0
    # (position, token_id) pairs in fill order
    filled:    list[tuple] = field(default_factory=list)


def beam_search(
    input_ids: torch.Tensor,          # [seq_len], already on the target device
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    *,
    beam_width: int = 5,
    temperature: float = 1.0,
    banned_token_ids: list[int] | None = None,   # e.g. [GAP_ID]
) -> list[Beam]:
    """
    Returns `beam_width` beams sorted by descending log-probability.
    """
    device = input_ids.device
    mask_id = tokenizer.mask_token_id
    banned = set(banned_token_ids or [])

    # Initialize a single starting beam
    beams: list[Beam] = [Beam(input_ids=input_ids.clone())]

    # Count how many masks need filling
    n_masks = (input_ids == mask_id).sum().item()

    with torch.no_grad():
        for _ in range(n_masks):
            candidates: list[Beam] = []

            for beam in beams:
                mask_positions = (beam.input_ids == mask_id).nonzero(
                    as_tuple=True)[0].tolist()

                if not mask_positions:
                    candidates.append(beam)
                    continue

                # ── Forward pass ───────────────────────────────────────────
                logits = model(
                    beam.input_ids.unsqueeze(0)
                ).logits[0]               # [seq_len, vocab]

                # ── Find the highest-confidence position ─────────────────
                # For each mask, take the probability of its most likely token.
                best_pos   = None
                best_conf  = -1.0

                for pos in mask_positions:
                    pos_logits = logits[pos] / max(temperature, 1e-6)
                    if banned:
                        pos_logits = pos_logits.clone()
                        for tid in banned:
                            if tid < pos_logits.shape[-1]:
                                pos_logits[tid] = float("-inf")
                    max_prob = pos_logits.softmax(dim=-1).max().item()
                    if max_prob > best_conf:
                        best_conf = max_prob
                        best_pos  = pos

                # ── Expand exactly this position ──────────────────────────
                pos_logits = logits[best_pos] / max(temperature, 1e-6)
                if banned:
                    pos_logits = pos_logits.clone()
                    for tid in banned:
                        if tid < pos_logits.shape[-1]:
                            pos_logits[tid] = float("-inf")

                probs    = pos_logits.softmax(dim=-1)
                top_probs, top_ids = probs.topk(beam_width)

                for prob, token_id in zip(top_probs.tolist(), top_ids.tolist()):
                    if prob <= 0:
                        continue
                    new_ids = beam.input_ids.clone()
                    new_ids[best_pos] = token_id
                    candidates.append(Beam(
                        input_ids = new_ids,
                        log_prob  = beam.log_prob + math.log(prob + 1e-12),
                        filled    = beam.filled + [(best_pos, token_id)],
                    ))

            # ── Pruning: keep the top-beam_width beams ────────────────
            beams = sorted(candidates, key=lambda b: b.log_prob, reverse=True)
            beams = beams[:beam_width]

    return beams


# ── Helper: decode beam results ───────────────────────────

def decode_beams(
    beams: list[Beam],
    original_ids: torch.Tensor,
    tokenizer: PreTrainedTokenizerBase,
) -> list[dict]:
    """
    Turns beams into a readable list of dicts.

    Returns:
        [
          {
            "text":          fully restored text,
            "filled_tokens": [(position, token_str), ...] in fill order,
            "score":         normalized probability (0..1),
            "log_prob":      cumulative log-prob,
          },
          ...
        ]
    """
    results = []
    # Normalize probabilities via softmax over beam log-probs
    log_probs = torch.tensor([b.log_prob for b in beams], dtype=torch.float)
    scores    = log_probs.softmax(dim=0).tolist()

    for beam, score in zip(beams, scores):
        text = tokenizer.decode(beam.input_ids, skip_special_tokens=True)

        filled_tokens = [
            (pos, tokenizer.decode([tid], skip_special_tokens=True,
                                   clean_up_tokenization_spaces=False).strip())
            for pos, tid in beam.filled
        ]

        results.append({
            "text":          text,
            "filled_tokens": filled_tokens,
            "score":         round(score, 4),
            "log_prob":      round(beam.log_prob, 4),
        })

    return results


# ── High-level interface ──────────────────────────────────

def restore(
    text: str,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    *,
    beam_width: int = 5,
    temperature: float = 1.0,
    gap_token: str = "[GAP]",
    max_length: int = 512,
) -> list[dict]:
    """
    High-level wrapper: takes a string with [MASK], returns a list of beams.

    Args:
        text:        text with one or more [MASK] tokens.
        gap_token:   the lacuna token — excluded from predictions.
        beam_width:  number of beams.
        temperature: <1 sharpens the distribution, >1 softens it.
    """
    device = next(model.parameters()).device

    enc = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    )
    input_ids = enc["input_ids"][0].to(device)

    # Exclude [GAP] from predictions
    banned = []
    if gap_token in tokenizer.get_vocab():
        banned.append(tokenizer.convert_tokens_to_ids(gap_token))

    beams = beam_search(
        input_ids, model, tokenizer,
        beam_width=beam_width,
        temperature=temperature,
        banned_token_ids=banned,
    )

    return decode_beams(beams, input_ids, tokenizer)


# ── CLI / quick check ─────────────────────────────────────

if __name__ == "__main__":
    import argparse
    from transformers import AutoModelForMaskedLM, AutoTokenizer

    parser = argparse.ArgumentParser()
    _HERE = Path(__file__).resolve().parent
    parser.add_argument("--model",  default=str(_HERE.parents[1] / "outputs/from_scratch/RoFormerBPE/final_model"))
    parser.add_argument("--text",   default="поклоне ѿ [MASK] к ѥва про [MASK] ѡкупи")
    parser.add_argument("--top_k",  type=int,   default=5)
    parser.add_argument("--temp",   type=float, default=1.0)
    args = parser.parse_args()

    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model     = AutoModelForMaskedLM.from_pretrained(args.model).to(device)
    model.eval()

    print(f"\nInput: {args.text}\n")
    results = restore(args.text, model, tokenizer,
                      beam_width=args.top_k, temperature=args.temp)

    for i, r in enumerate(results, 1):
        print(f"  [{i}] score={r['score']:.3f}  log_prob={r['log_prob']:.3f}")
        print(f"       {r['text']}")
        print(f"       filled: {r['filled_tokens']}")