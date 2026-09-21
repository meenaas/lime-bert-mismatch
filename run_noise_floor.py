"""
Noise-Floor Control for the Order-Swap Ablation

Question this answers: when nothing changes at all -- same sentence, same
model, just LIME run twice -- how much does LIME's own explanation wobble
on its own, purely from its random perturbation sampling?

This number is the baseline you compare the order-swap results against.
If swap-condition lime_topk_overlap (0.33-0.67, from run_order_swap.py)
is close to this identical-input noise floor, the "LIME is blind to
order" claim is weak -- you'd be looking at LIME's ordinary jitter, not a
context-sensitivity effect. If the noise floor is much higher (LIME is
highly self-consistent on an unchanged sentence), the swap-condition drop
is real signal.

Design: for each of the 10 original sentences (not the swapped ones),
run get_lime_word_importance TWICE with different random seeds and
compare the two runs against each other using the exact same metrics
(topk_overlap, divergence_score) as run_order_swap.py, so the two are
directly comparable.

Usage:
    python run_noise_floor.py
"""
import json
import os
import random

import numpy as np

from src.model_utils import load_model, predict_proba
from src.alignment import align_to_lime_words
from src.lime_explainer import get_lime_word_importance
from src.metrics import topk_overlap, divergence_score

DATA_PATH = "data/order_swap_pairs.json"
RESULTS_DIR = "results"
N_REPEATS = 2          # LIME runs per sentence (2 is enough to measure one noise gap;
                        # bump to 3-5 if you want a distribution rather than a point estimate)


def lime_run_with_seed(text, tokenizer, model, seed):
    random.seed(seed)
    np.random.seed(seed)
    words, scores, _ = get_lime_word_importance(text, tokenizer, model, predict_proba)
    return words, scores


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    with open(DATA_PATH) as f:
        pairs = json.load(f)

    print("Loading model (distilbert-base-uncased-finetuned-sst-2-english)...")
    tokenizer, model = load_model()

    all_results = []

    for i, pair in enumerate(pairs):
        pid = pair["id"]
        text = pair["original"]
        print(f"[{i + 1}/{len(pairs)}] {pid} (noise floor): {text}")

        run_a_words, run_a_scores = lime_run_with_seed(text, tokenizer, model, seed=1)
        run_b_words, run_b_scores = lime_run_with_seed(text, tokenizer, model, seed=2)

        overlap = topk_overlap(run_a_words, run_a_scores, run_b_words, run_b_scores, k=3)

        # align run B's scores onto run A's word order before computing divergence
        run_b_aligned = align_to_lime_words(run_a_words, run_b_words, run_b_scores)
        div = divergence_score(run_a_scores, run_b_aligned)

        all_results.append({
            "id": pid,
            "text": text,
            "lime_top3_run_a": run_a_words[:3],
            "lime_top3_run_b": run_b_words[:3],
            "noise_floor_topk_overlap": overlap,
            "noise_floor_divergence": div,
        })

    with open(f"{RESULTS_DIR}/noise_floor_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved per-sentence noise-floor results to {RESULTS_DIR}/noise_floor_results.json")

    mean_overlap = sum(r["noise_floor_topk_overlap"] for r in all_results) / len(all_results)
    mean_div = sum(r["noise_floor_divergence"] for r in all_results) / len(all_results)
    print(f"\n--- Noise floor summary (n={len(all_results)}) ---")
    print(f"mean noise_floor_topk_overlap: {mean_overlap:.4f}")
    print(f"mean noise_floor_divergence:   {mean_div:.4f}")
    print(
        "\nCompare these two numbers directly against the swap-condition\n"
        "mean_lime_topk_overlap printed by run_order_swap.py.\n"
        "  - If they're close            -> swap effect is mostly LIME's own sampling noise.\n"
        "  - If noise-floor overlap is\n"
        "    much HIGHER (LIME agrees with\n"
        "    itself on an unchanged input) -> the swap-condition drop is real signal,\n"
        "    not jitter."
    )


if __name__ == "__main__":
    main()
