"""
Order-Swap Ablation for "Does LIME Lie to You?"

Extends the existing stress-test design (negation / contrast / sarcasm /
long-range dependency) with a fifth category: order_sensitivity.

For each (original, swapped) sentence pair -- same words, clauses
reordered around a "but"/"although" connective -- this script:

  1. Runs BERT + LIME on BOTH versions using the EXACT same pipeline as
     run_experiment.py (src.model_utils, src.lime_explainer, src.alignment),
     so results are directly comparable to your existing stress-test data.
  2. Computes, per pair:
       - delta_bert_prob   : how much BERT's predicted-class probability
                              shifted between original and swapped
       - label_flip        : whether the predicted sentiment flipped
       - lime_topk_overlap : overlap between LIME's top-3 words
                              (via src.metrics.topk_overlap) original vs swapped
       - lime_divergence   : src.metrics.divergence_score between the two
                              LIME score vectors (aligned by word)
  3. Saves results.json + a scatter plot: x = delta_bert_prob,
     y = lime_topk_overlap. Hypothesis: points cluster top-right --
     BERT's prediction moves a lot, but LIME's own explanation for the
     sentence barely changes, meaning LIME is blind to exactly the
     signal (clause order / "but"-final dominance) that BERT is using.

This does NOT touch run_experiment.py or its outputs -- it's a
self-contained addition that reuses your existing src/ modules.

Usage:
    python run_order_swap.py
"""
import json
import os

from src.model_utils import load_model, predict_proba, get_attention_and_tokens
from src.alignment import merge_wordpieces, align_to_lime_words
from src.lime_explainer import get_lime_word_importance
from src.metrics import topk_overlap, divergence_score

DATA_PATH = "data/order_swap_pairs.json"
RESULTS_DIR = "results"


def analyze_one(text, tokenizer, model):
    """Run LIME + attention for a single sentence, same as run_experiment.py."""
    lime_words, lime_scores, _ = get_lime_word_importance(
        text, tokenizer, model, predict_proba
    )
    bert_tokens, bert_attn = get_attention_and_tokens(text, tokenizer, model)
    merged_words, merged_scores = merge_wordpieces(bert_tokens, bert_attn)
    aligned_attn = align_to_lime_words(lime_words, merged_words, merged_scores)

    probs = predict_proba([text], tokenizer, model)[0]
    pred_label = int(probs.argmax())
    pred_prob = float(probs[pred_label])

    return {
        "lime_words": lime_words,
        "lime_scores": lime_scores,
        "attn_scores": aligned_attn,
        "pred_label": pred_label,
        "pred_prob": pred_prob,
    }


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    with open(DATA_PATH) as f:
        pairs = json.load(f)

    print("Loading model (distilbert-base-uncased-finetuned-sst-2-english)...")
    tokenizer, model = load_model()

    all_results = []

    for i, pair in enumerate(pairs):
        pid = pair["id"]
        print(f"[{i + 1}/{len(pairs)}] {pid}: {pair['original']}")

        orig = analyze_one(pair["original"], tokenizer, model)
        swap = analyze_one(pair["swapped"], tokenizer, model)

        # BERT prediction shift, measured on the class predicted for the ORIGINAL sentence
        label_flip = orig["pred_label"] != swap["pred_label"]
        if orig["pred_label"] == swap["pred_label"]:
            delta_bert_prob = abs(orig["pred_prob"] - swap["pred_prob"])
        else:
            # label flipped: compare prob of the original's predicted class
            # on both sides (swap's prob for that same class = 1 - swap["pred_prob"]
            # since this is binary SST-2)
            delta_bert_prob = abs(orig["pred_prob"] - (1 - swap["pred_prob"]))

        # LIME comparison: original's LIME words/scores vs swapped's LIME words/scores
        overlap = topk_overlap(
            orig["lime_words"], orig["lime_scores"],
            swap["lime_words"], swap["lime_scores"],
            k=3,
        )
        # divergence_score expects two aligned score vectors of the same length/order;
        # align swap's LIME scores onto orig's LIME word order for a fair comparison
        swap_scores_aligned = align_to_lime_words(
            orig["lime_words"], swap["lime_words"], swap["lime_scores"]
        )
        div = divergence_score(orig["lime_scores"], swap_scores_aligned)

        all_results.append({
            "id": pid,
            "category": pair["category"],
            "original_text": pair["original"],
            "swapped_text": pair["swapped"],
            "bert_label_original": orig["pred_label"],
            "bert_label_swapped": swap["pred_label"],
            "label_flip": label_flip,
            "bert_prob_original": round(orig["pred_prob"], 4),
            "bert_prob_swapped": round(swap["pred_prob"], 4),
            "delta_bert_prob": round(delta_bert_prob, 4),
            "lime_top3_original": orig["lime_words"][:3],
            "lime_top3_swapped": swap["lime_words"][:3],
            "lime_topk_overlap": overlap,
            "lime_divergence": div,
            "meaningful_shift_ge_0.10": delta_bert_prob >= 0.10,
        })

    with open(f"{RESULTS_DIR}/order_swap_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved per-pair results to {RESULTS_DIR}/order_swap_results.json")

    # Summary
    n_flips = sum(r["label_flip"] for r in all_results)
    mean_delta = sum(r["delta_bert_prob"] for r in all_results) / len(all_results)
    mean_overlap = sum(r["lime_topk_overlap"] for r in all_results) / len(all_results)
    print(f"\n--- Summary ---")
    print(f"n pairs: {len(all_results)}")
    print(f"label flips: {n_flips}")
    print(f"mean delta_bert_prob: {mean_delta:.4f}")
    print(f"mean lime_topk_overlap: {mean_overlap:.4f}")

    # Scatter plot
    try:
        import matplotlib.pyplot as plt
        xs = [r["delta_bert_prob"] for r in all_results]
        ys = [r["lime_topk_overlap"] for r in all_results]
        flips = [r["label_flip"] for r in all_results]

        fig, ax = plt.subplots(figsize=(7, 6))
        colors = ["#C44E52" if f else "#4C72B0" for f in flips]
        ax.scatter(xs, ys, c=colors, s=90, alpha=0.85, edgecolors="black", linewidths=0.5)
        ax.axvline(0.10, color="gray", linestyle="--", linewidth=1, alpha=0.6,
                   label="meaningful-shift threshold (0.10)")
        ax.set_xlabel("|Δ BERT prediction probability| (original vs. swapped)")
        ax.set_ylabel("LIME top-3 word overlap (original vs. swapped)")
        ax.set_title("Order-swap ablation: does LIME notice what BERT notices?\n"
                      "(top-right = BERT shifts a lot, LIME's explanation doesn't)\n"
                      "red = label flip, blue = no flip")
        ax.legend()
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        fig.tight_layout()
        fig.savefig(f"{RESULTS_DIR}/order_swap_scatter.png", dpi=200)
        print(f"Saved scatter plot to {RESULTS_DIR}/order_swap_scatter.png")
    except ImportError:
        print("matplotlib not available -- skipped plot, results.json still saved.")


if __name__ == "__main__":
    main()
