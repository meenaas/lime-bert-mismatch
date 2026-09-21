"""
Main experiment script for "Does LIME Lie to You?"

Runs each stress-test sentence through:
  1. LIME (word-level, model-agnostic, perturbation-based explanation)
  2. BERT's own attention (from [CLS], last layer, averaged over heads)

then aligns the two word-importance vectors and measures how much they
diverge -- broken down by linguistic phenomenon (negation, contrast,
sarcasm, long-range dependency) -- to test the hypothesis that LIME's
local-linearity assumption breaks down precisely where BERT's
contextual, non-linear attention matters most.

Usage:
    pip install -r requirements.txt
    python run_experiment.py
"""
import json
import os

from src.model_utils import load_model, predict_proba, get_attention_and_tokens
from src.alignment import merge_wordpieces, align_to_lime_words
from src.lime_explainer import get_lime_word_importance
from src.metrics import rank_correlation, topk_overlap, divergence_score
from src.visualize import plot_comparison, plot_divergence_by_category

DATA_PATH = "data/stress_test_examples.json"
RESULTS_DIR = "results"


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    with open(DATA_PATH) as f:
        examples = json.load(f)

    print("Loading model (distilbert-base-uncased-finetuned-sst-2-english)...")
    tokenizer, model = load_model()

    all_results = []

    for i, ex in enumerate(examples):
        text, category = ex["text"], ex["category"]
        print(f"[{i + 1}/{len(examples)}] ({category}) {text}")

        # LIME importance
        lime_words, lime_scores, _ = get_lime_word_importance(
            text, tokenizer, model, predict_proba
        )

        # BERT attention importance, merged to whole words, aligned to LIME's word order
        bert_tokens, bert_attn = get_attention_and_tokens(text, tokenizer, model)
        merged_words, merged_scores = merge_wordpieces(bert_tokens, bert_attn)
        aligned_attn = align_to_lime_words(lime_words, merged_words, merged_scores)

        corr = rank_correlation(lime_scores, aligned_attn)
        overlap = topk_overlap(lime_words, lime_scores, lime_words, aligned_attn, k=3)
        div = divergence_score(lime_scores, aligned_attn)

        all_results.append({
            "text": text,
            "category": category,
            "lime_words": lime_words,
            "lime_scores": lime_scores,
            "attn_scores": aligned_attn,
            "spearman": corr["spearman"],
            "kendall": corr["kendall"],
            "topk_overlap": overlap,
            "divergence": div,
        })

        plot_comparison(
            lime_words, lime_scores, aligned_attn,
            title=f"[{category}] {text}",
            save_path=f"{RESULTS_DIR}/example_{i + 1}_{category}.png",
        )

    plot_divergence_by_category(
        [r["category"] for r in all_results],
        [r["divergence"] for r in all_results],
        save_path=f"{RESULTS_DIR}/divergence_by_category.png",
    )

    with open(f"{RESULTS_DIR}/results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nDone. Plots + results.json saved to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
