"""
Order-Swap Ablation: isolating whether LIME's blindness to BERT's attention-based
context sensitivity (rather than generic LIME sampling noise) explains instability.

WHAT THIS SCRIPT DOES
----------------------
For each clause pair (original vs. word-order-swapped, same word multiset):
  1. Runs LegalPro-BERT inference on both versions -> records label + probability.
  2. Runs LIME on both versions -> records top-5 words + full importance vector.
  3. Computes:
       - delta_bert_prob   : |p(original) - p(swapped)| on the predicted label
       - label_flip        : did the predicted label change?
       - lime_jaccard5     : Jaccard similarity of LIME's top-5 words (original vs swapped)
       - lime_cosine       : cosine similarity of full LIME importance vectors
  4. Writes a results CSV and the core scatter plot:
       x = delta_bert_prob, y = lime_jaccard5 (and lime_cosine)
     Hypothesis: points cluster top-right (BERT prediction shifts a lot,
     but LIME's explanation barely moves) -> LIME is blind to the ordering
     signal BERT is actually using.

HOW TO USE THIS WITH YOUR EXISTING PIPELINE
--------------------------------------------
This script reuses the same model + LIME setup style as lime_instability_round2.py.
Fill in the three functions in the "PLUG IN YOUR PIPELINE" section below with your
existing model-loading and inference code (same checkpoint path you already use:
LIME_Folder/checkpoint). Everything else (metrics, CSV, plot) is ready to run as-is.

INPUT
-----
clause_pairs_template.csv with columns:
  clause_id, clause_type, original_text, swapped_text, meaning_preserved_check, notes
Only rows with meaning_preserved_check == "Y" are used.

OUTPUT
------
order_swap_results.csv   -- per-pair metrics
order_swap_scatter.png   -- the core evidence plot
"""

import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lime.lime_text import LimeTextExplainer

# ============================================================
# PLUG IN YOUR PIPELINE (same model/tokenizer as round 2 script)
# ============================================================

MODEL_CHECKPOINT_PATH = "LIME_Folder/checkpoint"  # <- adjust to your local path
LABELS = ["not_applicable", "applicable"]         # <- adjust to your actual class names


def load_model():
    """
    Load your LegalPro-BERT model + tokenizer exactly as in
    lime_instability_round2.py. Return (model, tokenizer).
    """
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_CHECKPOINT_PATH)
    model.eval()
    return model, tokenizer


def predict_proba(texts, model, tokenizer):
    """
    Required by LIME: takes a list of strings, returns an (n, num_classes)
    probability array. Mirrors the predict_fn you already wrote for round 2.
    """
    import torch
    inputs = tokenizer(list(texts), return_tensors="pt", padding=True,
                        truncation=True, max_length=512)
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1).numpy()
    return probs


# ============================================================
# METRICS
# ============================================================

def jaccard(set_a, set_b):
    a, b = set(set_a), set(set_b)
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def cosine_sim(vec_a, vec_b, vocab):
    """Cosine similarity between two LIME importance vectors, aligned by vocab union."""
    a = np.array([vec_a.get(w, 0.0) for w in vocab])
    b = np.array([vec_b.get(w, 0.0) for w in vocab])
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def get_lime_explanation(text, predict_fn, explainer, num_features=10, num_samples=500):
    """Returns (top5_words, full_importance_dict, predicted_label_idx)."""
    exp = explainer.explain_instance(
        text, predict_fn, num_features=num_features, num_samples=num_samples
    )
    weights = dict(exp.as_list())
    top5 = [w for w, _ in exp.as_list()[:5]]
    probs = predict_fn([text])[0]
    pred_label = int(np.argmax(probs))
    return top5, weights, pred_label, probs


# ============================================================
# MAIN PIPELINE
# ============================================================

def main(input_csv="clause_pairs_template.csv",
         output_csv="order_swap_results.csv",
         output_plot="order_swap_scatter.png",
         num_samples=500,
         seed=42):

    np.random.seed(seed)

    model, tokenizer = load_model()
    predict_fn = lambda texts: predict_proba(texts, model, tokenizer)
    explainer = LimeTextExplainer(class_names=LABELS)

    df = pd.read_csv(input_csv)
    df = df[df["meaning_preserved_check"].str.upper() == "Y"].reset_index(drop=True)

    if len(df) == 0:
        raise ValueError(
            "No usable rows found. Check that meaning_preserved_check == 'Y' "
            "for at least one clause pair in the input CSV."
        )

    results = []
    for i, row in df.iterrows():
        clause_id = row["clause_id"]
        clause_type = row["clause_type"]
        orig_text = row["original_text"]
        swap_text = row["swapped_text"]

        print(f"[{i+1}/{len(df)}] Processing {clause_id} ({clause_type})...")

        top5_o, weights_o, label_o, probs_o = get_lime_explanation(
            orig_text, predict_fn, explainer, num_samples=num_samples
        )
        top5_s, weights_s, label_s, probs_s = get_lime_explanation(
            swap_text, predict_fn, explainer, num_samples=num_samples
        )

        # BERT prediction shift, measured on the class predicted for the ORIGINAL text
        p_orig_on_pred_class = probs_o[label_o]
        p_swap_on_pred_class = probs_s[label_o]
        delta_bert_prob = abs(p_orig_on_pred_class - p_swap_on_pred_class)
        label_flip = label_o != label_s

        lime_jaccard5 = jaccard(top5_o, top5_s)
        vocab = set(weights_o.keys()) | set(weights_s.keys())
        lime_cos = cosine_sim(weights_o, weights_s, vocab)

        results.append({
            "clause_id": clause_id,
            "clause_type": clause_type,
            "bert_label_original": label_o,
            "bert_label_swapped": label_s,
            "label_flip": label_flip,
            "bert_prob_original": round(p_orig_on_pred_class, 4),
            "bert_prob_swapped": round(p_swap_on_pred_class, 4),
            "delta_bert_prob": round(delta_bert_prob, 4),
            "lime_top5_original": ", ".join(top5_o),
            "lime_top5_swapped": ", ".join(top5_s),
            "lime_jaccard5": round(lime_jaccard5, 4),
            "lime_cosine_sim": round(lime_cos, 4),
            "meaningful_shift_ge_0.10": delta_bert_prob >= 0.10,
        })

    out_df = pd.DataFrame(results)
    out_df.to_csv(output_csv, index=False)
    print(f"\nSaved per-pair results to {output_csv}")

    # Summary by clause type
    print("\n--- Summary by clause type ---")
    summary = out_df.groupby("clause_type").agg(
        mean_delta_bert=("delta_bert_prob", "mean"),
        mean_lime_jaccard5=("lime_jaccard5", "mean"),
        mean_lime_cosine=("lime_cosine_sim", "mean"),
        n_label_flips=("label_flip", "sum"),
        n=("clause_id", "count"),
    )
    print(summary)

    # Core scatter plot: delta_bert_prob (x) vs lime_jaccard5 (y)
    fig, ax = plt.subplots(figsize=(7, 6))
    colors = {"Governing Law": "#4C72B0", "Liquidated Damages": "#C44E52"}
    for ctype, group in out_df.groupby("clause_type"):
        ax.scatter(
            group["delta_bert_prob"], group["lime_jaccard5"],
            label=ctype, color=colors.get(ctype, "gray"),
            s=90, alpha=0.8, edgecolors="black", linewidths=0.5,
        )
    ax.axvline(0.10, color="gray", linestyle="--", linewidth=1, alpha=0.6,
               label="meaningful-shift threshold (0.10)")
    ax.set_xlabel("|Δ BERT prediction probability| (original vs. swapped)")
    ax.set_ylabel("LIME top-5 Jaccard similarity (original vs. swapped)")
    ax.set_title("Does LIME notice what BERT notices?\n(top-right = BERT shifts, LIME doesn't)")
    ax.legend()
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    fig.tight_layout()
    fig.savefig(output_plot, dpi=200)
    print(f"Saved scatter plot to {output_plot}")


if __name__ == "__main__":
    main()
