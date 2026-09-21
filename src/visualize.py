"""
Side-by-side bar charts of LIME importance vs. BERT attention importance
for a single sentence, plus an aggregate chart of divergence by
linguistic-phenomenon category. These are the figures the LinkedIn post
and README should lead with.
"""
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt


def plot_comparison(words, lime_scores, attn_scores, title, save_path):
    fig, axes = plt.subplots(2, 1, figsize=(max(6, len(words) * 0.6), 4), sharex=True)

    x = np.arange(len(words))

    axes[0].bar(x, lime_scores, color="#4C72B0")
    axes[0].set_ylabel("LIME weight")
    axes[0].axhline(0, color="black", linewidth=0.5)

    axes[1].bar(x, attn_scores, color="#DD8452")
    axes[1].set_ylabel("Attention weight")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(words, rotation=45, ha="right")

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_divergence_by_category(categories, divergence_scores, save_path):
    by_cat = defaultdict(list)
    for c, d in zip(categories, divergence_scores):
        if d is not None:
            by_cat[c].append(d)

    cats = list(by_cat.keys())
    means = [np.mean(by_cat[c]) for c in cats]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(cats, means, color="#55A868")
    ax.set_ylabel("Mean divergence (1 - Spearman)")
    ax.set_title("LIME vs. attention divergence by linguistic phenomenon")
    plt.xticks(rotation=20, ha="right")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
