"""
Quantifies how much LIME's word-importance ranking diverges from
BERT's attention-derived word-importance ranking.
"""
from scipy.stats import spearmanr, kendalltau
import numpy as np


def rank_correlation(lime_scores, attn_scores):
    """Spearman + Kendall correlation between the two importance vectors."""
    if len(lime_scores) < 2:
        return {"spearman": None, "kendall": None}
    spearman_corr, _ = spearmanr(lime_scores, attn_scores)
    kendall_corr, _ = kendalltau(lime_scores, attn_scores)
    return {"spearman": spearman_corr, "kendall": kendall_corr}


def topk_overlap(lime_words, lime_scores, attn_words, attn_scores, k=3):
    """
    Fraction overlap between LIME's top-k most important words (by |score|)
    and attention's top-k most attended words. Assumes lime_words and
    attn_words are already the same list in the same order (post-alignment).
    """
    def topk(words, scores, k):
        order = np.argsort(-np.abs(np.array(scores)))[:k]
        return set(words[i] for i in order)

    lime_top = topk(lime_words, lime_scores, k)
    attn_top = topk(attn_words, attn_scores, k)

    if not lime_top:
        return 0.0
    return len(lime_top & attn_top) / len(lime_top)


def divergence_score(lime_scores, attn_scores):
    """
    A single scalar summarizing mismatch: 1 - Spearman correlation, so
    0 = perfect agreement, 1 = no relationship, 2 = perfect disagreement.
    """
    corr = rank_correlation(lime_scores, attn_scores)["spearman"]
    if corr is None:
        return None
    return 1 - corr
