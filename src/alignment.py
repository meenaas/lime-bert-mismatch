"""
Alignment between BERT wordpiece tokens (with attention scores) and
plain-word tokens (as used by LIME), so the two importance vectors can
be compared on a like-for-like basis.
"""
from typing import List, Tuple


def merge_wordpieces(tokens: List[str], scores) -> Tuple[List[str], List[float]]:
    """
    Merge BERT wordpiece tokens (marked with '##') into whole words,
    summing the attention score of continuation pieces into the word
    they belong to. Drops [CLS] / [SEP] / [PAD].
    """
    words, word_scores = [], []
    for tok, score in zip(tokens, scores):
        if tok in ("[CLS]", "[SEP]", "[PAD]"):
            continue
        if tok.startswith("##") and words:
            words[-1] = words[-1] + tok[2:]
            word_scores[-1] += float(score)
        else:
            words.append(tok)
            word_scores.append(float(score))
    return words, word_scores


def align_to_lime_words(lime_words: List[str], bert_words: List[str], bert_scores: List[float]):
    """
    LIME tokenizes on its own regex (roughly \\w+). BERT wordpiece merging
    above should produce very similar tokens for simple English text, but
    punctuation and casing can still cause mismatches. This does a
    case-insensitive best-effort alignment and returns a score vector
    indexed by LIME's word order. Words on the BERT side that don't match
    are skipped; LIME words with no BERT match get a score of 0.0.

    NOTE: this greedy first-match alignment is a known simplification —
    repeated words in a sentence may be paired out of order. Good enough
    for a stress-test-scale project; worth flagging as a limitation in
    the README/writeup rather than silently trusting it.
    """
    bert_lookup = {}
    for w, s in zip(bert_words, bert_scores):
        key = w.lower()
        bert_lookup.setdefault(key, []).append(s)

    aligned_bert = []
    for w in lime_words:
        key = w.lower()
        if key in bert_lookup and bert_lookup[key]:
            aligned_bert.append(bert_lookup[key].pop(0))
        else:
            aligned_bert.append(0.0)
    return aligned_bert
