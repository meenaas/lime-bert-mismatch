"""
Wraps LIME's text explainer around a BERT classifier, and exposes a
word-level importance vector in LIME's own tokenization order so it can
be compared directly against attention-derived importance.
"""
from functools import partial
from lime.lime_text import LimeTextExplainer

CLASS_NAMES = ["NEGATIVE", "POSITIVE"]


def get_lime_word_importance(text, tokenizer, model, predict_proba_fn, num_features=20, num_samples=200):
    """
    Returns (lime_words, lime_scores, exp) — the words LIME perturbed and
    the signed importance it assigned to each, for the model's predicted
    class, plus the raw LIME explanation object for further inspection.
    """
    explainer = LimeTextExplainer(class_names=CLASS_NAMES, bow=False)

    fn = partial(predict_proba_fn, tokenizer=tokenizer, model=model)

    exp = explainer.explain_instance(
        text,
        fn,
        num_features=num_features,
        num_samples=num_samples,
    )

    pairs = exp.as_list()  # [(word, weight), ...] for the predicted label
    words = [w for w, _ in pairs]
    scores = [s for _, s in pairs]
    return words, scores, exp
