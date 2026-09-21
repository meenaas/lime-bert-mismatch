# Does LIME Lie to You? A Local-Linearity Stress Test on BERT

**TL;DR:** LIME explains BERT's predictions by fitting a locally *linear*
surrogate model around perturbed inputs. BERT's attention is *contextual
and non-linear* — a word's importance depends on the other words around
it. This project stress-tests where those two assumptions collide:
sentences built around negation, contrast, sarcasm, and long-range
dependency, where meaning genuinely can't be captured word-by-word.

## Hypothesis

LIME and attention-derived importance should roughly agree on simple,
compositional sentences ("This is a wonderful film.") because a linear
approximation is a reasonable local model when words contribute
independently to the prediction.

They should **diverge** on sentences where a word's contribution flips
or depends on distant context — e.g. "not bad" (negation reverses local
polarity), "great, but a mess" (contrast requires weighing two clauses
against each other), or sarcasm (surface sentiment and actual sentiment
are opposite). In these cases, LIME's linear surrogate is fitting the
wrong kind of function to a non-linear decision boundary — a structural
mismatch, not just occasional noise.

## Method

1. **Model:** `distilbert-base-uncased-finetuned-sst-2-english` (binary
   sentiment), used as-is — no fine-tuning needed for this to be a valid
   interpretability comparison.
2. **LIME:** `lime.lime_text.LimeTextExplainer` perturbs the input by
   dropping words and refitting a linear model on the resulting
   prediction changes. Produces a signed importance score per word.
3. **Attention:** mean attention *from* the `[CLS]` token, averaged
   across all heads of the last transformer layer (`[CLS]` is what
   feeds the classification head, so this approximates "what the model
   looked at when deciding").
4. **Alignment:** BERT's wordpiece tokens are merged back into whole
   words (`##`-continuation pieces summed into their parent word), then
   matched to LIME's own word tokenization so the two importance vectors
   describe the same units. See `src/alignment.py` for the known
   limitation here (greedy matching on repeated words).
5. **Divergence metrics:** Spearman/Kendall rank correlation between the
   two importance vectors, top-k overlap, and a single divergence score
   (`1 - Spearman`), computed per sentence and aggregated by linguistic
   category.

## Results

Run `python run_experiment.py` to regenerate. Expect:
- `results/example_N_<category>.png` — per-sentence LIME vs. attention
  bar charts
- `results/divergence_by_category.png` — mean divergence per category
- `results/results.json` — raw scores for further analysis

*(Fill in the actual numbers here once you've run it — this is the
figure to lead the LinkedIn post with.)*

## Why this matters for interpretability research

This isn't "LIME is broken." It's a **mechanistic, causal diagnosis**:
LIME's local-linearity assumption and BERT's attention mechanism encode
fundamentally different notions of "importance," and the gap between
them is *predictable* — it should track linguistic phenomena that
require non-linear, contextual reasoning. That's the difference between
"post-hoc explainability tools can be unreliable" (a known, generic
claim) and "here is exactly which structural property causes the
unreliability, and here is a controlled test that demonstrates it."

That distinction — mechanism over anecdote — is the core of the research
identity this repo is meant to establish, and it generalizes directly to
network-facing ML: any interpretability method built on local
linearization (LIME, and some SHAP variants) will face the same failure
mode when explaining models over sequential, contextual data like
routing tables or telemetry streams.

## Setup

```bash
pip install -r requirements.txt
python run_experiment.py
```

## Extending this

- Add more stress categories (double negation, idioms, coreference).
- Swap `[CLS]`-attention for attention rollout or gradient-based
  attribution (Integrated Gradients, Captum) as a second "ground truth"
  to triangulate against.
- Fine-tune on a domain-specific dataset (e.g. network log sentiment /
  anomaly severity) to connect this directly to the networking
  interpretability angle.
