# LIME vs. Attention on BERT: A Pilot Stress Test

**TL;DR:** A small pilot comparing LIME word importances with last-layer [CLS] attention on a BERT sentiment model, across sentence types where word meaning depends on context (negation, contrast, sarcasm, long-range dependency, word order). Neither method is treated as ground truth: attention is not a faithful explanation in general, so disagreement here shows that the two methods diverge, not which one is right.

## Hypothesis

LIME and attention-derived importance should roughly agree on simple, compositional sentences ("This is a wonderful film.") because a linear approximation is a reasonable local model when words contribute independently to the prediction.

They should diverge on sentences where a word's contribution flips or depends on distant context — e.g. "not bad" (negation reverses local polarity), "great, but a mess" (contrast requires weighing two clauses against each other), sarcasm (surface sentiment and actual sentiment are opposite), or reordering (the same words, differently arranged, can flip the model's decision). In these cases, I expected LIME's linear surrogate to be fitting the wrong kind of function to a non-linear decision boundary — a structural mismatch, not just occasional noise.

## Method

- **Model:** `distilbert-base-uncased-finetuned-sst-2-english` (binary sentiment), used as released.
- **LIME:** `lime.lime_text.LimeTextExplainer` perturbs the input by dropping words and refitting a linear model on the resulting prediction changes. Produces a signed importance score per word.
- **Attention:** mean attention from the `[CLS]` token, averaged across all heads of the last transformer layer (`[CLS]` is what feeds the classification head, so this approximates "what the model looked at when deciding").
- **Alignment:** BERT's wordpiece tokens are merged back into whole words (`##`-continuation pieces summed into their parent word), then matched to LIME's own word tokenization so the two importance vectors describe the same units. See `src/alignment.py` for the known limitation here (greedy matching on repeated words).
- **Divergence metrics:** Spearman/Kendall rank correlation between the two importance vectors, top-k overlap, and a single divergence score (1 − Spearman), computed per sentence and aggregated by linguistic category.
- **LIME samples:** 5,000 perturbation samples per explanation (LIME's default).

## Results

### Stress-test categories (negation, contrast, sarcasm, long-range dependency)

Run `python run_experiment.py` to regenerate. n=2 sentences per category — a stress-test/qualitative scale, not a statistically powered study.

| Category | Mean divergence | Mean Spearman | Mean top-3 overlap |
|---|---|---|---|
| negation | 0.27 (lowest) | 0.73 (highest agreement) | 0.67 |
| control | 0.75 | 0.25 | 0.33 |
| sarcasm | 1.11 | −0.11 | 0.33 |
| contrast | 1.21 | −0.21 | 0.67 |
| long_range | 1.31 (highest) | −0.31 (most anti-correlated) | 0.33 |

**This partially disconfirms the hypothesis as originally stated, and that's worth reporting rather than smoothing over.** The hypothesis (above) predicted negation would be a hard case where LIME and attention diverge; instead negation shows the *best* agreement of any category — better than "control," where one sentence ("This is a terrible film.") diverged badly (Spearman −0.4) despite being simple and compositional. Long-range dependency and contrast diverge more, in the predicted direction, but with n=2 per category this is suggestive only.

See `results/example_N_<category>.png` for per-sentence LIME vs. attention bar charts and `results/divergence_by_category.png` for the aggregate view; raw scores in `results/results.json`.

### Order-swap ablation (order_sensitivity category)

`python run_order_swap.py` — tests whether reordering clauses around "but"/"although", using the exact same word set, changes BERT's prediction more than it changes LIME's explanation. n=10 sentence pairs.

| Metric | Result |
|---|---|
| Label flips | **8 / 10** pairs |
| Mean \|Δ BERT prediction probability\| | **~0.98** in flip cases |
| Mean LIME top-3 word overlap (original vs. swapped) | **0.53** |

Split by connective:
- **"but"-clauses:** 8/8 flipped (delta ≈ 0.97–0.999)
- **"although"-clauses:** 0/2 flipped (delta ≈ 0.0001)

This is consistent with the known "but-clause dominance" effect in sentiment models — the clause after "but" tends to dominate the prediction — but LIME's explanation shows only partial sensitivity to which clause that is.

### Noise-floor control

`python run_noise_floor.py` — runs LIME twice on the *same, unswapped* sentence to measure LIME's own baseline instability, independent of any real input change. This tests whether "LIME is just noisy" could fully explain the order-swap result above.

| Metric | Noise floor (identical input) | Swap condition |
|---|---|---|
| Mean top-3 overlap | 0.77 | 0.53 |
| Mean divergence | 0.39 | 0.68 |

Divergence nearly doubles under swapping relative to LIME's own noise floor — the order-swap effect is not fully explained by LIME's inherent sampling jitter, though the gap is moderate rather than total (LIME is not perfectly self-consistent even with zero input change). n=10 is a pilot scale; treat as a directional finding, not a significance-tested result.

## What this does and does not show

This is an exploratory pilot (n=2 sentences per category; n=10 order-swap pairs). It shows that LIME and attention diverge more on contrast and long-range sentences than on simple ones, and that LIME’s explanations change only partly when a clause swap flips the prediction. It does not show which method is faithful: that needs an intervention test, such as removing the top-ranked words and measuring the change in prediction. That is the direction I took in [cuad-lime-shortcut](https://github.com/meenaas/cuad-lime-shortcut).

## Setup

```bash
pip install -r requirements.txt
python run_experiment.py      # stress-test categories
python run_order_swap.py      # order-swap ablation
python run_noise_floor.py     # noise-floor control
```

## Extending this

- Add more stress categories (double negation, idioms, coreference).
- Compare against attention rollout or a gradient-based method (Integrated Gradients, via Captum), and test all methods with the same intervention, rather than treating any one as ground truth.
- Add a deletion test: remove each method's top-ranked words (and, separately, replace them with natural alternatives) and measure how much the prediction changes.
- Scale the order-swap set beyond n=10 (particularly more "although" pairs, to confirm the but/although split isn't an artifact of only having 2 examples).
