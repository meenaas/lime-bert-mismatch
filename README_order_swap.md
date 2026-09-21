# Order-Swap Ablation — Setup & Run Guide

Extends your existing `lime_instability_round2.py` pipeline to test the
mechanistic claim directly: is LIME's instability driven by its
blindness to BERT's context/order-dependent attention (not just generic
sampling noise)?

## Files

- `clause_pairs_template.csv` — starter file with 2 example pairs (1 Governing
  Law, 1 Liquidated Damages). **Replace/extend these with your own 15–20 pairs**
  built from real CUAD clauses you already have confidence scores for.
- `order_swap_ablation.py` — runs LIME + BERT on each pair and produces the
  results CSV and scatter plot.
- `README_order_swap.md` — this file.

## 1. Build your clause pairs

Open `clause_pairs_template.csv` and add rows following the same format:

- `clause_id` — short unique ID (e.g. `LD_03`)
- `clause_type` — `Governing Law` or `Liquidated Damages` (reuse your existing categories)
- `original_text` — the real CUAD clause text
- `swapped_text` — same clause with a segment reordered, **same word multiset**,
  no words added/removed/changed
- `meaning_preserved_check` — `Y` only if you (or Dr. Marshan) manually confirmed
  the swap doesn't change the legal meaning. Rows marked anything other than `Y`
  are skipped automatically.
- `notes` — optional

Aim for 8–10 pairs per clause type (15–20 total), reusing clauses from your
existing pilot where possible so BERT confidence is already known to be ≥ 0.5.

## 2. Wire up your existing model

In `order_swap_ablation.py`, edit the top of the file:

```python
MODEL_CHECKPOINT_PATH = "LIME_Folder/checkpoint"   # your existing checkpoint path
LABELS = ["not_applicable", "applicable"]          # your actual class names
```

If your `lime_instability_round2.py` loads the model differently (e.g. a
custom wrapper class, different tokenizer args, GPU device placement), copy
that logic into `load_model()` and `predict_proba()` — those two functions are
the only place you need to match your existing setup. Everything downstream
(metrics, CSV, plot) doesn't need to change.

## 3. Install dependencies (if not already present)

```bash
pip install lime transformers torch pandas numpy matplotlib --break-system-packages
```

## 4. Run

```bash
python order_swap_ablation.py
```

This will print progress per clause pair, then:
- save `order_swap_results.csv` (per-pair metrics)
- print a summary table grouped by clause type
- save `order_swap_scatter.png` (the core evidence plot)

## 5. What to look for

- **`order_swap_scatter.png`**: points in the **top-right** (high Δ BERT prob,
  high LIME Jaccard similarity) are your mechanistic evidence — BERT's
  prediction moved a lot, but LIME's explanation barely changed, meaning LIME
  is blind to the exact signal driving BERT's decision.
- **Summary table**: if `Liquidated Damages` (compositional) shows higher
  `mean_delta_bert` and similarly high `mean_lime_jaccard5` than `Governing Law`
  (fixed-formula), that's the class-level separation your original pilot
  didn't find — and it's the number to lead with in the SOP/paper.
- **`n_label_flips`**: any nonzero count here is strong, simple evidence to
  quote directly (a full label flip triggered by pure word reordering, with
  LIME's top-5 explanation unchanged, is the cleanest single example you can
  put in a figure or SOP paragraph).

## 6. Optional next step (not required to run now)

Once this runs cleanly, the "sampling-noise control" from the earlier
discussion (running LIME on a plain bag-of-words logistic regression model on
the same clause pairs, to confirm instability is BERT-specific and not just
LIME's own randomness) reuses this same script structure — swap `load_model`/
`predict_proba` for a scikit-learn logistic regression pipeline and rerun.
