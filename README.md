# Data Archive — Supplementary Materials

This archive accompanies the submission *"Does Compression Preserve What
Classification Needs? A Controlled Evaluation of Legal Contract
Summarisation for Clause Classification."*

## Contents

### `split_record_TRAINVALTEST_PERMANENT.json`
The permanent, verifiable three-way split of the 509 deduplicated CUAD
contracts used throughout the paper (seed = 2027; 356 train / 76
validation / 77 test). Saved immediately after the split was created,
before any training took place. Contains SHA-1 content hashes
(`train_context_hashes`, `val_context_hashes`, `test_context_hashes`)
rather than raw contract text, so the partition can be independently
verified against a copy of CUAD without redistributing CUAD itself.
Validation was used only for early stopping and threshold selection;
test was touched exactly once, after training was complete, to produce
the results reported in the paper.

### `hybrid_leakfree_TEST_57.csv`
The 57-contract evaluation set used for Table 1 (§5.1) and the
frequency-bucket and qualitative flip analyses (§5.2): the subset of
the 77 held-out test contracts that have matching pre-generated
summaries. Each row includes:
- `context` — full contract text
- `extractive_summary` — TextRank summary
- `pegasus_summary` — Legal-Pegasus summary
- `hybrid_leakfree` — the leak-free hybrid summary reported in the
  paper (MMR selection scored against a TF-IDF centroid of the
  contract's own text, not gold answer spans)
- `hybrid_blend` — an earlier intermediate candidate pool column,
  **not used in any reported result**; retained only because it was
  part of the original candidate-generation step
- `key_hash` — SHA-1 content hash, matched against
  `test_context_hashes` in the split record
- `in_test_set` — confirms membership in the genuine 77-contract test
  partition (all rows here are `True`)

### `compression_curve_dataset_TEST57.csv`
The matched-budget dataset underlying Table 2 and Figure 1 (§5.3):
1,710 rows = 6 character budgets (5/10/12/15/20/30%) × 5 methods
(`lead`, `random`, `extractive_budget`, `hybrid_budget`,
`pegasus_truncated`) × the same 57 test contracts as above, linked via
`context_hash`.

## What is *not* included

`CUAD_v1.json` is not re-hosted here. CUAD (Hendrycks et al., 2021) is
a public dataset with its own license; the split record's context
hashes are sufficient to verify contract membership against an
independently obtained copy of CUAD without redistributing it.

## Reproducing the reported results

1. Obtain `CUAD_v1.json` from the original CUAD release.
2. Verify example counts and hashes against
   `split_record_TRAINVALTEST_PERMANENT.json`.
3. Load `hybrid_leakfree_TEST_57.csv` and evaluate the fine-tuned
   LegalPro-BERT classifier (decision threshold = 0.35, selected via
   validation) on the `context`, `extractive_summary`,
   `pegasus_summary`, and `hybrid_leakfree` columns to reproduce
   Table 1.
4. Load `compression_curve_dataset_TEST57.csv` and evaluate the same
   classifier on the `summary` column, grouped by `budget` and
   `method`, to reproduce Table 2 and Figure 1.
