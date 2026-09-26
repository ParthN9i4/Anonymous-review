# Import note (2026-09-23)

Imported verbatim from draft PR #3 (`research/private-vit-evidence-20260920`,
`research/2026-09-20-private-vit/`) during the repository restructure. Internal paths in
this directory's README (`results/...`, `source_final/...`) are relative to this directory.

## One post-hoc edit

`source_final/native_source.py:59`: a code comment referring to the project rules file was
reworded to its new path (`docs/PROJECT_RULES.md`). No executable line changed.

| | SHA-256 |
|---|---|
| As executed (recorded in `source_final/PACKAGE_SHA256.json` and `results/source_hashes.json`) | `71ffb234235435c09a342a0d9dfb0f59c3914927aeb44f927aaf04758d2f62d2` |
| Current file | `b71583eae3cacefb4bbe0bd8ed4415853ffb3490be31a318c4a0747df31ffd84` |

The executed bytes remain on branch `research/private-vit-evidence-20260920` (kept):
`git show fcfb5c8:research/2026-09-20-private-vit/source_final/native_source.py`.

## Relationship to `paper/analysis_2026092*/`

| This bundle | Paper table | Relationship |
|---|---|---|
| `results/factorial_summary.csv` | `paper/analysis_20260918/factorial_summary.csv` | byte-identical |
| `results/baseline_summary.csv` | `paper/analysis_20260918/baseline_summary.csv` | byte-identical |
| `results/ckks.csv` | `paper/analysis_20260920/ckks_primitive_ledger.csv` | same 135 records, different column schema (only this bundle has `total_max_abs`, `configuration`, `profile`) |
| `results/detectors.csv` | `paper/analysis_20260920/detector_recall.csv` | same 312 rows, different columns (this bundle adds flips and false-positive rate) |
| `results/conversion_verified.csv` | `paper/analysis_20260920/conversion_fidelity.csv` | same 78 rows, different columns (paper adds FP64 accuracy and checkpoint hash) |

Both are derived views of the same campaign archive
(`campaign_23crox26_return(1).tgz`, SHA-256 `748c8c9e…a708ca29`, kept outside git).
