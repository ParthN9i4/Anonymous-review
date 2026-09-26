# results/ — evidence by campaign

| Directory | What it is | Split / status |
|---|---|---|
| `ablations/` | Per-script JSON outputs of `core/` and `experiments/` (formerly `experiment_results/`), including the H200 CIFAR-10/100 and BloodMNIST factorial seeds imported from the `results-h200` branch on 2026-09-23. `*.PARTIAL.json` and `*_RECOVERED.json` keep their status suffixes. `tools/audit_results.py` classifies every file as SAFE / TAINTED / SUPERSEDED / PARTIAL. | Mixed. Run `python tools/audit_results.py` before citing anything. |
| `figures/` | BloodMNIST collapse figures from `experiments/exploratory/investigate_collapse_bloodmnist.py`. | Descriptive |
| `campaign_2026-09-20/` | Frozen fixed-weight conversion + primitive CKKS campaign: executed source (`source_final/`), derived tables (`results/`), checkpoint inventory and hashes. Imported verbatim from draft PR #3; see its `IMPORT_NOTE.md`. | Fit/gate from train, clean test 10k, validation-selected references |
| `closeout_v2/` | Tables from `experiments/active/closeout_v2/` (oracle attribution, dose-response, layer localization, pre-registered screens, depth ledger, encrypted MLP sub-block). Written only by `analyze_v2.py`. | dev = validation split; test opened once, after the screen freeze |

Paper-facing tables live in `paper/analysis_*/`. They are derived from these campaigns;
`paper/analysis_20260920/PROVENANCE.md` records the mapping.
