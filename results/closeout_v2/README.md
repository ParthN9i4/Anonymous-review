# results/closeout_v2 — tables from the closeout_v2 campaign

Populated by `bash experiments/active/closeout_v2/launch_v2.sh report`, which runs
`analyze_v2.py`. Until the H200 runs finish this directory holds only this note. No number
here is written by hand.

| File | Source |
|---|---|
| `completion.json` | expected / recorded / failed jobs per group |
| `e1_rows.csv`, `e1_summary.csv` | E1 oracle attribution (dev split) |
| `e2_rows.csv`, `e2_spearman.csv` | E2 dose-response; rank correlation of each gate metric with the dev change rate |
| `e3_rows.csv`, `e3_summary.csv` | E3 layer localization |
| `confirm_rows.csv` | locked CIFAR-10 test pass of the E2 configurations |
| `screen_frozen.json`, `screen_evaluation.csv` | E4 frozen thresholds and their evaluation |
| `f0_rows.csv`, `f0_composition.json` | F0 measured depth ledger and labelled block composition |
| `f1_summary.json` | F1 encrypted MLP sub-block and hybrid replay |
