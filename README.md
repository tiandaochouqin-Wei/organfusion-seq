# OrganFusion-Seq — Reproducibility Package

Code and derived data for the paper:

> **Reliable Early Drug-Response Screening as Feedback Control of Observation:
> A Calibration-Robust Optimal-Stopping Controller for Tumour Organoids**
> Wei Zhang, Mingrui Lai, Wenhao Jiang, Jianfeng Wang, Mingqiang Wei, Xiaoping Zhang.
> *IEEE Transactions on Cybernetics* (under review), 2026.

We recast early drug-response screening as **feedback control of the observation
process**: a controller decides, per organoid, *when to stop observing and commit a
decision*, with provable links from learned-model calibration to closed-loop control
loss. This repository reproduces every results figure and table in the paper **from the
released calibrated posteriors** — no raw imaging data or model training required.

## What is in here

```
posterior/
  posterior_data.npz              calibrated posteriors p̂_t for 1,888 breast-cancer
                                  organoids: P (N×T), GRID (time points, h), xgrid,
                                  y (labels), drug, conc, cell-line
  unified_posterior.csv           long-form posterior table
  auc_by_horizon.csv              honest trajectory-up-to-T AUC vs horizon
  table2..table6_*.csv            derived result tables (baselines, window info,
                                  ablation, matched operating points, robustness)
figures_cs/data/cost.csv          per-decision cost inputs (supplementary cost figure)
E10_results.json                  anytime-valid bound U_n samples
tcyb_final_figure_data/
  shift_results.csv               distribution-shift / early-indicator results
build_fig3_motivation.py          Fig. 3  — window-information / saturation motivation
build_fig4_controller.py          Fig. 4  — optimal-stopping controller theory & bands
build_controller.py               Fig. 5  + Table V (matched operating points)
build_robustness.py               robustness / anytime-bound contraction figure
build_supp.py                     supplementary cost / endpoint / by-drug figures
gen_tables_tex.py                 assembles all main result tables as LaTeX
figures_regen/  figures_supp/     output directories (populated by the scripts)
```

## Data provenance

The calibrated posteriors are derived from the **public** high-throughput
holographic interferometry (HSLCI) breast-cancer organoid cohort. We release the
**derived posteriors** that the paper's analyses consume; the raw imaging data should
be obtained from its original public source. The training/posterior-estimation step
that produced `posterior_data.npz` from the raw cohort is **not** included here because
it depends on the raw dataset; the released posteriors are sufficient to reproduce all
reported figures and tables.

## Reproduce the figures and tables

```bash
pip install -r requirements.txt          # numpy, matplotlib
python build_fig3_motivation.py          # -> figures_regen/
python build_fig4_controller.py          # -> figures_regen/
python build_controller.py               # -> figures_regen/ + posterior/table5_matched_ops.csv
python build_robustness.py               # -> figures_regen/
python build_supp.py                     # -> figures_supp/
python gen_tables_tex.py                 # prints LaTeX for the main tables
```

Run from the repository root (the scripts use relative paths into `posterior/`).
All scripts are deterministic.

## Citation

See [`CITATION.cff`](CITATION.cff). Please cite the paper if you use this code or data.

## License

[MIT](LICENSE) © 2026 the OrganFusion-Seq authors.
