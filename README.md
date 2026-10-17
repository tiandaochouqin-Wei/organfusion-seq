# CertStop: certified cost-sensitive early classification — reproducibility package

Code and derived data for the paper

> **Certified Cost-Sensitive Early Classification: Calibration-Robust Optimal Stopping
> with Risk Control under Condition Drift**
> Wei Zhang, Mingrui Lai, Wenhao Jiang, Jianfeng Wang, Mingqiang Wei, Xiaoping Zhang.
> Manuscript under review at *IEEE Transactions on Knowledge and Data Engineering*, 2026.

Early classification commits a decision on a growing prefix of a time series. We cast
cost-sensitive early classification as constrained optimal stopping on the posterior
process and propose CertStop, a controller that generates a finite family of stopping
rules, selects one on calibration data and certifies the selection. We give (i) a regret bound that ties the realised cost of any deployed stopping
rule to its calibration error *at the stopping time*, (ii) finite-sample certificates for a
rule selected on data, and (iii) a condition-level conformal risk-control rule for new
deployment conditions. The certificates apply to any family of stopping rules, including
published triggers (CALIMERA, EconomyGamma, TEASER). Evidence: tumour-organoid drug
screening (1,850 organoids, 18 conditions, leave-condition-out), all 40 two-class UCR
datasets, PhysioNet/CinC 2019 sepsis, NASA C-MAPSS, and an independent pancreatic panel.

The tag [`tcyb-2026-06`](../../tree/tcyb-2026-06) keeps the June 2026 version of this
repository (an earlier manuscript, desk-rejected by IEEE TCYB on 2026-07-13 and not
submitted elsewhere); everything below refers to the current version.

## Layout

```
*.py                      every script behind the manuscript's tables and figures (see the map below)
paths.py                  where the raw public data live: $ORGANFUSION_DATA, or ./datasets
posterior/                derived posteriors (*.npz) and every result table (*.csv) cited in the paper
posterior/ucr40/          frozen MiniRocket posteriors of the 40 UCR datasets (raw and Platt-scaled)
E*_results.json           results of the cost sweep, cadence and certificate-validity experiments
figures_tkde/             main-text figures (vector PDF, final size); fig_overview_tkde.tex is Fig. 1
figures_supp/             supplementary figures
datasets/                 raw data go here (not distributed, except the derived 4,066-organoid table)
```

## Setup

```bash
python -m pip install -r requirements.txt          # tested with Python 3.13
git clone https://github.com/ML-EDM/ml_edm.git _external/ml_edm
git -C _external/ml_edm checkout 93ffd9380dc80b30f258e833ad847aa2bdb0559b
```

`ml_edm` provides the reference implementations of TEASER, EconomyGamma and CALIMERA;
the scripts that use it add `_external/ml_edm/src` to the import path themselves.

### Raw data

Most results can be regenerated from the derived files in `posterior/` alone. Scripts that
start from raw data read them from `$ORGANFUSION_DATA` (default `./datasets`):

| File or folder under the data directory | Source |
|---|---|
| `Tebon_SupplementaryData_MOESM6.xlsx` | Supplementary Data of Tebon *et al.*, *Nat. Commun.* 14, 3168 (2023) |
| `main_development_cohort_4066.csv` | included here; four-timepoint table derived from the workbook above |
| `Deben_NOGR_SupplementaryData1.xlsx` | Supplementary Data 1 of Deben *et al.*, *Commun. Biol.* 7, 1612 (2024) |
| `physionet_challenge2019/training_setA`, `training_setB` | PhysioNet/CinC Challenge 2019 |
| `cmapss/train_FD001.txt`, `train_FD004.txt` | NASA C-MAPSS turbofan degradation data |
| `ucr/<Name>/<Name>_TRAIN.txt`, `_TEST.txt` | UCR archive: `python download_ucr_binary.py` fetches the 40 two-class sets |

The two journal datasets come from open-access (CC BY) articles; please cite the original
studies when using the derived tables.

## Map: manuscript item → script → output (`posterior/` unless noted)

| Item | Script(s) | Output |
|---|---|---|
| Posteriors (leave-condition-out, shared encoder) | `build_posterior_grouped.py`, `build_posterior_shared.py`, `rebuild_chain_shared.py`, `producers.py` | `posterior_data_*.npz` |
| Table 3, operating points | `build_posterior_grouped.py` | `table_matched_ops_grouped.csv` |
| Table 4, rules on one posterior | `seq_baselines.py`, `extra_baselines.py`, `stage3a_dp_bins.py`, `etsc_baseline_comparison.py` | `table_seq_baselines_*.csv`, `table_extra_baselines.csv`, `table_etsc_baselines_*.csv` |
| Table 5, held-out selection | `protocol_3way.py` | `table_3way_protocol.csv` |
| Table 6, constrained controller (Thm. 3c) | `constrained_control.py` | `table_constrained_control.csv` |
| Table 7, drift certificates (Thm. 4) | `constrained_control_crc.py`, `crc_outer_split.py` | `table_constrained_control_crc.csv`, `table_crc_outer_split.csv` |
| Table 10, cross-domain audit | `crossdomain_realized.py`, `physionet_protocol.py`, `cmapss_*.py`, `etsc_ucr_binary40.py` (40 sets; `python etsc_ucr_binary40.py 1 ECG5000` for ECG5000) | `table_crossdomain_realized.csv`, `table_etsc_ucr_*.csv` |
| Table 8 and Tables S17–S18, certifying published triggers | `certify_any_trigger.py` → `tables_anytrigger.py` | `table_certify_triggers_*.csv` |
| Fig. 1 | `fig_overview_data.py` (mini-plots → `fig_overview_data.tex`), `fig_overview_images.py` (organoid phase-image tiles from BioImage Archive S-BIAD616, one 357 MB stack, see the script header), then `fig_overview_tkde.tex` (pdflatex) | `figures_tkde/fig_overview_tkde.pdf` |
| Figs. 2–7 | `figs_tkde.py` (with `panel_labels.py`) | `figures_tkde/` |
| Fig. 4(a), zero-shot shift | `shift_matrix.py` | `table_shift_matrix.csv` |
| Fig. 7(b)–(c), Tables S14–S15, UCR benchmark | `download_ucr_binary.py`, `etsc_ucr_binary40.py`, `ucr40_analysis.py` | `table_etsc_ucr_binary40.csv`, `table_ucr40_*.csv` |
| Figs. S1–S8 | `figs_supp.py` | `figures_supp/` |
| Table S1, pairwise DeLong | `auc_significance.py` | `table_auc_pairwise_delong.csv` |
| Table S2, UCR with a weak posterior | `etsc_baseline_comparison_ucr.py`, `etsc_ucr_fair.py`, `etsc_generality.py` | `table_etsc_baselines_ucr.csv`, `table_etsc_ucr_fair.csv` |
| Tables S4–S7, ablations, coherence, window bound | `ablation_fusion.py`, `build_ablation.py`, `build_posterior_shared.py`, `build_window_info.py` | `table_fusion_ablation.csv`, `table4_ablation.csv`, `table_martingale_coherence.csv`, `table3_window_info.csv` |
| Table S8, 4,066-organoid source | `expand_cohort_4066.py`, `cohort4066_feature_ablation.py` | `table_4066_*.csv` |
| Fig. 6, Tables S9–S10, calibration and regret | `stage3a_calib_regret.py`, `stage3a_dp_bins.py` | `table_stage3a_*.csv` |
| Tables S11–S12, fixed-time baselines, protocols | `build_table2_baselines.py`, `build_posterior_grouped.py`, `seq_baselines.py` | `table2_fixed_time_baselines.csv`, `table_cv_auc_comparison.csv`, `table_estimator_swap.csv` |
| Table S13, UCR with MiniRocket | `etsc_ucr_strong.py` | `table_etsc_ucr_strong.csv` |
| Table S16, many conditions (pancreatic panel) | `crc_many_conditions.py` | `table_crc_many_conditions.csv` |
| Table S19, Thm. 4 with a frozen posterior | `crc_outer_split.py` → `tables_outer_split.py` | `table_crc_outer_split.csv` |
| Table 9 and Tables S20–S22, exact check of Thm. 2, paired bootstrap, ECE binning | `semisynthetic_thm2.py`, `paired_bootstrap.py`, `ece_sensitivity.py` → `tables_3r.py` | `table_semisynthetic_thm2.csv`, `table_paired_bootstrap.csv`, `table_ece_sensitivity.csv` |
| Cost sweep, cadence, certificate validity, adaptive sampling | `E8_cost_perdrug.py`, `E_F.py`, `E10_ABC.py`, `adaptive_sampling_pilot.py` | `E*_results.json`, `table_adaptive_sampling_pilot.csv` |
| Reliability layer (pancreatic panel) | `E5_deben_generality.py` | printed |
| Running time of CertStop and the published triggers (organoids) | `runtime_comparison.py` | `table_runtime.csv` |

Figure checks used for submission: `check_text_line_overlap.py` (no lettering crossed by a
line), `check_label_overlap.py` (panel labels clear of other text), `export_figures_pdf.py`
(no lettering under 8 pt).

## Notes on reproducibility

* All organoid results use leave-condition-out cross-fitting over the 18 (cell line × drug ×
  dose) conditions; the producer's condition input is the drug (one-hot) and log dose, and a
  drug held out entirely is encoded as unknown (all-zero indicator).
* Scripts that fit MiniRocket and logistic models pin BLAS/numba threads to 2. Changing the
  thread count changes some posteriors at the 1e-3 level and can move individual
  win/tie/loss cells; summary statistics are stable.
* Seeds are fixed in every script; runtimes range from seconds to about 15 minutes per
  script on a desktop CPU.

## Licence and citation

Code: MIT (see `LICENSE`). If you use this code or data, please cite the paper (see
`CITATION.cff`) and the original data sources listed above.
