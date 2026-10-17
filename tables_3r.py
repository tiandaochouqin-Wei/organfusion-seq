#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stage 3R: Supplementary Tables S20-S22 written straight from their CSVs (no hand transcription).
  S20 semi-synthetic exact check of Theorem 2   <- posterior/table_semisynthetic_thm2.csv (semisynthetic_thm2.py)
  S21 paired condition-level bootstrap           <- posterior/table_paired_bootstrap.csv  (paired_bootstrap.py)
  S22 ECE_stop under five binnings                <- posterior/table_ece_sensitivity.csv   (ece_sensitivity.py)
Writes _supp_3r.tex.
"""
import csv
rd = lambda f: list(csv.DictReader(open("posterior/" + f, encoding="utf-8")))
L = []
# ---------------- S20
S = rd("table_semisynthetic_thm2.csv")
L += [r"\begin{table}[t]\centering\small",
      r"\caption{\textbf{Theorem~2 with every term known exactly (supports Sec.~6.5).} Semi-synthetic Bayesian model:",
      r"$y\sim\mathrm{Bernoulli}(1/2)$, twelve observations i.i.d.\ $\mathcal N(\pm0.5,1)$ given $y$, so the true posterior is a",
      r"Markov process (Assumption~1); $c_{\rm FP}{=}1$, $c_{\rm FN}{=}3$ ($L{=}3$, $p^\star{=}1/4$), $\lambda{=}0.006$ per step.",
      r"(a)~The deployed estimate is $\hat p=\sigma(\gamma\,\mathrm{logit}\,p)$, a monotone recalibration, so",
      r"$\Delta_{\rm info}{=}0$; the deployed rule is the exact recursion of Theorem~1 on $\hat p$ ($\eta{=}0$). All values are",
      r"computed by dynamic programming on a fine log-odds grid, except ECE$_{\rm stop}$, the binned estimate the experiments",
      r"use, here from $20{,}000$ simulated units. (b)~The estimator sees only the sign of each observation: the information",
      r"deficit $\Delta_{\rm info}=V_q-V$ is the same under every recalibration $\gamma$, and recalibration removes only the",
      r"excess over $V_q$.}\label{tab:semisyn}",
      r"\begin{tabular}{@{}lcccccc@{}}\toprule",
      r"\multicolumn{7}{@{}l}{\textit{(a) miscalibration only}}\\",
      r"$\gamma$ & $J_{\rm dep}{-}V$ & $L[\kappa(\hat\tau){+}\kappa(\tau_q)]$ & ratio & $2L\bar\kappa$ & $\kappa(\hat\tau)$ & ECE$_{\rm stop}$\\\midrule"]
for r in S:
    if r["block"].startswith("A"):
        b = float(r["bound_i"]); ex = float(r["excess"])
        L.append(f"{float(r['gamma']):.2f} & {ex:.4f} & {b:.4f} & {'--' if b == 0 else f'{ex / b:.2f}'} & "
                 f"{float(r['bound_2Lkbar']):.4f} & {float(r['kappa_hat']):.4f} & {float(r['ece_stop_mc']):.4f}\\\\")
L += [r"\midrule", r"\multicolumn{7}{@{}l}{\textit{(b) sign-only estimator}}\\",
      r"$\gamma$ & $J_{\rm dep}{-}V$ & $\Delta_{\rm info}$ & $J_{\rm dep}{-}V_q$ & & & \\\midrule"]
for r in S:
    if r["block"].startswith("B"):
        ex = float(r["excess"]); di = float(r["delta_info"])
        L.append(f"{float(r['gamma']):.2f} & {ex:.4f} & {di:.4f} & {ex - di:.4f} & & & \\\\")
L += [r"\bottomrule\end{tabular}\end{table}", ""]
# ---------------- S21
B = rd("table_paired_bootstrap.csv")
L += [r"\begin{table}[t]\centering\small",
      r"\caption{\textbf{Paired condition-level bootstrap of Table~3 (supports Sec.~6.2).} All rules on the same organoids and",
      r"the same leave-condition-out posterior; the $18$ conditions are resampled with replacement ($B{=}2000$), each rule's",
      r"error--time frontier is recomputed on every replicate, and the difference in hours to reach the target error is",
      r"recorded (negative: the first rule is faster). Intervals that cover $0$ mean no stable difference.}\label{tab:pairedboot}",
      r"\begin{tabular}{@{}lcccc@{}}\toprule",
      r"& \multicolumn{4}{c}{difference in hours at target error [95\% CI]}\\\cmidrule(lr){2-5}",
      r"Comparison & $0.12$ & $0.16$ & $0.20$ & $0.24$\\\midrule"]
NAME = {"margin - CALIMERA": "margin $-$ CALIMERA", "margin - fixed": "margin $-$ fixed horizon",
        "CALIMERA - fixed": "CALIMERA $-$ fixed horizon"}
for comp in NAME:
    rr = [r for r in B if r["comparison"] == comp]
    L.append(NAME[comp] + " & " + " & ".join(
        f"${float(r['diff_h']):+.1f}$ $[{float(r['ci_lo']):+.1f},{float(r['ci_hi']):+.1f}]$" for r in rr) + r"\\")
L += [r"\bottomrule\end{tabular}\end{table}", ""]
# ---------------- S22
E = rd("table_ece_sensitivity.csv")
SCH = [("width5", "5 equal-width"), ("width10", "10 equal-width (used)"), ("width15", "15 equal-width"),
       ("width20", "20 equal-width"), ("mass10", "10 equal-mass")]
L += [r"\begin{table}[t]\centering\small",
      r"\caption{\textbf{ECE at the stopping time under five binnings (supports Sec.~6.5 and Fig.~6).} Plug-in Bellman band on",
      r"the leave-condition-out organoid posterior after scaling the logits by $\gamma$ (as in Fig.~6); ECE$_{\rm stop}$ is",
      r"stratified by stopping horizon; brackets: condition-level bootstrap $95\%$ interval ($B{=}1000$). ``argmin'': the",
      r"$\gamma$ minimizing ECE$_{\rm stop}$ (the cost is minimized at $\gamma{=}0.9$); $\rho$: Spearman correlation of",
      r"ECE$_{\rm stop}$ with excess cost over the twelve values of $\gamma$. At least $24$ organoids stop at each of the",
      r"$13$ horizons ($\gamma{=}0.9$).}\label{tab:ecesens}",
      r"\begin{tabular}{@{}lccccc@{}}\toprule",
      r"Binning & $\gamma{=}0.5$ & $\gamma{=}0.9$ & $\gamma{=}3$ & argmin & $\rho$\\\midrule"]
from scipy.stats import spearmanr
for key, lab in SCH:
    rr = {float(r["gamma"]): r for r in E if r["scheme"] == key}
    gs = sorted(rr); cmin = min(float(rr[g]["cost"]) for g in gs)
    rho = spearmanr([float(rr[g]["ece_stop"]) for g in gs], [float(rr[g]["cost"]) - cmin for g in gs]).statistic
    arg = min(gs, key=lambda g: float(rr[g]["ece_stop"]))
    cell = lambda g: f"{float(rr[g]['ece_stop']):.3f} $[{float(rr[g]['ci_lo']):.3f},{float(rr[g]['ci_hi']):.3f}]$"
    L.append(f"{lab} & {cell(0.5)} & {cell(0.9)} & {cell(3.0)} & {arg:g} & {rho:.2f}\\\\")
L += [r"\bottomrule\end{tabular}\end{table}"]
open("_supp_3r.tex", "w", encoding="utf-8").write("\n".join(L) + "\n")
print("wrote _supp_3r.tex")
