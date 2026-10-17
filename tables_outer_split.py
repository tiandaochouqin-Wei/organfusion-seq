#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stage 3Q: LaTeX for Supplementary Table S19 from posterior/table_crc_outer_split.csv (crc_outer_split.py)."""
import csv
R = list(csv.DictReader(open("posterior/table_crc_outer_split.csv", encoding="utf-8")))
LAB = {"none": "none", "ltt": "Learn-then-Test, organoids", "hoeffding": "Hoeffding, organoids (Thm.~3)",
       "crc-condition": "conditions (Thm.~4)"}
L = [r"\begin{table}[t]\centering\small",
     r"\caption{\textbf{Theorem~4 exactly as stated: a posterior frozen before the calibration conditions are seen",
     r"(supports Sec.~4.4).} In each of $50$ random splits the shared prefix encoder is fitted on $6$ conditions and frozen,",
     r"and the other $12$ are assigned at random to $8$ calibration and $4$ test conditions, so the margin family does",
     r"not depend on the calibration conditions and calibration and test conditions are exchangeable---no cross-fitting.",
     r"Training on six conditions makes the shift harder than in Table~7 (held-out $24$\,h AUC $0.84$, range",
     r"$0.78$--$0.91$). Columns as in Table~7; ``endpoint'': splits on which the rule returns the endpoint assay. The",
     r"guarantee of Theorem~4 is on $\mathbb E[\mathrm{FP}]$ and $\mathbb E[\mathrm{FN}]$, not on every split.}",
     r"\label{tab:outersplit}",
     r"\begin{tabular}{@{}llcccccc@{}}\toprule",
     r"$\alpha$ & selection & holds & time (h) & $\mathbb E[\mathrm{FP}]$ & $\mathbb E[\mathrm{FN}]$ & cond.\ $\le\alpha$ & endpoint\\\midrule"]
for i, a in enumerate(("0.15", "0.2", "0.25")):
    for sel in ("none", "ltt", "hoeffding", "crc-condition"):
        r = next(x for x in R if x["alpha"] == a and x["selection"] == sel)
        L.append(f"{'$%.2f$' % float(a) if sel == 'none' else ''} & {LAB[sel]} & {r['holds']}/{r['n_splits']} & "
                 f"{float(r['mean_time_h']):.1f} & {float(r['E_FP_new']):.3f} & {float(r['E_FN_new']):.3f} & "
                 f"{float(r['frac_cond_within']):.2f} & {r['endpoint_returned']}/{r['n_splits']}\\\\")
    if i < 2: L.append(r"\midrule")
L += [r"\bottomrule\end{tabular}\end{table}"]
open("_outer_split_supp.tex", "w", encoding="utf-8").write("\n".join(L) + "\n")
print("wrote _outer_split_supp.tex")
