#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stage 3Q: LaTeX for the certify-any-trigger results, written straight from the CSVs of
certify_any_trigger.py so that no number is transcribed by hand.

Writes: _anytrigger_main.tex (main-text Table, tab:anytrigger),
        _anytrigger_supp.tex (Supplementary Tables, tab:anytrigger_org and tab:anytrigger_ucr)
and prints the numbers quoted in the text.
"""
import csv, sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
A = list(csv.DictReader(open("posterior/table_certify_triggers_organoid.csv", encoding="utf-8")))
U = list(csv.DictReader(open("posterior/table_certify_triggers_ucr40.csv", encoding="utf-8")))
FAMS = ["constant margin (ours)", "CALIMERA", "EconomyGamma", "TEASER-cost"]
LAB = {"constant margin (ours)": r"\method{} (margin)", "CALIMERA": "CALIMERA", "EconomyGamma": "EconomyGamma",
       "TEASER-cost": "TEASER, cost crit."}
ARMLAB = {"none": "none", "ltt": "LTT", "hoeffding": "Thm.~3", "crc-condition": "Thm.~4"}


def a(part, fam, alpha, sel):
    return next(r for r in A if r["part"] == part and r["family"] == fam and abs(float(r["alpha"]) - alpha) < 1e-9
                and r["selection"] == sel)


def u(fam, alpha, sel):
    rr = [r for r in U if r["family"] == fam and abs(float(r["alpha"]) - alpha) < 1e-9 and r["certificate"] == sel]
    fe = sum(int(r["splits_certifiable"]) for r in rr); ho = sum(int(r["splits_holding"]) for r in rr)
    return sum(int(r["splits_certifiable"]) > 0 for r in rr), fe, ho


# ---------------- main-text table
L = [r"\begin{table}[t]\centering\footnotesize\caption{Certifying published triggers as the margin family of Table~\ref{tab:crc}.",
     r"Organoids: splits (of fifty, test conditions new) whose pooled test FP and FN both meet $\alpha{=}0.20$, by",
     r"selection rule; h: mean decision time of the condition-level rule. UCR: share of certifiable (dataset, split)",
     r"pairs on which Learn-then-Test meets $\alpha{=}0.10$ (datasets with any certifiable rule). Supplementary",
     r"Tables~S17--S18 give full results.}\label{tab:anytrigger}",
     r"\begin{tabular*}{\columnwidth}{@{\extracolsep{\fill}}lccccc c@{}}\toprule",
     r"& \multicolumn{5}{c}{organoids, new conditions} & UCR\\\cmidrule(lr){2-6}\cmidrule(lr){7-7}",
     r"Family & none & LTT & Thm.~3 & Thm.~4 & h & LTT\\\midrule"]
for f in FAMS:
    c = [a("A-drift", f, 0.20, s) for s in ("none", "ltt", "hoeffding", "crc-condition")]
    cells = [f"{r['holds']}/{r['n_splits']}" for r in c]
    cells[3] = r"\textbf{" + cells[3] + "}" if c[3]["holds"] == c[3]["n_splits"] else cells[3]
    if f == "TEASER-cost": uc = "--"
    else:
        nd, fe, ho = u(f, 0.10, "ltt"); uc = f"${100 * ho / fe:.1f}\\%$ ({nd})"
    L.append(f"{LAB[f]} & " + " & ".join(cells) + f" & {float(c[3]['mean_time_h']):.1f} & {uc}\\\\")
L += [r"\bottomrule\end{tabular*}\end{table}"]
open("_anytrigger_main.tex", "w", encoding="utf-8").write("\n".join(L) + "\n")

# ---------------- supplementary tables
S = [r"\begin{table*}[t]\centering\small",
     r"\caption{\textbf{Certifying the published triggers under condition drift: organoids (supports Sec.~6.4 and",
     r"Table~\ref{tab:anytrigger_main_ref}).} Each trigger is swept over $24$ delay weights",
     r"($2\cdot10^{-4}$--$6\cdot10^{-2}$/h; consecutive weights that give identical decisions are merged), fitted",
     r"out-of-fold by condition (5-fold GroupKFold over the $18$ conditions) on the same leave-condition-out posterior,",
     r"and certified with the endpoint assay appended, exactly as the margin family of Table~6; same fifty $9/9$",
     r"condition splits. Columns as in Table~6; ``random halves'': the same selection when organoids are split at",
     r"random, so that calibration and test units are exchangeable. At $\alpha{=}0.10$ the condition-level rule",
     r"returns the endpoint assay for every family ($m{=}9$). Our margin family needs no fitting; the triggers' losses",
     r"carry the same cross-fitting caveat as the posterior.}\label{tab:anytrigger_org}",
     r"{\setlength{\tabcolsep}{3pt}%",
     r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}llcccccc cc@{}}\toprule",
     r"& & \multicolumn{6}{c}{new conditions (fifty $9/9$ splits)} & \multicolumn{2}{c}{random halves: holds}\\",
     r"\cmidrule(lr){3-8}\cmidrule(lr){9-10}",
     r"Family & $\alpha$ & none & LTT & Thm.~3 & Thm.~4 & time (h) & $\mathbb E[\mathrm{FP}]/\mathbb E[\mathrm{FN}]$ & LTT & Thm.~3\\\midrule"]
for f in FAMS:
    for al in (0.15, 0.20, 0.25):
        c = {s: a("A-drift", f, al, s) for s in ("none", "ltt", "hoeffding", "crc-condition")}
        k = {s: a("A-control-exchangeable", f, al, s) for s in ("ltt", "hoeffding")}
        cr = c["crc-condition"]
        S.append(f"{LAB[f] if al == 0.15 else ''} & ${al:.2f}$ & " + " & ".join(f"{c[s]['holds']}/50" for s in c)
                 + f" & {float(cr['mean_time_h']):.1f} & ${float(cr['E_FP_new']):.3f}/{float(cr['E_FN_new']):.3f}$"
                 + f" & {k['ltt']['holds']}/50 & {k['hoeffding']['holds']}/50\\\\")
    if f != FAMS[-1]: S.append(r"\midrule")
S += [r"\bottomrule\end{tabular*}}\end{table*}", ""]
S += [r"\begin{table}[t]\centering\small",
      r"\caption{\textbf{Certifying the published triggers with exchangeable units: the $40$ UCR datasets (supports",
      r"Sec.~6.4).} Each trigger is fitted on the out-of-fold training posteriors at twelve delay weights and applied to the",
      r"test split; the test series are halved at random into calibration and deployment ($R{=}50$ per dataset, the",
      r"splits of Table~\ref{tab:ucrcert}). ``sets'': datasets on which some rule is certifiable; ``splits'': certifiable",
      r"(dataset, split) pairs; ``holds'': share of those on which both deployment-half rates meet $\alpha$.}",
      r"\label{tab:anytrigger_ucr}",
      r"\begin{tabular}{@{}llcccccc@{}}\toprule",
      r"& & \multicolumn{2}{c}{none} & \multicolumn{2}{c}{Hoeffding (Thm.~3)} & \multicolumn{2}{c}{Learn-then-Test}\\",
      r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}",
      r"$\alpha$ & family & sets & holds & sets & holds & sets & holds\\\midrule"]
for al in (0.05, 0.10, 0.20):
    for f in FAMS[:3]:
        cells = []
        for s in ("none", "hoeffding", "ltt"):
            nd, fe, ho = u(f, al, s); cells += [str(nd), f"${100 * ho / fe:.1f}\\%$" if fe else "--"]
        S.append(f"{'$%.2f$' % al if f == FAMS[0] else ''} & {LAB[f]} & " + " & ".join(cells) + r"\\")
    if al != 0.20: S.append(r"\midrule")
S += [r"\bottomrule\end{tabular}\end{table}"]
open("_anytrigger_supp.tex", "w", encoding="utf-8").write("\n".join(S).replace(
    r"Table~\ref{tab:anytrigger_main_ref}", "Table~8 of the main text") + "\n")

# ---------------- numbers quoted in the text
for al in (0.15, 0.20):
    print(f"alpha={al}: " + "; ".join(
        f"{LAB[f]}: Thm4 {a('A-drift', f, al, 'crc-condition')['holds']}/50 @ {a('A-drift', f, al, 'crc-condition')['mean_time_h']} h,"
        f" LTT {a('A-drift', f, al, 'ltt')['holds']}/50, Thm3 {a('A-drift', f, al, 'hoeffding')['holds']}/50,"
        f" LTT random halves {a('A-control-exchangeable', f, al, 'ltt')['holds']}/50" for f in FAMS))
for f in FAMS[:3]:
    nd, fe, ho = u(f, 0.10, "ltt"); print(f"UCR a=0.10 LTT {LAB[f]}: {nd} sets, {ho}/{fe} = {100 * ho / fe:.1f}%")
print("wrote _anytrigger_main.tex, _anytrigger_supp.tex")
