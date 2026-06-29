#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Assemble all 6 main tables (LaTeX) from the real CSVs in posterior/."""
import csv
def rd(p): return list(csv.DictReader(open(p,encoding="utf-8-sig")))
out=[]
out.append(r"""% ===== TABLE I : dataset & task summary =====
\begin{table}[t]\centering\small\caption{Dataset and task summary (real cohorts).}\label{tab:data}
\begin{tabular}{@{}lll@{}}\toprule
Attribute & Primary cohort & Independent cohort\\\midrule
Cohort & breast-organoid HSLCI & pancreatic (NOGR)\\
Cancer & breast & pancreatic\\
Sample unit & single organoid & single organoid\\
$n$ & $1{,}888$ & $1{,}934$\\
Lines & MCF-7, BT-474 & patient-derived\\
Drugs & Stauro., Nerat., Lapat. & Oxal., 5-FU, Pacli., Gemcit., SN38\\
Conc. & $0.1/1/10\,\mu$M & per protocol\\
Window & $6$--$72$\,h & early growth window\\
Input & $x(\tau),z{=}(d,c)$ & early growth indicator\\
Endpoint & $\mathbf 1[x(72)/x(6){<}0.9]$ & NOGR viability\\
Role & main evaluation & external transfer\\\bottomrule
\end{tabular}\end{table}""")

# Table II : pivot model x horizon
r2=rd("posterior/table2_fixed_time_baselines.csv")
m={}
for r in r2: m.setdefault(r["model"],{})[int(float(r["horizon_h"]))]=r
order=["LR","RF","HGB","SVM-RBF","kNN","trajectory encoder","near-Bayes (LDA)"]
t2=[r"""% ===== TABLE II : fixed-time baselines =====
\begin{table}[t]\centering\small\caption{Fixed-time discrimination baselines (real, 5-fold cross-fit). Diverse models cluster at the same level and the dense trajectory encoder does not exceed them: fixed-time early discrimination is saturated.}\label{tab:baselines}
\begin{tabular}{@{}lcccccc@{}}\toprule
& \multicolumn{3}{c}{$24$\,h} & \multicolumn{3}{c}{$48$\,h}\\ \cmidrule(lr){2-4}\cmidrule(lr){5-7}
Model & AUC & ECE & Brier & AUC & ECE & Brier\\\midrule"""]
for mod in order:
    a=m[mod][24]; b=m[mod][48]
    t2.append(f"{mod} & {float(a['AUC']):.3f} & {float(a['ECE']):.3f} & {float(a['Brier']):.3f} & {float(b['AUC']):.3f} & {float(b['ECE']):.3f} & {float(b['Brier']):.3f}\\\\")
t2.append(r"\bottomrule\end{tabular}\end{table}"); out.append("\n".join(t2))

# Table III
r3=rd("posterior/table3_window_info.csv")
t3=[r"""% ===== TABLE III : window-information bound =====
\begin{table}[t]\centering\small\caption{Window-information bound verification (real). Measured trajectory-over-endpoint AUC gain stays below $\sqrt{\mathcal I_W}/(2\sqrt\pi)$ and is ${\le}0$ for Neratinib/Lapatinib.}\label{tab:bound}
\begin{tabular}{@{}lccccc@{}}\toprule
Drug & $\mathcal D_{\rm end}$ & $\mathcal D_{\rm full}$ & $\mathcal I_W$ & bound & $\Delta$AUC\\\midrule"""]
nm={"Stauro":"Staurosporine","Ner":"Neratinib","Lap":"Lapatinib"}
for r in r3:
    t3.append(f"{nm[r['drug']]} & {float(r['D_end']):.3f} & {float(r['D_full']):.3f} & {float(r['I_W']):.3f} & {float(r['bound_sqrtIW_2sqrtpi']):.3f} & {float(r['measured_dAUC']):+.3f}\\\\")
t3.append(r"\bottomrule\end{tabular}\end{table}"); out.append("\n".join(t3))

# Table IV
r4=rd("posterior/table4_ablation.csv")
t4=[r"""% ===== TABLE IV : posterior-producer / calibration ablation =====
\begin{table}[t]\centering\small\caption{Posterior-producer and calibration ablation (real). Component rows use a common logistic posterior + the same controller; simplex/unconstrained rows are the GRU fusion variants. Regret stays within $2L\varepsilon$.}\label{tab:ablation}
\begin{tabular}{@{}lcccccc@{}}\toprule
Producer & AUC & ECE & MCE & Brier & $\bar t$ & error\\\midrule"""]
for r in r4:
    mt=r["mean_time_h"]; er=r["decision_error"]
    mt=f"{float(mt):.1f}" if mt else "--"; er=f"{float(er):.3f}" if er else "--"
    t4.append(f"{r['producer']} & {float(r['AUC']):.3f} & {float(r['ECE']):.3f} & {float(r['MCE']):.3f} & {float(r['Brier']):.3f} & {mt} & {er}\\\\")
t4.append(r"\bottomrule\end{tabular}\end{table}"); out.append("\n".join(t4))

# Table V
r5=rd("posterior/table5_matched_ops.csv")
t5=[r"""% ===== TABLE V : matched operating points =====
\begin{table}[t]\centering\small\caption{Matched operating points (real). At every point the controller saves observation time at matched decision error; PPV/NPV rise as the band tightens.}\label{tab:matched}
\begin{tabular}{@{}lcccccc@{}}\toprule
OP & ctrl err & $\bar t_{\rm ctrl}$ & $t_{\rm fix}$ & saved & PPV & NPV\\\midrule"""]
for r in r5:
    t5.append(f"{r['OP']} & {float(r['controller_error']):.3f} & {float(r['controller_mean_time_h']):.1f} & {float(r['matched_fixed_time_h']):.1f} & {float(r['hours_saved']):.1f} & {float(r['PPV']):.3f} & {float(r['NPV']):.3f}\\\\")
t5.append(r"\bottomrule\end{tabular}\end{table}"); out.append("\n".join(t5))

# Table VI
r6=rd("posterior/table6_robustness.csv")
t6=[r"""% ===== TABLE VI : robustness & transfer =====
\begin{table}[t]\centering\small\caption{Robustness and transfer (real). Dominated matched operating points under each shift; the single exception is the extreme $10\,\mu$M dose. Sparse-imaging cost stays flat; the framework transfers to an external pancreatic cohort.}\label{tab:robust}
\begin{tabular}{@{}llc@{}}\toprule
Setting & Dominated & Time saving\\\midrule"""]
for r in r6:
    sv=r["mean_time_saving_h"]; sv=f"{float(sv):.1f}\\,h" if sv else "--"
    setting=r['setting'].replace('->','$\\to$').replace('uM','~$\\mu$M')
    t6.append(f"{setting} & {r['dominated_points']} & {sv}\\\\")
t6.append(r"\bottomrule\end{tabular}\end{table}"); out.append("\n".join(t6))

open("tables_all.tex","w",encoding="utf-8").write("\n\n".join(out)+"\n")
print("wrote tables_all.tex with 6 tables")
