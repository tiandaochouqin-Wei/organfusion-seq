#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stage 3P: Theorem 4 where the number of conditions is not nine.

On the primary cohort the condition-level rule has m = 9 calibration conditions, so its
deflated target alpha - (B - alpha)/m is zero at alpha = 0.10 and the rule returns the endpoint
assay: the theorem is shown valid but, at tight targets, vacuous. The independent pancreatic
panel (Deben et al., 2024) has 62 (patient line x drug) conditions, each imaged at 72 h and at
120 h in the same well, which makes a two-horizon early-decision problem with many conditions:

  unit      a well (967; each has a 72 h and a 120 h record)
  evidence  the 72 h read-outs (NOGR, growth rate, relative viability), dose, drug
  label     y = 1[relative viability at 120 h < 0.5]   (the endpoint the plate is run for)
  rule      call at 72 h if |p_hat - 1/2| >= margin, otherwise wait for the 120 h read-out
            (the endpoint assay itself: zero decision risk, 48 h later)
  condition (patient line, drug); deployment = line-drug pairs never seen in calibration

The posterior is cross-fitted leave-condition-out (GroupKFold over the 62 conditions), the
family is the fit-free margin family plus the endpoint assay, and conditions are split at random
into calibration and test, exactly as constrained_control_crc.py does on the primary cohort.
Certificates compared: none; Hoeffding over wells (Theorem 3); Learn-then-Test over wells (the
calibration of Ringel et al., ICML 2024); conformal risk control over conditions (Theorem 4).
A second sweep varies the number of calibration conditions m.

Output: posterior/table_crc_many_conditions.csv
Run with the miniconda interpreter (openpyxl):  python crc_many_conditions.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import DATA_DIR  # $ORGANFUSION_DATA, or ./datasets (see README)
import os, sys, csv
import numpy as np
import openpyxl
from scipy.stats import binom
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

SRC = os.path.join(DATA_DIR, "Deben_NOGR_SupplementaryData1.xlsx")
OUT = "posterior"
DELTA = 0.10; B = 1.0; R = 50; N_BOOT = 2000
ALPHAS = (0.05, 0.10, 0.15, 0.20)
T_EARLY, T_END = 72.0, 120.0
RV_SENSITIVE = 0.5

# ---------------------------------------------------------------- wells
ws = openpyxl.load_workbook(SRC, read_only=True)["Sheet1"]
rows = list(ws.iter_rows(values_only=True)); col = {h: i for i, h in enumerate(rows[0])}
rec = {}
for r in rows[1:]:
    key = (str(r[col["CellLine"]]), str(r[col["Well"]]), str(r[col["Conditions"]]), float(r[col["Conc1"]]))
    rec.setdefault(key, {})[int(r[col["Timepoint"]])] = r
keys = [k for k, v in rec.items() if set(v) == {72, 120}]
f72 = lambda k, h: float(rec[k][72][col[h]])
line = np.array([k[0] for k in keys]); drug = np.array([k[2] for k in keys])
conc = np.array([k[3] for k in keys])
nogr = np.array([f72(k, "NOGR") for k in keys]); grow = np.array([f72(k, "GrowthRate") for k in keys])
rv72 = np.array([f72(k, "RV") for k in keys])
rv120 = np.array([float(rec[k][120][col["RV"]]) for k in keys])
y = (rv120 < RV_SENSITIVE).astype(int)
cond = np.array([f"{a}|{b}" for a, b in zip(line, drug)])
conds = np.array(sorted(set(cond))); C = len(conds); N = len(y)
cidx = [np.where(cond == c)[0] for c in conds]
print(f"wells N={N}  conditions (line x drug) C={C}  wells per condition "
      f"{min(map(len, cidx))}-{max(map(len, cidx))}  prevalence {y.mean():.3f}")

# ---------------------------------------------------------------- leave-condition-out posterior at 72 h
DR = sorted(set(drug)); doh = np.array([np.eye(len(DR))[DR.index(d)] for d in drug])
X = np.column_stack([nogr, grow, rv72, np.log10(conc), doh])
P = np.zeros(N)
for tr, te in GroupKFold(5).split(X, y, cond):
    P[te] = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]


def ece(p, yy, nbins=10):
    e = 0.0
    for b in range(nbins):
        m = (p > b / nbins) & (p <= (b + 1) / nbins) if b else (p <= 1 / nbins)
        if m.any(): e += m.mean() * abs(yy[m].mean() - p[m].mean())
    return e


print(f"72 h posterior, leave-condition-out: AUC {roc_auc_score(y, P):.3f}  ECE {ece(P, y):.3f}  "
      f"error of calling everything at 72 h {((P >= 0.5) != y).mean():.3f}")

# ---------------------------------------------------------------- family: margins + endpoint assay
MARGINS = np.linspace(0.0, 0.5, 51)
early = np.array([np.abs(P - 0.5) >= m for m in MARGINS] + [np.zeros(N, bool)])        # [S, N]
dec = (P >= 0.5).astype(int)
FPu = early & (dec == 1)[None, :] & (y == 0)[None, :]                                   # per-well losses
FNu = early & (dec == 0)[None, :] & (y == 1)[None, :]
TMu = np.where(early, T_EARLY, T_END)
S = len(early)
Lfp = np.array([[FPu[s, ix].mean() for ix in cidx] for s in range(S)])                  # [S, C]
Lfn = np.array([[FNu[s, ix].mean() for ix in cidx] for s in range(S)])


def crc_select(L, alpha):
    Lt = np.maximum.accumulate(L[::-1], axis=0)[::-1]
    ok = np.where((Lt.sum(1) + B) / (L.shape[1] + 1) <= alpha)[0]
    return int(ok[0]) if len(ok) else None


def hoeffding_select(fp, fn, alpha, n, slack=True):
    U = np.sqrt(np.log(2 * S / DELTA) / (2 * n)) if slack else 0.0
    ok = np.where((fp + U <= alpha) & (fn + U <= alpha))[0]
    return int(ok[0]) if len(ok) else None


def ltt_select(fp, fn, alpha, n):
    sel = None
    for s in range(S - 1, -1, -1):
        pv = max(binom.cdf(int(round(fp[s] * n)), n, alpha), binom.cdf(int(round(fn[s] * n)), n, alpha))
        if pv <= DELTA: sel = s
        else: break
    return sel


def select(arm, ci, alpha):
    w = np.concatenate([cidx[c] for c in ci]); n = len(w)
    if arm == "crc-condition":
        a, b = crc_select(Lfp[:, ci], alpha), crc_select(Lfn[:, ci], alpha)
        return None if (a is None or b is None) else max(a, b)
    fp, fn = FPu[:, w].mean(1), FNu[:, w].mean(1)
    if arm == "ltt-well": return ltt_select(fp, fn, alpha, n)
    return hoeffding_select(fp, fn, alpha, n, slack=(arm == "hoeffding-well"))


def run(n_cal, alphas, arms, seed=0):
    rng = np.random.default_rng(seed); out = []
    splits = [rng.permutation(C) for _ in range(R)]
    for alpha in alphas:
        for arm in arms:
            holds = 0; tm = []; fpc = []; fnc = []; inside = []; endpoint = 0
            for perm in splits:
                ci, ti = perm[:n_cal], perm[n_cal:]
                s = select(arm, ci, alpha)
                if s is None: s = S - 1                    # nothing certifiable: run the endpoint assay
                endpoint += (s == S - 1)
                w = np.concatenate([cidx[c] for c in ti])
                holds += bool(FPu[s, w].mean() <= alpha and FNu[s, w].mean() <= alpha)
                tm.append(TMu[s, w].mean()); fpc.append(Lfp[s, ti]); fnc.append(Lfn[s, ti])
                inside.append(np.maximum(Lfp[s, ti], Lfn[s, ti]) <= alpha)
            rb = np.random.default_rng(1)
            boots = [np.mean(np.concatenate([inside[i] for i in rb.integers(0, R, R)])) for _ in range(N_BOOT)]
            lo, hi = np.percentile(boots, [2.5, 97.5])
            tmean = float(np.mean(tm))
            out.append(dict(n_cal_conditions=n_cal, alpha=alpha, selection=arm, holds=holds, n_splits=R,
                            mean_time_h=round(tmean, 2),
                            frac_called_at_72h=round((T_END - tmean) / (T_END - T_EARLY), 3),
                            E_FP_new_condition=round(float(np.concatenate(fpc).mean()), 4),
                            E_FN_new_condition=round(float(np.concatenate(fnc).mean()), 4),
                            frac_conditions_within_alpha=round(float(np.mean(np.concatenate(inside))), 3),
                            frac_ci_lo=round(float(lo), 3), frac_ci_hi=round(float(hi), 3),
                            splits_returning_endpoint=int(endpoint)))
    return out


ARMS = ("no-slack", "hoeffding-well", "ltt-well", "crc-condition")
main = run(C // 2, ALPHAS, ARMS)
print(f"\n=== {C // 2} calibration / {C - C // 2} test conditions, R={R} random condition splits ===")
print(f"{'alpha':>5} {'certificate':>15} | {'holds':>7} {'time':>6} {'at 72h':>7} | {'E[FP]':>7} {'E[FN]':>7} {'cond<=a [95% CI]':>20} {'endpoint':>8}")
for r in main:
    print(f"{r['alpha']:>5.2f} {r['selection']:>15} | {r['holds']:>4}/{R} {r['mean_time_h']:>6.1f} {r['frac_called_at_72h']:>7.2f} | "
          f"{r['E_FP_new_condition']:>7.4f} {r['E_FN_new_condition']:>7.4f} {r['frac_conditions_within_alpha']:>6.2f} "
          f"[{r['frac_ci_lo']:.2f},{r['frac_ci_hi']:.2f}] {r['splits_returning_endpoint']:>8}")

print("\n=== condition-level rule against the number of calibration conditions m (test: the other conditions) ===")
sweep = []
for m in (9, 15, 19, 21, 25, 31, 41, 51):
    sweep += run(m, (0.05, 0.10), ("crc-condition", "hoeffding-well", "ltt-well"), seed=10 + m)
for r in sweep:
    defl = r["alpha"] - (B - r["alpha"]) / r["n_cal_conditions"]
    print(f"  m={r['n_cal_conditions']:>2} alpha={r['alpha']:.2f} {r['selection']:>14}: deflated target {max(defl, 0):.3f}  "
          f"holds {r['holds']:>2}/{R}  E[FP]/E[FN] {r['E_FP_new_condition']:.3f}/{r['E_FN_new_condition']:.3f}  "
          f"called at 72 h {r['frac_called_at_72h']:.2f}  endpoint on {r['splits_returning_endpoint']}/{R}")

with open(os.path.join(OUT, "table_crc_many_conditions.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["experiment"] + list(main[0].keys()))
    w.writeheader()
    for r in main: w.writerow(dict(experiment="half-split", **r))
    for r in sweep: w.writerow(dict(experiment="m-sweep", **r))
print("\nwrote table_crc_many_conditions.csv")
