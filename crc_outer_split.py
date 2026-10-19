#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stage 3Q: Theorem 4 exactly as stated -- a family frozen before the calibration conditions are seen.

Tables 7 and 8 certify families evaluated on a posterior (and, for the triggers, rules) cross-fitted
over all 18 conditions, so each condition's losses come from a model that never saw that condition
but the fold structure couples conditions: exchangeability holds only up to the cross-fitting.
Here nothing is cross-fitted. In each of R random splits of the 18 conditions into
    6 TRAIN  : fit the shared prefix encoder (the posterior producer of Table 7), then freeze it
    8 CAL    : compute per-condition losses of the fit-free margin family and select
    4 TEST   : new conditions, evaluated once
the margin family is a fixed function of the TRAIN conditions only, and the CAL and TEST conditions
are exchangeable by the random assignment -- precisely the premises of Theorem 4 (m = 8). The
unit-level selections (none / Learn-then-Test / Hoeffding over CAL organoids) are run on the same
frozen posterior for comparison. No recalibration is fitted (it would make the family depend on CAL).

Output: posterior/table_crc_outer_split.csv
Run with the miniconda interpreter (torch):  python crc_outer_split.py
"""
import os, csv, sys, time
import numpy as np
import torch
import torch.nn as nn
from scipy.stats import binom
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
torch.manual_seed(0); np.random.seed(0)
torch.use_deterministic_algorithms(True)
OUT = "posterior"
R = 50; N_TR, N_CAL, N_TE = 6, 8, 4
DELTA = 0.10; B = 1.0
ALPHAS = (0.15, 0.20, 0.25)

d = np.load(os.path.join(OUT, "posterior_data_shared.npz"), allow_pickle=True)
xgrid = d["xgrid"]; y = d["y"].astype(int); GRID = d["GRID"].astype(float)
drug = d["drug"]; conc = d["conc"].astype(float); groups = d["groups"]
N, T = xgrid.shape
DRUGS = ["Stauro", "Ner", "Lap"]
tnorm = np.tile((GRID / 72.0)[None, :], (N, 1))
inc = np.concatenate([np.zeros((N, 1)), np.diff(xgrid, axis=1)], axis=1)
SEQ = np.stack([xgrid, tnorm, inc], axis=-1).astype(np.float32)
doh = np.array([np.eye(3)[DRUGS.index(x)] for x in drug], dtype=np.float32)
COND = np.concatenate([doh, np.log1p(conc)[:, None].astype(np.float32)], axis=1)
conds = np.array(sorted(set(groups.tolist())))
print(f"N={N}, conditions={len(conds)}; split train {N_TR} / cal {N_CAL} / test {N_TE}, R={R}")


class SharedPosterior(nn.Module):                     # as in protocol_3way.py
    def __init__(self, d_hidden=32, d_cond=8):
        super().__init__()
        self.gru = nn.GRU(input_size=3, hidden_size=d_hidden, batch_first=True)
        self.cond = nn.Linear(4, d_cond)
        self.head = nn.Linear(d_hidden + d_cond, 1)

    def forward(self, seq, cond):
        h, _ = self.gru(seq)
        e = torch.tanh(self.cond(cond)).unsqueeze(1).expand(-1, h.size(1), -1)
        return self.head(torch.cat([h, e], -1)).squeeze(-1)


def fit(tr, epochs=300, lr=0.01):
    m = SharedPosterior(); opt = torch.optim.Adam(m.parameters(), lr=lr); lf = nn.BCEWithLogitsLoss()
    xs = torch.tensor(SEQ[tr]); cs = torch.tensor(COND[tr])
    ys = torch.tensor(y[tr].astype(np.float32)).unsqueeze(1).expand(-1, T)
    m.train()
    for _ in range(epochs):
        opt.zero_grad(); lf(m(xs, cs), ys).backward(); opt.step()
    m.eval(); return m


MARGINS = np.linspace(0.0, 0.5, 51)


def family(P, yy):
    """per-unit FP/FN/time for each margin + the endpoint assay (index increasing = more conservative)."""
    FP, FN, TM = [], [], []
    for m in MARGINS:
        ex = np.abs(P - 0.5) >= m; j = np.where(ex.any(1), ex.argmax(1), T - 1)
        dd = (P[np.arange(len(P)), j] >= 0.5).astype(int)
        FP.append((dd == 1) & (yy == 0)); FN.append((dd == 0) & (yy == 1)); TM.append(GRID[j])
    FP.append(np.zeros(len(yy), bool)); FN.append(np.zeros(len(yy), bool)); TM.append(np.full(len(yy), GRID[-1]))
    return np.array(FP), np.array(FN), np.array(TM)


def crc_select(L, alpha):
    Lt = np.maximum.accumulate(L[::-1], axis=0)[::-1]
    ok = np.where((Lt.sum(1) + B) / (L.shape[1] + 1) <= alpha)[0]
    return int(ok[0]) if len(ok) else None


def unit_select(arm, fp, fn, alpha, n):
    if arm == "ltt":
        sel = None
        for s in range(len(fp) - 1, -1, -1):
            pv = max(binom.cdf(int(round(fp[s] * n)), n, alpha), binom.cdf(int(round(fn[s] * n)), n, alpha))
            if pv <= DELTA: sel = s
            else: break
        return sel
    U = np.sqrt(np.log(2 * len(fp) / DELTA) / (2 * n)) if arm == "hoeffding" else 0.0
    ok = np.where((fp + U <= alpha) & (fn + U <= alpha))[0]
    return int(ok[0]) if len(ok) else None


rng = np.random.default_rng(7)
recs = {(a, arm): [] for a in ALPHAS for arm in ("none", "ltt", "hoeffding", "crc-condition")}
aucs = []
t0 = time.time()
for rep in range(R):
    perm = rng.permutation(conds)
    c_tr, c_cal, c_te = perm[:N_TR], perm[N_TR:N_TR + N_CAL], perm[N_TR + N_CAL:]
    tr = np.where(np.isin(groups, c_tr))[0]
    model = fit(tr)
    with torch.no_grad():
        P = torch.sigmoid(model(torch.tensor(SEQ), torch.tensor(COND))).numpy()
    from sklearn.metrics import roc_auc_score
    te_all = np.where(np.isin(groups, c_te))[0]
    if len(set(y[te_all])) > 1: aucs.append(roc_auc_score(y[te_all], P[te_all, 4]))
    FP, FN, TM = family(P, y)                        # fixed by TRAIN only; evaluated per unit
    cal_c = [np.where(groups == c)[0] for c in c_cal]; te_c = [np.where(groups == c)[0] for c in c_te]
    Lfp = np.array([[FP[s, ix].mean() for ix in cal_c] for s in range(len(FP))])
    Lfn = np.array([[FN[s, ix].mean() for ix in cal_c] for s in range(len(FP))])
    cal = np.concatenate(cal_c); te = np.concatenate(te_c)
    for a in ALPHAS:
        for arm in ("none", "ltt", "hoeffding", "crc-condition"):
            if arm == "crc-condition":
                s1, s2 = crc_select(Lfp, a), crc_select(Lfn, a)
                s = None if (s1 is None or s2 is None) else max(s1, s2)
            else:
                s = unit_select(arm, FP[:, cal].mean(1), FN[:, cal].mean(1), a, len(cal))
            if s is None: s = len(FP) - 1
            fpc = np.array([FP[s, ix].mean() for ix in te_c]); fnc = np.array([FN[s, ix].mean() for ix in te_c])
            recs[a, arm].append(dict(holds=bool(FP[s, te].mean() <= a and FN[s, te].mean() <= a), t=TM[s, te].mean(),
                                     fpc=fpc, fnc=fnc, endpoint=s == len(FP) - 1))
    if (rep + 1) % 10 == 0:
        print(f"  {rep + 1}/{R} splits  [{time.time() - t0:.0f}s]", flush=True)

print(f"\nheld-out 24 h AUC of the frozen posterior on test conditions: {np.mean(aucs):.3f} "
      f"(range {min(aucs):.3f}-{max(aucs):.3f})")
rows = []
for (a, arm), rr in recs.items():
    fp = np.concatenate([r["fpc"] for r in rr]); fn = np.concatenate([r["fnc"] for r in rr])
    row = dict(alpha=a, selection=arm, holds=sum(r["holds"] for r in rr), n_splits=R,
               mean_time_h=round(float(np.mean([r["t"] for r in rr])), 2), E_FP_new=round(float(fp.mean()), 4),
               E_FN_new=round(float(fn.mean()), 4), frac_cond_within=round(float(np.mean(np.maximum(fp, fn) <= a)), 3),
               endpoint_returned=sum(r["endpoint"] for r in rr), deflated_target=round(max(a - (B - a) / N_CAL, 0), 4))
    rows.append(row)
    print(f"  a={a:.2f} {arm:>14}: holds {row['holds']:2d}/{R}  time {row['mean_time_h']:5.1f} h  "
          f"E[FP]/E[FN] {row['E_FP_new']:.3f}/{row['E_FN_new']:.3f}  cond<=a {row['frac_cond_within']:.2f}  "
          f"endpoint {row['endpoint_returned']}/{R}")
with open(os.path.join(OUT, "table_crc_outer_split.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
print("wrote table_crc_outer_split.csv")
