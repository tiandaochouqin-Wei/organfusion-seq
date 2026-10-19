#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stage 3R: how much does ECE_stop depend on the binning, and how noisy is it?

ECE_stop in Fig. 6 / Table S22 is the stopped-posterior ECE with 10 equal-width bins, stratified by
stopping horizon. Here, for the plug-in Bellman band of stage3a_calib_regret.py (same posterior, same
lambda, same gamma sweep), it is recomputed with 5, 10, 15 and 20 equal-width bins and with 10
equal-mass bins, with a condition-level bootstrap 95% interval (B = 1000), and the number of stopped
organoids per horizon is reported. The question is whether the conclusions drawn from it -- minimum
at the same gamma as the excess cost, monotone on both sides -- survive the choice of binning.
Output: posterior/table_ece_sensitivity.csv
"""
import os, csv, sys
import numpy as np
from scipy.stats import spearmanr
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
OUT = "posterior"
d = np.load(os.path.join(OUT, "posterior_data_grouped.npz"), allow_pickle=True)
P = d["P_group"]; y = d["y"].astype(int); GRID = d["GRID"].astype(float); groups = d["groups"]
N, T = P.shape; LAM = 0.004
GAMMAS = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0]
Bbin = np.linspace(0, 1, 51); bc = (Bbin[:-1] + Bbin[1:]) / 2; nb = len(bc); rho_bc = np.minimum(1 - bc, bc)
conds = np.array(sorted(set(groups.tolist()))); cidx = [np.where(groups == c)[0] for c in conds]


def scale(Pm, g):
    p = np.clip(Pm, 1e-6, 1 - 1e-6); return 1 / (1 + np.exp(-g * np.log(p / (1 - p))))


def rule_dp(Pm):                                   # as stage3a_calib_regret.rule_dp
    Ks = []
    for j in range(T - 1):
        K = np.zeros((nb, nb))
        a = np.clip(np.digitize(Pm[:, j], Bbin) - 1, 0, nb - 1); b = np.clip(np.digitize(Pm[:, j + 1], Bbin) - 1, 0, nb - 1)
        np.add.at(K, (a, b), 1); K += 1e-3; K /= K.sum(1, keepdims=True); Ks.append(K)
    V = rho_bc.copy(); ab = np.full((T, 2), 0.5)
    for j in range(T - 2, -1, -1):
        cont = LAM * (GRID[j + 1] - GRID[j]) + Ks[j] @ V
        cs = np.where(cont < rho_bc - 1e-9)[0]
        if len(cs): ab[j] = [bc[cs.min()], bc[cs.max()]]
        V = np.minimum(rho_bc, cont)
    inside = (Pm[:, :T - 1] > ab[:T - 1, 0]) & (Pm[:, :T - 1] < ab[:T - 1, 1])
    js = np.where((~inside).any(1), (~inside).argmax(1), T - 1)
    return GRID[js], (Pm[np.arange(N), js] >= 0.5).astype(int), Pm[np.arange(N), js]


def ece(p, yy, w, edges):
    e = 0.0; W = w.sum()
    for i in range(len(edges) - 1):
        mk = (p >= edges[i]) & ((p < edges[i + 1]) if i < len(edges) - 2 else (p <= edges[i + 1]))
        ww = w[mk]
        if ww.sum() > 0: e += ww.sum() / W * abs((yy[mk] * ww).sum() / ww.sum() - (p[mk] * ww).sum() / ww.sum())
    return e


def ece_stop(ps, ts, yy, w, scheme):
    e = 0.0
    for t in np.unique(ts):
        mk = ts == t
        if w[mk].sum() == 0: continue
        if scheme == "mass10":
            q = np.quantile(ps[mk], np.linspace(0, 1, 11)); q[0], q[-1] = 0.0, 1.0
            edges = np.unique(q)
        else:
            edges = np.linspace(0, 1, int(scheme[5:]) + 1)
        e += w[mk].sum() / w.sum() * ece(ps[mk], yy[mk], w[mk], edges)
    return e


SCHEMES = ["width5", "width10", "width15", "width20", "mass10"]
rng = np.random.default_rng(0); B = 1000
W = [np.bincount(np.concatenate([cidx[c] for c in rng.integers(0, len(conds), len(conds))]), minlength=N).astype(float)
     for _ in range(B)]
rows = []; cost = {}; est = {s: {} for s in SCHEMES}
for g in GAMMAS:
    tt, dd, ps = rule_dp(scale(P, g))
    cost[g] = float(np.mean(LAM * tt + (dd != y)))
    counts = {int(t): int((tt == t).sum()) for t in np.unique(tt)}
    for s in SCHEMES:
        pt = ece_stop(ps, tt, y, np.ones(N), s)
        bs = [ece_stop(ps, tt, y, w, s) for w in W]
        lo, hi = np.percentile(bs, [2.5, 97.5]); est[s][g] = pt
        rows.append(dict(gamma=g, scheme=s, ece_stop=round(pt, 6), ci_lo=round(lo, 4), ci_hi=round(hi, 4),
                         cost=round(cost[g], 4), min_stopped_per_horizon=min(counts.values()),
                         horizons_used=len(counts)))
cmin = min(cost.values())
print(f"{'scheme':>8} | argmin gamma | Spearman(ECE_stop, excess) | ECE_stop at gamma=0.5 / 0.9 / 3.0  [95% CI at 0.9]")
for s in SCHEMES:
    gs = sorted(est[s]); ex = [cost[g] - cmin for g in gs]; ev = [est[s][g] for g in gs]
    ci = next(r for r in rows if r["scheme"] == s and r["gamma"] == 0.9)
    print(f"{s:>8} | {min(est[s], key=est[s].get):>12} | {spearmanr(ev, ex).statistic:>26.3f} | "
          f"{est[s][0.5]:.3f} / {est[s][0.9]:.3f} / {est[s][3.0]:.3f}  [{ci['ci_lo']:.3f}, {ci['ci_hi']:.3f}]")
r09 = next(r for r in rows if r["scheme"] == "width10" and r["gamma"] == 0.9)
print(f"cost-minimising gamma: {min(cost, key=cost.get)};  smallest number of organoids stopping at one horizon "
      f"(gamma=0.9): {r09['min_stopped_per_horizon']} over {r09['horizons_used']} horizons")
with open(os.path.join(OUT, "table_ece_sensitivity.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
print("wrote table_ece_sensitivity.csv")
