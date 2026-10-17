#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stage 3A, optional item from the Stage 3B record: is the Bayes-DP band's error floor
(0.035 vs CALIMERA's 0.026 on the organoid cohort) a property of the STOPPING RULE or of its
50-bin discretisation of the posterior transition kernel? In the rewritten Theorem 2 the
discretisation enters only through the surrogate-optimisation slack eta, so this is a direct
measurement of eta's practical size.

Reruns seq_baselines.py's Bayes-DP band (same lambda grid, same frontier/envelope logic, same
in-sample kernel construction) with histogram kernels at 25/50/100/200/400 bins (pseudo-count
mass per row held at 0.05 so smoothing does not scale with the bin count) and with a
Nadaraya-Watson kernel-smoothed transition (Gaussian bandwidth h in p-space, 200 bins), and
prints the same min-error / time-at-target-error frontier columns as
table_etsc_baselines_frontier.csv next to the CALIMERA/EconomyGamma rows of that file.

Output: posterior/table_stage3a_dp_bins.csv
"""
import os, sys, csv, numpy as np
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

OUT = "posterior"
d = np.load(os.path.join(OUT, "posterior_data_grouped.npz"), allow_pickle=True)
P = d["P_group"]; y = d["y"].astype(int); G = d["GRID"].astype(float); N, nG = P.shape
LAMS = np.logspace(-4, -0.7, 30)
TARGETS = np.array([0.12, 0.16, 0.20, 0.24])


def make_grid(nb):
    Bbin = np.linspace(0, 1, nb + 1); bc = (Bbin[:-1] + Bbin[1:]) / 2
    return Bbin, bc


def kernel_hist(nb):
    Bbin, bc = make_grid(nb); Ks = []
    for j in range(nG - 1):
        K = np.zeros((nb, nb))
        a = np.clip(np.digitize(P[:, j], Bbin) - 1, 0, nb - 1)
        b = np.clip(np.digitize(P[:, j + 1], Bbin) - 1, 0, nb - 1)
        np.add.at(K, (a, b), 1); K += 0.05 / nb; K /= K.sum(1, keepdims=True); Ks.append(K)
    return Ks, Bbin, bc


def kernel_nw(nb, h):
    """Nadaraya-Watson: row a = Gaussian-weighted (bandwidth h) empirical distribution of
    p_{j+1} over organoids whose p_j is near bc[a]; next-state binned on the same grid."""
    Bbin, bc = make_grid(nb); Ks = []
    for j in range(nG - 1):
        W = np.exp(-0.5 * ((bc[:, None] - P[None, :, j]) / h) ** 2)          # [nb, N]
        b = np.clip(np.digitize(P[:, j + 1], Bbin) - 1, 0, nb - 1)
        K = np.zeros((nb, nb))
        for col in range(nb):
            mk = b == col
            if mk.any(): K[:, col] = W[:, mk].sum(1)
        K += 1e-9; K /= K.sum(1, keepdims=True); Ks.append(K)
    return Ks, Bbin, bc


def dp_frontier(Ks, bc):
    rho_bc = np.minimum(1 - bc, bc); ts, es = [], []
    for lam in LAMS:
        V = rho_bc.copy(); ab = np.full((nG, 2), 0.5)
        for j in range(nG - 2, -1, -1):
            cont = lam * (G[j + 1] - G[j]) + Ks[j] @ V
            cs = np.where(cont < rho_bc - 1e-9)[0]
            if len(cs): ab[j] = [bc[cs.min()], bc[cs.max()]]
            V = np.minimum(rho_bc, cont)
        times = np.empty(N); dec = np.empty(N, int)
        for i in range(N):
            js = nG - 1
            for j in range(nG - 1):
                if not (ab[j, 0] < P[i, j] < ab[j, 1]): js = j; break
            times[i] = G[js]; dec[i] = int(P[i, js] >= 0.5)
        ts.append(times.mean()); es.append((dec != y).mean())
    return np.array(ts), np.array(es)


def envelope(t, e):
    o = np.argsort(t); t = np.asarray(t)[o]; e = np.asarray(e)[o]
    return t, np.minimum.accumulate(e)


def time_at(t, e):
    t2, e2 = envelope(t, e); lo, hi = e2.min(), e2.max()
    return [np.nan if (x < lo or x > hi) else float(np.interp(-x, -e2, t2)) for x in TARGETS], float(e2.min())


VARIANTS = [("hist", 25, None), ("hist", 50, None), ("hist", 100, None), ("hist", 200, None), ("hist", 400, None),
            ("nw", 200, 0.02), ("nw", 200, 0.04), ("nw", 200, 0.08)]
rows = []
print(f"{'variant':>22} {'min_err':>8} " + " ".join(f"t@{x:.2f}" for x in TARGETS))
for kind, nb, h in VARIANTS:
    Ks, Bbin, bc = kernel_hist(nb) if kind == "hist" else kernel_nw(nb, h)
    t, e = dp_frontier(Ks, bc); ta, me = time_at(t, e)
    name = f"dp_{kind}_nb{nb}" + (f"_h{h}" if h else "")
    rows.append([name, round(me, 4)] + [("" if np.isnan(x) else round(x, 2)) for x in ta])
    print(f"{name:>22} {me:>8.4f} " + " ".join(f"{'  n/a':>6}" if np.isnan(x) else f"{x:>6.1f}" for x in ta))

ref = os.path.join(OUT, "table_etsc_baselines_frontier.csv")
if os.path.exists(ref):
    print("\nexternal rules on the same posterior (table_etsc_baselines_frontier.csv):")
    with open(ref, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            print(f"{r['policy']:>22} {float(r['min_achievable_err']):>8.4f} " + " ".join(
                f"{'  n/a':>6}" if r[k] in ('', None) else f"{float(r[k]):>6.1f}" for k in r if k.startswith('time_at')))

# ---- cross-fitted kernel (leave-condition-out): a fine in-sample histogram kernel could
# memorise the very trajectories it is applied to, so the honest version estimates the
# kernel on the other folds' conditions and forward-simulates on the held-out fold only.
from sklearn.model_selection import GroupKFold
groups = d["groups"]
folds = list(GroupKFold(5).split(P, y, groups))


def kernel_hist_idx(nb, idx):
    Bbin, bc = make_grid(nb); Ks = []
    for j in range(nG - 1):
        K = np.zeros((nb, nb))
        a = np.clip(np.digitize(P[idx, j], Bbin) - 1, 0, nb - 1)
        b = np.clip(np.digitize(P[idx, j + 1], Bbin) - 1, 0, nb - 1)
        np.add.at(K, (a, b), 1); K += 0.05 / nb; K /= K.sum(1, keepdims=True); Ks.append(K)
    return Ks, Bbin, bc


def kernel_nw_idx(nb, h, idx):
    Bbin, bc = make_grid(nb); Ks = []
    for j in range(nG - 1):
        W = np.exp(-0.5 * ((bc[:, None] - P[None, idx, j]) / h) ** 2)
        b = np.clip(np.digitize(P[idx, j + 1], Bbin) - 1, 0, nb - 1)
        K = np.zeros((nb, nb))
        for col in range(nb):
            mk = b == col
            if mk.any(): K[:, col] = W[:, mk].sum(1)
        K += 1e-9; K /= K.sum(1, keepdims=True); Ks.append(K)
    return Ks, Bbin, bc


def dp_frontier_cv(kind, nb, h):
    ts, es = [], []
    for lam in LAMS:
        times = np.empty(N); dec = np.empty(N, int)
        for tr, te in folds:
            Ks, Bbin, bc = kernel_hist_idx(nb, tr) if kind == "hist" else kernel_nw_idx(nb, h, tr)
            rho_bc = np.minimum(1 - bc, bc); V = rho_bc.copy(); ab = np.full((nG, 2), 0.5)
            for j in range(nG - 2, -1, -1):
                cont = lam * (G[j + 1] - G[j]) + Ks[j] @ V
                cs = np.where(cont < rho_bc - 1e-9)[0]
                if len(cs): ab[j] = [bc[cs.min()], bc[cs.max()]]
                V = np.minimum(rho_bc, cont)
            for i in te:
                js = nG - 1
                for j in range(nG - 1):
                    if not (ab[j, 0] < P[i, j] < ab[j, 1]): js = j; break
                times[i] = G[js]; dec[i] = int(P[i, js] >= 0.5)
        ts.append(times.mean()); es.append((dec != y).mean())
    return np.array(ts), np.array(es)


print("\nleave-condition-out kernel (fitted on the other folds, applied to the held-out fold):")
for kind, nb, h in [("hist", 50, None), ("hist", 200, None), ("hist", 400, None),
                    ("nw", 200, 0.02), ("nw", 200, 0.04)]:
    t, e = dp_frontier_cv(kind, nb, h); ta, me = time_at(t, e)
    name = f"dp_{kind}_nb{nb}" + (f"_h{h}" if h else "") + "_cvkernel"
    rows.append([name, round(me, 4)] + [("" if np.isnan(x) else round(x, 2)) for x in ta])
    print(f"{name:>32} {me:>8.4f} " + " ".join(f"{'  n/a':>6}" if np.isnan(x) else f"{x:>6.1f}" for x in ta))

with open(os.path.join(OUT, "table_stage3a_dp_bins.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f); w.writerow(["variant", "min_achievable_err"] + [f"time_at_err_{x:.2f}" for x in TARGETS])
    for r in rows: w.writerow(r)
print("\nwrote table_stage3a_dp_bins.csv")
