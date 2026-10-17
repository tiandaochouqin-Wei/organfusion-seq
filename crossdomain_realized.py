#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stage 3C pre-check: re-score the cross-domain experiments (PhysioNet 2019, C-MAPSS FD001/FD004)
on REALISED cost with held-out selection.

The original scripts (physionet_protocol.py, cmapss_multi.py) score both policies by the
model-based objective J = E[lambda*tau + rho(p_hat_tau)] -- the cost the posterior BELIEVES it
incurs -- and pick the controller's margin and the fixed horizon on the same organoids/patients
they are scored on (oracle vs oracle). By Theorem 2 the realised cost differs from the believed
one by at most L*kappa, so the two agree only under calibration at the stopping time; a
controller that stops when p_hat is extreme is exactly where over-confidence would flatter it.
Here, for the same cross-fitted posteriors:
  realised cost  = lambda*tau + c_FP*1{delta=1,y=0} + c_FN*1{delta=0,y=1},  delta = 1[p_hat_tau >= p*];
  selection      = margin m and fixed horizon T each chosen on a random half (stratified) and
                   scored on the other half, R repeats;
  also reported  = the original believed-cost oracle comparison and a realised-cost oracle one.
Output: posterior/table_crossdomain_realized.csv
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import DATA_DIR  # $ORGANFUSION_DATA, or ./datasets (see README)
import os, sys, csv, numpy as np
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from sklearn.model_selection import GroupKFold, cross_val_predict, StratifiedShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans

OUT = "posterior"; TOL = 1e-3


def policies(P, G, ms, pstar):
    """precompute (tau, delta, p_at_stop) for every margin and every fixed horizon."""
    N, T = P.shape; mar = []
    for m in ms:
        ex = np.abs(P - pstar) >= m
        j = np.where(ex.any(1), ex.argmax(1), T - 1)
        ps = P[np.arange(N), j]; mar.append((G[j], (ps >= pstar).astype(int), ps))
    fix = [(np.full(N, G[j]), (P[:, j] >= pstar).astype(int), P[:, j]) for j in range(T)]
    return mar, fix


def realised(tau, d, y, lam, cfp, cfn):
    return lam * tau + cfp * ((d == 1) & (y == 0)) + cfn * ((d == 0) & (y == 1))


def believed(tau, ps, lam, cfp, cfn):
    return lam * tau + np.minimum(cfp * (1 - ps), cfn * ps)


def evaluate(name, P, y, G, ms, lambdas, cfp, cfn, R, seed=0):
    pstar = cfp / (cfp + cfn)
    from sklearn.metrics import roc_auc_score
    aucs = [roc_auc_score(y, P[:, j]) for j in range(P.shape[1])]
    print(f"[{name}] AUC by horizon {min(aucs):.3f}-{max(aucs):.3f}")
    mar, fix = policies(P, G, ms, pstar)
    sss = StratifiedShuffleSplit(n_splits=R, test_size=0.5, random_state=seed)
    splits = list(sss.split(np.zeros(len(y)), y))
    rows = []
    for lam in lambdas:
        Rm = np.array([realised(t, d, y, lam, cfp, cfn) for t, d, _ in mar])      # [M, N]
        Rf = np.array([realised(t, d, y, lam, cfp, cfn) for t, d, _ in fix])      # [T, N]
        Bm = np.array([believed(t, p, lam, cfp, cfn) for t, _, p in mar])
        Bf = np.array([believed(t, p, lam, cfp, cfn) for t, _, p in fix])
        g_believed_oracle = Bf.mean(1).min() - Bm.mean(1).min()
        g_realised_oracle = Rf.mean(1).min() - Rm.mean(1).min()
        gs = []
        for a, b in splits:
            im = int(np.argmin(Rm[:, a].mean(1))); jf = int(np.argmin(Rf[:, a].mean(1)))
            gs.append(Rf[jf, b].mean() - Rm[im, b].mean())
        gs = np.array(gs)
        rows.append((name, lam, g_believed_oracle, g_realised_oracle, gs.mean(), gs.std(), (gs > 0).mean()))
    print(f"\n=== {name}  N={len(y)} prevalence={y.mean():.3f} c_FN/c_FP={cfn/cfp:.0f} R={R} ===")
    print(f"{'lambda':>9} {'believed-oracle':>15} {'realised-oracle':>15} {'held-out mean':>13} {'sd':>7} {'frac>0':>6}")
    for r in rows:
        print(f"{r[1]:>9.5f} {r[2]:>+15.4f} {r[3]:>+15.4f} {r[4]:>+13.4f} {r[5]:>7.4f} {r[6]:>6.2f}")
    nb = sum(r[2] > TOL for r in rows); nro = sum(r[3] > TOL for r in rows); nh = sum(r[4] > TOL for r in rows)
    print(f"wins (gain > {TOL}): believed-oracle {nb}/{len(rows)}, realised-oracle {nro}/{len(rows)}, "
          f"held-out {nh}/{len(rows)}; max held-out gain {max(r[4] for r in rows):+.4f}")
    return rows


allrows = []
# ---- PhysioNet 2019 (posterior saved by physionet_protocol.py) ----
d = np.load(os.path.join(OUT, "physionet_posterior_data.npz"), allow_pickle=True)
allrows += evaluate("PhysioNet2019", d["P"], d["y"].astype(int), d["GRID"].astype(float),
                    np.linspace(0.0, 0.15, 49), np.geomspace(1e-4, 2e-2, 25), 1.0, 9.0, R=20)

# ---- C-MAPSS FD001 / FD004 (posterior rebuilt exactly as cmapss_multi.py) ----
DDIR = os.path.join(DATA_DIR, "cmapss")


def cmapss_posterior(subset, grid, regimes):
    raw = np.loadtxt(os.path.join(DDIR, f"train_{subset}.txt"))
    eng = raw[:, 0].astype(int); cyc = raw[:, 1].astype(int); opset = raw[:, 2:5]; S = raw[:, 5:26].copy()
    if regimes > 1:
        lab = KMeans(n_clusters=regimes, n_init=5, random_state=0).fit(opset).labels_
        for r in range(regimes):
            m = lab == r; mu = S[m].mean(0); sd = S[m].std(0); sd[sd < 1e-8] = 1.0; S[m] = (S[m] - mu) / sd
    S = S[:, np.where(S.var(0) > 1e-6)[0]]
    engines = np.unique(eng); maxcyc = {e: cyc[eng == e].max() for e in engines}
    med = np.median(list(maxcyc.values())); y_all = np.array([int(maxcyc[e] <= med) for e in engines])
    N = len(engines); Pm = np.zeros((N, len(grid)))
    for j, T in enumerate(grid):
        X, yy, keep = [], [], []
        for i, e in enumerate(engines):
            m = (eng == e) & (cyc <= T)
            if m.sum() < 2: continue
            Sp = S[m]; tp = cyc[m]
            X.append(sum(([Sp[-1, k], (Sp[-1, k] - Sp[0, k]) / max(tp[-1] - tp[0], 1)] for k in range(Sp.shape[1])), []))
            yy.append(y_all[i]); keep.append(i)
        X = np.array(X); yy = np.array(yy); keep = np.array(keep)
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
        Pm[keep, j] = cross_val_predict(clf, X, yy, cv=GroupKFold(5).split(X, yy, keep), method="predict_proba")[:, 1]
    return Pm, y_all, np.array(grid, float)


for subset, regimes in (("FD001", 1), ("FD004", 6)):
    Pm, yc, Gc = cmapss_posterior(subset, [20, 30, 40, 50, 60, 70, 80], regimes)
    allrows += evaluate(f"C-MAPSS-{subset}", Pm, yc, Gc, np.linspace(0.0, 0.35, 40),
                        np.geomspace(2e-3, 1e-1, 20), 1.0, 3.0, R=50)

with open(os.path.join(OUT, "table_crossdomain_realized.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["domain", "lambda", "gain_believed_oracle", "gain_realised_oracle", "gain_heldout_mean",
                "gain_heldout_sd", "frac_splits_won"])
    for r in allrows: w.writerow([r[0]] + [round(float(x), 6) for x in r[1:]])
print("\nwrote table_crossdomain_realized.csv")
