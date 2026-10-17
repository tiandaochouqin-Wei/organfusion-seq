#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""B: deepen the industrial domain -- run the SAME controller on C-MAPSS FD001 AND
FD004. FD004 is the hardest subset (6 operating conditions x 2 fault modes), so raw
sensor values are not comparable across conditions; we apply the standard fix
(cluster the 3 operating settings into regimes, z-normalize each sensor within its
regime) before the identical prefix-feature + stop/DP-band pipeline. Same label as
FD001 (short-lived engine: lifetime <= fleet median) and same native-J comparison.

Outputs (extends the FD001 result):
  posterior/table_cmapss_multi.csv   one row per (subset, lambda)
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import DATA_DIR  # $ORGANFUSION_DATA, or ./datasets (see README)
import os, csv, sys, numpy as np
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score

OUT = "posterior"; os.makedirs(OUT, exist_ok=True)
DDIR = os.path.join(DATA_DIR, "cmapss")
c_FP, c_FN = 1.0, 3.0
P_STAR = c_FP / (c_FP + c_FN)
rho = lambda p: np.minimum(c_FP * (1 - p), c_FN * p)


def run_subset(subset, grid, regimes):
    raw = np.loadtxt(os.path.join(DDIR, f"train_{subset}.txt"))
    eng = raw[:, 0].astype(int); cyc = raw[:, 1].astype(int)
    opset = raw[:, 2:5]                          # 3 operating settings
    S = raw[:, 5:26].copy()                       # 21 sensors
    # regime normalization (FD002/FD004): z-normalize sensors within operating regime
    if regimes > 1:
        km = KMeans(n_clusters=regimes, n_init=5, random_state=0).fit(opset)
        lab = km.labels_
        for r in range(regimes):
            m = lab == r
            mu = S[m].mean(0); sd = S[m].std(0); sd[sd < 1e-8] = 1.0
            S[m] = (S[m] - mu) / sd
        var = S.var(0)
    else:
        var = S.var(0)
    sens = np.where(var > 1e-6)[0]
    S = S[:, sens]
    engines = np.unique(eng)
    maxcyc = {e: cyc[eng == e].max() for e in engines}
    med = np.median([maxcyc[e] for e in engines])
    y_all = {e: int(maxcyc[e] <= med) for e in engines}
    N = len(engines)
    print(f"\n=== {subset}: engines={N}, regimes={regimes}, informative sensors={len(sens)}, "
          f"median life={med:.0f}, short-rate={np.mean(list(y_all.values())):.2f} ===")

    def feats(e, T):
        m = (eng == e) & (cyc <= T)
        if m.sum() < 2:
            return None
        Sp = S[m]; tp = cyc[m]; out = []
        for k in range(Sp.shape[1]):
            v = Sp[:, k]; out += [v[-1], (v[-1] - v[0]) / max(tp[-1] - tp[0], 1)]
        return out

    Pm = np.zeros((N, len(grid)))
    aucs = []
    for j, T in enumerate(grid):
        X, y, keep = [], [], []
        for i, e in enumerate(engines):
            f = feats(e, T)
            if f is None:
                continue
            X.append(f); y.append(y_all[e]); keep.append(i)
        X = np.array(X); y = np.array(y); keep = np.array(keep)
        if len(set(y)) < 2:
            aucs.append(np.nan); continue
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
        p = cross_val_predict(clf, X, y, cv=GroupKFold(5).split(X, y, keep), method="predict_proba")[:, 1]
        Pm[keep, j] = p
        aucs.append(roc_auc_score(y, p))
    G = np.array(grid, float)
    aucs = np.array(aucs)
    print(f"  AUC by horizon: {np.round(aucs, 3).tolist()}  (range {np.nanmin(aucs):.3f}-{np.nanmax(aucs):.3f})")

    def controller(m):
        tau = np.empty(N); pdec = np.empty(N)
        for i in range(N):
            ex = np.where(np.abs(Pm[i] - P_STAR) >= m)[0]
            jj = ex[0] if len(ex) else len(G) - 1
            tau[i] = G[jj]; pdec[i] = Pm[i, jj]
        return tau, pdec

    ms = np.linspace(0.0, 0.35, 40); lambdas = np.geomspace(2e-3, 1e-1, 20)
    rows = []
    for lam in lambdas:
        Jf = lam * G + np.array([rho(Pm[:, j]).mean() for j in range(len(G))])
        jb = int(np.argmin(Jf)); J_fix = float(Jf[jb]); T_fix = float(G[jb])
        Jc = np.empty(len(ms)); tc = np.empty(len(ms))
        for k, m in enumerate(ms):
            tau, pd = controller(m); Jc[k] = (lam * tau + rho(pd)).mean(); tc[k] = tau.mean()
        kb = int(np.argmin(Jc))
        rows.append((subset, round(lam, 5), round(J_fix, 4), round(T_fix, 0),
                     round(float(Jc[kb]), 4), round(float(tc[kb]), 1), round(J_fix - float(Jc[kb]), 5)))
    nwin = sum(1 for r in rows if r[-1] > 0)
    print(f"  controller lowers J at {nwin}/{len(rows)} delay weights "
          f"(AUC only {np.nanmin(aucs):.2f}-{np.nanmax(aucs):.2f})")
    return rows, (float(np.nanmin(aucs)), float(np.nanmax(aucs)), nwin, len(rows))


allrows = []
summary = {}
for subset, grid, regimes in [("FD001", [20, 30, 40, 50, 60, 70, 80], 1),
                               ("FD004", [20, 30, 40, 50, 60, 70, 80], 6)]:
    rows, summ = run_subset(subset, grid, regimes)
    allrows += rows; summary[subset] = summ

with open(os.path.join(OUT, "table_cmapss_multi.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f); w.writerow(["subset", "lambda", "J_fixed", "T_fixed", "J_ctrl", "mean_T_ctrl", "gain"])
    for r in allrows:
        w.writerow(r)
print("\n=== SUMMARY ===")
for s, (amin, amax, nwin, ntot) in summary.items():
    print(f"  {s}: AUC {amin:.2f}-{amax:.2f}, controller wins {nwin}/{ntot} lambda")
print("wrote table_cmapss_multi.csv")
