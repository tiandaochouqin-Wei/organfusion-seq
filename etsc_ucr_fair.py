#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stage 3C pre-check: a FAIR UCR head-to-head for our stopping rules.

Stage 3B juxtaposed the external triggers (etsc_baseline_comparison_ucr.py: symmetric cost,
8 lambdas, trigger FIT ON TRAIN, realised error on test) against Supplementary Table S2
(etsc_generality.py: c_FN=3, 16 lambdas, margin chosen ON TEST, model-based rho(p_hat) cost).
Those are different protocols, so "ours 12/16 vs theirs 0-2/8" is not a comparison. This script
re-runs OUR rules under exactly the external protocol:
  - identical posterior: per-horizon LR on 5 prefix features; out-of-fold (StratifiedKFold(5),
    seed 0) probabilities on TRAIN, full-train fit scored on TEST (same code as the 3B script);
  - identical cost: symmetric, J = lambda*tau + 1[wrong], realised on TEST;
  - identical lambda grid: geomspace(1e-3, 5e-2, 8) / (L/100);
  - every learnable choice made on TRAIN only:
      const-margin band: margin m minimising realised train-OOF J;
      Bayes-DP band: transition kernel estimated on train-OOF paths (50 bins), no free parameter;
  - reference: the best fixed length chosen ON TEST (an oracle, as in the 3B script) and, for
    completeness, the fixed length chosen on train.
A "win" is a gain over the oracle fixed length of more than 1e-3 (ties like +0.0001 are not wins).

Output: posterior/table_etsc_ucr_fair.csv   (ours) ; externals are read from
        posterior/table_etsc_baselines_ucr.csv and summarised with the same win rule.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import DATA_DIR  # $ORGANFUSION_DATA, or ./datasets (see README)
import os, sys, csv, numpy as np
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

UCR = os.path.join(DATA_DIR, "ucr")
DATASETS = ["ECG5000", "Wafer", "Earthquakes", "FordA"]
OUT = "posterior"
TOL = 1e-3


def load(ds, split):
    a = np.loadtxt(os.path.join(UCR, ds, f"{ds}_{split}.txt")); return a[:, 1:], a[:, 0]


def binarize(y_raw, maj=None):
    vals, cnts = np.unique(y_raw, return_counts=True)
    maj = vals[np.argmax(cnts)] if maj is None else maj
    return (y_raw != maj).astype(int), maj


def feats(X, T):
    P = X[:, :T]
    return np.column_stack([P[:, -1], P.mean(1), (P[:, -1] - P[:, 0]) / max(T - 1, 1),
                            P.min(1), P.max(1) - P.min(1)])


def margin_rule(P, G, m):
    ex = np.abs(P - 0.5) >= m
    j = np.where(ex.any(1), ex.argmax(1), P.shape[1] - 1)
    return G[j], (P[np.arange(len(P)), j] >= 0.5).astype(int)


Bbin = np.linspace(0, 1, 51); bc = (Bbin[:-1] + Bbin[1:]) / 2; nb = len(bc)
rho_bc = np.minimum(1 - bc, bc)


def dp_band(Ptr, G, lam):
    T = Ptr.shape[1]; V = rho_bc.copy(); ab = np.full((T, 2), 0.5)
    for j in range(T - 2, -1, -1):
        K = np.zeros((nb, nb))
        a = np.clip(np.digitize(Ptr[:, j], Bbin) - 1, 0, nb - 1)
        b = np.clip(np.digitize(Ptr[:, j + 1], Bbin) - 1, 0, nb - 1)
        np.add.at(K, (a, b), 1); K += 1e-3; K /= K.sum(1, keepdims=True)
        cont = lam * (G[j + 1] - G[j]) + K @ V
        cs = np.where(cont < rho_bc - 1e-9)[0]
        if len(cs): ab[j] = [bc[cs.min()], bc[cs.max()]]
        V = np.minimum(rho_bc, cont)
    return ab


def apply_band(P, G, ab):
    T = P.shape[1]; inside = (P[:, :T - 1] > ab[:T - 1, 0]) & (P[:, :T - 1] < ab[:T - 1, 1])
    out = ~inside
    j = np.where(out.any(1), out.argmax(1), T - 1)
    return G[j], (P[np.arange(len(P)), j] >= 0.5).astype(int)


rows = []
MS = np.linspace(0.0, 0.49, 50)
for ds in DATASETS:
    Xtr, ytr_raw = load(ds, "TRAIN"); Xte, yte_raw = load(ds, "TEST")
    L = Xtr.shape[1]
    G = np.array(sorted(set(max(3, int(round(f * L))) for f in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])), float)
    ytr, maj = binarize(ytr_raw); yte, _ = binarize(yte_raw, maj)
    Ptr = np.zeros((len(Xtr), len(G))); Pte = np.zeros((len(Xte), len(G)))
    kf = StratifiedKFold(5, shuffle=True, random_state=0)
    for j, T_ in enumerate(G.astype(int)):
        Xf = feats(Xtr, T_); oof = np.zeros(len(Xtr))
        for tr_i, te_i in kf.split(Xf, ytr):
            clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(Xf[tr_i], ytr[tr_i])
            oof[te_i] = clf.predict_proba(Xf[te_i])[:, 1]
        Ptr[:, j] = oof
        Pte[:, j] = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(Xf, ytr).predict_proba(feats(Xte, T_))[:, 1]
    auc = [roc_auc_score(yte, Pte[:, j]) for j in range(len(G))]
    lambdas = np.geomspace(1e-3, 5e-2, 8) / (L / 100.0)
    err_fix_te = np.array([((Pte[:, j] >= 0.5) != yte).mean() for j in range(len(G))])
    err_fix_tr = np.array([((Ptr[:, j] >= 0.5) != ytr).mean() for j in range(len(G))])
    print(f"=== {ds} len={L} n_tr={len(ytr)} n_te={len(yte)} AUC[{min(auc):.3f},{max(auc):.3f}] ===")
    for lam in lambdas:
        J_fix_oracle = float((lam * G + err_fix_te).min())
        jtr = int(np.argmin(lam * G + err_fix_tr)); J_fix_train = float(lam * G[jtr] + err_fix_te[jtr])
        # const-margin: m chosen on train OOF
        Jtr = [np.mean(lam * t + (d != ytr)) for t, d in (margin_rule(Ptr, G, m) for m in MS)]
        m_star = MS[int(np.argmin(Jtr))]
        t, d = margin_rule(Pte, G, m_star); J_cm = float(np.mean(lam * t + (d != yte))); t_cm = float(t.mean())
        # Bayes-DP band: kernel from train OOF
        ab = dp_band(Ptr, G, lam); t2, d2 = apply_band(Pte, G, ab)
        J_dp = float(np.mean(lam * t2 + (d2 != yte))); t_dp = float(t2.mean())
        for name, J, tm in (("const_margin(train-selected)", J_cm, t_cm), ("bayes_dp_band(train-kernel)", J_dp, t_dp)):
            rows.append((ds, name, round(float(lam), 6), round(tm, 3), round(J, 4), round(J_fix_oracle, 4),
                         round(J_fix_oracle - J, 4), round(J_fix_train, 4), round(J_fix_train - J, 4)))
        print(f"  lam={lam:.5f}  fix(oracle)={J_fix_oracle:.4f} fix(train)={J_fix_train:.4f} | "
              f"margin m*={m_star:.2f} J={J_cm:.4f} ({J_fix_oracle-J_cm:+.4f}) | DP J={J_dp:.4f} ({J_fix_oracle-J_dp:+.4f})")

with open(os.path.join(OUT, "table_etsc_ucr_fair.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["dataset", "method", "lambda", "mean_time", "J_method", "J_fixed_oracle", "gain_vs_fixed_oracle",
                "J_fixed_trainselected", "gain_vs_fixed_trainselected"])
    for r in rows: w.writerow(r)

# ---- unified summary: ours + externals, same win rule ------------------------------
ext = list(csv.DictReader(open(os.path.join(OUT, "table_etsc_baselines_ucr.csv"), encoding="utf-8")))
print(f"\n=== summary: wins over the ORACLE fixed length (gain > {TOL}), 8 lambdas, identical posterior/cost ===")
print(f"{'dataset':>12} {'method':>30} {'wins':>6} {'best gain':>10} {'mean gain':>10}")
for ds in DATASETS:
    for name in ("const_margin(train-selected)", "bayes_dp_band(train-kernel)"):
        g = [r[6] for r in rows if r[0] == ds and r[1] == name]
        print(f"{ds:>12} {name:>30} {sum(x > TOL for x in g):>3}/{len(g)} {max(g):>+10.4f} {np.mean(g):>+10.4f}")
    for name in ("EconomyGamma", "CALIMERA", "TEASER"):
        g = [float(r["gain_vs_fixed"]) for r in ext if r["dataset"] == ds and r["method"] == name and r["gain_vs_fixed"]]
        if g:
            print(f"{ds:>12} {name:>30} {sum(x > TOL for x in g):>3}/{len(g)} {max(g):>+10.4f} {np.mean(g):>+10.4f}")
        else:
            print(f"{ds:>12} {name:>30}   failed (implementation did not fit)")
print("\nwrote table_etsc_ucr_fair.csv")
