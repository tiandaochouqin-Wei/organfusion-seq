#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stage 3F, A1: does "nothing beats fixed time on UCR" survive a STRONGER posterior?

etsc_ucr_fair.py and etsc_baseline_comparison_ucr.py both used a deliberately weak
posterior producer (5 hand-crafted prefix features: last value, mean, slope, min, range)
shared by every stopping rule -- adequate for the paper's own controller (which only needs
p_t, not a state-of-the-art classifier) but a genuine confound when the conclusion is
"no rule, published or ours, beats fixed time here": a rule cannot out-decide a posterior
that carries no more information than five numbers. This script reruns the identical fair
protocol (train-only fitting of every free parameter; realised cost; the archive's own
train/test split; same lambda grid, same cost structure) with the posterior replaced by
MiniRocket convolutional features (aeon, n_kernels=1000) + logistic regression, refit
independently at each prefix horizon. Same horizons as etsc_ucr_fair.py (fractions 0.2-0.9
of series length) so the comparison isolates the posterior, not the discretisation.

Design (unchanged from etsc_ucr_fair.py): our rules' free parameters (constant-margin m,
Bellman-band transition kernel) are chosen on TRAIN-fold out-of-fold predictions only,
scored once on TEST; published triggers (EconomyGamma, CALIMERA, TEASER via ml_edm) are
fit on train out-of-fold probabilities the same way; the fixed-horizon reference is the
oracle (best on TEST) as before, so if anything this is unfavourable to every adaptive rule.

Output: posterior/table_etsc_ucr_strong.csv
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import DATA_DIR  # $ORGANFUSION_DATA, or ./datasets (see README)
import os, sys, csv, time, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "_external", "ml_edm", "src"))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from aeon.transformations.collection.convolution_based import MiniRocket
from ml_edm.cost_matrices import CostMatrices
from ml_edm.trigger import EconomyGamma, CALIMERA, TEASER

UCR = os.path.join(DATA_DIR, "ucr")
DATASETS = ["ECG5000", "Wafer", "Earthquakes", "FordA"]
OUT = "posterior"
TOL = 1e-3
N_KERNELS = 1000


def load(ds, split):
    a = np.loadtxt(os.path.join(UCR, ds, f"{ds}_{split}.txt")); return a[:, 1:], a[:, 0]


def binarize(y_raw, maj=None):
    vals, cnts = np.unique(y_raw, return_counts=True)
    maj = vals[np.argmax(cnts)] if maj is None else maj
    return (y_raw != maj).astype(int), maj


def minirocket_posterior(Xtr, ytr, Xte, T):
    """MiniRocket-transform the length-T prefix once, then 5-fold OOF logistic regression
    on train (for fitting triggers/margins) and a full-train fit scored on test."""
    Ptr_raw = Xtr[:, :T].astype(np.float32)[:, None, :]
    Pte_raw = Xte[:, :T].astype(np.float32)[:, None, :]
    mr = MiniRocket(n_kernels=N_KERNELS, random_state=0)
    Ztr = mr.fit_transform(Ptr_raw)
    Zte = mr.transform(Pte_raw)
    mu = Ztr.mean(0); sd = Ztr.std(0) + 1e-8
    Ztr = (Ztr - mu) / sd; Zte = (Zte - mu) / sd
    kf = StratifiedKFold(5, shuffle=True, random_state=0)
    oof = np.zeros(len(ytr))
    for tr_i, te_i in kf.split(Ztr, ytr):
        clf = LogisticRegression(max_iter=2000, C=1.0)
        clf.fit(Ztr[tr_i], ytr[tr_i])
        oof[te_i] = clf.predict_proba(Ztr[te_i])[:, 1]
    clf_full = LogisticRegression(max_iter=2000, C=1.0).fit(Ztr, ytr)
    pte = clf_full.predict_proba(Zte)[:, 1]
    return oof, pte


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


def run_ext_method(cls, kwargs, cm, GRID, probas_tr, y_tr, probas_te, y_te, folds_tr):
    T = len(GRID); Ntr = len(y_tr); Nte = len(y_te)
    trig = cls(timestamps=GRID, **kwargs)
    trig.fit(np.zeros((Ntr, T)), probas_tr, y_tr, cm)
    triggered = np.zeros(Nte, bool)
    preds = np.full(Nte, -1, dtype=int); ts = np.full(Nte, -1.0)
    for t_idx, length in enumerate(GRID):
        p_arg = probas_te[:, :t_idx + 1, :] if trig.require_past_probas else probas_te[:, t_idx, :]
        X_now = np.zeros((Nte, int(length)))
        fire = np.asarray(trig.predict(X_now, p_arg, cm))
        newly = fire & (~triggered)
        preds[newly] = probas_te[:, t_idx, :][newly].argmax(-1)
        ts[newly] = length
        triggered |= fire
        if triggered.all(): break
    if not triggered.all():
        rem = ~triggered
        preds[rem] = probas_te[:, -1, :][rem].argmax(-1); ts[rem] = GRID[-1]
    return float(ts.mean()), float((preds != y_te).mean())


METHODS = [("EconomyGamma", EconomyGamma, dict(nb_intervals=5)), ("CALIMERA", CALIMERA, {}),
           ("TEASER", TEASER, dict(n_jobs=2))]
MS = np.linspace(0.0, 0.49, 50)
rows = []
t_start = time.time()
for ds in DATASETS:
    Xtr, ytr_raw = load(ds, "TRAIN"); Xte, yte_raw = load(ds, "TEST")
    L = Xtr.shape[1]
    G = np.array(sorted(set(max(9, int(round(f * L))) for f in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])), float)
    ytr, maj = binarize(ytr_raw); yte, _ = binarize(yte_raw, maj)
    Ptr = np.zeros((len(ytr), len(G))); Pte = np.zeros((len(yte), len(G)))
    print(f"=== {ds} len={L} n_tr={len(ytr)} n_te={len(yte)} horizons={G.astype(int).tolist()} ===")
    for j, T_ in enumerate(G.astype(int)):
        t0 = time.time()
        Ptr[:, j], Pte[:, j] = minirocket_posterior(Xtr, ytr, Xte, T_)
        print(f"  T={T_:4d}  ({time.time()-t0:.1f}s)  [elapsed {time.time()-t_start:.0f}s]")
    auc = [roc_auc_score(yte, Pte[:, j]) for j in range(len(G))]
    print(f"  MiniRocket posterior AUC range: {min(auc):.3f}-{max(auc):.3f} (vs weak-feature "
          f"posterior in Supp Table S2/etsc_ucr_fair.py, for reference)")

    lambdas = np.geomspace(1e-3, 5e-2, 8) / (L / 100.0)
    err_fix_te = np.array([((Pte[:, j] >= 0.5) != yte).mean() for j in range(len(G))])
    probas_tr = np.stack([1 - Ptr, Ptr], axis=-1); probas_te = np.stack([1 - Pte, Pte], axis=-1)
    kf_dummy = None  # ml_edm trigger.fit doesn't need fold objects here, matches etsc_baseline_comparison_ucr.py
    for lam in lambdas:
        J_fix_oracle = float((lam * G + err_fix_te).min())
        Jtr = [np.mean(lam * t + (d != ytr)) for t, d in (margin_rule(Ptr, G, m) for m in MS)]
        m_star = MS[int(np.argmin(Jtr))]
        t_, d_ = margin_rule(Pte, G, m_star); J_cm = float(np.mean(lam * t_ + (d_ != yte))); t_cm = float(t_.mean())
        ab = dp_band(Ptr, G, lam); t2, d2 = apply_band(Pte, G, ab)
        J_dp = float(np.mean(lam * t2 + (d2 != yte))); t_dp = float(t2.mean())
        for name, J, tm in (("const_margin(train-selected)", J_cm, t_cm), ("bayes_dp_band(train-kernel)", J_dp, t_dp)):
            rows.append((ds, name, round(float(lam), 6), round(tm, 3), round(J, 4),
                        round(J_fix_oracle, 4), round(J_fix_oracle - J, 4)))
        cm = CostMatrices(timestamps=G.astype(int), n_classes=2, misclf_cost=2.0,
                          delay_cost=(lambda t, L_=lam: 2 * L_ * t), alpha=0.5)
        for name, cls, kwargs in METHODS:
            try:
                mt, err = run_ext_method(cls, kwargs, cm, G.astype(int), probas_tr, ytr, probas_te, yte, None)
                J = lam * mt + err
                rows.append((ds, name, round(float(lam), 6), round(mt, 3), round(J, 4),
                            round(J_fix_oracle, 4), round(J_fix_oracle - J, 4)))
            except Exception as e:
                rows.append((ds, name, round(float(lam), 6), "", "", round(J_fix_oracle, 4), ""))
                print(f"    {name} FAILED at lambda={lam:.5f}: {e}")

    for name in ["const_margin(train-selected)", "bayes_dp_band(train-kernel)"] + [m[0] for m in METHODS]:
        g = [r[6] for r in rows if r[0] == ds and r[1] == name and r[6] != ""]
        if g:
            print(f"  {name:30s} wins {sum(x > TOL for x in g)}/{len(g)}  best_gain={max(g):+.4f}  "
                  f"mean_gain={np.mean(g):+.4f}")
        else:
            print(f"  {name:30s} FAILED on all lambdas")

with open(os.path.join(OUT, "table_etsc_ucr_strong.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["dataset", "method", "lambda", "mean_time", "J_method", "J_fixed_oracle", "gain_vs_fixed_oracle"])
    for r in rows: w.writerow(r)
print(f"\nwrote table_etsc_ucr_strong.csv  (total runtime {time.time()-t_start:.0f}s)")
