#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stage 3P: the UCR audit on a benchmark that is not hand-picked.

etsc_ucr_strong.py scored four UCR datasets chosen for having a plausible early-decision cost.
A reviewer cannot tell four chosen datasets from four favourable ones, so this script reruns
the IDENTICAL protocol on every two-class, equal-length, complete dataset of the UCR 2018
archive (40 datasets; the rule and the list are in download_ucr_binary.py and were fixed before
any result was seen). Wafer, Earthquakes and FordA are in both sets and must reproduce
table_etsc_ucr_strong.csv exactly -- checked at the end.

Protocol (unchanged): per-horizon MiniRocket (1,000 kernels) + logistic regression; horizons at
fractions 0.2..0.9 of the series length; every free parameter of every rule (our margin, the
Bellman band's kernel, the published triggers) is fitted on 5-fold out-of-fold TRAIN posteriors
and scored once on TEST by realised cost  lambda*tau + 1{wrong};  eight delay weights
geomspace(1e-3, 5e-2, 8) / (L/100).

Added here:
  * a fixed horizon chosen on TRAIN the same way (the like-for-like competitor used for ranks),
    next to the TEST-oracle fixed horizon of the earlier table (kept for continuity);
  * a second, recalibrated posterior (per-horizon Platt scaling fitted on the out-of-fold TRAIN
    scores -- line 1 of Algorithm 1, which the earlier UCR run skipped), as a sensitivity run;
  * the per-dataset posteriors are saved (posterior/ucr40/<name>.npz) so that the certificate
    check (ucr40_analysis.py) does not have to refit anything.

Outputs: posterior/table_etsc_ucr_binary40.csv, posterior/ucr40/*.npz
Run with the miniconda interpreter (aeon, ml_edm):  python -I etsc_ucr_binary40.py [workers]
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import DATA_DIR  # $ORGANFUSION_DATA, or ./datasets (see README)
import os, sys, csv, time, warnings
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "_external", "ml_edm", "src"))
sys.path.insert(0, HERE)
os.environ.setdefault("NUMBA_NUM_THREADS", "2")
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
import numpy as np

from download_ucr_binary import BINARY_UCR

UCR = os.path.join(DATA_DIR, "ucr")
OUT = os.path.join(HERE, "posterior")
NPZ = os.path.join(OUT, "ucr40")
TOL = 1e-3
N_KERNELS = 1000
FRACS = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
MS = np.linspace(0.0, 0.49, 50)
Bbin = np.linspace(0, 1, 51); bc = (Bbin[:-1] + Bbin[1:]) / 2; nb = len(bc)
rho_bc = np.minimum(1 - bc, bc)


def load(ds, split):
    a = np.loadtxt(os.path.join(UCR, ds, f"{ds}_{split}.txt")); return a[:, 1:], a[:, 0]


def binarize(y_raw, maj=None):
    vals, cnts = np.unique(y_raw, return_counts=True)
    maj = vals[np.argmax(cnts)] if maj is None else maj
    return (y_raw != maj).astype(int), maj


def minirocket_posterior(Xtr, ytr, Xte, T):
    """Identical to etsc_ucr_strong.minirocket_posterior, except that the number of folds drops
    below 5 when a class has fewer than 5 training series (never the case for the earlier four)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from aeon.transformations.collection.convolution_based import MiniRocket
    Ptr_raw = Xtr[:, :T].astype(np.float32)[:, None, :]
    Pte_raw = Xte[:, :T].astype(np.float32)[:, None, :]
    mr = MiniRocket(n_kernels=N_KERNELS, random_state=0)
    Ztr = mr.fit_transform(Ptr_raw)
    Zte = mr.transform(Pte_raw)
    mu = Ztr.mean(0); sd = Ztr.std(0) + 1e-8
    Ztr = (Ztr - mu) / sd; Zte = (Zte - mu) / sd
    k = int(min(5, np.bincount(ytr).min()))
    kf = StratifiedKFold(k, shuffle=True, random_state=0)
    oof = np.zeros(len(ytr))
    for tr_i, te_i in kf.split(Ztr, ytr):
        clf = LogisticRegression(max_iter=2000, C=1.0)
        clf.fit(Ztr[tr_i], ytr[tr_i])
        oof[te_i] = clf.predict_proba(Ztr[te_i])[:, 1]
    clf_full = LogisticRegression(max_iter=2000, C=1.0).fit(Ztr, ytr)
    return oof, clf_full.predict_proba(Zte)[:, 1]


def platt(ptr, ytr, pte):
    """Two-parameter recalibration of one horizon, fitted on the out-of-fold TRAIN scores."""
    from sklearn.linear_model import LogisticRegression
    lg = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
    lr = LogisticRegression(C=1e4, max_iter=1000).fit(lg(ptr)[:, None], ytr)
    return lr.predict_proba(lg(ptr)[:, None])[:, 1], lr.predict_proba(lg(pte)[:, None])[:, 1]


def margin_rule(P, G, m):
    ex = np.abs(P - 0.5) >= m
    j = np.where(ex.any(1), ex.argmax(1), P.shape[1] - 1)
    return G[j], (P[np.arange(len(P)), j] >= 0.5).astype(int)


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


def fit_trigger(cls, kwargs, cm, GRID, L, probas_tr, y_tr):
    """ml_edm reads the full series length from the width of X (TriggerModel.fit sets
    max_length = X.shape[1]); only TEASER uses it, for its earliness term. The earlier scripts
    passed a dummy X with one column per HORIZON, which made TEASER's earliness negative, its
    harmonic mean undefined and its fit fail -- our calling error, not TEASER's (Stage 3P)."""
    trig = cls(timestamps=GRID, **kwargs)
    trig.fit(np.zeros((len(y_tr), int(L))), probas_tr, y_tr, cm)
    return trig


def predict_trigger(trig, cm, GRID, probas_te, y_te):
    Nte = len(y_te)
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


def make_teaser_cached():
    """TEASER whose per-horizon one-class SVMs are fitted once per posterior and reused across
    delay weights and objectives (they depend on neither); the search for v is ml_edm's own."""
    from ml_edm.trigger import TEASER

    class TEASERCached(TEASER):
        def _fit_master_clf(self, probas, masks_pos_probas):
            key = (probas.tobytes(), masks_pos_probas.tobytes())
            if key not in self._cache:
                self._cache[key] = super()._fit_master_clf(probas, masks_pos_probas)
            return self._cache[key]
    return TEASERCached


def score_posterior(ds, tag, Ptr, Pte, ytr, yte, G, L):
    """Every rule on one (train out-of-fold, test) posterior pair; returns CSV rows."""
    import random
    from ml_edm.cost_matrices import CostMatrices
    from ml_edm.trigger import EconomyGamma, CALIMERA
    TEASERCached = make_teaser_cached()
    methods = [("EconomyGamma", EconomyGamma, dict(nb_intervals=5)), ("CALIMERA", CALIMERA, {})]
    rows = []
    GI = G.astype(int)
    lambdas = np.geomspace(1e-3, 5e-2, 8) / (L / 100.0)
    err_fix_te = np.array([((Pte[:, j] >= 0.5) != yte).mean() for j in range(len(G))])
    err_fix_tr = np.array([((Ptr[:, j] >= 0.5) != ytr).mean() for j in range(len(G))])
    probas_tr = np.stack([1 - Ptr, Ptr], axis=-1); probas_te = np.stack([1 - Pte, Pte], axis=-1)
    svm_cache = {}; teaser_eval = {}

    def teaser(objective, cm):
        """(time, error, v) of TEASER with ml_edm's 'hmean' (as published: harmonic mean of
        accuracy and earliness, blind to the delay weight) or cost objective."""
        random.seed(0)
        trig = TEASERCached(timestamps=GI, objective=objective, n_jobs=1); trig._cache = svm_cache
        trig.fit(np.zeros((len(ytr), int(L))), probas_tr, ytr, cm)
        if trig.best_v not in teaser_eval:
            teaser_eval[trig.best_v] = predict_trigger(trig, cm, GI, probas_te, yte)
        return teaser_eval[trig.best_v]

    for lam in lambdas:
        J_orc = float((lam * G + err_fix_te).min())

        def add(name, tm, err):
            J = lam * tm + err
            rows.append((ds, tag, name, round(float(lam), 6), round(tm, 3), round(err, 5), round(J, 5),
                         round(J_orc, 5), round(J_orc - J, 5)))

        def fail(name, e):
            rows.append((ds, tag, name, round(float(lam), 6), "", "", "", round(J_orc, 5), ""))
            if lam == lambdas[0]:
                print(f"    [{ds}/{tag}] {name} failed: {type(e).__name__}: {str(e)[:110]}", flush=True)
        jf = int(np.argmin(lam * G + err_fix_tr))            # fixed horizon chosen on TRAIN
        add("fixed(train-selected)", float(G[jf]), float(err_fix_te[jf]))
        Jtr = [np.mean(lam * t + (d != ytr)) for t, d in (margin_rule(Ptr, G, m) for m in MS)]
        t_, d_ = margin_rule(Pte, G, MS[int(np.argmin(Jtr))])
        add("const_margin(train-selected)", float(t_.mean()), float((d_ != yte).mean()))
        t2, d2 = apply_band(Pte, G, dp_band(Ptr, G, lam))
        add("bayes_dp_band(train-kernel)", float(t2.mean()), float((d2 != yte).mean()))
        cm = CostMatrices(timestamps=GI, n_classes=2, misclf_cost=2.0,
                          delay_cost=(lambda t, L_=lam: 2 * L_ * t), alpha=0.5)
        for name, cls, kwargs in methods:
            try:
                trig = fit_trigger(cls, kwargs, cm, GI, L, probas_tr, ytr)
                add(name, *predict_trigger(trig, cm, GI, probas_te, yte))
            except Exception as e:
                fail(name, e)
        for name, objective in (("TEASER", "hmean"), ("TEASER-cost", "avg_cost")):
            try:
                add(name, *teaser(objective, cm))
            except Exception as e:
                fail(name, e)
    return rows


def run_dataset(ds):
    from sklearn.metrics import roc_auc_score
    t0 = time.time()
    Xtr, ytr_raw = load(ds, "TRAIN"); Xte, yte_raw = load(ds, "TEST")
    L = Xtr.shape[1]
    G = np.array(sorted(set(max(9, int(round(f * L))) for f in FRACS)), float)
    ytr, maj = binarize(ytr_raw); yte, _ = binarize(yte_raw, maj)
    cache = os.path.join(NPZ, ds + ".npz")
    if os.path.isfile(cache):
        z = np.load(cache); Ptr, Pte = z["Ptr"], z["Pte"]
    else:
        Ptr = np.zeros((len(ytr), len(G))); Pte = np.zeros((len(yte), len(G)))
        for j, T_ in enumerate(G.astype(int)):
            Ptr[:, j], Pte[:, j] = minirocket_posterior(Xtr, ytr, Xte, T_)
        np.savez_compressed(cache, Ptr=Ptr, Pte=Pte, ytr=ytr, yte=yte, G=G, L=L)
    Ctr = np.zeros_like(Ptr); Cte = np.zeros_like(Pte)
    for j in range(len(G)):
        Ctr[:, j], Cte[:, j] = platt(Ptr[:, j], ytr, Pte[:, j])
    np.savez_compressed(os.path.join(NPZ, ds + "_platt.npz"), Ptr=Ctr, Pte=Cte, ytr=ytr, yte=yte, G=G, L=L)
    auc = [roc_auc_score(yte, Pte[:, j]) if len(set(yte)) > 1 else np.nan for j in range(len(G))]
    rows = score_posterior(ds, "raw", Ptr, Pte, ytr, yte, G, L)
    rows += score_posterior(ds, "platt", Ctr, Cte, ytr, yte, G, L)
    info = (ds, L, len(ytr), len(yte), len(G), round(float(np.nanmin(auc)), 3), round(float(np.nanmax(auc)), 3),
            round(time.time() - t0, 1))
    print(f"  done {ds:30s} L={L:5d} n_tr={len(ytr):5d} n_te={len(yte):5d} AUC {info[5]:.3f}-{info[6]:.3f} "
          f"({info[7]:.0f}s)", flush=True)
    return rows, info


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    from concurrent.futures import ProcessPoolExecutor
    os.makedirs(NPZ, exist_ok=True)
    args = [a for a in sys.argv[1:]]
    workers = int(args[0]) if args and args[0].isdigit() else 8
    names = [a for a in args if not a.isdigit()] or BINARY_UCR
    t_start = time.time(); rows = []; infos = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for r, info in ex.map(run_dataset, names):
            rows += r; infos.append(info)
    if names == BINARY_UCR:
        f_rows, f_info = "table_etsc_ucr_binary40.csv", "table_ucr40_datasets.csv"
    elif names == ["ECG5000"]:                              # the multi-class set of Table 7 (majority vs rest)
        f_rows, f_info = "table_etsc_ucr_ecg5000.csv", "table_ucr_ecg5000_dataset.csv"
    else:
        f_rows, f_info = "table_etsc_ucr_binary40_partial.csv", "table_ucr40_datasets_partial.csv"
    with open(os.path.join(OUT, f_rows), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "posterior", "method", "lambda", "mean_time", "error", "J_method",
                    "J_fixed_oracle", "gain_vs_fixed_oracle"])
        for r in rows: w.writerow(r)
    with open(os.path.join(OUT, f_info), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "length", "n_train", "n_test", "n_horizons", "auc_min", "auc_max", "seconds"])
        for i in infos: w.writerow(i)
    print(f"\nwrote {f_rows} ({len(rows)} rows, {len(infos)} datasets, "
          f"{time.time() - t_start:.0f}s)")

    # ---- continuity check: the three datasets shared with etsc_ucr_strong.py must reproduce ----
    old = os.path.join(OUT, "table_etsc_ucr_strong.csv")
    if os.path.isfile(old):
        import csv as _c
        o = {(r["dataset"], r["method"], r["lambda"]): r["J_method"] for r in _c.DictReader(open(old, encoding="utf-8"))}
        n = bad = 0
        for r in rows:
            k = (r[0], r[2], f"{r[3]}")
            if r[1] == "raw" and k in o and o[k] != "" and r[6] != "":
                n += 1; bad += abs(float(o[k]) - float(r[6])) > 6e-5
        print(f"continuity with table_etsc_ucr_strong.csv: {n} shared (dataset, method, lambda) cells, "
              f"{bad} differ by more than rounding")
