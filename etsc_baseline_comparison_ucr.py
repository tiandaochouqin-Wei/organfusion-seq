#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stage 3B, part 2: same TEASER/EconomyGamma/CALIMERA head-to-head as
etsc_baseline_comparison.py, run on the UCR benchmark datasets already used in
Supplementary Table S2 (ECG5000, Wafer, Earthquakes; FordA excluded there for
near-chance early AUC, included here anyway since a published trigger's own
sample-adaptive threshold might behave differently than our fixed-margin band on a
weak-signal dataset). Same design: all methods share ONE posterior (our own
cross-fitted logistic-regression prefix classifier, matching etsc_generality.py's
existing pipeline exactly), so the comparison isolates the stopping rule.

Cost structure: J = lambda*tau + rho(p_tau), rho symmetric with L=1, matching every
other experiment in the paper; lambda scaled by series length (matching
etsc_generality.py) so waiting-cost is comparable across datasets of very different
raw length.

Outputs:
  posterior/table_etsc_baselines_ucr.csv
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import DATA_DIR  # $ORGANFUSION_DATA, or ./datasets (see README)
import random
import os, sys, csv, time, warnings
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "_external", "ml_edm", "src"))
warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from ml_edm.cost_matrices import CostMatrices
from ml_edm.trigger import EconomyGamma, CALIMERA, TEASER

UCR = os.path.join(DATA_DIR, "ucr")
DATASETS = ["ECG5000", "Wafer", "Earthquakes", "FordA"]
OUT = "posterior"


def load(ds, split):
    path = os.path.join(UCR, ds, f"{ds}_{split}.txt")
    a = np.loadtxt(path)
    return a[:, 1:], a[:, 0]


def binarize(y_raw, maj=None):
    vals, cnts = np.unique(y_raw, return_counts=True)
    maj = vals[np.argmax(cnts)] if maj is None else maj
    return (y_raw != maj).astype(int), maj


def feats(X, T):
    P = X[:, :T]
    last = P[:, -1]; mean = P.mean(1)
    slope = (P[:, -1] - P[:, 0]) / max(T - 1, 1)
    mn = P.min(1); rng = P.max(1) - P.min(1)
    return np.column_stack([last, mean, slope, mn, rng])


class TriggerFailure(Exception):
    pass


def run_method(cls, kwargs, cm, GRID, probas_tr, y_tr, probas_te, y_te, folds_tr):
    """Fit trigger on out-of-fold TRAIN probas (avoids the trigger memorising
    train-only quirks), apply to TEST probas, report (mean_time, error) on TEST.
    Raises TriggerFailure (caught by the caller, reported as N/A, not a crash) if
    the third-party implementation fails on this dataset.

    Stage 3P correction: until then TEASER "failed" here on every dataset (self.best_v
    never set). That was OUR calling error, not a fragility of TEASER: ml_edm takes the
    full series length from X.shape[1], and the dummy X we passed had one column per
    HORIZON (8), so TEASER's earliness term 1 - mean(t)/max_length was negative and its
    harmonic mean undefined. X now has the true series length L."""
    T = len(GRID); Ntr = len(y_tr); Nte = len(y_te)
    trig = cls(timestamps=GRID, **kwargs)
    try:
        random.seed(0)
        trig.fit(np.zeros((Ntr, int(L))), probas_tr, y_tr, cm)
    except Exception as e:
        raise TriggerFailure(f"fit failed: {e}")
    triggered = np.zeros(Nte, bool)
    preds = np.full(Nte, -1, dtype=int); ts = np.full(Nte, -1.0)
    for t_idx, length in enumerate(GRID):
        p_arg = probas_te[:, :t_idx + 1, :] if trig.require_past_probas else probas_te[:, t_idx, :]
        X_now = np.zeros((Nte, int(length)))
        try:
            fire = np.asarray(trig.predict(X_now, p_arg, cm))
        except Exception as e:
            raise TriggerFailure(f"predict failed at t={length}: {e}")
        newly = fire & (~triggered)
        preds[newly] = probas_te[:, t_idx, :][newly].argmax(-1)
        ts[newly] = length
        triggered |= fire
        if triggered.all():
            break
    if not triggered.all():
        rem = ~triggered
        preds[rem] = probas_te[:, -1, :][rem].argmax(-1); ts[rem] = GRID[-1]
    return float(ts.mean()), float((preds != y_te).mean())


class TEASERCached(TEASER):
    """TEASER with its per-horizon one-class SVMs fitted once per posterior and reused across
    delay weights (they do not depend on the weight); the search for v is ml_edm's own."""
    _cache = {}

    def _fit_master_clf(self, probas, masks_pos_probas):
        key = (probas.tobytes(), masks_pos_probas.tobytes())
        if key not in TEASERCached._cache:
            TEASERCached._cache[key] = super()._fit_master_clf(probas, masks_pos_probas)
        return TEASERCached._cache[key]


METHODS = [
    ("EconomyGamma", EconomyGamma, dict(nb_intervals=5)),
    ("CALIMERA", CALIMERA, {}),
    ("TEASER", TEASERCached, dict(n_jobs=1)),
]

rows_out = []
t_start = time.time()
for ds in DATASETS:
    Xtr, ytr_raw = load(ds, "TRAIN")
    Xte, yte_raw = load(ds, "TEST")
    L = Xtr.shape[1]
    fracs = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    horizons = sorted(set(max(3, int(round(f * L))) for f in fracs))
    GRID = np.array(horizons, dtype=int)
    ytr, maj = binarize(ytr_raw)
    yte, _ = binarize(yte_raw, maj)
    if len(set(ytr)) < 2 or len(set(yte)) < 2:
        print(f"{ds}: degenerate binary split, skipped"); continue

    # cross-fitted TRAIN posterior (out-of-fold, for trigger fitting) + TEST posterior
    Ptr = np.zeros((len(Xtr), len(GRID))); Pte = np.zeros((len(Xte), len(GRID)))
    kf = StratifiedKFold(5, shuffle=True, random_state=0)
    for j, T_ in enumerate(GRID):
        Xf = feats(Xtr, T_)
        # out-of-fold train probas
        oof = np.zeros(len(Xtr))
        for tr_i, te_i in kf.split(Xf, ytr):
            clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
            clf.fit(Xf[tr_i], ytr[tr_i])
            oof[te_i] = clf.predict_proba(Xf[te_i])[:, 1]
        Ptr[:, j] = oof
        clf_full = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(Xf, ytr)
        Pte[:, j] = clf_full.predict_proba(feats(Xte, T_))[:, 1]
    probas_tr = np.stack([1 - Ptr, Ptr], axis=-1)
    probas_te = np.stack([1 - Pte, Pte], axis=-1)

    lambdas = np.geomspace(1e-3, 5e-2, 8) / (L / 100.0)
    print(f"=== {ds}  n_train={len(Xtr)} n_test={len(Xte)} len={L} horizons={GRID.tolist()} ===")
    for name, cls, kwargs in METHODS:
        best_J = {}
        failed = False
        for lam in lambdas:
            cm = CostMatrices(timestamps=GRID, n_classes=2, misclf_cost=2.0,
                               delay_cost=(lambda t, L_=lam: 2 * L_ * t), alpha=0.5)
            try:
                mt, err = run_method(cls, kwargs, cm, GRID, probas_tr, ytr, probas_te, yte, kf)
            except TriggerFailure as e:
                print(f"  {name:14s} FAILED at lambda={lam:.5f}: {e}")
                rows_out.append((ds, name, lam, "", "", "", "", ""))
                failed = True
                continue
            best_J[round(float(lam), 6)] = (mt, err, lam * mt + err)
        # also compute the best fixed-length policy's J at the same lambdas, for a
        # direct per-dataset win/loss tally against OUR controller AND this baseline
        for lam, (mt, err, J) in best_J.items():
            Jf_by_h = []
            for j, h in enumerate(GRID):
                pred_fixed = (Pte[:, j] >= 0.5).astype(int)
                err_fixed = (pred_fixed != yte).mean()
                Jf_by_h.append(lam * h + err_fixed)
            J_fix = float(min(Jf_by_h))
            rows_out.append((ds, name, lam, round(mt, 3), round(err, 4), round(J, 4),
                              round(J_fix, 4), round(J_fix - J, 4)))
        gains = [r[7] for r in rows_out if r[0] == ds and r[1] == name and r[7] != ""]
        if gains:
            print(f"  {name:14s} wins {sum(1 for g in gains if g>0)}/{len(gains)} lambda  "
                  f"best_gain={max(gains):+.4f}  ({time.time()-t_start:.0f}s elapsed)"
                  + ("  [PARTIAL: some lambdas failed]" if failed else ""))
        else:
            print(f"  {name:14s} FAILED on all {len(lambdas)} lambdas for {ds}  "
                  f"({time.time()-t_start:.0f}s elapsed)")

with open(os.path.join(OUT, "table_etsc_baselines_ucr.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["dataset", "method", "lambda", "mean_time", "decision_error", "J_method", "J_fixed_time", "gain_vs_fixed"])
    for r in rows_out: w.writerow(r)
print("\nwrote table_etsc_baselines_ucr.csv")
print(f"total runtime {time.time()-t_start:.0f}s")
