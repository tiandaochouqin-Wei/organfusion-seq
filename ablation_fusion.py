#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""(c1) The fusion ablation, run properly.

The design: compare last-step feature / mean pooling / max pooling /
unconstrained temporal attention / simplex temporal attention, each also with
temperature scaling, plus isotonic and beta calibration -- and report AUC, Brier,
ECE, MCE, calibration slope, calibration intercept, decision error, mean stopping
time and total control cost for each. Decision rule:
"if Simplex only improves ECE but does not improve control cost, list it as a
secondary module rather than a core innovation."

Every variant shares one encoder implementation and differs ONLY in the causal
prefix pooling (producers.causal_pool), so the comparison isolates the fusion.
All pooling is causal: g_t aggregates {h_1..h_t} only.

Protocol: leave-condition-out GroupKFold for the posterior; calibrators are fit on
an inner calibration split carved out of the training conditions (never on the fold
being scored), matching the "do not tune calibration on test" requirement.

Outputs:
  posterior/table_fusion_ablation.csv
"""
import os, csv, sys, numpy as np
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
from sklearn.model_selection import GroupKFold
from producers import (load_data, fit_producer, logits_of, full_metrics,
                       CALIBRATORS, set_seed)

torch.use_deterministic_algorithms(True)
OUT = "posterior"
D = load_data()
y = D["y"]; GRID = D["GRID"]; N, T = D["N"], D["T"]; groups = D["groups"]

POOLS = ["last", "mean", "max", "attn_unconstrained", "attn_simplex"]
# plain-attention+temperature and simplex+temperature are included explicitly;
# we run every pooling x every calibrator so the claim cannot be cherry-picked.
CALS = ["none", "temperature", "isotonic", "beta"]

rng = np.random.default_rng(0)
conds = np.array(sorted(set(groups.tolist())))

rows = []
for pool in POOLS:
    # out-of-fold posteriors, with an inner cal split inside each training fold
    P_raw = np.zeros((N, T))
    P_cal = {c: np.zeros((N, T)) for c in CALS}
    for f, (tr_all, te) in enumerate(GroupKFold(5).split(D["SEQ"], y, groups), 1):
        tr_conds = np.array(sorted(set(groups[tr_all].tolist())))
        cal_conds = rng.choice(tr_conds, size=max(2, len(tr_conds) // 4), replace=False)
        inner_tr = tr_all[~np.isin(groups[tr_all], cal_conds)]
        inner_cal = tr_all[np.isin(groups[tr_all], cal_conds)]
        m = fit_producer(D, inner_tr, enc="GRU", pool=pool, seed=0)
        lo_cal = logits_of(m, D, inner_cal); lo_te = logits_of(m, D, te)
        P_raw[te] = 1 / (1 + np.exp(-lo_te))
        for c in CALS:
            fn, _ = CALIBRATORS[c](lo_cal, y[inner_cal])
            P_cal[c][te] = fn(lo_te)
    for c in CALS:
        mt = full_metrics(P_cal[c], y, GRID)
        rows.append((pool, c, mt["auc"], mt["brier"], mt["ece"], mt["mce"], mt["slope"],
                     mt["intercept"], mt["dec_err"], mt["mean_t"], mt["control_cost"]))
        print(f"  {pool:<20} {c:<12} AUC={mt['auc']:.3f} Brier={mt['brier']:.3f} "
              f"ECE={mt['ece']:.3f} MCE={mt['mce']:.3f} slope={mt['slope']:.2f} "
              f"int={mt['intercept']:+.2f} err={mt['dec_err']:.3f} t={mt['mean_t']:.1f}h "
              f"J={mt['control_cost']:.4f}")

with open(os.path.join(OUT, "table_fusion_ablation.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["pooling", "calibrator", "AUC", "Brier", "ECE", "MCE", "cal_slope",
                "cal_intercept", "decision_error", "mean_stop_time_h", "control_cost_J"])
    for r in rows:
        w.writerow([r[0], r[1]] + [round(float(x), 5) for x in r[2:]])

# ---- the decision rule, applied ----
print("\n=== decision rule: does simplex improve CONTROL COST, or only ECE? ===")
def get(pool, cal, key):
    i = {"ECE": 4, "J": 10}[key]
    return [r[i] for r in rows if r[0] == pool and r[1] == cal][0]
for cal in CALS:
    s_e, u_e = get("attn_simplex", cal, "ECE"), get("attn_unconstrained", cal, "ECE")
    s_j, u_j = get("attn_simplex", cal, "J"), get("attn_unconstrained", cal, "J")
    print(f"  cal={cal:<12} dECE(simplex-unconstr)={s_e-u_e:+.4f}   dJ={s_j-u_j:+.4f}  "
          f"({'simplex better on J' if s_j < u_j else 'simplex NOT better on J'})")
best = min(rows, key=lambda r: r[10])
print(f"\nlowest control cost overall: pooling={best[0]}, calibrator={best[1]}, J={best[10]:.4f}")
print("wrote table_fusion_ablation.csv")
