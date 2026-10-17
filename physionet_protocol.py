#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Phase 2 (generality): applies the SAME feedback-observation-control protocol
used for organoid screening (build_posterior.py's cross-fitted posterior +
build_controller.py's constant-margin stopping band vs fixed-time baseline) to
PhysioNet/CinC Challenge 2019 (early sepsis prediction from ICU vitals) -- a
structurally analogous but domain-different early-warning task, to test the
Discussion section's generality claim empirically for the first time.

Honest scope (documented, not hidden): predicts "ever septic during the
recorded stay" (patient-level binary outcome, mirroring the organoid's late
endpoint label) from the first 24h of 5 well-covered vitals (HR, O2Sat, SBP,
MAP, Resp; Temp/DBP dropped for poor coverage per physionet_census.py).
This is NOT a reproduction of the official Challenge utility function (which
credits/penalizes on a precise pre-onset time window) -- it is a test of
whether the SAME controller machinery transfers, using symmetric decision
costs for direct methodological parity with the organoid protocol.

Outputs:
  posterior/table_physionet_auc_by_horizon.csv
  posterior/table_physionet_matched_ops.csv
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import DATA_DIR  # $ORGANFUSION_DATA, or ./datasets (see README)
import os, glob, csv, sys, numpy as np
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

OUT = "posterior"; os.makedirs(OUT, exist_ok=True)
DATA_DIR = os.path.join(DATA_DIR, "physionet_challenge2019")
CORE = ["HR", "O2Sat", "SBP", "MAP", "Resp"]          # dropped Temp/DBP: poor coverage (census)
CUTOFF_H = 24                                          # require >=24h stay (census: retains ~77%)
GRID = [4, 6, 8, 10, 12, 16, 20, 24]                   # fine early, coarser late (mirrors organoid GRID shape)

files = sorted(glob.glob(os.path.join(DATA_DIR, "training_set*", "*.psv")))
print(f"files found: {len(files)}")

with open(files[0], encoding="utf-8") as f:
    header = f.readline().strip().split("|")
col = {c: i for i, c in enumerate(header)}
core_idx = [col[c] for c in CORE]
sep_idx = col["SepsisLabel"]; age_idx = col["Age"]; gender_idx = col["Gender"]

recs = []  # (t[T], vit[T,5], age, gender, y)
n_parsed = 0; n_kept = 0
for fp in files:
    try:
        with open(fp, encoding="utf-8") as f:
            lines = f.read().splitlines()[1:]
    except OSError:
        continue
    if len(lines) < CUTOFF_H:
        continue
    n_parsed += 1
    rows = [ln.split("|") for ln in lines]

    def fv(s):
        return np.nan if s == "NaN" else float(s)

    vit = np.array([[fv(r[i]) for i in core_idx] for r in rows], float)  # [T,5]
    y = int(any(r[sep_idx] == "1" for r in rows))
    age = fv(rows[0][age_idx]); gender = fv(rows[0][gender_idx])
    if not np.isfinite(age):
        age = 60.0  # rare missing-age fallback (population median-ish)
    t = np.arange(1, len(rows) + 1, dtype=float)
    # require each core vital have >=2 real readings within the first CUTOFF_H hours
    # (enough to interpolate a prefix trend, mirrors organoid's finite-count gate)
    ok = True
    for k in range(len(CORE)):
        if np.isfinite(vit[:CUTOFF_H, k]).sum() < 2:
            ok = False; break
    if not ok:
        continue
    recs.append((t, vit, age, gender, y))
    n_kept += 1

print(f"parsed (stay>={CUTOFF_H}h): {n_parsed}   kept (enough vital coverage): {n_kept}")
y = np.array([r[4] for r in recs])
print(f"sepsis prevalence: {y.mean():.4f} ({y.sum()}/{len(y)})")


def interp_vital(t, v, T):
    ok = np.isfinite(v)
    if ok.sum() == 0:
        return np.full(1, np.nan)
    return np.interp(T, t[ok], v[ok])


def feats(t, vit, T):
    """[last_value, mean_slope] per core vital over prefix 1..T, plus nothing else
    here -- age/gender appended separately as the condition (mirrors organoid's
    drug/conc condition)."""
    out = []
    g = np.linspace(1, T, 6)
    for k in range(len(CORE)):
        v = vit[:, k]
        w = np.array([interp_vital(t, v, tt) for tt in g])
        last = w[-1]
        slope = (w[-1] - w[0]) / max(T - 1, 1e-6)
        out += [last, slope]
    return out


N = len(recs)
folds = StratifiedKFold(5, shuffle=True, random_state=0)
demo = np.array([[r[2], r[3]] for r in recs], float)  # age, gender

P = np.zeros((N, len(GRID)))
rows_auc = []
for j, T in enumerate(GRID):
    X = np.column_stack([np.array([feats(t, vit, T) for (t, vit, _, _, _) in recs], float), demo])
    X = np.nan_to_num(X, nan=0.0)  # residual NaNs (vital never observed) -> 0 after scaling context
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    P[:, j] = cross_val_predict(clf, X, y, cv=folds, method="predict_proba")[:, 1]
    auc = roc_auc_score(y, P[:, j])
    rows_auc.append((T, auc))
    print(f"  T={T:>2}h  cross-fitted AUC={auc:.3f}")

with open(os.path.join(OUT, "table_physionet_auc_by_horizon.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f); w.writerow(["horizon_h", "auc"])
    for T, a in rows_auc:
        w.writerow([T, round(a, 5)])
print("wrote table_physionet_auc_by_horizon.csv")

np.savez(os.path.join(OUT, "physionet_posterior_data.npz"), P=P, y=y, GRID=np.array(GRID), demo=demo)

# ---------------------------------------------------------------------------
# SAME controller machinery as build_controller.py: constant-margin band vs
# fixed-time frontier, matched-error dominance count.
# ---------------------------------------------------------------------------
G = np.array(GRID, float)

# The paper's controller/fixed-time convention thresholds at p*=c_FP/(c_FP+c_FN),
# which is 0.5 under symmetric costs -- correct for the ~roughly-balanced organoid
# label, but DEGENERATE here: diagnostics show p_hat never exceeds ~0.31 at any
# horizon (median~0.054, tracking the 5.7% prevalence), so a 0.5 threshold never
# fires positive for anyone (fixed-time "error" = prevalence at every horizon,
# an empty/uninformative comparison). This reflects a real clinical asymmetry
# (missing sepsis >> one false alarm), not a bug: we set c_FN=9*c_FP up front,
# from the diagnostic requirement that p* must fall within the ACHIEVABLE
# p_hat range (p*=0.1 sits near the 97th percentile of p_hat -- a clinically
# plausible top-few-percent alerting rate), not tuned to the resulting dominance
# count. Both the fixed-time baseline and the controller use the SAME p*, so the
# comparison stays fair.
P_STAR = 0.10


def controller(m):
    """Returns (decision_time, decision, p_hat_at_decision) per patient."""
    times = np.empty(N); dec = np.empty(N, int); p_dec = np.empty(N)
    for i in range(N):
        ex = np.where(np.abs(P[i] - P_STAR) >= m)[0]
        j = ex[0] if len(ex) else len(G) - 1
        times[i] = G[j]; p_dec[i] = P[i, j]; dec[i] = int(P[i, j] >= P_STAR)
    return times, dec, p_dec


ft = G.copy()
fe = np.array([(np.array(P[:, j] >= P_STAR, int) != y).mean() for j in range(len(G))])
print(f"\n[p*={P_STAR}] fixed-time error by horizon:", [f"{e:.3f}" for e in fe])

# The organoid protocol's "matched decision-error, hours-saved" framing does not
# transfer cleanly here: with weak AUC (0.58-0.60) and 5.7% prevalence, raw 0/1
# decision error is prevalence-dominated and roughly FLAT (even non-monotonic)
# across horizons at p*=0.10 (checked: error is NOT monotonically decreasing in
# T), so there is no clean matched-error frontier to interpolate. This is an
# honest property of hard-threshold error under heavy class imbalance, not a
# bug -- so we switch to the controller's own NATIVE objective instead:
# J(policy) = E[lambda*tau + rho(p_hat_tau)] (Sec. sec:seq, Eq. rho), rho(p) =
# min(c_FP*(1-p), c_FN*p) with c_FP=1, c_FN=9 (matching P_STAR=0.10). This is
# MORE faithful to Theorem 1's actual objective than the matched-error proxy,
# and it is well-defined (a smooth function of lambda) even when decision error
# itself is uninformative.
c_FP, c_FN = 1.0, 9.0
rho = lambda p: np.minimum(c_FP * (1 - p), c_FN * p)

ms = np.linspace(0.0, 0.15, 49)  # scaled to the achievable p_hat range (diagnostics above)
lambdas = np.geomspace(1e-4, 2e-2, 25)

rows_cost = []
for lam in lambdas:
    # fixed-time: best single horizon at this lambda
    J_fixed_by_T = lam * G + np.array([rho(P[:, j]).mean() for j in range(len(G))])
    j_best = int(np.argmin(J_fixed_by_T))
    J_fixed = float(J_fixed_by_T[j_best]); T_fixed = float(G[j_best])
    # controller: best margin at this lambda
    J_ctrl_by_m = np.empty(len(ms)); t_ctrl_by_m = np.empty(len(ms))
    for i, m in enumerate(ms):
        tt, dd, pd = controller(m)
        J_ctrl_by_m[i] = (lam * tt + rho(pd)).mean(); t_ctrl_by_m[i] = tt.mean()
    i_best = int(np.argmin(J_ctrl_by_m))
    J_ctrl = float(J_ctrl_by_m[i_best]); t_ctrl = float(t_ctrl_by_m[i_best])
    gain = J_fixed - J_ctrl
    rows_cost.append((lam, J_fixed, T_fixed, J_ctrl, t_ctrl, gain))
    print(f"  lambda={lam:.5f}  fixed(J={J_fixed:.4f},T={T_fixed:.0f}h)  "
          f"ctrl(J={J_ctrl:.4f},meanT={t_ctrl:.1f}h)  gain={gain:+.5f}")

with open(os.path.join(OUT, "table_physionet_matched_ops.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["lambda", "J_fixed_best", "T_fixed_best", "J_controller_best", "mean_T_controller", "gain"])
    for r in rows_cost:
        w.writerow([round(x, 6) if isinstance(x, float) else x for x in r])
print("wrote table_physionet_matched_ops.csv")

n_dom = sum(1 for r in rows_cost if r[-1] > 0)
print(f"\nSUMMARY: adaptive controller achieves lower expected cost J than the best fixed-time policy at "
      f"{n_dom}/{len(rows_cost)} lambda values on PhysioNet 2019 early sepsis prediction "
      f"(N={N}, prevalence={y.mean():.3f}, c_FP=1, c_FN=9). "
      f"This directly tests the Discussion section's 'recurs in early sepsis warning' generality claim, "
      f"using the controller's native objective J=E[lambda*tau+rho(p_hat)] rather than the organoid "
      f"protocol's matched-decision-error proxy, which is uninformative under this task's class imbalance.")
