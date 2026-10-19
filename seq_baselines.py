#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Phase 2 (sequential baselines): compares the DP-derived time-varying band
(the realization of Theorem 1) and the deployed constant-margin controller
(the headline "ours" in Table 3) against three additional stopping baselines,
all evaluated on the SAME cross-fitted posterior process P[N,G]. Main protocol
(2026-09-28): P is the leave-condition-out posterior (posterior_data_grouped.npz's
P_group), not the organoid-level stratified-CV posterior -- the latter is an
upper-bound reference only (Supplementary Table S12(a)), so differences here
reflect the STOPPING RULE only, under the protocol the paper actually deploys:

  1. fixed-time            (no adaptivity; existing baseline)
  2. constant-margin band  (deployed "ours", build_controller.py)
  3. SPRT-on-posterior-odds (Wald boundaries applied to logit(p_t); an
                              approximation since p_t is a posterior, not a
                              raw per-increment likelihood ratio -- documented)
  4. myopic one-step VOI   (greedy: stop unless ONE more step's expected risk
                              reduction beats its cost; ignores further options)
  5. Bayes-DP time-varying band (backward induction; the actual optimum of
                              Theorem 1; F2_band.py's kernel machinery)

Then a REVERSE experiment: fix the decision procedure (constant-margin AND
DP-band) and swap the posterior PRODUCER (LR / RF / HGB), to separate the
controller's contribution from the estimator's.

Outputs:
  posterior/table_seq_baselines_frontier.csv   all 5 policies' full frontiers
  posterior/table_seq_baselines_matched.csv    mean-time at 9 matched target errors
  posterior/table_estimator_swap.csv           hours-saved under LR/RF/HGB x {const-margin, DP-band}
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import DATA_DIR  # $ORGANFUSION_DATA, or ./datasets (see README)
import os, csv, sys, numpy as np
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

OUT = "posterior"
os.makedirs(OUT, exist_ok=True)

d = np.load("posterior/posterior_data_grouped.npz", allow_pickle=True)
P = d["P_group"]; y = d["y"]; G = d["GRID"].astype(float); N = len(y)
GROUPS = d["groups"]
nG = len(G)
rho = lambda p: np.minimum(1 - p, p)  # symmetric costs c_FP=c_FN=1, as used throughout


def isotonic_lower_envelope(t, e):
    """Sort by time, then enforce error is the running MINIMUM seen so far at
    or below that time (the achievable lower envelope), for clean interpolation.
    This mirrors the honest 'best achievable at this budget' reading used for
    fixed-time frontiers elsewhere in the codebase."""
    o = np.argsort(t)
    t = np.asarray(t)[o]; e = np.asarray(e)[o]
    e_env = np.minimum.accumulate(e)
    return t, e_env


def frontier_time_at_error(t, e, target_errs):
    """Minimal mean-time to reach each target error, via the lower envelope.
    Returns np.nan (NOT a silently clamped/extrapolated value) for any target
    error the policy's envelope never reaches -- np.interp would otherwise
    silently flatten to its boundary value outside [min(e2), max(e2)], which
    is exactly the kind of overclaim to avoid."""
    t2, e2 = isotonic_lower_envelope(t, e)
    lo, hi = e2.min(), e2.max()
    out = []
    for te in target_errs:
        if te < lo or te > hi:
            out.append(np.nan)
        else:
            out.append(float(np.interp(-te, -e2, t2)))
    return out


# ---------------------------------------------------------------------------
# Policy 1: fixed-time (baseline, no adaptivity)
# ---------------------------------------------------------------------------
def policy_fixed():
    ts, es = [], []
    for j in range(nG):
        dec = (P[:, j] >= 0.5).astype(int)
        ts.append(G[j]); es.append((dec != y).mean())
    return np.array(ts), np.array(es)


# ---------------------------------------------------------------------------
# Policy 2: constant-margin band (deployed "ours" in the main text)
# ---------------------------------------------------------------------------
def controller_const_margin(Pmat, m):
    times = np.empty(N); dec = np.empty(N, int)
    for i in range(N):
        ex = np.where(np.abs(Pmat[i] - 0.5) >= m)[0]
        j = ex[0] if len(ex) else nG - 1
        times[i] = G[j]; dec[i] = int(Pmat[i, j] >= 0.5)
    return times, dec


def policy_const_margin(Pmat=None):
    Pmat = P if Pmat is None else Pmat
    ts, es = [], []
    for m in np.linspace(0.0, 0.48, 49):
        tt, dd = controller_const_margin(Pmat, m)
        ts.append(tt.mean()); es.append((dd != y).mean())
    return np.array(ts), np.array(es)


# ---------------------------------------------------------------------------
# Policy 3: SPRT on posterior log-odds (Wald boundaries; approximation --
# p_t is a posterior probability, not a raw per-increment likelihood ratio,
# so this is "SPRT-style thresholding of accumulated evidence", a standard
# practical stand-in when only p_t is observed).
# ---------------------------------------------------------------------------
def controller_sprt(Pmat, eps):
    eps = min(max(eps, 1e-6), 0.499)
    A = np.log((1 - eps) / eps)   # upper boundary on logit(p)
    B = np.log(eps / (1 - eps))   # lower boundary on logit(p)
    Pc = np.clip(Pmat, 1e-6, 1 - 1e-6)
    logit = np.log(Pc / (1 - Pc))
    times = np.empty(N); dec = np.empty(N, int)
    for i in range(N):
        hi = np.where(logit[i] >= A)[0]
        lo = np.where(logit[i] <= B)[0]
        cand = sorted(list(hi) + list(lo))
        j = cand[0] if cand else nG - 1
        times[i] = G[j]; dec[i] = int(Pmat[i, j] >= 0.5)
    return times, dec


def policy_sprt(Pmat=None):
    Pmat = P if Pmat is None else Pmat
    ts, es = [], []
    for eps in np.concatenate([np.linspace(0.02, 0.49, 40)]):
        tt, dd = controller_sprt(Pmat, eps)
        ts.append(tt.mean()); es.append((dd != y).mean())
    return np.array(ts), np.array(es)


# ---------------------------------------------------------------------------
# Shared: binned Markov transition kernel between consecutive horizons
# (identical construction to F2_band.py), used by both the myopic-VOI and
# the full Bayes-DP policies.
# ---------------------------------------------------------------------------
Bbin = np.linspace(0, 1, 51); bc = (Bbin[:-1] + Bbin[1:]) / 2; nb = len(bc)
rho_bc = np.minimum(1 - bc, bc)


def build_kernel(Pmat):
    def kernel(pj, pj1):
        K = np.zeros((nb, nb))
        bi = np.clip(np.digitize(pj, Bbin) - 1, 0, nb - 1)
        bj = np.clip(np.digitize(pj1, Bbin) - 1, 0, nb - 1)
        for a, b in zip(bi, bj):
            K[a, b] += 1
        K += 1e-3
        K /= K.sum(1, keepdims=True)
        return K
    return [kernel(Pmat[:, j], Pmat[:, j + 1]) for j in range(nG - 1)]


# ---------------------------------------------------------------------------
# Policy 4: myopic one-step VOI -- stop unless ONE more step's expected risk
# reduction beats its cost; ignores option value beyond the immediate step
# (weaker than full DP, which is the point of the comparison).
# ---------------------------------------------------------------------------
def controller_myopic(Pmat, Ks, lam):
    times = np.full(N, G[-1]); dec = np.empty(N, int)
    decided = np.zeros(N, bool)
    for j in range(nG - 1):
        p = Pmat[:, j]
        bidx = np.clip(np.digitize(p, Bbin) - 1, 0, nb - 1)
        Ej = (Ks[j][bidx] * rho_bc[None, :]).sum(1)          # E[rho(p_{j+1}) | p_j]
        c1 = lam * (G[j + 1] - G[j]) + Ej
        stop_now = (rho(p) <= c1) & (~decided)
        idx = np.where(stop_now)[0]
        times[idx] = G[j]; dec[idx] = (p[idx] >= 0.5).astype(int); decided[idx] = True
    idx = np.where(~decided)[0]
    times[idx] = G[-1]; dec[idx] = (Pmat[idx, -1] >= 0.5).astype(int)
    return times, dec


def policy_myopic(Pmat=None, Ks=None):
    Pmat = P if Pmat is None else Pmat
    Ks = build_kernel(Pmat) if Ks is None else Ks
    ts, es = [], []
    for lam in np.logspace(-4, -0.7, 30):
        tt, dd = controller_myopic(Pmat, Ks, lam)
        ts.append(tt.mean()); es.append((dd != y).mean())
    return np.array(ts), np.array(es)


# ---------------------------------------------------------------------------
# Policy 5: Bayes-DP time-varying band -- the actual optimum of Theorem 1
# (backward induction on the binned value function; F2_band.py's machinery),
# then FORWARD-SIMULATED on each organoid's real posterior path.
# ---------------------------------------------------------------------------
def dp_band_boundaries(Ks, lam):
    V = rho_bc.copy()
    ab = np.full((nG, 2), 0.5)  # default: empty band -> immediate decision
    for j in range(nG - 2, -1, -1):
        cont = lam * (G[j + 1] - G[j]) + Ks[j] @ V
        cs = np.where(cont < rho_bc - 1e-9)[0]
        if len(cs):
            ab[j] = [bc[cs.min()], bc[cs.max()]]
        V = np.minimum(rho_bc, cont)
    return ab  # ab[j] = (a_j, b_j); ab[-1] left at (0.5,0.5) i.e. decide at horizon


def controller_dp(Pmat, Ks, lam):
    ab = dp_band_boundaries(Ks, lam)
    times = np.empty(N); dec = np.empty(N, int)
    for i in range(N):
        j_stop = nG - 1
        for j in range(nG - 1):
            a, b = ab[j]
            if not (a < Pmat[i, j] < b):
                j_stop = j; break
        times[i] = G[j_stop]; dec[i] = int(Pmat[i, j_stop] >= 0.5)
    return times, dec


def policy_dp(Pmat=None, Ks=None):
    Pmat = P if Pmat is None else Pmat
    Ks = build_kernel(Pmat) if Ks is None else Ks
    ts, es = [], []
    for lam in np.logspace(-4, -0.7, 30):
        tt, dd = controller_dp(Pmat, Ks, lam)
        ts.append(tt.mean()); es.append((dd != y).mean())
    return np.array(ts), np.array(es)


# ---------------------------------------------------------------------------
# Run all 5 on the primary (stratified-CV) posterior; save frontiers.
# ---------------------------------------------------------------------------
print("computing frontiers (fixed / const-margin / SPRT / myopic-VOI / Bayes-DP) ...")
Ks_primary = build_kernel(P)
frontiers = {
    "fixed_time": policy_fixed(),
    "const_margin": policy_const_margin(),
    "sprt_posterior_odds": policy_sprt(),
    "myopic_voi": policy_myopic(P, Ks_primary),
    "bayes_dp_band": policy_dp(P, Ks_primary),
}
with open(os.path.join(OUT, "table_seq_baselines_frontier.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f); w.writerow(["policy", "mean_time_h", "decision_error"])
    for name, (t, e) in frontiers.items():
        for tt, ee in zip(t, e):
            w.writerow([name, round(float(tt), 3), round(float(ee), 4)])
print("wrote table_seq_baselines_frontier.csv")

target_errs = np.linspace(0.12, 0.24, 9)
rows = []
for name, (t, e) in frontiers.items():
    times_at = frontier_time_at_error(t, e, target_errs)
    t2, e2 = isotonic_lower_envelope(t, e)
    rows.append((name, times_at, float(e2.min())))
    fmt = lambda x: "  n/a" if np.isnan(x) else "%5.1f" % x
    print(f"  {name:20s} min_err={e2.min():.4f}  time@err:", " ".join(fmt(x) for x in times_at))

with open(os.path.join(OUT, "table_seq_baselines_matched.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    # Stage 3G fix: the grid is linspace(0.12, 0.24, 9) = 0.120, 0.135, ..., 0.225, 0.240; the old
    # ".2f" header labelled 0.165 as "0.16" and 0.195 as "0.20", and Table 4 copied those columns next
    # to rows computed at the exact 0.16/0.20. Headers now carry three decimals, and the exact-target
    # table below is the one Table 4 reads.
    w.writerow(["policy", "min_achievable_err"] + [f"time_at_err_{e:.3f}" for e in target_errs])
    for name, times_at, min_err in rows:
        w.writerow([name, round(min_err, 4)] + [("" if np.isnan(x) else round(x, 2)) for x in times_at])
print("wrote table_seq_baselines_matched.csv")
TABLE3_TARGETS = [0.12, 0.16, 0.20, 0.24]   # the columns of Table 4 / Supplementary Table S10
with open(os.path.join(OUT, "table_seq_baselines_matched_exact.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["policy", "min_achievable_err"] + [f"time_at_err_{e:.2f}" for e in TABLE3_TARGETS])
    for name, (t, e) in frontiers.items():
        ta = frontier_time_at_error(t, e, TABLE3_TARGETS)
        w.writerow([name, round(float(isotonic_lower_envelope(t, e)[1].min()), 4)]
                   + [("" if np.isnan(x) else round(x, 2)) for x in ta])
print("wrote table_seq_baselines_matched_exact.csv")

# ranking sanity check: at each target error, which policy needs the LEAST time?
# (policies that cannot reach a target error at all -- NaN -- are excluded from
# that column's ranking rather than silently omitted or misranked)
print("\nranking (least time = best) at each matched target error (NaN = target unreachable by that policy):")
name_list = [r[0] for r in rows]
mat = np.array([r[1] for r in rows])  # [policy, target_err]
for k, te in enumerate(target_errs):
    col = mat[:, k]
    valid = [(name_list[i], col[i]) for i in range(len(name_list)) if not np.isnan(col[i])]
    valid.sort(key=lambda x: x[1])
    unreached = [name_list[i] for i in range(len(name_list)) if np.isnan(col[i])]
    line = " < ".join(f"{n}({v:.1f}h)" for n, v in valid)
    if unreached:
        line += "  [unreachable: " + ", ".join(unreached) + "]"
    print(f"  err={te:.2f}: {line}")

# ---------------------------------------------------------------------------
# Reverse experiment: fix the decision procedure, swap the posterior producer.
# Retrain LR (existing features/pipeline, reused) plus RF and HGB under
# leave-condition-out GroupKFold (matches the main leave-condition-out
# protocol; build_posterior_grouped.py's grouping key), then apply BOTH
# const-margin and DP-band procedures to each producer's P.
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("reverse experiment: fixed decision procedure, swap posterior producer")
print("=" * 70)
import openpyxl
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier

SEED = 0
XLSX = os.path.join(DATA_DIR, "Tebon_SupplementaryData_MOESM6.xlsx")
DMAP = {"Staurosporine": "Stauro", "Neratinib": "Ner", "Lapatinib": "Lap"}
DRUGS = ["Stauro", "Ner", "Lap"]


def parse(ws, cell):
    rows_ = list(ws.iter_rows(values_only=True))
    ci = [i for i, r in enumerate(rows_) if isinstance(r[1], str) and any(dd in r[1] for dd in DMAP)]
    out = []
    for k, idx in enumerate(ci):
        end = ci[k + 1] if k + 1 < len(ci) else len(rows_)
        lab = rows_[idx][1].split(); conc = float(lab[0]); drg = DMAP[lab[1]]
        blk = [r for r in (rows_[j] for j in range(idx, end)) if isinstance(r[2], (int, float)) and r[2] <= 72.5]
        t = np.array([r[2] for r in blk], float); nc = max(len(r) for r in blk)
        for c in range(3, nc):
            mm = np.array([(r[c] if c < len(r) and isinstance(r[c], (int, float)) else np.nan) for r in blk], float)
            if np.isfinite(mm).sum() >= 100:
                out.append((drg, conc, cell, t, mm))
    return out


wb = openpyxl.load_workbook(XLSX, read_only=True)
recs = parse(wb["Supp Fig 9 MCF"], "MCF-7") + parse(wb["Supp Fig 9 BT"], "BT-474")
from data_filter import filter_recs   # data-integrity filter (2026-07-16): must match posterior_data.npz (1888 -> 1850)
recs = filter_recs(recs)
drug_r = np.array([r[0] for r in recs]); conc_r = np.array([r[1] for r in recs]); cell_r = np.array([r[2] for r in recs])
doh = np.array([np.eye(3)[DRUGS.index(dd)] for dd in drug_r]); logc = np.array([[np.log(c + 1)] for c in conc_r])
# leave-condition-out grouping key, identical construction to build_posterior_grouped.py
cond_key_r = np.array([f"{c}|{dd}|{cc}" for c, dd, cc in zip(cell_r, drug_r, conc_r)])
cond_ids_r = {k: i for i, k in enumerate(sorted(set(cond_key_r)))}
groups_r = np.array([cond_ids_r[k] for k in cond_key_r])
assert np.array_equal(groups_r, GROUPS), "condition grouping mismatch between rebuilt and cached posterior data"


def val(t, m, tt):
    ok = np.isfinite(m)
    return np.interp(tt, t[ok], m[ok])


y_check = np.array([int(val(t, m, 72) / (val(t, m, 6) + 1e-8) < 0.90) for (_, _, _, t, m) in recs])
assert np.array_equal(y_check, y), "label mismatch between rebuilt and cached posterior data"


def feats(t, m, T, k=12):
    ok = np.isfinite(m)
    g = np.linspace(6, T, k); w = np.interp(g, t[ok], m[ok]); w = w / (w[0] + 1e-8)
    dt = (T - 6) / (k - 1); v = np.gradient(w, dt)
    return [w[-1], v.mean(), v[-1], w.min(), w.max() - w.min()]


folds = GroupKFold(5)
producers = {
    "LR": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
    "RF": RandomForestClassifier(n_estimators=300, max_depth=6, random_state=SEED, n_jobs=-1),
    "HGB": HistGradientBoostingClassifier(max_depth=4, random_state=SEED),
}
P_by_producer = {}
for name, clf in producers.items():
    Pp = np.zeros((N, nG))
    for j, T in enumerate(G):
        X = np.column_stack([np.array([feats(t, m, T) for (_, _, _, t, m) in recs], float), doh, logc])
        Pp[:, j] = cross_val_predict(clf, X, y, cv=folds.split(X, y, groups_r), method="predict_proba")[:, 1]
    P_by_producer[name] = Pp
    print(f"  built posterior for producer={name} (leave-condition-out)")

swap_rows = []
for name, Pp in P_by_producer.items():
    Ksp = build_kernel(Pp)
    fe = np.array([((Pp[:, j] >= 0.5).astype(int) != y).mean() for j in range(nG)])
    t_cm, e_cm = policy_const_margin(Pp)
    t_dp, e_dp = policy_dp(Pp, Ksp)
    for proc_name, (t_pol, e_pol) in [("const_margin", (t_cm, e_cm)), ("bayes_dp_band", (t_dp, e_dp))]:
        times_at = frontier_time_at_error(t_pol, e_pol, target_errs)
        fixed_times_at = frontier_time_at_error(G.copy(), fe, target_errs)
        saved = [ft_ - t_ for ft_, t_ in zip(fixed_times_at, times_at)
                 if not (np.isnan(ft_) or np.isnan(t_))]
        n_reached = len(saved)
        n_dom = sum(1 for s in saved if s > 0)
        mean_saved = float(np.mean(saved)) if saved else float("nan")
        swap_rows.append((name, proc_name, n_dom, n_reached, mean_saved))
        print(f"  producer={name:4s} procedure={proc_name:14s} dominates {n_dom}/{n_reached} reached "
              f"matched points (of 9 targets), mean hours saved={mean_saved:.2f}")

with open(os.path.join(OUT, "table_estimator_swap.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f); w.writerow(["posterior_producer", "decision_procedure", "n_dominated", "n_targets_reached_of_9", "mean_hours_saved"])
    for r in swap_rows:
        w.writerow([r[0], r[1], r[2], r[3], round(float(r[4]), 2)])
print("wrote table_estimator_swap.csv")
print("\nDONE.")
