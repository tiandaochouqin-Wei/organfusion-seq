#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stage 3A: a certificate that COVERS CONDITION DRIFT (answers the E7 gap: at alpha=0.20 the
organoid-level Hoeffding certificate held on only 7/10 leave-condition-out test splits).

Diagnosis. Theorem 3's slack U_n = B*sqrt(log(2|G|/delta)/(2n)) treats the n ~ 900 calibration
ORGANOIDS as exchangeable with the test organoids. Under leave-condition-out the test organoids
come from CONDITIONS never seen in calibration, so the exchangeable unit is the condition and
the effective sample size is m = 9, not n = 900. A high-probability distribution-free
certificate at the condition level would need width B*sqrt(log(2|G|/delta)/(2m)) ~ 0.6 --
vacuous. That is not a weakness of one bound: with 9 exchangeable units no distribution-free
high-probability statement about a 0.20 rate is possible.

Fix (Theorem 3', condition-level risk control). What CAN be certified with m = 9 exchangeable
conditions is the risk IN EXPECTATION over a new condition: conformal risk control
(Angelopoulos, Bates, Fisch, Lei, Schuster 2024) applied at the condition level. For a
1-D family pi_s ordered by conservativeness and per-condition losses L_c(s) in [0,B], with the
monotone envelope Lt_c(s) = max_{s'>=s} L_c(s'), choose
      s_hat = min{ s : ( sum_{c in cal} Lt_c(s) + B ) / (m+1) <= alpha }.
Then E[L_new(s_hat)] <= alpha whenever the calibration conditions and the new condition are
exchangeable -- no organoid-level exchangeability, no i.i.d. within condition, no assumption on
how conditions differ. The price is explicit: the calibration mean must be at most
alpha - (B - alpha)/m  (= 0.122 for alpha=0.20, m=9), and the guarantee is in expectation.

To keep the guarantee rigorous the certified family must NOT be fitted on the calibration
conditions (else the calibration losses are in-sample and no longer exchangeable with a new
condition's loss). The primary certified family is therefore the deployed constant-margin band
(|p_hat - 1/2| >= m, the "ours" of Table 3; E2b showed it within 1 h of the Bayes-DP band),
which needs no fitting; fixed-time is certified the same way for the comparison. The
Lagrangian-DP family of Algorithm 1 is also run, with its transition kernel fitted on the
calibration conditions, and flagged as a kernel-reuse arm (mild in-sample dependence).

Same 10 random 9/9 condition splits as constrained_control.py (same rng seed and permutation
order), same posterior (leave-condition-out shared-encoder P_group), same lambda_t.

Outputs:
  posterior/table_constrained_control_crc.csv
"""
import os, csv, sys, numpy as np
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

OUT = "posterior"
LAM_T = 0.004; DELTA = 0.10; R = 50; B = 1.0  # R raised from 10 to 50 (Stage 3F, A3):
# R=10 let a single unlucky split flip "holds" between 9/10 and 10/10 by one condition;
# R=50 (still <1s to run) gives a stable estimate with a bootstrap CI on the fraction of
# held-out conditions within the risk target, resampling SPLITS (not conditions) so the
# within-split correlation across a split's 9 test conditions is preserved.
ALPHAS = (0.10, 0.15, 0.20, 0.25)
N_BOOT = 2000

d = np.load(os.path.join(OUT, "posterior_data_shared_strat.npz"), allow_pickle=True)
P = d["P_group"]; y = d["y"].astype(int); GRID = d["GRID"].astype(float); groups = d["groups"]
N, T = P.shape
conds = np.array(sorted(set(groups.tolist()))); C = len(conds)
cidx = {c: np.where(groups == c)[0] for c in conds}
print(f"N={N} T={T} conditions={C}")

# identical split sequence to constrained_control.py (rng re-created per run(), reps 1..R)
rng = np.random.default_rng(0)
SPLITS = []
for rep in range(1, R + 1):
    perm = rng.permutation(conds); SPLITS.append((rep, perm[:9], perm[9:]))


# ---- families (index increasing = more conservative = later stopping) ------------
MARGINS = np.linspace(0.0, 0.5, 51)
SCALES = np.geomspace(0.05, 50.0, 40)


def apply_margin(m):
    tau = np.empty(N); dec = np.empty(N, int)
    for i in range(N):
        e = np.where(np.abs(P[i] - 0.5) >= m)[0]
        j = e[0] if len(e) else T - 1
        tau[i] = GRID[j]; dec[i] = int(P[i, j] >= 0.5)
    return tau, dec


def apply_fixed(j):
    return np.full(N, GRID[j]), (P[:, j] >= 0.5).astype(int)


def per_condition(tau, dec):
    """per-condition FP=Pr(delta=1,y=0|c), FN=Pr(delta=0,y=1|c), mean time; arrays over conds."""
    fp = np.array([np.mean((dec[ix] == 1) & (y[ix] == 0)) for ix in cidx.values()])
    fn = np.array([np.mean((dec[ix] == 0) & (y[ix] == 1)) for ix in cidx.values()])
    tm = np.array([tau[ix].mean() for ix in cidx.values()])
    return fp, fn, tm


print("precomputing per-condition losses for the fit-free families ...")
# Every family ends with the endpoint assay itself (wait to 72 h and read the label):
# zero decision risk, maximal time. Theorem 3' needs it so that (sum + B)/(m+1) <= alpha is
# always attainable for alpha >= B/(m+1); selecting it means "no early decision certifiable".
ENDPOINT = np.stack([np.zeros(C), np.zeros(C), np.full(C, GRID[-1])])              # [3, C]
FAM = {}
FAM["margin"] = np.array([per_condition(*apply_margin(m)) for m in MARGINS] + [ENDPOINT])   # [S+1, 3, C]
FAM["fixed"] = np.array([per_condition(*apply_fixed(j)) for j in range(T)] + [ENDPOINT])    # [T+1, 3, C]
MARGINS = np.append(MARGINS, np.nan); GRID_FAM = np.append(GRID, np.nan)   # nan = endpoint assay

# Lagrangian-DP family (Algorithm 1) at p*=1/2, kernel fitted on calibration organoids
Bbin = np.linspace(0, 1, 51); bc = (Bbin[:-1] + Bbin[1:]) / 2; nb = len(bc)


def kernels(idx):
    Ks = []
    for j in range(T - 1):
        K = np.zeros((nb, nb))
        a = np.clip(np.digitize(P[idx, j], Bbin) - 1, 0, nb - 1)
        b = np.clip(np.digitize(P[idx, j + 1], Bbin) - 1, 0, nb - 1)
        np.add.at(K, (a, b), 1); K += 1e-3; K /= K.sum(1, keepdims=True); Ks.append(K)
    return Ks


def dp_apply(Ks, s):
    rho = s * np.minimum(1 - bc, bc); V = rho.copy(); band = np.full((T, 2), 0.5)
    for j in range(T - 2, -1, -1):
        cont = LAM_T * (GRID[j + 1] - GRID[j]) + Ks[j] @ V
        cs = np.where(cont < rho - 1e-12)[0]
        if len(cs): band[j] = (bc[cs.min()], bc[cs.max()])
        V = np.minimum(rho, cont)
    tau = np.empty(N); dec = np.empty(N, int)
    for i in range(N):
        js = T - 1
        for j in range(T - 1):
            if not (band[j, 0] < P[i, j] < band[j, 1]): js = j; break
        tau[i] = GRID[js]; dec[i] = int(P[i, js] >= 0.5)
    return tau, dec


# ---- selection rules ----------------------------------------------------------------
def crc_select(L, alpha):
    """L: [S, m] per-condition losses of the calibration conditions. Returns the smallest
    index s with (sum_c envelope_c(s) + B)/(m+1) <= alpha, or None."""
    Lt = np.maximum.accumulate(L[::-1], axis=0)[::-1]
    m = L.shape[1]
    Rt = (Lt.sum(1) + B) / (m + 1)
    ok = np.where(Rt <= alpha)[0]
    return int(ok[0]) if len(ok) else None


def crc_two_sided(Lfp, Lfn, alpha):
    a = crc_select(Lfp, alpha); b = crc_select(Lfn, alpha)
    return None if (a is None or b is None) else max(a, b)


def hoeffding_select(fp_pool, fn_pool, alpha, n_cal, S, slack=True):
    U = np.sqrt(np.log(2 * S / DELTA) / (2 * n_cal)) if slack else 0.0
    ok = np.where((fp_pool + U <= alpha) & (fn_pool + U <= alpha))[0]
    return (int(ok[0]) if len(ok) else None), U


def ltt_select(fp_pool, fn_pool, alpha, n_cal):
    """Learn-then-Test with fixed-sequence testing (Angelopoulos et al., 2025) -- the calibration
    that Ringel et al. (ICML 2024) use for early stopping, here on our two risks. Walk from the
    most conservative rule towards faster ones; reject H_s: FP(s) > alpha or FN(s) > alpha while
    both exact binomial tail p-values are <= delta (intersection-union, so no correction between
    the two risks), stop at the first non-rejection. The selected rule violates either risk with
    probability <= delta -- if the calibration and deployment UNITS are exchangeable. It is the
    tightest sample-level certificate we know of for a monotone family (no union bound, no
    Hoeffding slack), and is added in Stage 3P to show that the failure under condition drift is
    not a matter of how tight the sample-level bound is."""
    from scipy.stats import binom
    sel = None
    for s in range(len(fp_pool) - 1, -1, -1):
        pv = max(binom.cdf(int(round(fp_pool[s] * n_cal)), n_cal, alpha),
                 binom.cdf(int(round(fn_pool[s] * n_cal)), n_cal, alpha))
        if pv <= DELTA: sel = s
        else: break
    return sel


def pooled(arr_c, cal_or_te):
    """organoid-weighted pooled rate from per-condition rates."""
    w = np.array([len(cidx[c]) for c in cal_or_te], float)
    return (arr_c * w).sum() / w.sum()


def evaluate(fam, s, c_te):
    """returns pooled test FP/FN/time and the per-condition test losses."""
    ti = [np.where(conds == c)[0][0] for c in c_te]
    fp_c, fn_c, tm_c = fam[s, 0, ti], fam[s, 1, ti], fam[s, 2, ti]
    return pooled(fp_c, c_te), pooled(fn_c, c_te), pooled(tm_c, c_te), fp_c, fn_c


# Stage 3Q: the Lagrangian family is cross-fitted by condition, like the posterior itself: the
# kernel used for the organoids of a condition is estimated on the other folds of conditions
# (5-fold GroupKFold), so no condition's losses come from a kernel that saw it. Until 3Q the
# kernel was refitted on each split's calibration conditions and the same conditions then
# selected -- a family that depends on the calibration data, which Theorem 4 excludes.
from sklearn.model_selection import GroupKFold
_tau = np.zeros((len(SCALES), N)); _dec = np.zeros((len(SCALES), N), int)
for _tr, _te in GroupKFold(5).split(P, y, groups):
    _Ks = kernels(_tr)
    for _k, _s in enumerate(SCALES):
        _t, _d = dp_apply(_Ks, _s); _tau[_k, _te] = _t[_te]; _dec[_k, _te] = _d[_te]
FAM["dp-lagrangian(cross-fitted)"] = np.array([per_condition(_tau[k], _dec[k]) for k in range(len(SCALES))] + [ENDPOINT])

ARMS = [("margin", "no-slack"), ("margin", "hoeffding-organoid"), ("margin", "ltt-organoid"),
        ("margin", "crc-condition"),
        ("fixed", "no-slack"), ("fixed", "hoeffding-organoid"), ("fixed", "ltt-organoid"),
        ("fixed", "crc-condition"),
        ("dp-lagrangian(cross-fitted)", "crc-condition")]
all_rows = []
summary = []
for alpha in ALPHAS:
    for fam_name, arm in ARMS:
        rec = []
        for rep, c_cal, c_te in SPLITS:
            ci = [np.where(conds == c)[0][0] for c in c_cal]
            fam = FAM[fam_name]
            S = fam.shape[0]
            Lfp, Lfn = fam[:, 0, ci], fam[:, 1, ci]
            n_cal = sum(len(cidx[c]) for c in c_cal)
            if arm == "crc-condition":
                s = crc_two_sided(Lfp, Lfn, alpha); U = np.nan
            else:
                fp_pool = np.array([pooled(fam[k, 0, ci], c_cal) for k in range(S)])
                fn_pool = np.array([pooled(fam[k, 1, ci], c_cal) for k in range(S)])
                if arm == "ltt-organoid":
                    s, U = ltt_select(fp_pool, fn_pool, alpha, n_cal), np.nan
                else:
                    s, U = hoeffding_select(fp_pool, fn_pool, alpha, n_cal, S, slack=(arm != "no-slack"))
            if s is None:
                rec.append(dict(rep=rep, ok=False)); continue
            fp_te, fn_te, tm_te, fp_c, fn_c = evaluate(fam, s, c_te)
            param = MARGINS[s] if fam_name == "margin" else (GRID_FAM[s] if fam_name == "fixed" else
                                                              (np.append(SCALES, np.nan)[s]))
            rec.append(dict(rep=rep, ok=True, s=s, param=float(param), fp=fp_te, fn=fn_te, t=tm_te,
                            holds=(fp_te <= alpha and fn_te <= alpha), fp_c=fp_c, fn_c=fn_c, U=U))
            all_rows.append((alpha, fam_name, arm, rep, round(float(param), 4), round(tm_te, 2),
                             round(fp_te, 4), round(fn_te, 4), fp_te <= alpha and fn_te <= alpha,
                             round(float(fp_c.mean()), 4), round(float(fn_c.mean()), 4),
                             round(float(np.mean(np.maximum(fp_c, fn_c) <= alpha)), 3),
                             "" if np.isnan(U) else round(U, 4)))
        good = [r for r in rec if r["ok"]]
        if good:
            fp_all = np.concatenate([r["fp_c"] for r in good]); fn_all = np.concatenate([r["fn_c"] for r in good])
            frac_point = float(np.mean(np.maximum(fp_all, fn_all) <= alpha))
            # bootstrap over SPLITS (not conditions): resample which of the `good` splits are
            # included, recompute the pooled fraction each time, so the 9-condition correlation
            # within a split is preserved rather than treating 9*R conditions as independent.
            rng_b = np.random.default_rng(1)
            within = [np.maximum(r["fp_c"], r["fn_c"]) <= alpha for r in good]
            boots = np.empty(N_BOOT)
            for b in range(N_BOOT):
                pick = rng_b.integers(0, len(good), len(good))
                boots[b] = np.mean(np.concatenate([within[i] for i in pick]))
            ci_lo, ci_hi = np.percentile(boots, [2.5, 97.5])
            summary.append((alpha, fam_name, arm, len(good), sum(r["holds"] for r in good),
                            float(np.mean([r["t"] for r in good])), float(fp_all.mean()), float(fn_all.mean()),
                            frac_point, float(ci_lo), float(ci_hi)))
        else:
            summary.append((alpha, fam_name, arm, 0, 0, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan))

print(f"\n{'alpha':>5} {'family':>28} {'arm':>19} | {'cal-feas':>8} {'test-holds':>10} {'time':>6} | "
      f"{'E[FP_new]':>9} {'E[FN_new]':>9} {'cond<=a [95% CI]':>22}")
for a, f, arm, k, h, t, efp, efn, frac, lo, hi in summary:
    print(f"{a:>5.2f} {f:>28} {arm:>19} | {k:>5}/{R:<2} {h:>6}/{k:<3} {t:>6.1f} | {efp:>9.4f} {efn:>9.4f} "
          f"{frac:>6.2f} [{lo:.2f},{hi:.2f}]"
          if k else f"{a:>5.2f} {f:>28} {arm:>19} | {0:>5}/{R:<2} {'--':>10} {'--':>6} | {'--':>9} {'--':>9} {'--':>7}")

with open(os.path.join(OUT, "table_constrained_control_crc.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["alpha", "family", "selection", "repeat", "selected_param", "test_mean_time_h",
                "test_FP_pooled", "test_FN_pooled", "test_feasible_pooled",
                "test_FP_condmean", "test_FN_condmean", "frac_test_conditions_within_alpha", "U_n"])
    for r in all_rows: w.writerow(r)
    w.writerow([]); w.writerow(["# summary"])
    w.writerow(["alpha", "family", "selection", "cal_feasible", "test_holds_pooled", "mean_time_h",
                "E_FP_new_condition", "E_FN_new_condition", "frac_conditions_within_alpha",
                "frac_ci_lo", "frac_ci_hi"])
    for s_ in summary: w.writerow([round(x, 4) if isinstance(x, float) else x for x in s_])
print(f"\nwrote table_constrained_control_crc.csv (R={R} splits, {N_BOOT}-resample split-level bootstrap CI)")

# ---- control (Stage 3P): the same sample-level rules when units ARE exchangeable ----------
# Organoids are split at random, ignoring conditions, so calibration and test units are
# exchangeable and Learn-then-Test must hold with probability >= 1 - delta. If it does here and
# does not above, what breaks it above is the drift to new conditions, not the procedure.
print("\ncontrol: random organoid-level halves (exchangeable units), margin family")
UNIT = [apply_margin(mm) for mm in MARGINS[:-1]] + [(np.full(N, GRID[-1]), y.copy())]
FPu = np.array([(dd == 1) & (y == 0) for _, dd in UNIT]); FNu = np.array([(dd == 0) & (y == 1) for _, dd in UNIT])
TMu = np.array([tt for tt, _ in UNIT])
rng_c = np.random.default_rng(2)
HALVES = [rng_c.permutation(N) for _ in range(R)]
control = []
for alpha in ALPHAS:
    for arm in ("no-slack", "hoeffding-organoid", "ltt-organoid"):
        holds = 0; tms = []
        for perm in HALVES:
            ca, te = perm[:N // 2], perm[N // 2:]
            fp_pool, fn_pool = FPu[:, ca].mean(1), FNu[:, ca].mean(1)
            if arm == "ltt-organoid": s_ = ltt_select(fp_pool, fn_pool, alpha, len(ca))
            else: s_, _ = hoeffding_select(fp_pool, fn_pool, alpha, len(ca), len(UNIT), slack=(arm != "no-slack"))
            holds += bool(FPu[s_, te].mean() <= alpha and FNu[s_, te].mean() <= alpha); tms.append(TMu[s_, te].mean())
        control.append((alpha, arm, holds, float(np.mean(tms))))
        print(f"  alpha={alpha:.2f} {arm:>19}: holds {holds}/{R}, mean time {np.mean(tms):.1f} h")
with open(os.path.join(OUT, "table_constrained_control_crc.csv"), "a", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow([]); w.writerow(["# control: random organoid-level halves (exchangeable units), margin family"])
    w.writerow(["alpha", "selection", "test_holds_pooled", "n_splits", "mean_time_h"])
    for a_, arm, h, t_ in control: w.writerow([a_, arm, h, R, round(t_, 2)])

m = 9
print("\nDeflated calibration targets (alpha*(m+1)-B)/m for m=9, B=1:",
      {a: round((a * (m + 1) - B) / m, 4) for a in ALPHAS})
for eps in (0.05, 0.10):
    print(f"conditions needed for a HIGH-PROBABILITY condition-level Hoeffding certificate of width {eps} "
          f"(delta={DELTA}, |G|=51): m >= {np.log(2*51/DELTA)/(2*eps**2):.0f};  single policy: m >= {np.log(2/DELTA)/(2*eps**2):.0f}")
print("Reading: 'E[FP_new]'/'E[FN_new]' average the per-condition losses over all R x 9 = 450 held-out (split, condition) pairs;")
print("the condition-level certificate promises these are <= alpha (in expectation), NOT that every split's pooled")
print("rate is <= alpha. 'test-holds' (pooled rate <= alpha on that split) is reported for continuity with Table 6.")
