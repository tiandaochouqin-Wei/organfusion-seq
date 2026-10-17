#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stage 3A (theory strengthening), empirical companion of the rewritten Theorem 2.

The old Theorem 2 bounded stopping regret by 2L*eps with eps = ess-sup_t |p_hat_t - p_t|,
a per-path worst-case posterior-approximation error that no experiment can measure; E9 used
ECE at 24 h as a "proxy" and had to concede that ECE is a binned, average-case quantity that
the theorem does not speak to. The rewritten theorem bounds the realised cost of the deployed
controller against the best controller WITH THE SAME INFORMATION (the sigma-algebra generated
by the estimate) by  L*[kappa(tau_hat) + kappa(tau_q)] + eta,  where
    kappa(tau) = E|p_hat_tau - E[y | G_tau]|      (calibration error AT THE STOPPING TIME),
and, under the self-sufficiency assumption E[y | p_hat_{<=t}] = E[y | p_hat_t],
    kappa(tau_hat) = E[g_tau(p_hat_tau)],  g_t(u) = |u - E[y | p_hat_t = u]|,
i.e. exactly the population ECE of the stopped posteriors, stratified by stopping horizon.
Every term is now an average-case, population-level, measurable quantity.

This script measures, on the leave-condition-out posterior and the same gamma-sweep as E9:
  (a) realised cost, ECE@24h (E9's old proxy), ECE at the stopping time (pooled and
      horizon-stratified), MCE at the stopping time, max over horizons of ECE_t and MCE_t;
  (b) excess cost over the best gamma of the sweep against the new bound's measurable part
      2L * ECE_stop (heuristic, drops the unobservable oracle term kappa(tau_q)) and against
      2L * MCE (rigorous under self-sufficiency, since kappa(tau_q) <= sup_u g(u));
  (c) a cross-fitted check of self-sufficiency: does the prefix history p_hat_{<t} predict y
      beyond p_hat_t?  leak_t := E|p_hat_t - q_path_t| - E|p_hat_t - q_marg_t|, both fitted by
      the same HGB class, out-of-fold by condition (GroupKFold), plus the log-loss difference.
Both the deployed fixed-margin rule of E9 (m = 0.30) and the DP band re-optimised on the
gamma-scaled surrogate are reported; the theorem allows any G-stopping time with an
explicit surrogate-suboptimality slack eta, so both are covered.

Outputs:
  posterior/table_stage3a_calib_regret.csv       (a)+(b), one row per (rule, gamma)
  posterior/table_stage3a_selfsufficiency.csv    (c), one row per horizon
"""
import os, sys, csv, numpy as np
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import log_loss

OUT = "posterior"
d = np.load(os.path.join(OUT, "posterior_data_grouped.npz"), allow_pickle=True)
P = d["P_group"]; y = d["y"].astype(int); GRID = d["GRID"].astype(float); groups = d["groups"]
N, T = P.shape
LAM = 0.004; M_FIXED = 0.30
GAMMAS = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0]
NB = 10; MIN_COUNT_MCE = 20
print(f"N={N} T={T} GRID={GRID.astype(int).tolist()}  lambda={LAM}  L=1 (symmetric)")


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6); return np.log(p / (1 - p))


def scale(Pm, g):
    return 1 / (1 + np.exp(-g * logit(Pm)))


def ece_mce(p, yy, nb=NB, min_count=MIN_COUNT_MCE):
    bb = np.linspace(0, 1, nb + 1); e = 0.0; mce = 0.0
    for i in range(nb):
        mk = (p >= bb[i]) & (p < bb[i + 1]) if i < nb - 1 else (p >= bb[i]) & (p <= bb[i + 1])
        if mk.sum():
            gap = abs(yy[mk].mean() - p[mk].mean()); e += mk.mean() * gap
            if mk.sum() >= min_count: mce = max(mce, gap)
    return e, mce


def ece_stop_strat(ps, ts, yy):
    """ECE of the stopped posteriors, stratified by stopping horizon (estimates
    E[g_tau(p_hat_tau)] = kappa(tau_hat) under self-sufficiency); MCE likewise."""
    e = 0.0; mce = 0.0
    for t in np.unique(ts):
        mk = ts == t
        et, mt = ece_mce(ps[mk], yy[mk]); e += mk.mean() * et; mce = max(mce, mt)
    return e, mce


# ---- deployed rules -------------------------------------------------------------
def rule_margin(Pm, m):
    tt = np.empty(N); dd = np.empty(N, int); ps = np.empty(N)
    for i in range(N):
        e = np.where(np.abs(Pm[i] - 0.5) >= m)[0]
        j = e[0] if len(e) else T - 1
        tt[i] = GRID[j]; dd[i] = int(Pm[i, j] >= 0.5); ps[i] = Pm[i, j]
    return tt, dd, ps


Bbin = np.linspace(0, 1, 51); bc = (Bbin[:-1] + Bbin[1:]) / 2; nb = len(bc)
rho_bc = np.minimum(1 - bc, bc)


def build_kernel(Pm):
    Ks = []
    for j in range(T - 1):
        K = np.zeros((nb, nb))
        a = np.clip(np.digitize(Pm[:, j], Bbin) - 1, 0, nb - 1)
        b = np.clip(np.digitize(Pm[:, j + 1], Bbin) - 1, 0, nb - 1)
        np.add.at(K, (a, b), 1)
        K += 1e-3; K /= K.sum(1, keepdims=True); Ks.append(K)
    return Ks


def rule_dp(Pm, lam):
    Ks = build_kernel(Pm); V = rho_bc.copy(); ab = np.full((T, 2), 0.5)
    for j in range(T - 2, -1, -1):
        cont = lam * (GRID[j + 1] - GRID[j]) + Ks[j] @ V
        cs = np.where(cont < rho_bc - 1e-9)[0]
        if len(cs): ab[j] = [bc[cs.min()], bc[cs.max()]]
        V = np.minimum(rho_bc, cont)
    tt = np.empty(N); dd = np.empty(N, int); ps = np.empty(N)
    for i in range(N):
        js = T - 1
        for j in range(T - 1):
            if not (ab[j, 0] < Pm[i, j] < ab[j, 1]): js = j; break
        tt[i] = GRID[js]; dd[i] = int(Pm[i, js] >= 0.5); ps[i] = Pm[i, js]
    return tt, dd, ps


# ---- (a)+(b): gamma sweep ---------------------------------------------------------
rows = []
for rule_name in ("margin_0.30", "dp_band"):
    res = []
    for g in GAMMAS:
        Pg = scale(P, g)
        tt, dd, ps = rule_margin(Pg, M_FIXED) if rule_name.startswith("margin") else rule_dp(Pg, LAM)
        cost = float(np.mean(LAM * tt + (dd != y)))
        ece24, mce24 = ece_mce(Pg[:, 4], y)
        ece_pool, mce_pool = ece_mce(ps, y)
        ece_strat, mce_strat = ece_stop_strat(ps, tt, y)
        per_t = [ece_mce(Pg[:, j], y) for j in range(T)]
        max_ece_t = max(e for e, _ in per_t); max_mce_t = max(m for _, m in per_t)
        res.append(dict(rule=rule_name, gamma=g, cost=cost, mean_time=float(tt.mean()),
                        err=float((dd != y).mean()), ece24=ece24, ece_stop_pooled=ece_pool,
                        ece_stop_strat=ece_strat, mce_stop_strat=mce_strat,
                        max_ece_t=max_ece_t, max_mce_t=max_mce_t))
    cmin = min(r["cost"] for r in res)
    for r in res:
        r["excess"] = r["cost"] - cmin
        r["bound_2ece_stop"] = 2 * r["ece_stop_strat"]
        r["bound_ece_stop_plus_mce"] = r["ece_stop_strat"] + max(r["mce_stop_strat"], r["max_mce_t"])
        r["bound_2mce"] = 2 * max(r["mce_stop_strat"], r["max_mce_t"])
    rows += res
    print(f"\n=== rule: {rule_name} ===")
    print(f"{'gamma':>5} {'cost':>7} {'time':>6} {'err':>6} | {'ECE24':>6} {'ECEstp':>6} {'ECEstr':>6} {'MCEstr':>6} {'maxECEt':>7} {'maxMCEt':>7} | {'excess':>6} {'2ECEs':>6} {'ECE+MCE':>7} {'2MCE':>6}")
    for r in res:
        print(f"{r['gamma']:>5.2f} {r['cost']:>7.4f} {r['mean_time']:>6.1f} {r['err']:>6.3f} | "
              f"{r['ece24']:>6.3f} {r['ece_stop_pooled']:>6.3f} {r['ece_stop_strat']:>6.3f} {r['mce_stop_strat']:>6.3f} "
              f"{r['max_ece_t']:>7.3f} {r['max_mce_t']:>7.3f} | {r['excess']:>6.4f} {r['bound_2ece_stop']:>6.3f} "
              f"{r['bound_ece_stop_plus_mce']:>7.3f} {r['bound_2mce']:>6.3f}"
              f"{'' if r['excess'] <= r['bound_2ece_stop'] + 1e-12 else '   <-- excess > 2*ECE_stop'}")
    g_cost = min(res, key=lambda r: r["cost"])["gamma"]
    g_e24 = min(res, key=lambda r: r["ece24"])["gamma"]
    g_est = min(res, key=lambda r: r["ece_stop_strat"])["gamma"]
    print(f"argmin gamma: cost -> {g_cost}   ECE@24h -> {g_e24}   ECE@stop(strat) -> {g_est}")

with open(os.path.join(OUT, "table_stage3a_calib_regret.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader()
    for r in rows: w.writerow({k: (round(v, 5) if isinstance(v, float) else v) for k, v in r.items()})
print("\nwrote table_stage3a_calib_regret.csv")

# ---- (c): self-sufficiency check --------------------------------------------------
print("\n=== self-sufficiency check: does the prefix history predict y beyond p_hat_t? (out-of-fold by condition) ===")
LG = logit(P)
folds = list(GroupKFold(5).split(P, y, groups))


def oof_prob(X):
    q = np.zeros(N)
    for tr, te in folds:
        clf = HistGradientBoostingClassifier(max_iter=150, learning_rate=0.05, max_depth=3,
                                             min_samples_leaf=40, l2_regularization=1.0, random_state=0)
        clf.fit(X[tr], y[tr]); q[te] = clf.predict_proba(X[te])[:, 1]
    return q


ss_rows = []
print(f"{'t(h)':>5} {'kappa_marg':>10} {'kappa_path':>10} {'leak':>8} {'ll_marg':>8} {'ll_path':>8} {'d_ll':>8}")
for j in range(T):
    q_m = oof_prob(LG[:, [j]]); q_p = oof_prob(LG[:, :j + 1])
    km = float(np.mean(np.abs(P[:, j] - q_m))); kp = float(np.mean(np.abs(P[:, j] - q_p)))
    llm = log_loss(y, np.clip(q_m, 1e-6, 1 - 1e-6)); llp = log_loss(y, np.clip(q_p, 1e-6, 1 - 1e-6))
    ss_rows.append(dict(t=GRID[j], kappa_marg=km, kappa_path=kp, leak=kp - km, logloss_marg=llm,
                        logloss_path=llp, d_logloss=llp - llm))
    print(f"{GRID[j]:>5.0f} {km:>10.4f} {kp:>10.4f} {kp-km:>+8.4f} {llm:>8.4f} {llp:>8.4f} {llp-llm:>+8.4f}")
with open(os.path.join(OUT, "table_stage3a_selfsufficiency.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(ss_rows[0].keys())); w.writeheader()
    for r in ss_rows: w.writerow({k: round(float(v), 5) for k, v in r.items()})
print("wrote table_stage3a_selfsufficiency.csv")
print("Reading: leak ~ 0 and d_logloss ~ 0 (or >0, i.e. history HURTS out-of-fold) means the current estimate is")
print("self-sufficient for the label, so kappa(tau) coincides with the stopped ECE and the theorem's bound is")
print("stated in the very quantity measured above; a clearly negative d_logloss would flag a history leak.")
