#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""(1) Unify the three boundary constructions into ONE constrained optimal control
problem (reviewer Sec. 7).

The objection is correct and substantive: the paper currently contains three
different objects all called "the band" --
  (i)   the Bellman-optimal band of Theorem 1,
  (ii)  the PPV/NPV thresholds of Prop. riskcov, which Algorithm 1 actually deploys,
  (iii) the fixed symmetric margin |p-0.5|>=m that the experiments sweep,
and they cannot be written as one optimal controller.

The reviewer's prescription, implemented here verbatim:

    min_pi  E[ lambda_t * tau + lambda_m * N_tau ]      (observation cost)
    s.t.    Pr(delta=1, Y=0) <= alpha_FP
            Pr(delta=0, Y=1) <= alpha_FN

  Lagrangian:  J_mu(pi) = E[ lambda_t*tau + lambda_m*N_tau + mu_FP*l_FP + mu_FN*l_FN ]

  -> DP over a grid of (mu_FP, mu_FN) GENERATES candidate policies
  -> the CALIBRATION set SELECTS the feasible policy with lowest observation cost
  -> the TEST set EVALUATES, once.

Why this is the right unification rather than a third patch: stopping at belief p
under the Lagrangian costs min{mu_FP*(1-p), mu_FN*p}, which is exactly the paper's
rho with (c_FP, c_FN) := (mu_FP, mu_FN). So (i) IS the Lagrangian DP; sweeping mu
traces out a family whose indifference point p*_mu = mu_FP/(mu_FP+mu_FN) moves --
recovering (iii) as the symmetric special case mu_FP=mu_FN, and replacing (ii)'s
calibration-error-inflated thresholds by an explicit feasibility test on held-out
data. The candidate family is finite, so it IS the grid G of Theorem 3, whose
uniform certificate is precisely what licenses selecting on the calibration set.

Note on lambda_m: on a fixed observation grid N_tau is a deterministic function of
tau, so lambda_m is collinear with lambda_t and adds no independent lever. It only
becomes independent under cadence control -- which E2e shows buys nothing here. We
carry it symbolically and set lambda_m=0 in the reported runs, rather than pretend
it is doing work.

Outputs:
  posterior/table_constrained_control.csv
"""
import os, csv, sys, numpy as np
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

OUT = "posterior"
LAM_T = 0.004      # per-hour waiting cost
LAM_M = 0.0        # per-measurement cost (collinear with LAM_T on a fixed grid; see docstring)
ALPHA_FP, ALPHA_FN = 0.10, 0.10
DELTA = 0.10       # certificate level
R = 50             # raised from 10 (Stage 3F, A3): a single-digit split count let one unlucky
                    # split move "holds" by one; 50 gives a stable read of how often the
                    # organoid-level certificate fails under condition drift (it is expected to
                    # fail sometimes here -- that failure motivates Theorem 4/Table 8, not a bug)

d = np.load(os.path.join(OUT, "posterior_data_shared_strat.npz"), allow_pickle=True)
P = d["P_group"]                      # leave-condition-out shared-encoder posterior
y = d["y"].astype(int); GRID = d["GRID"].astype(float); groups = d["groups"]
N, T = P.shape
conds = np.array(sorted(set(groups.tolist())))
print(f"N={N} T={T} conditions={len(conds)}  constraints: FP<={ALPHA_FP}, FN<={ALPHA_FN}")

Bbin = np.linspace(0, 1, 51); bc = (Bbin[:-1] + Bbin[1:]) / 2; nb = len(bc)


def kernels(P, idx):
    Ks = []
    for j in range(T - 1):
        K = np.zeros((nb, nb))
        a = np.clip(np.digitize(P[idx, j], Bbin) - 1, 0, nb - 1)
        b = np.clip(np.digitize(P[idx, j + 1], Bbin) - 1, 0, nb - 1)
        for u, v in zip(a, b): K[u, v] += 1
        K += 1e-3; K /= K.sum(1, keepdims=True); Ks.append(K)
    return Ks


def dp_band(Ks, mu_FP, mu_FN):
    """Backward induction under the Lagrangian stop-cost rho_mu(p)=min(mu_FP(1-p), mu_FN p).
    Returns per-horizon (a_t,b_t); the continuation region is where continuing is cheaper."""
    rho = np.minimum(mu_FP * (1 - bc), mu_FN * bc)
    V = rho.copy()
    band = np.zeros((T, 2))
    pstar = mu_FP / (mu_FP + mu_FN)
    band[T - 1] = (pstar, pstar)
    for j in range(T - 2, -1, -1):
        cont = LAM_T * (GRID[j + 1] - GRID[j]) + LAM_M + Ks[j] @ V
        cs = np.where(cont < rho - 1e-12)[0]
        band[j] = (bc[cs.min()], bc[cs.max()]) if len(cs) else (pstar, pstar)
        V = np.minimum(rho, cont)
    return band, pstar


def apply_band(P, idx, band, pstar):
    tau = np.empty(len(idx)); dec = np.empty(len(idx), int); nobs = np.empty(len(idx), int)
    for n, i in enumerate(idx):
        js = T - 1
        for j in range(T - 1):
            a, b = band[j]
            if not (a < P[i, j] < b):
                js = j; break
        tau[n] = GRID[js]; nobs[n] = js + 1
        dec[n] = int(P[i, js] >= pstar)
    return tau, dec, nobs


def rates(dec, yy, tau, nobs):
    fp = float(np.mean((dec == 1) & (yy == 0)))     # Pr(delta=1, Y=0)
    fn = float(np.mean((dec == 0) & (yy == 1)))     # Pr(delta=0, Y=1)
    obs = float(np.mean(LAM_T * tau + LAM_M * nobs))
    return fp, fn, obs, float(tau.mean())


# candidate family: sweep the Lagrangian multipliers (ratio -> p*, scale -> vs delay)
PSTARS = np.linspace(0.20, 0.80, 13)
SCALES = np.geomspace(0.2, 20.0, 12)
CANDS = [(s * p, s * (1 - p)) for p in PSTARS for s in SCALES]
print(f"candidate policies |G| = {len(CANDS)}  (Lagrangian grid; this IS Theorem 3's G)")

def run(alpha_fp, alpha_fn, use_slack=True):
    rng = np.random.default_rng(0)
    rows = []
    for rep in range(1, R + 1):
        perm = rng.permutation(conds)
        c_cal, c_te = perm[:9], perm[9:]
        cal_all = np.where(np.isin(groups, c_cal))[0]
        te = np.where(np.isin(groups, c_te))[0]
        # Stage 3Q: Theorem 3's union bound is over a family fixed BEFORE the selection sample is
        # seen. Until 3Q the transition kernel that generates the 156 candidates was fitted on the
        # same calibration organoids that then selected among them -- a data-dependent family the
        # bound does not cover. Now the kernel is fitted on a random half of the calibration
        # organoids and feasibility/selection use the other half (the fixed-horizon competitor is
        # selected on the same half), so the family is independent of the selection sample.
        halves = np.random.default_rng(1000 + rep).permutation(cal_all)
        cal_k, cal = halves[:len(halves) // 2], halves[len(halves) // 2:]
        Ks = kernels(P, cal_k)                 # kernel on one half of CAL; selection on the other
        n_cal = len(cal)
        # Theorem 3 slack: uniform over the finite candidate grid G
        U = np.sqrt(np.log(2 * len(CANDS) / DELTA) / (2 * n_cal)) if use_slack else 0.0

        feas = []
        for (mfp, mfn) in CANDS:
            band, ps = dp_band(Ks, mfp, mfn)
            t_, d_, n_ = apply_band(P, cal, band, ps)
            fp, fn, obs, tbar = rates(d_, y[cal], t_, n_)
            if fp + U <= alpha_fp and fn + U <= alpha_fn:
                feas.append((obs, mfp, mfn, band, ps))
        fx = []
        for j in range(T):
            dj = (P[cal, j] >= 0.5).astype(int)
            fp = float(np.mean((dj == 1) & (y[cal] == 0))); fn = float(np.mean((dj == 0) & (y[cal] == 1)))
            if fp + U <= alpha_fp and fn + U <= alpha_fn:
                fx.append((LAM_T * GRID[j], j))

        if not feas:
            rows.append(dict(rep=rep, ok=False)); continue
        feas.sort(key=lambda r: r[0]); _, mfp, mfn, band, ps = feas[0]
        t_, d_, n_ = apply_band(P, te, band, ps)
        fp_te, fn_te, _, tbar_te = rates(d_, y[te], t_, n_)
        if fx:
            fx.sort(key=lambda r: r[0]); jstar = fx[0][1]
            dj = (P[te, jstar] >= 0.5).astype(int)
            fp_f = float(np.mean((dj == 1) & (y[te] == 0))); fn_f = float(np.mean((dj == 0) & (y[te] == 1)))
            t_f = GRID[jstar]; ok_f = (fp_f <= alpha_fp) and (fn_f <= alpha_fn)
        else:
            t_f, ok_f = np.nan, None
        rows.append(dict(rep=rep, ok=True, pstar=ps, mu_FP=mfp, mu_FN=mfn, t=tbar_te,
                         fp=fp_te, fn=fn_te, feas_te=(fp_te <= alpha_fp and fn_te <= alpha_fn),
                         t_fix=t_f, feas_fix=ok_f, U=U))
    return rows


all_rows = []
print(f"\n{'alpha':>6} {'slack':>6} | {'cal-feasible':>13} {'test-holds':>11} {'ctrl time':>10} | "
      f"{'fixed feasible':>15} {'fixed time':>11}")
for use_slack in (True, False):
    for a in (0.10, 0.15, 0.20):
        rr = run(a, a, use_slack)
        good = [r for r in rr if r["ok"]]
        if good:
            okte = sum(1 for r in good if r["feas_te"])
            tc = np.mean([r["t"] for r in good])
            gf = [r for r in good if r["feas_fix"] is not None]
            okf = sum(1 for r in gf if r["feas_fix"]); tf = np.mean([r["t_fix"] for r in gf]) if gf else float("nan")
            print(f"{a:>6.2f} {str(use_slack):>6} | {len(good):>6}/{R:<6} {okte:>5}/{len(good):<5} "
                  f"{tc:>9.1f}h | {okf:>8}/{len(gf):<6} {tf:>10.1f}h")
            for r in good:
                all_rows.append((a, use_slack, r["rep"], round(r["pstar"], 2), round(r["mu_FP"], 3),
                                 round(r["mu_FN"], 3), round(r["t"], 2), round(r["fp"], 4),
                                 round(r["fn"], 4), r["feas_te"], round(r["t_fix"], 1), r["feas_fix"],
                                 round(r["U"], 4)))
        else:
            print(f"{a:>6.2f} {str(use_slack):>6} | {0:>6}/{R:<6} {'--':>11} {'--':>10} | {'--':>15} {'--':>11}")

with open(os.path.join(OUT, "table_constrained_control.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["alpha", "thm3_slack", "repeat", "selected_pstar", "mu_FP", "mu_FN",
                "test_mean_time_h", "test_FP", "test_FN", "test_feasible",
                "fixed_T_star_h", "fixed_feasible", "U_n"])
    for r in all_rows: w.writerow(r)
print("\nwrote table_constrained_control.csv")
print("Reading: with the Theorem-3 uniform slack the certificate is conservative -- at alpha=0.10 the")
print("slack consumes most of the risk budget, so many calibration splits admit NO feasible policy.")
print("That conservatism is a real deployment cost of certification, and we report it rather than")
print("quietly dropping the infeasible splits.")
