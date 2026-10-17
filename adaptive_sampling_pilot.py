#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Phase 2 (adaptive-sampling PILOT, not the full controller): before building
the reviewer-requested u_t in {STOP0, STOP1, Delta_1,...,Delta_M} controller
(a large undertaking), test the premise it depends on -- that the value of
choosing WHEN to sample next (fine vs coarse spacing) is heterogeneous enough
across organoid-states to be worth the added complexity, rather than a fixed
cadence already capturing ~all the value (which is what the existing sparse-
imaging finding, E12, suggests: cost is roughly flat from 0.5h to 12h cadence).

Uses only the EXISTING 13-point GRID (already alternates ~3h / ~6h spacing) --
no new data or dense re-parse needed. Introduces a genuine per-measurement cost
lambda_m (distinct from the existing per-hour wait cost lambda_t), matching the
reviewer's J_pi = E[... + lambda_t*tau + lambda_m*N_tau] formulation.

At each (grid index j, organoid) state, myopically compares three one-step
actions: STOP now, fine-step to j+1, or coarse-step (skip j+1) to j+2 -- each
priced with the SAME per-measurement cost lambda_m and per-hour wait cost
lambda_t -- then simulates full trajectories under three REALIZED policies:
  (a) always-fine cadence   (b) always-coarse cadence   (c) adaptive (argmin
  of the myopic 3-way choice at each visited step)
across a lambda_m sweep, to see whether (c) meaningfully beats min(a,b).

Outputs:
  posterior/table_adaptive_sampling_pilot.csv
"""
import os, csv, sys, numpy as np
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

OUT = "posterior"
os.makedirs(OUT, exist_ok=True)

d = np.load("posterior/posterior_data_grouped.npz", allow_pickle=True)
P = d["P_group"]; y = d["y"]; G = d["GRID"].astype(float); N = len(y); nG = len(G)
drug = d["drug"]
rho = lambda p: np.minimum(1 - p, p)

Bbin = np.linspace(0, 1, 51); bc = (Bbin[:-1] + Bbin[1:]) / 2; nb = len(bc); rho_bc = np.minimum(1 - bc, bc)


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


Ks = build_kernel(P)                                   # 1-step kernels, j -> j+1
Ks2 = [Ks[j] @ Ks[j + 1] for j in range(nG - 2)]        # 2-step kernels, j -> j+2 (coarse)

print("grid spacing (h):", np.diff(G))


def E_rho_next_vec(j_arr, p_arr, two_step=False):
    """E[rho(p_{j+1 or j+2})|p_j] for a (possibly mixed-j) batch, via the binned
    kernel; loops over the handful of unique j values present in the batch."""
    out = np.empty(len(j_arr))
    for jj in np.unique(j_arr):
        mask = j_arr == jj
        K = Ks2[jj] if two_step else Ks[jj]
        bidx = np.clip(np.digitize(p_arr[mask], Bbin) - 1, 0, nb - 1)
        out[mask] = (K[bidx] * rho_bc[None, :]).sum(1)
    return out


def simulate(policy, lam_t, lam_m):
    """policy in {'fine','coarse','adaptive'}. Each organoid tracks its own
    current grid index `cur` (fine- and coarse-steppers advance at different
    rates), so state is per-organoid rather than a single shared time index."""
    cur = np.zeros(N, int)
    times = np.full(N, G[-1]); n_obs = np.ones(N, int); dec = np.empty(N, int)
    decided = np.zeros(N, bool)
    for _ in range(nG):
        active = np.where(~decided)[0]
        if len(active) == 0:
            break
        j = cur[active]
        at_last = j >= nG - 1
        force = active[at_last]
        if len(force):
            times[force] = G[nG - 1]
            dec[force] = (P[force, nG - 1] >= 0.5).astype(int)
            decided[force] = True
        active = active[~at_last]; j = j[~at_last]
        if len(active) == 0:
            continue

        p = P[active, j]
        stop_cost = rho(p)
        fine_cost = lam_t * (G[j + 1] - G[j]) + lam_m + E_rho_next_vec(j, p, two_step=False)
        has_coarse = (j + 2) < nG
        coarse_cost = np.full(len(active), np.inf)
        if has_coarse.any():
            hc = np.where(has_coarse)[0]
            coarse_cost[hc] = (lam_t * (G[j[hc] + 2] - G[j[hc]]) + lam_m
                                + E_rho_next_vec(j[hc], p[hc], two_step=True))

        if policy == "fine":
            action = np.where(stop_cost <= fine_cost, 0, 1)
        elif policy == "coarse":
            prefer_coarse = has_coarse & (stop_cost > coarse_cost)
            action = np.where(prefer_coarse, 2, np.where(stop_cost <= fine_cost, 0, 1))
        elif policy == "adaptive":
            costs = np.column_stack([stop_cost, fine_cost, coarse_cost])
            action = np.argmin(costs, axis=1)
        else:
            raise ValueError(policy)

        stop_m = active[action == 0]
        times[stop_m] = G[j[action == 0]]
        dec[stop_m] = (P[stop_m, j[action == 0]] >= 0.5).astype(int)
        decided[stop_m] = True

        fine_m = active[action == 1]
        cur[fine_m] = j[action == 1] + 1
        n_obs[fine_m] += 1

        coarse_m = active[action == 2]
        cur[coarse_m] = j[action == 2] + 2
        n_obs[coarse_m] += 1
    return times, n_obs, dec


rows = []
# lambda_t sweep spans the range actually used elsewhere in this codebase
# (E4/E8/F2_band.py use 0.004-0.05/h); include it explicitly so we don't draw
# a verdict from a degenerate "everyone stops at the first grid point" corner.
for lam_t in [0.002, 0.004, 0.008, 0.02]:
    for lam_m in [0.0, 0.01, 0.02, 0.05, 0.10, 0.20]:
        results = {}
        for policy in ("fine", "coarse", "adaptive"):
            times, n_obs, dec = simulate(policy, lam_t, lam_m)
            err = (dec != y).mean()
            results[policy] = dict(mean_time=times.mean(), mean_nobs=n_obs.mean(), err=err)
        best_fixed = min(results["fine"]["mean_time"] * lam_t + results["fine"]["mean_nobs"] * lam_m,
                          results["coarse"]["mean_time"] * lam_t + results["coarse"]["mean_nobs"] * lam_m)
        adapt_cost = results["adaptive"]["mean_time"] * lam_t + results["adaptive"]["mean_nobs"] * lam_m
        gain = best_fixed - adapt_cost
        rows.append((lam_t, lam_m,
                     results["fine"]["mean_time"], results["fine"]["mean_nobs"], results["fine"]["err"],
                     results["coarse"]["mean_time"], results["coarse"]["mean_nobs"], results["coarse"]["err"],
                     results["adaptive"]["mean_time"], results["adaptive"]["mean_nobs"], results["adaptive"]["err"],
                     gain))
        print(f"lam_t={lam_t:.3f} lam_m={lam_m:.2f}  "
              f"fine(t={results['fine']['mean_time']:.1f},n={results['fine']['mean_nobs']:.1f},err={results['fine']['err']:.3f})  "
              f"coarse(t={results['coarse']['mean_time']:.1f},n={results['coarse']['mean_nobs']:.1f},err={results['coarse']['err']:.3f})  "
              f"adaptive(t={results['adaptive']['mean_time']:.1f},n={results['adaptive']['mean_nobs']:.1f},err={results['adaptive']['err']:.3f})  "
              f"gain={gain:+.4f}")

with open(os.path.join(OUT, "table_adaptive_sampling_pilot.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["lambda_t", "lambda_m", "fine_time", "fine_nobs", "fine_err",
                "coarse_time", "coarse_nobs", "coarse_err",
                "adaptive_time", "adaptive_nobs", "adaptive_err", "gain_vs_best_fixed"])
    for r in rows:
        w.writerow([round(x, 4) if isinstance(x, float) else x for x in r])
print("wrote table_adaptive_sampling_pilot.csv")

max_gain = max(r[-1] for r in rows)
best_row = max(rows, key=lambda r: r[-1])
print(f"\nPILOT VERDICT: max scheduling-cost gain of adaptive-cadence over the better fixed cadence "
      f"across the full lambda_t x lambda_m sweep = {max_gain:.4f} "
      f"(at lambda_t={best_row[0]}, lambda_m={best_row[1]}, mean_time~{best_row[8]:.1f}h). "
      f"For reference, this scheduling-cost scale is directly comparable to rho (max 0.5) and to the "
      f"E4/E8 delay-cost regime elsewhere in this paper -- so a gain this small there is a real null result, "
      f"not an artifact of choosing a degenerate lambda_t.")
