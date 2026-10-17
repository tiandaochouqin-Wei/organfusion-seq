#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""(c2/c3) The two stopping baselines missing from seq_baselines.py, which the
reviewer listed explicitly (Sec. 13.1): static selective classification with a
waiting mechanism, and a reinforcement-learning stopping policy.

Both are evaluated on the SAME leave-condition-out posterior process as every other
policy (posterior_data_grouped.npz's P_group, matching seq_baselines.py's main
protocol as of 2026-09-28), so any difference is attributable to the stopping rule
alone. Added to the existing five (fixed-time / constant-margin / SPRT / myopic-VOI /
Bayes-DP) for a complete Table.

  static selective classification + wait (Chow's rule at a FIXED horizon T0):
      at T0, decide if |p_T0 - 0.5| >= k, else abstain and wait to the horizon and
      decide there. This is the natural "static selective prediction" baseline: the
      abstention is static (one fixed decision time + one fixed rule), unlike our
      controller whose stopping time is a function of the path.

  RL stopping (tabular Q-learning on the discretised belief state):
      state = (binned p_t, t), actions = {stop, continue}; reward = -(decision cost)
      on stop, -(lambda * dt) on continue. Trained by Q-learning on the TRAINING
      organoids' real posterior paths, evaluated on held-out organoids. Tabular Q is
      the right RL baseline here: the state is 2-D and low-cardinality, so a deep
      policy would add variance without adding capacity.

Outputs:
  posterior/table_extra_baselines.csv
"""
import os, csv, sys, numpy as np
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sklearn.model_selection import GroupKFold

OUT = "posterior"; LAM = 0.004
d = np.load(os.path.join(OUT, "posterior_data_grouped.npz"), allow_pickle=True)
P = d["P_group"]; y = d["y"].astype(int); GRID = d["GRID"].astype(float)
groups = d["groups"]
N, T = P.shape
print(f"N={N}, horizons={GRID.tolist()} (leave-condition-out posterior)")


def isotonic_envelope(t, e):
    o = np.argsort(t); return np.asarray(t)[o], np.minimum.accumulate(np.asarray(e)[o])


def time_at_error(t, e, targets):
    t2, e2 = isotonic_envelope(t, e)
    out = []
    for te in targets:
        out.append(np.nan if (te < e2.min() or te > e2.max()) else float(np.interp(-te, -e2, t2)))
    return out


# ---------------- baseline: static selective classification + wait ----------------
def static_selective(T0_idx, k):
    times = np.empty(N); dec = np.empty(N, int)
    conf = np.abs(P[:, T0_idx] - 0.5) >= k
    times[conf] = GRID[T0_idx]; dec[conf] = (P[conf, T0_idx] >= 0.5).astype(int)
    times[~conf] = GRID[-1]; dec[~conf] = (P[~conf, -1] >= 0.5).astype(int)
    return times, dec


# ---------------- baseline: tabular Q-learning stopping ----------------
NB = 20
bins = np.linspace(0, 1, NB + 1)

def qlearn(train_idx, lam=LAM, episodes=30, alpha=0.25, seed=0):
    """lam is swept by the caller: the delay weight is baked into the reward, so a
    single Q gives a single operating point. Sweeping lam traces the RL frontier --
    without this the RL baseline would be handicapped relative to the threshold
    rules, whose frontiers come free from sweeping their margin."""
    rng = np.random.default_rng(seed)
    Q = np.zeros((NB, T, 2))          # state (p-bin, t) x action {0=stop,1=continue}
    b_all = np.clip(np.digitize(P, bins) - 1, 0, NB - 1)
    for _ in range(episodes):
        for i in rng.permutation(train_idx):
            for j in range(T):
                s = b_all[i, j]
                r_stop = -float(int(P[i, j] >= 0.5) != int(y[i]))  # decision cost now
                if j == T - 1:
                    Q[s, j, 0] += alpha * (r_stop - Q[s, j, 0])
                    Q[s, j, 1] += alpha * (r_stop - Q[s, j, 1])   # cannot continue
                    continue
                r_cont = -lam * (GRID[j + 1] - GRID[j])
                s2 = b_all[i, j + 1]
                Q[s, j, 0] += alpha * (r_stop - Q[s, j, 0])
                Q[s, j, 1] += alpha * (r_cont + Q[s2, j + 1].max() - Q[s, j, 1])
    return Q


def qpolicy(Q, idx):
    b_all = np.clip(np.digitize(P, bins) - 1, 0, NB - 1)
    times = np.empty(len(idx)); dec = np.empty(len(idx), int)
    for n, i in enumerate(idx):
        j_stop = T - 1
        for j in range(T - 1):
            if Q[b_all[i, j], j, 0] >= Q[b_all[i, j], j, 1]:
                j_stop = j; break
        times[n] = GRID[j_stop]; dec[n] = int(P[i, j_stop] >= 0.5)
    return times, dec


# RL is a LEARNED policy, so it must be fit and evaluated out-of-fold like any model,
# and its delay weight must be swept to give it a frontier comparable to the others.
print("\ntraining tabular-Q stopping policy (leave-condition-out, sweeping lambda) ...")
RL_LAMS = np.geomspace(2e-4, 5e-2, 12)
rl_t_list, rl_e_list = [], []
for lam in RL_LAMS:
    q_t = np.zeros(N); q_d = np.zeros(N, int)
    for tr, te in GroupKFold(5).split(P, y, groups):
        Q = qlearn(tr, lam=lam)
        tt, dd = qpolicy(Q, te)
        q_t[te] = tt; q_d[te] = dd
    rl_t_list.append(q_t.mean()); rl_e_list.append((q_d != y).mean())
    print(f"  lam={lam:.5f}  err={rl_e_list[-1]:.4f}  mean_t={rl_t_list[-1]:.2f}h")
rl_t_list = np.array(rl_t_list); rl_e_list = np.array(rl_e_list)
i_ref = int(np.argmin(np.abs(RL_LAMS - LAM)))
rl_err, rl_t = rl_e_list[i_ref], rl_t_list[i_ref]

# ---------------- frontiers ----------------
targets = np.array([0.12, 0.16, 0.20, 0.24])
rows = []

# static selective: sweep both the fixed decision time and the confidence threshold
ss_t, ss_e = [], []
for T0 in range(T):
    for k in np.linspace(0.0, 0.45, 25):
        t_, d_ = static_selective(T0, k)
        ss_t.append(t_.mean()); ss_e.append((d_ != y).mean())
ss_min = isotonic_envelope(ss_t, ss_e)[1].min()
rows.append(("static_selective_wait", ss_min, time_at_error(ss_t, ss_e, targets)))

# RL: frontier from the lambda sweep, same treatment as the threshold rules
rl_min = isotonic_envelope(rl_t_list, rl_e_list)[1].min()
rows.append(("rl_qlearning_stopping", rl_min, time_at_error(rl_t_list, rl_e_list, targets)))

with open(os.path.join(OUT, "table_extra_baselines.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["policy", "min_achievable_err"] + [f"time_at_err_{t:.2f}" for t in targets]
               + ["single_op_err", "single_op_time_h"])
    for nm, mn, ts in rows:
        extra = [round(rl_err, 4), round(rl_t, 2)] if nm.startswith("rl") else ["", ""]
        w.writerow([nm, round(float(mn), 4)] + [("" if np.isnan(x) else round(x, 2)) for x in ts] + extra)

print("\n=== added baselines (same posterior as Table seqbase) ===")
for nm, mn, ts in rows:
    s = "  ".join("n/a" if np.isnan(x) else f"{x:.1f}h" for x in ts)
    print(f"  {nm:<26} min_err={mn:.4f}   time@err[0.12,0.16,0.20,0.24] = {s}")
print(f"\n  (compare against table_seq_baselines_matched.csv, same leave-condition-out posterior)")
print("wrote table_extra_baselines.csv")
