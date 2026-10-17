#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Phase 2/3 fix (3): a strict train / calibration / test protocol.

Why this exists (reviewer point, Sec. 10 of the revision plan): the deployed
evaluation cross-fits the posterior and then sweeps the band margin on the SAME
organoids it reports. Even with leakage-free grouping, the operating point is chosen
using the data it is scored on -- exactly the "do not tune calibration parameters on
the test set" objection. The finite-sample certificate (Thm. 3) is stated for a
policy selected on an independent calibration sample, so the reported protocol should
match the theory.

Protocol (the reviewer's outer/inner structure, at the finest grouping the public
data supports -- 18 (cell x drug x conc) conditions; there are no well/plate ids):
  TRAIN  (10 conditions) : fit the shared prefix encoder (build_posterior_shared.py)
  CAL     (4 conditions) : fit probability calibration (temperature / isotonic / beta)
                           AND select the band margin m* AND select the best fixed T*
  TEST    (4 conditions) : evaluated once, never used for any choice
Repeated over R random condition partitions to report spread rather than one lucky
split. Every set is a disjoint group of whole conditions, so no condition (hence no
plate/well) is shared across roles.

Outputs:
  posterior/table_3way_protocol.csv
"""
import os, csv, sys, numpy as np
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import torch
import torch.nn as nn
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

SEED = 0
torch.manual_seed(SEED); np.random.seed(SEED)
torch.use_deterministic_algorithms(True)
DEV = "cpu"; OUT = "posterior"
R = 50                      # random condition partitions (10 until Stage 3P; the first 10 are unchanged)
N_TR, N_CAL, N_TE = 10, 4, 4

d = np.load(os.path.join(OUT, "posterior_data_shared.npz"), allow_pickle=True)
xgrid = d["xgrid"]; y = d["y"].astype(np.float32); GRID = d["GRID"].astype(float)
drug = d["drug"]; conc = d["conc"].astype(float); groups = d["groups"]
N, T = xgrid.shape
DRUGS = ["Stauro", "Ner", "Lap"]
tnorm = np.tile((GRID / 72.0)[None, :], (N, 1))
inc = np.concatenate([np.zeros((N, 1)), np.diff(xgrid, axis=1)], axis=1)
SEQ = np.stack([xgrid, tnorm, inc], axis=-1).astype(np.float32)
doh = np.array([np.eye(3)[DRUGS.index(x)] for x in drug], dtype=np.float32)
COND = np.concatenate([doh, np.log1p(conc)[:, None].astype(np.float32)], axis=1)
conds = np.array(sorted(set(groups.tolist())))
print(f"N={N}, conditions={len(conds)}  protocol: train {N_TR} / cal {N_CAL} / test {N_TE}, R={R}")


class SharedPosterior(nn.Module):
    def __init__(self, d_hidden=32, d_cond=8):
        super().__init__()
        self.gru = nn.GRU(input_size=3, hidden_size=d_hidden, batch_first=True)
        self.cond = nn.Linear(4, d_cond)
        self.head = nn.Linear(d_hidden + d_cond, 1)

    def forward(self, seq, cond):
        h, _ = self.gru(seq)
        e = torch.tanh(self.cond(cond)).unsqueeze(1).expand(-1, h.size(1), -1)
        return self.head(torch.cat([h, e], -1)).squeeze(-1)


def fit(tr, epochs=300, lr=0.01):
    m = SharedPosterior().to(DEV)
    opt = torch.optim.Adam(m.parameters(), lr=lr); lf = nn.BCEWithLogitsLoss()
    xs = torch.tensor(SEQ[tr]); cs = torch.tensor(COND[tr])
    ys = torch.tensor(y[tr]).unsqueeze(1).expand(-1, T)
    m.train()
    for _ in range(epochs):
        opt.zero_grad(); lf(m(xs, cs), ys).backward(); opt.step()
    m.eval(); return m


def logits_of(m, idx):
    with torch.no_grad():
        return m(torch.tensor(SEQ[idx]), torch.tensor(COND[idx])).cpu().numpy()


# ---- calibrators: fit on CAL only, applied to TEST ----
def cal_temperature(lo_cal, y_cal):
    lo = torch.tensor(lo_cal.ravel(), dtype=torch.float32)
    yy = torch.tensor(np.repeat(y_cal, lo_cal.shape[1]), dtype=torch.float32)
    logT = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([logT], lr=0.1, max_iter=60)
    lf = nn.BCEWithLogitsLoss()
    def closure():
        opt.zero_grad(); l = lf(lo / torch.exp(logT), yy); l.backward(); return l
    opt.step(closure)
    Tv = float(torch.exp(logT).detach())
    return lambda L: 1.0 / (1.0 + np.exp(-L / Tv)), f"T={Tv:.2f}"


def cal_isotonic(lo_cal, y_cal):
    p = 1 / (1 + np.exp(-lo_cal.ravel()))
    ir = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1)
    ir.fit(p, np.repeat(y_cal, lo_cal.shape[1]))
    return lambda L: ir.predict(np.clip(1 / (1 + np.exp(-L)), 0, 1).ravel()).reshape(L.shape), "isotonic"


def cal_beta(lo_cal, y_cal):
    p = np.clip(1 / (1 + np.exp(-lo_cal.ravel())), 1e-6, 1 - 1e-6)
    X = np.column_stack([np.log(p), -np.log(1 - p)])
    lr = LogisticRegression(max_iter=1000).fit(X, np.repeat(y_cal, lo_cal.shape[1]))
    def f(L):
        q = np.clip(1 / (1 + np.exp(-L)), 1e-6, 1 - 1e-6)
        Xt = np.column_stack([np.log(q.ravel()), -np.log(1 - q.ravel())])
        return lr.predict_proba(Xt)[:, 1].reshape(L.shape)
    return f, "beta"


def controller(P, m, p_star=0.5):
    times = np.empty(len(P)); dec = np.empty(len(P), int)
    for i in range(len(P)):
        ex = np.where(np.abs(P[i] - p_star) >= m)[0]
        j = ex[0] if len(ex) else T - 1
        times[i] = GRID[j]; dec[i] = int(P[i, j] >= p_star)
    return times, dec


rng = np.random.default_rng(SEED)
MARGINS = np.linspace(0.02, 0.45, 30)
rows = []
for rep in range(1, R + 1):
    perm = rng.permutation(conds)
    c_tr, c_cal, c_te = perm[:N_TR], perm[N_TR:N_TR + N_CAL], perm[N_TR + N_CAL:N_TR + N_CAL + N_TE]
    tr = np.where(np.isin(groups, c_tr))[0]
    ca = np.where(np.isin(groups, c_cal))[0]
    te = np.where(np.isin(groups, c_te))[0]

    model = fit(tr)
    lo_cal, lo_te = logits_of(model, ca), logits_of(model, te)

    for calname, mk in [("temperature", cal_temperature), ("isotonic", cal_isotonic), ("beta", cal_beta)]:
        f, tag = mk(lo_cal, y[ca])
        P_cal, P_te = f(lo_cal), f(lo_te)

        # --- ALL choices made on CAL only ---
        cost_cal = []
        for m in MARGINS:
            t_, d_ = controller(P_cal, m)
            cost_cal.append(((d_ != y[ca]).mean(), t_.mean()))
        err_cal = np.array([c[0] for c in cost_cal]); tt_cal = np.array([c[1] for c in cost_cal])
        lam = 0.004
        m_star = MARGINS[int(np.argmin(err_cal + lam * tt_cal))]
        fe_cal = np.array([((P_cal[:, j] >= 0.5).astype(int) != y[ca]).mean() for j in range(T)])
        T_star_j = int(np.argmin(fe_cal + lam * GRID))

        # --- TEST: evaluated once, no choices made here ---
        t_te, d_te = controller(P_te, m_star)
        err_ctrl = (d_te != y[te]).mean(); time_ctrl = t_te.mean()
        err_fix = ((P_te[:, T_star_j] >= 0.5).astype(int) != y[te]).mean(); time_fix = GRID[T_star_j]
        J_ctrl = err_ctrl + lam * time_ctrl; J_fix = err_fix + lam * time_fix
        auc_te = roc_auc_score(y[te], P_te[:, 4]) if len(set(y[te])) > 1 else np.nan
        rows.append((rep, calname, m_star, float(GRID[T_star_j]), err_ctrl, time_ctrl,
                     err_fix, time_fix, J_ctrl, J_fix, J_fix - J_ctrl, auc_te))
    r = [x for x in rows if x[0] == rep]
    print(f"  rep {rep:>2}: " + "  ".join(f"{x[1][:4]}:J_gain={x[10]:+.4f}" for x in r))

with open(os.path.join(OUT, "table_3way_protocol.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["repeat", "calibrator", "m_star_from_cal", "T_star_from_cal_h", "test_err_ctrl",
                "test_time_ctrl_h", "test_err_fixed", "test_time_fixed_h", "test_J_ctrl",
                "test_J_fixed", "test_J_gain", "test_auc_24h"])
    for r in rows:
        w.writerow([r[0], r[1]] + [round(float(x), 5) for x in r[2:]])

print("\n=== TEST-set results (all choices made on CAL; TEST used once) ===")
for calname in ["temperature", "isotonic", "beta"]:
    sub = [r for r in rows if r[1] == calname]
    g = np.array([r[10] for r in sub]); win = (g > 0).sum()
    ec = np.array([r[4] for r in sub]); tc = np.array([r[5] for r in sub])
    ef = np.array([r[6] for r in sub]); tf = np.array([r[7] for r in sub])
    print(f"  {calname:<12} J_gain mean={g.mean():+.4f} (sd {g.std():.4f})  controller wins {win}/{len(sub)} splits |"
          f"  ctrl err={ec.mean():.3f}@{tc.mean():.1f}h   fixed err={ef.mean():.3f}@{tf.mean():.1f}h")
print("\nwrote table_3way_protocol.csv")
