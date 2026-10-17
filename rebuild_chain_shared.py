#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""(a) Recompute the WHOLE downstream chain on the shared prefix encoder, so the
headline numbers rest on a coherent posterior rather than 13 independently-fitted
per-horizon models.

Recomputes, under BOTH cross-fitting schemes (stratified organoid-level = the
protocol the submitted numbers used; leave-condition-out = leakage-free):
  E2   frontier + 9 matched operating points (hours saved, PPV/NPV/coverage)
  E13  backward-DP continuation band
  E8   cost sweep vs best fixed-time
  E10  finite-sample uniform certificate (coverage failure rate, width)
  E7   cross-cell-line transfer   (retrained per direction)
  E11  leave-one-drug / leave-one-concentration transfer (retrained per stratum)
and prints old (per-horizon LR) vs new (shared encoder) side by side.

Outputs:
  posterior/posterior_data_shared_strat.npz
  posterior/table_chain_shared_matched_ops.csv
  posterior/table_chain_shared_summary.csv
  posterior/table_chain_shared_transfer.csv
"""
import os, csv, sys, numpy as np
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # portable py: script dir not on path
import torch
from sklearn.model_selection import StratifiedKFold, GroupKFold
from sklearn.metrics import roc_auc_score
from producers import load_data, fit_producer, logits_of, controller, set_seed

torch.use_deterministic_algorithms(True)
OUT = "posterior"; LAM = 0.004
D = load_data()
y = D["y"]; GRID = D["GRID"]; N, T = D["N"], D["T"]
groups = D["groups"]; drug = D["drug"]; conc = D["conc"]; cell = D["cell"]
P_old = D["P_legacy"]
print(f"N={N} T={T} conditions={len(set(groups.tolist()))}")


def crossfit(splitter_iter, tag):
    P = np.zeros((N, T))
    for f, (tr, te) in enumerate(splitter_iter, 1):
        m = fit_producer(D, tr, enc="GRU", pool="last", seed=0)
        P[te] = 1 / (1 + np.exp(-logits_of(m, D, te)))
        print(f"    [{tag}] fold {f}/5 (train {len(tr)}, test {len(te)})")
    return P


print("\ncross-fitting shared encoder ...")
P_strat = crossfit(StratifiedKFold(5, shuffle=True, random_state=0).split(D["SEQ"], y), "strat")
P_group = crossfit(GroupKFold(5).split(D["SEQ"], y, groups), "group")
np.savez(os.path.join(OUT, "posterior_data_shared_strat.npz"),
         P=P_strat, P_group=P_group, xgrid=D["xgrid"], y=y, drug=drug, conc=conc,
         cell=cell, GRID=GRID.astype(int), groups=groups)


# ---------------- E2: frontier + matched operating points ----------------
def frontier(P):
    ms = np.linspace(0.0, 0.48, 49)
    ct, ce = [], []
    for m in ms:
        t_, d_ = controller(P, GRID, m)
        ct.append(t_.mean()); ce.append((d_ != y).mean())
    return np.array(ct), np.array(ce)


def fixed_frontier(P):
    return GRID.copy(), np.array([((P[:, j] >= 0.5).astype(int) != y).mean() for j in range(T)])


def envelope(t, e):
    o = np.argsort(t); return t[o], np.minimum.accumulate(e[o])


def matched_ops(P, tag):
    ft, fe = fixed_frontier(P)
    ft2, fe2 = envelope(ft, fe)
    rows = []
    for k, m in enumerate(np.linspace(0.10, 0.42, 9), 1):
        t_, d_ = controller(P, GRID, m)
        err = (d_ != y).mean(); mt = t_.mean()
        # matched fixed-time only if reachable (never extrapolate)
        if err < fe2.min() or err > fe2.max():
            tfix, saved = np.nan, np.nan
        else:
            tfix = float(np.interp(-err, -fe2, ft2)); saved = tfix - mt
        ex = np.abs(P - 0.5).max(1) >= m
        pos = ex & (d_ == 1); neg = ex & (d_ == 0)
        ppv = (y[pos] == 1).mean() if pos.sum() else np.nan
        npv = (y[neg] == 0).mean() if neg.sum() else np.nan
        rows.append((tag, f"OP{k}", round(m, 3), round(err, 4), round(mt, 2),
                     round(tfix, 2) if tfix == tfix else "", round(saved, 2) if saved == saved else "",
                     round(ex.mean(), 3), round(ppv, 3) if ppv == ppv else "",
                     round(npv, 3) if npv == npv else ""))
    return rows


rows_ops = []
for tag, P in [("shared_strat", P_strat), ("shared_group", P_group), ("legacy_perhorizon", P_old)]:
    rows_ops += matched_ops(P, tag)
with open(os.path.join(OUT, "table_chain_shared_matched_ops.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f); w.writerow(["posterior", "OP", "margin", "err", "t_ctrl", "t_fixed_matched",
                                    "hours_saved", "coverage", "PPV", "NPV"])
    for r in rows_ops: w.writerow(r)

print("\n=== E2 matched operating points: hours saved (dominance count) ===")
for tag in ["legacy_perhorizon", "shared_strat", "shared_group"]:
    sub = [r for r in rows_ops if r[0] == tag]
    sv = [r[6] for r in sub if r[6] != ""]
    ndom = sum(1 for s in sv if s > 0)
    print(f"  {tag:<20} dominance {ndom}/{len(sv)} reachable OPs   max saved={max(sv) if sv else float('nan'):.1f}h"
          f"   err range [{min(r[3] for r in sub):.3f},{max(r[3] for r in sub):.3f}]")


# ---------------- E13: backward-DP band ----------------
Bbin = np.linspace(0, 1, 51); bc = (Bbin[:-1] + Bbin[1:]) / 2; nb = len(bc)
rho_bc = np.minimum(1 - bc, bc)

def kernels(P):
    Ks = []
    for j in range(T - 1):
        K = np.zeros((nb, nb))
        a = np.clip(np.digitize(P[:, j], Bbin) - 1, 0, nb - 1)
        b = np.clip(np.digitize(P[:, j + 1], Bbin) - 1, 0, nb - 1)
        for u, v in zip(a, b): K[u, v] += 1
        K += 1e-3; K /= K.sum(1, keepdims=True); Ks.append(K)
    return Ks

def dp_band(P, lam):
    Ks = kernels(P); V = rho_bc.copy(); out = {}
    for j in range(T - 2, -1, -1):
        cont = lam * (GRID[j + 1] - GRID[j]) + Ks[j] @ V
        cs = np.where(cont < rho_bc - 1e-9)[0]
        if len(cs): out[int(GRID[j])] = (bc[cs.min()], bc[cs.max()])
        V = np.minimum(rho_bc, cont)
    return out

print("\n=== E13 DP band width (lambda=0.02): tightening toward horizon? ===")
for tag, P in [("legacy_perhorizon", P_old), ("shared_strat", P_strat)]:
    bd = dp_band(P, 0.02); ts = sorted(bd)
    if len(ts) >= 2:
        w18 = bd[min(ts, key=lambda t: abs(t - 18))]; w60 = bd[min(ts, key=lambda t: abs(t - 60))]
        print(f"  {tag:<20} width@~18h={w18[1]-w18[0]:.3f}  width@~60h={w60[1]-w60[0]:.3f}")
    else:
        print(f"  {tag:<20} band empty")


# ---------------- E8: cost sweep ----------------
print("\n=== E8 cost sweep: controller vs best fixed-time ===")
rows_sum = []
for tag, P in [("legacy_perhorizon", P_old), ("shared_strat", P_strat), ("shared_group", P_group)]:
    ct, ce = frontier(P); ft, fe = fixed_frontier(P)
    for lam in [0.002, 0.004, 0.01, 0.02]:
        Jc = (ce + lam * ct).min(); Jf = (fe + lam * ft).min()
        rows_sum.append((tag, "cost", lam, round(Jc, 4), round(Jf, 4), round(Jf - Jc, 4)))
    g = [r for r in rows_sum if r[0] == tag and r[1] == "cost"]
    print(f"  {tag:<20} " + "  ".join(f"lam={r[2]}:gain={r[5]:+.4f}" for r in g))


# ---------------- E10: finite-sample uniform certificate ----------------
print("\n=== E10 certificate: empirical coverage-failure rate (target <= delta=0.10) ===")
rng = np.random.default_rng(0)
for tag, P in [("legacy_perhorizon", P_old), ("shared_strat", P_strat)]:
    mgrid = np.linspace(0.10, 0.42, 9)
    costs = np.zeros((N, len(mgrid)))
    for k, m in enumerate(mgrid):
        t_, d_ = controller(P, GRID, m)
        costs[:, k] = (d_ != y) + LAM * t_
    Jpop = costs.mean(0)
    for n in [150, 1000]:
        fails = 0; B = 300
        for _ in range(B):
            idx = rng.integers(0, N, n)
            Jhat = costs[idx].mean(0)
            khat = int(np.argmin(Jhat))
            Bnd = (1.0 + LAM * 72)
            Un = Bnd * np.sqrt(np.log(2 * len(mgrid) / 0.10) / (2 * n))
            if Jpop[khat] > Jhat[khat] + Un: fails += 1
        rows_sum.append((tag, "cert", n, round(fails / B, 4), round(Un, 4), ""))
        print(f"  {tag:<20} n={n:<5} failure_rate={fails/B:.3f}  width U_n={Un:.3f}")


# ---------------- E7/E11: transfer (retrain per stratum) ----------------
print("\n=== E7/E11 transfer with shared encoder (retrained per held-out stratum) ===")
rows_tr = []
def transfer(mask_tr, mask_te, name):
    tr = np.where(mask_tr)[0]; te = np.where(mask_te)[0]
    if len(set(y[tr].tolist())) < 2 or len(set(y[te].tolist())) < 2 or len(te) < 30:
        return
    m = fit_producer(D, tr, enc="GRU", pool="last", seed=0)
    Pte = 1 / (1 + np.exp(-logits_of(m, D, te)))
    yte = y[te]
    fe = np.array([((Pte[:, j] >= 0.5).astype(int) != yte).mean() for j in range(T)])
    ft2, fe2 = envelope(GRID.copy(), fe)
    ndom = 0; tot = 0
    for mm in np.linspace(0.10, 0.42, 4):
        t_, d_ = controller(Pte, GRID, mm)
        err = (d_ != yte).mean()
        if err < fe2.min() or err > fe2.max(): continue
        tfix = float(np.interp(-err, -fe2, ft2)); tot += 1
        if tfix - t_.mean() > 0: ndom += 1
    auc24 = roc_auc_score(yte, Pte[:, int(np.where(GRID == 24)[0][0])])
    rows_tr.append((name, len(te), round(auc24, 3), f"{ndom}/{tot}"))
    print(f"  {name:<28} n_test={len(te):<5} AUC@24h={auc24:.3f}  dominance {ndom}/{tot}")

for a, b in [("MCF-7", "BT-474"), ("BT-474", "MCF-7")]:
    transfer(cell == a, cell == b, f"cross-cell {a}->{b}")
for dd in sorted(set(drug.tolist())):
    transfer(drug != dd, drug == dd, f"leave-drug {dd}")
for cc in sorted(set(conc.tolist())):
    transfer(conc != cc, conc == cc, f"leave-conc {cc}uM")

with open(os.path.join(OUT, "table_chain_shared_transfer.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f); w.writerow(["setting", "n_test", "auc_24h", "dominance"])
    for r in rows_tr: w.writerow(r)
with open(os.path.join(OUT, "table_chain_shared_summary.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f); w.writerow(["posterior", "kind", "param", "val1", "val2", "gain"])
    for r in rows_sum: w.writerow(r)
print("\nwrote table_chain_shared_{matched_ops,summary,transfer}.csv + posterior_data_shared_strat.npz")
