#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Per-operating-point detail for the shift experiment (E7/E11 of rebuild_chain_shared.py).

rebuild_chain_shared.py only stored the dominance COUNT per held-out stratum ("4/4"). This script
re-runs the identical transfer loop (same producer, same seed, same four margins, same matched
fixed-time interpolation, never extrapolating) and stores, per stratum and margin, the controller's
error and time, the matched fixed time and the hours saved -- the data behind the Fig. 4(a) matrix.
Built-in check: the AUC@24h and the dominance counts must reproduce table_chain_shared_transfer.csv.

Output: posterior/table_shift_matrix.csv
"""
import os, sys, csv, numpy as np
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
from sklearn.metrics import roc_auc_score
from producers import load_data, fit_producer, logits_of, controller

torch.use_deterministic_algorithms(True)
OUT = "posterior"
D = load_data()
y = D["y"]; GRID = D["GRID"]; N, T = D["N"], D["T"]
drug = D["drug"]; conc = D["conc"]; cell = D["cell"]


def envelope(t, e):
    o = np.argsort(t); return t[o], np.minimum.accumulate(e[o])


ref = {}
with open(os.path.join(OUT, "table_chain_shared_transfer.csv"), encoding="utf-8") as f:
    for r in csv.DictReader(f): ref[r["setting"]] = r

rows = []


def transfer(mask_tr, mask_te, name):
    tr = np.where(mask_tr)[0]; te = np.where(mask_te)[0]
    if len(set(y[tr].tolist())) < 2 or len(set(y[te].tolist())) < 2 or len(te) < 30:
        return
    m = fit_producer(D, tr, enc="GRU", pool="last", seed=0)
    # Stage 3R: a drug absent from training has an all-zero one-hot column there, so its weights in the
    # condition layer never receive a gradient and keep their random initial values; feeding the test
    # organoids' indicator through them would add an arbitrary offset. An unseen drug is therefore encoded
    # as "unknown" (all-zero drug indicator) at test time. Dose (log concentration) is continuous and kept;
    # the cell line is not an input of the producer. (Until 3R the leave-drug rows used the untrained
    # weights, so they no longer reproduce table_chain_shared_transfer.csv; the other rows still do.)
    Dte = dict(D); C = D["COND"].copy()
    unseen = [k for k in range(3) if D["COND"][tr, k].max() == 0]
    if unseen: C[np.ix_(te, unseen)] = 0.0
    Dte["COND"] = C
    Pte = 1 / (1 + np.exp(-logits_of(m, Dte, te)))
    yte = y[te]
    fe = np.array([((Pte[:, j] >= 0.5).astype(int) != yte).mean() for j in range(T)])
    ft2, fe2 = envelope(GRID.copy(), fe)
    auc24 = roc_auc_score(yte, Pte[:, int(np.where(GRID == 24)[0][0])])
    ndom = tot = 0
    for k, mm in enumerate(np.linspace(0.10, 0.42, 4), 1):
        t_, d_ = controller(Pte, GRID, mm)
        err = (d_ != yte).mean(); tc = t_.mean()
        if err < fe2.min() or err > fe2.max():
            tfix = saved = np.nan
        else:
            tfix = float(np.interp(-err, -fe2, ft2)); saved = tfix - tc; tot += 1; ndom += saved > 0
        rows.append((name, len(te), round(auc24, 3), k, round(mm, 3), round(err, 4), round(tc, 2),
                     "" if tfix != tfix else round(tfix, 2), "" if saved != saved else round(saved, 2)))
    r0 = ref.get(name)
    ok = r0 is not None and abs(float(r0["auc_24h"]) - round(auc24, 3)) < 1e-9 and r0["dominance"] == f"{ndom}/{tot}"
    print(f"  {name:<28} AUC@24h={auc24:.3f} dominance {ndom}/{tot}   reproduces table: {ok}")


for a, b in [("MCF-7", "BT-474"), ("BT-474", "MCF-7")]:
    transfer(cell == a, cell == b, f"cross-cell {a}->{b}")
for dd in sorted(set(drug.tolist())):
    transfer(drug != dd, drug == dd, f"leave-drug {dd}")
for cc in sorted(set(conc.tolist())):
    transfer(conc != cc, conc == cc, f"leave-conc {cc}uM")

with open(os.path.join(OUT, "table_shift_matrix.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["setting", "n_test", "auc_24h", "op", "margin", "err", "t_ctrl", "t_fixed_matched", "hours_saved"])
    for r in rows: w.writerow(r)
print("wrote posterior/table_shift_matrix.csv")
