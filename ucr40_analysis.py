#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stage 3P: what the 40-dataset UCR benchmark says (reads etsc_ucr_binary40.py's outputs).

  1. ranks      every rule's realised cost, averaged over the eight delay weights, ranked within
                each dataset; Friedman test, Nemenyi critical difference, Holm-corrected Wilcoxon
                tests of the constant-margin rule against every other rule;
  2. W/T/L      delay weights won / tied / lost (margin 1e-3) against the fixed horizon, both the
                TEST-oracle one of Supplementary Table S14 and the like-for-like one chosen on TRAIN;
  3. certificates  with the frozen posterior, the TEST set is split at random into a calibration
                and a deployment half (exchangeable units, R = 50): no slack / Hoeffding
                (Theorem 3) / Learn-then-Test, alpha_FP = alpha_FN = alpha; how often a rule is
                certifiable, how often the certified rule holds, and what it saves over the
                fastest fixed horizon certified the same way.

A trigger that could not be fitted is scored as never triggering (it decides at the last
horizon), which is what an operator would fall back to.

Outputs: posterior/table_ucr40_ranks.csv, table_ucr40_wtl.csv, table_ucr40_certificates.csv
"""
import os, sys, csv, itertools
import numpy as np
from scipy.stats import friedmanchisquare, wilcoxon, rankdata, binom
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

OUT = "posterior"; NPZ = os.path.join(OUT, "ucr40"); TOL = 1e-3
DELTA = 0.10; R = 50
METHODS = ["fixed(train-selected)", "const_margin(train-selected)", "bayes_dp_band(train-kernel)",
           "EconomyGamma", "CALIMERA", "TEASER", "TEASER-cost"]
SHORT = {"fixed(train-selected)": "fixed horizon", "const_margin(train-selected)": "constant margin (ours)",
         "bayes_dp_band(train-kernel)": "Bellman band (ours)", "EconomyGamma": "EconomyGamma",
         "CALIMERA": "CALIMERA", "TEASER": "TEASER", "TEASER-cost": "TEASER (cost)"}
Q_NEMENYI = {5: 2.728, 6: 2.850, 7: 2.949}          # alpha = 0.05 (Demsar, 2006)

rows = list(csv.DictReader(open(os.path.join(OUT, "table_etsc_ucr_binary40.csv"), encoding="utf-8")))
DATASETS = sorted({r["dataset"] for r in rows})
info = {r["dataset"]: r for r in csv.DictReader(open(os.path.join(OUT, "table_ucr40_datasets.csv"), encoding="utf-8"))}
print(f"{len(DATASETS)} datasets, {len(rows)} rows")


def table(post):
    """J[d, method, lambda], oracle fixed J[d, lambda], number of failed fits per method."""
    J = {}; orc = {}; failed = {m: set() for m in METHODS}
    for d in DATASETS:
        z = np.load(os.path.join(NPZ, d + ("_platt" if post == "platt" else "") + ".npz"))
        last_err = float(((z["Pte"][:, -1] >= 0.5) != z["yte"]).mean()); G = z["G"]
        sub = [r for r in rows if r["dataset"] == d and r["posterior"] == post]
        lams = sorted({float(r["lambda"]) for r in sub})
        orc[d] = np.array([float(next(r for r in sub if float(r["lambda"]) == l)["J_fixed_oracle"]) for l in lams])
        for m in METHODS:
            v = []
            for l in lams:
                r = next(r for r in sub if r["method"] == m and float(r["lambda"]) == l)
                if r["J_method"] == "":
                    failed[m].add(d); v.append(l * G[-1] + last_err)
                else: v.append(float(r["J_method"]))
            J[d, m] = np.array(v)
    return J, orc, failed


def holm(p):
    o = np.argsort(p); adj = np.empty(len(p)); run = 0.0
    for k, i in enumerate(o):
        run = max(run, (len(p) - k) * p[i]); adj[i] = min(1.0, run)
    return adj


rank_rows = []; wtl_rows = []
for post in ("raw", "platt"):
    J, orc, failed = table(post)
    M = np.array([[J[d, m].mean() for m in METHODS] for d in DATASETS])            # [D, K] mean cost
    rk = np.array([rankdata(np.round(r, 6)) for r in M]); avg = rk.mean(0)
    stat, p = friedmanchisquare(*[M[:, k] for k in range(len(METHODS))])
    cd = Q_NEMENYI[len(METHODS)] * np.sqrt(len(METHODS) * (len(METHODS) + 1) / (6 * len(DATASETS)))
    print(f"\n===== posterior: {post} =====  Friedman chi2={stat:.1f}, p={p:.2e};  Nemenyi CD(0.05)={cd:.3f}")
    ours = METHODS.index("const_margin(train-selected)")
    pw = []
    for k in range(len(METHODS)):
        if k == ours: pw.append(np.nan); continue
        dlt = M[:, ours] - M[:, k]
        pw.append(1.0 if np.allclose(dlt, 0) else wilcoxon(dlt, zero_method="wilcox").pvalue)
    adj = np.full(len(METHODS), np.nan); idx = [k for k in range(len(METHODS)) if k != ours]
    adj[idx] = holm(np.array([pw[k] for k in idx]))
    for k in np.argsort(avg):
        m = METHODS[k]
        better = int((M[:, ours] < M[:, k] - 1e-9).sum()); worse = int((M[:, ours] > M[:, k] + 1e-9).sum())
        print(f"  {SHORT[m]:24s} mean rank {avg[k]:.2f}   best on {int((rk[:, k] == rk.min(1)).sum()):2d}   "
              f"failed fits {len(failed[m]):2d}   ours better/worse {better:2d}/{worse:2d}   "
              f"Wilcoxon-Holm p={adj[k]:.3f}" if k != ours else
              f"  {SHORT[m]:24s} mean rank {avg[k]:.2f}   best on {int((rk[:, k] == rk.min(1)).sum()):2d}   "
              f"failed fits {len(failed[m]):2d}")
        rank_rows.append((post, m, round(float(avg[k]), 3), int((rk[:, k] == rk.min(1)).sum()), len(failed[m]),
                          "" if k == ours else better, "" if k == ours else worse,
                          "" if k == ours else round(float(adj[k]), 4), round(float(stat), 2), f"{p:.3g}", round(float(cd), 3)))
    # ---- W/T/L against the fixed horizon ----
    print("  delay weights won/tied/lost against the fixed horizon, summed over datasets; datasets with more wins than losses")
    for m in METHODS[1:]:
        for ref_name, ref in (("oracle(test)", orc), ("train-selected", {d: J[d, METHODS[0]] for d in DATASETS})):
            W = T = L = 0; dw = dl = dmaj = 0; gains = []
            for d in DATASETS:
                g = ref[d] - J[d, m]; w = int((g > TOL).sum()); l = int((g < -TOL).sum())
                W += w; L += l; T += len(g) - w - l; dw += w > l; dl += l > w; dmaj += w >= 5; gains.append(g.mean())
                if ref_name == "oracle(test)":
                    wtl_rows.append((post, d, m, w, len(g) - w - l, l, round(float(g.max()), 4), round(float(g.min()), 4),
                                     round(float(g.mean()), 5)))
            print(f"    {SHORT[m]:24s} vs {ref_name:14s}: {W:3d}/{T:3d}/{L:3d}   datasets W>L {dw:2d}, L>W {dl:2d}, "
                  f"won on >=5 of 8 weights {dmaj:2d}   mean gain {np.mean(gains):+.4f}")

with open(os.path.join(OUT, "table_ucr40_ranks.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["posterior", "method", "mean_rank", "datasets_best", "datasets_fit_failed", "margin_rule_better_on",
                "margin_rule_worse_on", "wilcoxon_holm_p_vs_margin_rule", "friedman_chi2", "friedman_p", "nemenyi_cd_0.05"])
    w.writerows(rank_rows)
with open(os.path.join(OUT, "table_ucr40_wtl.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["posterior", "dataset", "method", "wins", "ties", "losses", "best_gain", "worst_gain", "mean_gain"])
    w.writerows(wtl_rows)

# ------------------------------------------------------------------ certificates, exchangeable units
MS = np.linspace(0.0, 0.49, 50)


def families(P, y, G):
    fam = {}
    ex = [np.abs(P - 0.5) >= m for m in MS]
    j = [np.where(e.any(1), e.argmax(1), P.shape[1] - 1) for e in ex]
    dm = [(P[np.arange(len(P)), jj] >= 0.5).astype(int) for jj in j]
    fam["margin"] = (np.array([(d == 1) & (y == 0) for d in dm]), np.array([(d == 0) & (y == 1) for d in dm]),
                     np.array([G[jj] for jj in j]))
    df = [(P[:, t] >= 0.5).astype(int) for t in range(P.shape[1])]
    fam["fixed"] = (np.array([(d == 1) & (y == 0) for d in df]), np.array([(d == 0) & (y == 1) for d in df]),
                    np.array([np.full(len(y), G[t]) for t in range(P.shape[1])]))
    return fam                      # index increasing = more conservative (later) in both families


def pick(arm, fp, fn, alpha, n):
    S = len(fp)
    if arm == "ltt":
        sel = None
        for s in range(S - 1, -1, -1):
            pv = max(binom.cdf(int(round(fp[s] * n)), n, alpha), binom.cdf(int(round(fn[s] * n)), n, alpha))
            if pv <= DELTA: sel = s
            else: break
        return sel
    U = np.sqrt(np.log(2 * S / DELTA) / (2 * n)) if arm == "hoeffding" else 0.0
    ok = np.where((fp + U <= alpha) & (fn + U <= alpha))[0]
    return int(ok[0]) if len(ok) else None


cert_rows = []
print("\n===== certificates on exchangeable units (random halves of TEST, R=50, delta=0.10), raw posterior =====")
for alpha in (0.05, 0.10, 0.20):
    for arm in ("no-slack", "hoeffding", "ltt"):
        tot_feas = tot_hold = n_ds = 0; saved = []; ds_hold = []
        for d in DATASETS:
            z = np.load(os.path.join(NPZ, d + ".npz")); P, y, G = z["Pte"], z["yte"], z["G"]; L = float(z["L"])
            fam = families(P, y, G); rng = np.random.default_rng(abs(hash(d)) % (2**32) if False else sum(map(ord, d)))
            feas = hold = 0; sv = []
            for _ in range(R):
                perm = rng.permutation(len(y)); ca, te = perm[:len(y) // 2], perm[len(y) // 2:]
                FP, FN, TM = fam["margin"]
                s = pick(arm, FP[:, ca].mean(1), FN[:, ca].mean(1), alpha, len(ca))
                if s is None: continue
                feas += 1; hold += bool(FP[s, te].mean() <= alpha and FN[s, te].mean() <= alpha)
                FPf, FNf, TMf = fam["fixed"]
                sf = pick(arm, FPf[:, ca].mean(1), FNf[:, ca].mean(1), alpha, len(ca))
                if sf is not None: sv.append((TMf[sf, te].mean() - TM[s, te].mean()) / L)
            cert_rows.append((alpha, arm, d, len(y), feas, hold, "" if not sv else round(float(np.mean(sv)), 4)))
            tot_feas += feas; tot_hold += hold; n_ds += feas > 0
            if feas: ds_hold.append(hold / feas)
            saved += sv
        print(f"  alpha={alpha:.2f} {arm:>10}: certifiable on {n_ds:2d}/40 datasets ({tot_feas:4d} of {40 * R} splits); "
              f"holds on {tot_hold}/{tot_feas} = {tot_hold / max(tot_feas, 1):.3f}; datasets with holds >= 90%: "
              f"{sum(h >= 0.9 for h in ds_hold)}/{len(ds_hold)}; mean saving over certified fixed horizon "
              f"{100 * np.mean(saved) if saved else float('nan'):.1f}% of the series")
with open(os.path.join(OUT, "table_ucr40_certificates.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["alpha", "certificate", "dataset", "n_test", "splits_certifiable", "splits_holding",
                "mean_saving_vs_certified_fixed_fraction_of_length"])
    w.writerows(cert_rows)
print("\nwrote table_ucr40_ranks.csv, table_ucr40_wtl.csv, table_ucr40_certificates.csv")
