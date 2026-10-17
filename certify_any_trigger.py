#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stage 3Q: the certificates are not tied to our rule -- certify the published triggers.

Theorems 3 and 4 certify a selection from ANY finite family of stopping rules: Theorem 3 needs
exchangeable units, Theorem 4 needs exchangeable conditions and a family that does not depend on
the calibration conditions. So far the paper certified only its own fit-free margin family. Here
the family is a published trigger swept over its delay weight -- CALIMERA, EconomyGamma, and
TEASER under ml_edm's cost criterion -- plus the endpoint assay, ordered by conservativeness
(large delay weight = early = least conservative first).

Part A (organoids, condition drift). Same leave-condition-out posterior and the same 50 random
9/9 condition splits as constrained_control_crc.py (Table 6). Each trigger is fitted out-of-fold
by condition (5-fold GroupKFold over the 18 conditions, as in etsc_baseline_comparison.py), so
every condition's losses come from a trigger that never saw that condition -- the same
cross-fitting caveat that already applies to the posterior. Selections: none; Hoeffding over
organoids (Theorem 3); Learn-then-Test over organoids; conformal risk control over conditions
(Theorem 4). Our margin family is run through the same code and must reproduce Table 6. A control
splits organoids at random (exchangeable units) to separate "the procedure fails" from "drift
breaks it".

Part B (UCR, exchangeable units). The 40 two-class datasets with the frozen MiniRocket posterior
of etsc_ucr_binary40.py; each trigger fitted on the out-of-fold TRAIN posteriors at twelve delay
weights and applied to TEST; test series split at random into calibration and deployment halves
(R = 50, the seeds of ucr40_analysis.py); none / Hoeffding / Learn-then-Test, as in Table S15.

Outputs: posterior/table_certify_triggers_organoid.csv, posterior/table_certify_triggers_ucr40.csv
Run with the miniconda interpreter (ml_edm, aeon):  python -I certify_any_trigger.py [A|B|AB]
"""
import os, sys, csv, time, random, warnings
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "_external", "ml_edm", "src"))
sys.path.insert(0, HERE)
for v in ("NUMBA_NUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(v, "2")
import numpy as np
from scipy.stats import binom

OUT = os.path.join(HERE, "posterior"); NPZ = os.path.join(OUT, "ucr40")
DELTA = 0.10; R = 50; B = 1.0; N_BOOT = 2000


# ------------------------------------------------------------------ selection rules (as in constrained_control_crc.py)
def crc_select(L, alpha):
    """L: [S, m] per-condition losses, index increasing = more conservative."""
    Lt = np.maximum.accumulate(L[::-1], axis=0)[::-1]
    ok = np.where((Lt.sum(1) + B) / (L.shape[1] + 1) <= alpha)[0]
    return int(ok[0]) if len(ok) else None


def crc_two_sided(Lfp, Lfn, alpha):
    a, b = crc_select(Lfp, alpha), crc_select(Lfn, alpha)
    return None if (a is None or b is None) else max(a, b)


def hoeffding_select(fp, fn, alpha, n, slack=True):
    U = np.sqrt(np.log(2 * len(fp) / DELTA) / (2 * n)) if slack else 0.0
    ok = np.where((fp + U <= alpha) & (fn + U <= alpha))[0]
    return int(ok[0]) if len(ok) else None


def ltt_select(fp, fn, alpha, n):
    sel = None
    for s in range(len(fp) - 1, -1, -1):
        pv = max(binom.cdf(int(round(fp[s] * n)), n, alpha), binom.cdf(int(round(fn[s] * n)), n, alpha))
        if pv <= DELTA: sel = s
        else: break
    return sel


def unit_select(arm, fp, fn, alpha, n):
    if arm == "ltt": return ltt_select(fp, fn, alpha, n)
    return hoeffding_select(fp, fn, alpha, n, slack=(arm == "hoeffding"))


# ------------------------------------------------------------------ triggers
def make_teaser_cached():
    from ml_edm.trigger import TEASER

    class TEASERCached(TEASER):
        _cache = {}

        def _fit_master_clf(self, probas, masks_pos_probas):
            key = (probas.tobytes(), masks_pos_probas.tobytes())
            if key not in TEASERCached._cache:
                TEASERCached._cache[key] = super()._fit_master_clf(probas, masks_pos_probas)
            return TEASERCached._cache[key]
    return TEASERCached


def trigger_specs():
    from ml_edm.trigger import EconomyGamma, CALIMERA
    return [("CALIMERA", CALIMERA, {}), ("EconomyGamma", EconomyGamma, dict(nb_intervals=5)),
            ("TEASER-cost", make_teaser_cached(), dict(objective="avg_cost", n_jobs=1))]


def fit_predict(cls, kwargs, lam, GI, full_len, probas_tr, y_tr, probas_te):
    """ml_edm trigger fitted on (probas_tr, y_tr) at delay weight lam; returns per-unit (tau, call)
    on probas_te. X carries the true series length (ml_edm reads it from X.shape[1])."""
    from ml_edm.cost_matrices import CostMatrices
    cm = CostMatrices(timestamps=GI, n_classes=2, misclf_cost=2.0, delay_cost=(lambda t, L_=lam: 2 * L_ * t), alpha=0.5)
    random.seed(0)
    trig = cls(timestamps=GI, **kwargs)
    trig.fit(np.zeros((len(y_tr), int(full_len))), probas_tr, y_tr, cm)
    n = len(probas_te); fired = np.zeros(n, bool); call = np.zeros(n, int); tau = np.full(n, float(GI[-1]))
    for j, length in enumerate(GI):
        p_arg = probas_te[:, :j + 1, :] if trig.require_past_probas else probas_te[:, j, :]
        fire = np.asarray(trig.predict(np.zeros((n, int(length))), p_arg, cm)) & ~fired
        call[fire] = probas_te[fire, j, :].argmax(-1); tau[fire] = length; fired |= fire
        if fired.all(): break
    rem = ~fired; call[rem] = probas_te[rem, -1, :].argmax(-1)
    return tau, call


def dedupe(members):
    """drop consecutive members that make identical decisions on every unit."""
    out = []
    for t, c in members:
        if not out or not (np.array_equal(t, out[-1][0]) and np.array_equal(c, out[-1][1])): out.append((t, c))
    return out


# ================================================================== Part A: organoids, condition drift
def part_a():
    from sklearn.model_selection import GroupKFold
    d = np.load(os.path.join(OUT, "posterior_data_shared_strat.npz"), allow_pickle=True)
    P = d["P_group"]; y = d["y"].astype(int); GRID = d["GRID"].astype(float); groups = d["groups"]
    N, T = P.shape; GI = GRID.astype(int)
    conds = np.array(sorted(set(groups.tolist()))); C = len(conds)
    cidx = [np.where(groups == c)[0] for c in conds]; w_c = np.array([len(ix) for ix in cidx], float)
    probas = np.stack([1 - P, P], axis=-1)
    folds = list(GroupKFold(5).split(P, y, groups))
    LAMS = np.geomspace(2e-4, 6e-2, 24)[::-1]                      # large weight (early) first

    fams = {}
    # our fit-free margin family, exactly as constrained_control_crc.py
    mem = []
    for m in np.linspace(0.0, 0.5, 51):
        ex = np.abs(P - 0.5) >= m; j = np.where(ex.any(1), ex.argmax(1), T - 1)
        mem.append((GRID[j], (P[np.arange(N), j] >= 0.5).astype(int)))
    fams["constant margin (ours)"] = mem
    t0 = time.time()
    for name, cls, kw in trigger_specs():
        mem = []
        for lam in LAMS:
            tau = np.zeros(N); call = np.zeros(N, int)
            for tr, te in folds:
                tau[te], call[te] = fit_predict(cls, kw, lam, GI, GI[-1], probas[tr], y[tr], probas[te])
            mem.append((tau, call))
        fams[name] = dedupe(mem)
        print(f"  [A] {name}: {len(fams[name])} distinct rules over {len(LAMS)} delay weights "
              f"(mean time {fams[name][0][0].mean():.1f}-{fams[name][-1][0].mean():.1f} h)  [{time.time() - t0:.0f}s]", flush=True)

    def per_unit(mem):                       # + endpoint assay: wait to 72 h and read the label
        tau = np.array([t for t, _ in mem] + [np.full(N, GRID[-1])])
        call = np.array([c for _, c in mem] + [y.copy()])
        return (call == 1) & (y == 0)[None], (call == 0) & (y == 1)[None], tau

    rng = np.random.default_rng(0)
    SPLITS = [rng.permutation(conds) for _ in range(R)]
    ALPHAS = (0.10, 0.15, 0.20, 0.25)
    rows = []
    print("\n[A] organoids, 50 random 9/9 condition splits (test conditions new)")
    for name, mem in fams.items():
        FPu, FNu, TMu = per_unit(mem)
        Lfp = np.array([[FPu[s, ix].mean() for ix in cidx] for s in range(len(TMu))])
        Lfn = np.array([[FNu[s, ix].mean() for ix in cidx] for s in range(len(TMu))])
        Ltm = np.array([[TMu[s, ix].mean() for ix in cidx] for s in range(len(TMu))])
        for alpha in ALPHAS:
            for arm in ("none", "ltt", "hoeffding", "crc-condition"):
                holds = 0; tm = []; fpc = []; fnc = []; inside = []; endpoint = 0
                for perm in SPLITS:
                    ci = np.array([np.where(conds == c)[0][0] for c in perm[:9]])
                    ti = np.array([np.where(conds == c)[0][0] for c in perm[9:]])
                    if arm == "crc-condition":
                        s = crc_two_sided(Lfp[:, ci], Lfn[:, ci], alpha)
                    else:
                        wc = w_c[ci]; n = wc.sum()
                        s = unit_select(arm, (Lfp[:, ci] * wc).sum(1) / n, (Lfn[:, ci] * wc).sum(1) / n, alpha, int(n))
                    if s is None: s = len(TMu) - 1
                    endpoint += s == len(TMu) - 1
                    wt = w_c[ti]
                    fp_p = (Lfp[s, ti] * wt).sum() / wt.sum(); fn_p = (Lfn[s, ti] * wt).sum() / wt.sum()
                    holds += bool(fp_p <= alpha and fn_p <= alpha); tm.append((Ltm[s, ti] * wt).sum() / wt.sum())
                    fpc.append(Lfp[s, ti]); fnc.append(Lfn[s, ti]); inside.append(np.maximum(Lfp[s, ti], Lfn[s, ti]) <= alpha)
                rb = np.random.default_rng(1)
                boots = [np.mean(np.concatenate([inside[i] for i in rb.integers(0, R, R)])) for _ in range(N_BOOT)]
                lo, hi = np.percentile(boots, [2.5, 97.5])
                r = dict(part="A-drift", family=name, alpha=alpha, selection=arm, holds=holds, n_splits=R,
                         mean_time_h=round(float(np.mean(tm)), 2),
                         E_FP_new=round(float(np.concatenate(fpc).mean()), 4), E_FN_new=round(float(np.concatenate(fnc).mean()), 4),
                         frac_cond_within=round(float(np.mean(np.concatenate(inside))), 3),
                         ci_lo=round(float(lo), 3), ci_hi=round(float(hi), 3), endpoint_returned=int(endpoint))
                rows.append(r)
                print(f"  {name:24s} a={alpha:.2f} {arm:>14}: holds {holds:2d}/{R}  time {r['mean_time_h']:5.1f} h  "
                      f"E[FP]/E[FN] {r['E_FP_new']:.3f}/{r['E_FN_new']:.3f}  cond<=a {r['frac_cond_within']:.2f} "
                      f"[{r['ci_lo']:.2f},{r['ci_hi']:.2f}]  endpoint {endpoint}/{R}", flush=True)
        # control: organoids split at random, so units are exchangeable
        rng_c = np.random.default_rng(2)              # the same 50 halves as constrained_control_crc.py
        HALVES = [rng_c.permutation(N) for _ in range(R)]
        for alpha in ALPHAS:
            for arm in ("none", "ltt", "hoeffding"):
                holds = 0; tm = []
                for perm in HALVES:
                    ca, te = perm[:N // 2], perm[N // 2:]
                    s = unit_select(arm, FPu[:, ca].mean(1), FNu[:, ca].mean(1), alpha, len(ca))
                    if s is None: s = len(TMu) - 1
                    holds += bool(FPu[s, te].mean() <= alpha and FNu[s, te].mean() <= alpha); tm.append(TMu[s, te].mean())
                rows.append(dict(part="A-control-exchangeable", family=name, alpha=alpha, selection=arm, holds=holds,
                                 n_splits=R, mean_time_h=round(float(np.mean(tm)), 2)))
                print(f"  {name:24s} a={alpha:.2f} {arm:>14} [random organoid halves]: holds {holds:2d}/{R}  "
                      f"time {np.mean(tm):5.1f} h", flush=True)
    keys = ["part", "family", "alpha", "selection", "holds", "n_splits", "mean_time_h", "E_FP_new", "E_FN_new",
            "frac_cond_within", "ci_lo", "ci_hi", "endpoint_returned"]
    with open(os.path.join(OUT, "table_certify_triggers_organoid.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
    print("wrote table_certify_triggers_organoid.csv")


# ================================================================== Part B: UCR, exchangeable units
def run_ucr(ds):
    z = np.load(os.path.join(NPZ, ds + ".npz"))
    Ptr, Pte, ytr, yte, G = z["Ptr"], z["Pte"], z["ytr"], z["yte"], z["G"]; L = float(z["L"])
    GI = G.astype(int); n = len(yte); T = len(G)
    probas_tr = np.stack([1 - Ptr, Ptr], -1); probas_te = np.stack([1 - Pte, Pte], -1)
    fams = {}
    mem = []
    for m in np.linspace(0.0, 0.49, 50):
        ex = np.abs(Pte - 0.5) >= m; j = np.where(ex.any(1), ex.argmax(1), T - 1)
        mem.append((G[j], (Pte[np.arange(n), j] >= 0.5).astype(int)))
    fams["constant margin (ours)"] = mem
    fams["fixed horizon"] = [(np.full(n, G[j]), (Pte[:, j] >= 0.5).astype(int)) for j in range(T)]
    failed = []
    for name, cls, kw in trigger_specs()[:2]:                      # CALIMERA, EconomyGamma
        mem = []
        for lam in (np.geomspace(1e-3, 5e-2, 12) / (L / 100.0))[::-1]:
            try:
                mem.append(fit_predict(cls, kw, lam, GI, L, probas_tr, ytr, probas_te))
            except Exception:
                failed.append((name, float(lam)))
        if mem: fams[name] = dedupe(mem)
    unit = {k: (np.array([(c == 1) & (yte == 0) for _, c in v]), np.array([(c == 0) & (yte == 1) for _, c in v]),
                np.array([t for t, _ in v])) for k, v in fams.items()}
    out = []
    for alpha in (0.05, 0.10, 0.20):
        for arm in ("none", "hoeffding", "ltt"):
            rng = np.random.default_rng(sum(map(ord, ds)))
            halves = [rng.permutation(n) for _ in range(R)]
            for fam in [k for k in unit if k != "fixed horizon"]:
                FP, FN, TM = unit[fam]; FPf, FNf, TMf = unit["fixed horizon"]
                feas = hold = 0; sv = []
                for perm in halves:
                    ca, te = perm[:n // 2], perm[n // 2:]
                    s = unit_select(arm, FP[:, ca].mean(1), FN[:, ca].mean(1), alpha, len(ca))
                    if s is None: continue
                    feas += 1; hold += bool(FP[s, te].mean() <= alpha and FN[s, te].mean() <= alpha)
                    sf = unit_select(arm, FPf[:, ca].mean(1), FNf[:, ca].mean(1), alpha, len(ca))
                    if sf is not None: sv.append((TMf[sf, te].mean() - TM[s, te].mean()) / L)
                out.append((alpha, arm, fam, ds, n, feas, hold, "" if not sv else round(float(np.mean(sv)), 4)))
    print(f"  [B] {ds:30s} n_test={n:5d}  families {[(k, len(v)) for k, v in fams.items()]}"
          + (f"  failed fits {len(failed)}" if failed else ""), flush=True)
    return out


def part_b(workers=9):
    from concurrent.futures import ProcessPoolExecutor
    from download_ucr_binary import BINARY_UCR
    rows = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for r in ex.map(run_ucr, BINARY_UCR): rows += r
    with open(os.path.join(OUT, "table_certify_triggers_ucr40.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["alpha", "certificate", "family", "dataset", "n_test", "splits_certifiable", "splits_holding",
                    "mean_saving_vs_certified_fixed_fraction_of_length"])
        w.writerows(rows)
    print("\n[B] UCR, 40 datasets, random calibration/deployment halves (exchangeable units)")
    for alpha in (0.05, 0.10, 0.20):
        for arm in ("none", "hoeffding", "ltt"):
            for fam in ("constant margin (ours)", "CALIMERA", "EconomyGamma"):
                rr = [r for r in rows if r[0] == alpha and r[1] == arm and r[2] == fam]
                fe = sum(r[5] for r in rr); ho = sum(r[6] for r in rr); nd = sum(r[5] > 0 for r in rr)
                sv = [r[7] for r in rr if r[7] != ""]
                print(f"  a={alpha:.2f} {arm:>9} {fam:24s}: certifiable on {nd:2d}/40 datasets ({fe:4d} splits), "
                      f"holds {ho}/{fe} = {ho / max(fe, 1):.3f}; mean saving {100 * np.mean(sv) if sv else float('nan'):.1f}%")
    print("wrote table_certify_triggers_ucr40.csv")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    which = sys.argv[1] if len(sys.argv) > 1 else "AB"
    t0 = time.time()
    if "A" in which: part_a()
    if "B" in which: part_b()
    print(f"total {time.time() - t0:.0f}s")
