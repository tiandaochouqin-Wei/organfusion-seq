"""Running time of CertStop and of the published triggers on the organoid posterior (Stage 3V).

Same posterior and folds as certify_any_trigger.py (leave-condition-out shared-encoder posterior,
5-fold GroupKFold over the 18 conditions). For every method we time
  * building its rule family on the training folds: CertStop's Lagrangian family (transition kernels +
    backward induction for the 156 multiplier pairs of constrained_control.py), CertStop's fit-free
    margin family (51 margins, nothing to fit), and each ml_edm trigger fitted at the 24 delay weights
    certify_any_trigger.py sweeps;
  * fitting one rule (one delay weight / one multiplier pair);
  * deciding for one unit at deployment (all horizons, averaged over the test folds).
Selection among family members is the same counting step for every method and is not timed.
Wall-clock seconds with BLAS pinned to two threads; the mean over the five folds is reported.

Run with the miniconda interpreter (ml_edm, aeon):  python -I runtime_comparison.py
Output: posterior/table_runtime.csv
"""
import os
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(v, "2")
import sys, csv, time, random, warnings
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "_external", "ml_edm", "src"))
warnings.filterwarnings("ignore")
import numpy as np
from sklearn.model_selection import GroupKFold
from certify_any_trigger import trigger_specs, fit_predict

OUT = os.path.join(HERE, "posterior")
d = np.load(os.path.join(OUT, "posterior_data_shared_strat.npz"), allow_pickle=True)
P = d["P_group"]; y = d["y"].astype(int); GRID = d["GRID"].astype(float); groups = d["groups"]
N, T = P.shape; GI = GRID.astype(int)
probas = np.stack([1 - P, P], axis=-1)
folds = list(GroupKFold(5).split(P, y, groups))
LAMS = np.geomspace(2e-4, 6e-2, 24)[::-1]          # the trigger sweep of certify_any_trigger.py
LAM_ONE = LAMS[12]                                  # one rule: a mid-range delay weight

# ---- CertStop: Lagrangian family, as in constrained_control.py
LAM_T = 0.004
Bbin = np.linspace(0, 1, 51); bc = (Bbin[:-1] + Bbin[1:]) / 2; nb = len(bc)
PSTARS = np.linspace(0.20, 0.80, 13); SCALES = np.geomspace(0.2, 20.0, 12)
CANDS = [(s * p, s * (1 - p)) for p in PSTARS for s in SCALES]


def kernels(idx):
    Ks = []
    for j in range(T - 1):
        K = np.zeros((nb, nb))
        a = np.clip(np.digitize(P[idx, j], Bbin) - 1, 0, nb - 1)
        b = np.clip(np.digitize(P[idx, j + 1], Bbin) - 1, 0, nb - 1)
        np.add.at(K, (a, b), 1.0)
        K += 1e-3; K /= K.sum(1, keepdims=True); Ks.append(K)
    return Ks


def dp_band(Ks, mu_FP, mu_FN):
    rho = np.minimum(mu_FP * (1 - bc), mu_FN * bc); V = rho.copy()
    band = np.zeros((T, 2)); pstar = mu_FP / (mu_FP + mu_FN); band[T - 1] = (pstar, pstar)
    for j in range(T - 2, -1, -1):
        cont = LAM_T * (GRID[j + 1] - GRID[j]) + Ks[j] @ V
        cs = np.where(cont < rho - 1e-12)[0]
        band[j] = (bc[cs.min()], bc[cs.max()]) if len(cs) else (pstar, pstar)
        V = np.minimum(rho, cont)
    return band, pstar


def apply_band(idx, band, pstar):
    Pi = P[idx]; inside = (Pi > band[:, 0]) & (Pi < band[:, 1]); inside[:, -1] = False
    j = np.argmax(~inside, axis=1)
    return GRID[j], (Pi[np.arange(len(idx)), j] >= pstar).astype(int)


def margin_family(idx):
    Pi = P[idx]; out = []
    for m in np.linspace(0.0, 0.5, 51):
        ex = np.abs(Pi - 0.5) >= m; ex[:, -1] = True; j = ex.argmax(1)
        out.append((GRID[j], (Pi[np.arange(len(idx)), j] >= 0.5).astype(int)))
    return out


def timed(f, *a, reps=1):
    t0 = time.perf_counter()
    for _ in range(reps): r = f(*a)
    return (time.perf_counter() - t0) / reps, r


rows = []
# ---------------- CertStop
fam_lag, one_lag, dec_lag, fam_mar, dec_mar = [], [], [], [], []
for tr, te in folds:
    def build_lag():
        Ks = kernels(tr)
        return Ks, [dp_band(Ks, a, b) for a, b in CANDS]
    t, (Ks, bands) = timed(build_lag, reps=3); fam_lag.append(t)
    t, _ = timed(lambda: (kernels(tr), dp_band(kernels(tr), *CANDS[78])), reps=3); one_lag.append(t)
    band, ps = bands[78]
    t, _ = timed(apply_band, te, band, ps, reps=20); dec_lag.append(t / len(te))
    t, _ = timed(margin_family, tr, reps=20); fam_mar.append(t)
    t, _ = timed(lambda: margin_family(te)[30], reps=20); dec_mar.append(t / len(te) / 51)
rows.append(dict(method="CertStop, Lagrangian family", members=len(CANDS), family_s=np.mean(fam_lag),
                 one_rule_s=np.mean(one_lag), decide_us=1e6 * np.mean(dec_lag)))
rows.append(dict(method="CertStop, constant margin", members=51, family_s=np.mean(fam_mar),
                 one_rule_s=0.0, decide_us=1e6 * np.mean(dec_mar)))

# ---------------- published triggers (ml_edm)
for name, cls, kw in trigger_specs():
    if name.startswith("TEASER"):                    # time the plain class: the cached subclass would hide refits
        from ml_edm.trigger import TEASER as cls
    fam, one, dec = [], [], []
    for tr, te in folds:
        t0 = time.perf_counter()
        for lam in LAMS:
            fit_predict(cls, kw, lam, GI, GI[-1], probas[tr], y[tr], probas[te[:1]])
        fam.append(time.perf_counter() - t0)
        t0 = time.perf_counter()
        fit_predict(cls, kw, LAM_ONE, GI, GI[-1], probas[tr], y[tr], probas[te[:1]])
        one.append(time.perf_counter() - t0)
        # decision time: fit once more, then time prediction alone on the whole test fold
        from ml_edm.cost_matrices import CostMatrices
        cm = CostMatrices(timestamps=GI, n_classes=2, misclf_cost=2.0, delay_cost=(lambda t, L_=LAM_ONE: 2 * L_ * t), alpha=0.5)
        random.seed(0); trig = cls(timestamps=GI, **kw)
        trig.fit(np.zeros((len(tr), int(GI[-1]))), probas[tr], y[tr], cm)
        t0 = time.perf_counter()
        for j, length in enumerate(GI):
            p_arg = probas[te][:, :j + 1, :] if trig.require_past_probas else probas[te][:, j, :]
            trig.predict(np.zeros((len(te), int(length))), p_arg, cm)
        dec.append((time.perf_counter() - t0) / len(te))
    label = {"TEASER-cost": "TEASER (cost criterion)"}.get(name, name)
    rows.append(dict(method=label, members=len(LAMS), family_s=np.mean(fam), one_rule_s=np.mean(one),
                     decide_us=1e6 * np.mean(dec)))
    print(f"  {label}: family {np.mean(fam):.2f}s  one rule {np.mean(one):.3f}s  decide {1e6*np.mean(dec):.1f}us", flush=True)

with open(os.path.join(OUT, "table_runtime.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader()
    for r in rows: w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in r.items()})
print(f"\nN={N} units, T={T} horizons, {len(folds)} folds (~{len(folds[0][0])} training units per fold)")
for r in rows:
    print(f"  {r['method']:30s} members {r['members']:4d}  family {r['family_s']:8.3f}s  "
          f"one rule {r['one_rule_s']:7.3f}s  decide/unit {r['decide_us']:9.1f} us")
print("wrote posterior/table_runtime.csv")
