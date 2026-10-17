r"""Real-data mini-plots for Fig. 1 (Stage 3AC).

Writes fig_overview_data.tex, a set of TikZ macros that fig_overview_tkde.tex draws inside its panel
boxes, so that every mini-plot of the framework diagram shows the organoid cohort itself:
  \panelA  normalized dry-mass trajectories x(t)/x(6) of six organoids (three sensitive, three resistant)
  \panelB  the leave-condition-out posterior paths p_t of the same six organoids (posterior_data_grouped.npz)
  \panelC  three members of the Lagrangian band family (backward induction on kernels from the cohort)
  \panelD  all 156 candidates on D_sel: mean decision time against risk + Hoeffding slack, alpha, selected rule
  \panelE  the selected band with the posterior path of a test organoid (new condition) up to its stopping time
C-E follow Algorithm 1 on one 9/9 condition split (kernels from D_gen, selection on D_sel, as in Table 6).
Same parsing, label and data-integrity filter as build_posterior_grouped.py; same kernels, band recursion and
candidate grid as runtime_comparison.py / constrained_control.py.

    python fig_overview_data.py      (needs openpyxl; then pdflatex fig_overview_tkde.tex)
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import DATA_DIR  # $ORGANFUSION_DATA, or ./datasets (see README)
import os, io
import numpy as np
import openpyxl

HERE = os.path.dirname(os.path.abspath(__file__)); os.chdir(HERE)
XLSX = os.path.join(DATA_DIR, "Tebon_SupplementaryData_MOESM6.xlsx")
DMAP = {"Staurosporine": "Stauro", "Neratinib": "Ner", "Lapatinib": "Lap"}


def parse(ws, cell):
    rows = list(ws.iter_rows(values_only=True))
    ci = [i for i, r in enumerate(rows) if isinstance(r[1], str) and any(d in r[1] for d in DMAP)]
    out = []
    for k, idx in enumerate(ci):
        end = ci[k + 1] if k + 1 < len(ci) else len(rows)
        lab = rows[idx][1].split()
        conc = float(lab[0]); drug = DMAP[lab[1]]
        blk = [r for r in (rows[j] for j in range(idx, end)) if isinstance(r[2], (int, float)) and r[2] <= 72.5]
        t = np.array([r[2] for r in blk], float)
        nc = max(len(r) for r in blk)
        for c in range(3, nc):
            m = np.array([(r[c] if c < len(r) and isinstance(r[c], (int, float)) else np.nan) for r in blk], float)
            if np.isfinite(m).sum() >= 100:
                out.append((drug, conc, cell, t, m))
    return out


def rmax(t, m):
    ok = np.isfinite(m)
    if ok.sum() < 2: return np.inf
    b = np.interp(6, t[ok], m[ok]); return np.inf if b <= 1e-3 else float(np.nanmax(m[ok]) / b)


wb = openpyxl.load_workbook(XLSX, read_only=True)
recs = parse(wb["Supp Fig 9 MCF"], "MCF-7") + parse(wb["Supp Fig 9 BT"], "BT-474")
recs = [r for r in recs if rmax(r[3], r[4]) <= 5.0]
d = np.load(os.path.join("posterior", "posterior_data_grouped.npz"), allow_pickle=True)
P = d["P_group"]; y = d["y"].astype(int); GRID = d["GRID"].astype(float); N, T = P.shape
assert len(recs) == N
assert all(r[0] == a and r[1] == b and r[2] == c for r, a, b, c in zip(recs, d["drug"], d["conc"], d["cell"]))
TT = np.arange(6.0, 72.01, 1.0)
X = np.array([np.interp(TT, r[3][np.isfinite(r[4])], r[4][np.isfinite(r[4])]) for r in recs])
X = X / X[:, :1]
assert ((X[:, -1] < 0.90).astype(int) == y).all()          # label x(72)/x(6) < 0.9 reproduced

# ---------------------------------------------------------------- one split exactly as in Algorithm 1 / Table 6
# 9 calibration and 9 test conditions; the calibration units split at random into D_gen (kernels) and D_sel
# (selection, slack U_n with n_sel units); alpha_FP = alpha_FN = 0.15; the online example is a test unit.
LAM_T = 0.004
Bbin = np.linspace(0, 1, 51); bc = (Bbin[:-1] + Bbin[1:]) / 2; nb = len(bc)
PSTARS = np.linspace(0.20, 0.80, 13); SCALES = np.geomspace(0.2, 20.0, 12)
CANDS = [(s * p, s * (1 - p)) for p in PSTARS for s in SCALES]
ALPHA, DELTA = 0.15, 0.10
groups = d["groups"]; conds = np.unique(groups)


def kernels(idx):
    Ks = []
    for j in range(T - 1):
        K = np.zeros((nb, nb))
        a_ = np.clip(np.digitize(P[idx, j], Bbin) - 1, 0, nb - 1); b_ = np.clip(np.digitize(P[idx, j + 1], Bbin) - 1, 0, nb - 1)
        np.add.at(K, (a_, b_), 1.0); K += 1e-3; K /= K.sum(1, keepdims=True); Ks.append(K)
    return Ks


def dp_band(Ks, mu_FP, mu_FN):
    rho = np.minimum(mu_FP * (1 - bc), mu_FN * bc); V = rho.copy()
    band = np.zeros((T, 2)); ps = mu_FP / (mu_FP + mu_FN); band[T - 1] = (ps, ps)
    for j in range(T - 2, -1, -1):
        cont = LAM_T * (GRID[j + 1] - GRID[j]) + Ks[j] @ V
        cs = np.where(cont < rho - 1e-12)[0]
        band[j] = (bc[cs.min()], bc[cs.max()]) if len(cs) else (ps, ps)
        V = np.minimum(rho, cont)
    return band, ps


def apply_band(band, ps):                                # Deploy on every unit
    inside = (P > band[:, 0]) & (P < band[:, 1]); inside[:, -1] = False
    j = np.argmax(~inside, axis=1)
    return j, (P[np.arange(N), j] >= ps).astype(int)


for SEED in range(100):
    rng = np.random.default_rng(SEED)
    cal_c = rng.choice(conds, 9, replace=False)
    cal = np.where(np.isin(groups, cal_c))[0]; te = np.where(~np.isin(groups, cal_c))[0]
    perm = rng.permutation(cal); gen = np.sort(perm[:len(cal) // 2]); seln = np.sort(perm[len(cal) // 2:])
    Ks = kernels(gen)
    U = np.sqrt(np.log(2 * len(CANDS) / DELTA) / (2 * len(seln)))
    cand = []
    for mf, mn in CANDS:
        band, ps = dp_band(Ks, mf, mn); j, dec = apply_band(band, ps)
        fp = np.mean((dec[seln] == 1) & (y[seln] == 0)); fn = np.mean((dec[seln] == 0) & (y[seln] == 1))
        cand.append(dict(band=band, ps=ps, j=j, dec=dec, t=GRID[j[seln]].mean(), risk=max(fp, fn) + U, s=mf + mn))
    feas = [c for c in cand if c["risk"] <= ALPHA]
    if feas: break
sel = min(feas, key=lambda c: c["t"])
fpt = np.mean((sel["dec"][te] == 1) & (y[te] == 0)); fnt = np.mean((sel["dec"][te] == 0) & (y[te] == 1))
print(f"split seed {SEED}: |D_gen|={len(gen)} |D_sel|={len(seln)} |test|={len(te)}  U_n={U:.4f}  "
      f"feasible={len(feas)}/{len(cand)}  selected p*={sel['ps']:.2f} scale={sel['s']:.2f} "
      f"mean time on D_sel={sel['t']:.1f} h  risk+U={sel['risk']:.3f}  test FP/FN={fpt:.3f}/{fnt:.3f}")

# ---------------------------------------------------------------- example organoids for A/B (calibration units)
first_out = lambda i, lo, hi: next((jj for jj in range(T) if P[i, jj] <= lo or P[i, jj] >= hi), T - 1)
fo = np.array([first_out(i, 0.2, 0.8) for i in range(N)])
in_cal = np.isin(np.arange(N), cal)
picks = []
for cls, lo_hi in ((1, (0.55, 0.85)), (0, (1.15, 1.70))):
    ok = np.where(in_cal & (y == cls) & (X[:, -1] > lo_hi[0]) & (X[:, -1] < lo_hi[1])
                  & ((P[:, -1] >= 0.5) == (cls == 1)) & (np.abs(np.diff(X, axis=1)).max(1) < 0.08))[0]
    ok = ok[np.argsort(fo[ok], kind="stable")]
    for q in (0.1, 0.5, 0.9):                              # early, middle and late deciders
        picks.append((cls, ok[int(q * (len(ok) - 1))]))
print("A/B organoids:", [(c, int(i), d["drug"][i], d["conc"][i], d["cell"][i], int(GRID[fo[i]])) for c, i in picks])

# online example: a test unit (new condition), sensitive, called correctly by the selected band, exiting upwards
cands_e = [i for i in te if y[i] == 1 and sel["dec"][i] == 1 and 30 <= GRID[sel["j"][i]] <= 48
           and P[i, sel["j"][i]] >= sel["band"][sel["j"][i], 1]]
ex = max(cands_e, key=lambda i: np.ptp(P[i, :sel["j"][i]]))
print("online example (test condition):", ex, d["drug"][ex], d["conc"][ex], d["cell"][ex], "tau =", GRID[sel["j"][ex]], "h")

# ---------------------------------------------------------------- TikZ emission
BLUE = "ours,line width=0.7pt"; PURP = "cal,line width=0.7pt,dash pattern=on 2.6pt off 1.2pt"
f2 = lambda v: f"{v:.3f}"


def poly(xs, ys):
    return " -- ".join(f"({f2(a)},{f2(b)})" for a, b in zip(xs, ys))


def lin(v, a, b, lo, hi):
    return lo + (np.asarray(v, float) - a) / (b - a) * (hi - lo)


XL, XR, YB, YT = -1.32, 1.32, -0.54, 0.54                # drawing area inside the 2.84 x 1.25 cm boxes
out = ["% generated by fig_overview_data.py -- do not edit"]

# A: trajectories (subsampled every 2 h)
Xa = X[:, ::2]; Ta = TT[::2]
lo = min(Xa[i].min() for _, i in picks); hi = max(Xa[i].max() for _, i in picks)
pad = 0.04 * (hi - lo); lo -= pad; hi += pad
cmds = []
for cls, i in picks:
    cmds.append(f"\\draw[{BLUE if cls == 1 else PURP}] {poly(lin(Ta, 6, 72, -0.70, 0.70), lin(Xa[i], lo, hi, YB, YT))};")
out.append("\\newcommand{\\panelA}{" + "\n".join(cmds) + "}")
out.append(f"\\newcommand{{\\panelAthr}}{{{f2(lin(0.9, lo, hi, YB, YT))}}}")

# B: posterior paths of the same organoids
cmds = [f"\\draw[gry,line width=0.5pt,dash pattern=on 0.6pt off 1.6pt] ({f2(XL)},0) -- ({f2(XR)},0);"]
for cls, i in picks:
    cmds.append(f"\\draw[{BLUE if cls == 1 else PURP}] {poly(lin(GRID, 12, 72, XL, XR), lin(P[i], 0, 1, YB, YT))};")
out.append("\\newcommand{\\panelB}{" + "\n".join(cmds) + "}")

# C: three members of the family at the selected p* -- a narrower one, the selected one and a wider one
# (bands drawn up to 66 h; the last horizon always stops)
fam = sorted([c for c in cand if abs(c["ps"] - sel["ps"]) < 1e-9], key=lambda c: c["s"])
k0 = next(k for k, c in enumerate(fam) if c is sel)
showC = [fam[k0 - 2], sel, fam[k0 + 2]]
print("panel C scales:", [round(c["s"], 2) for c in showC])
cmds = []
for k, c in enumerate(reversed(showC)):
    bx = lin(GRID[:-1], 12, 66, XL, XR); up = lin(c["band"][:-1, 1], 0, 1, YB, YT); dn = lin(c["band"][:-1, 0], 0, 1, YB, YT)
    shade = ["ours!10", "ours!20", "ours!32"][k]
    cmds.append(f"\\fill[{shade}] {poly(bx, up)} -- {poly(bx[::-1], dn[::-1])} -- cycle;")
for c in showC:
    bx = lin(GRID[:-1], 12, 66, XL, XR)
    cmds.append(f"\\draw[oursd,line width=0.5pt] {poly(bx, lin(c['band'][:-1, 1], 0, 1, YB, YT))};")
    cmds.append(f"\\draw[oursd,line width=0.5pt] {poly(bx, lin(c['band'][:-1, 0], 0, 1, YB, YT))};")
yp = f2(lin(sel["ps"], 0, 1, YB, YT))
cmds.append(f"\\draw[gry,line width=0.5pt,dash pattern=on 0.6pt off 1.6pt] ({f2(XL)},{yp}) -- ({f2(XR)},{yp});")
out.append("\\newcommand{\\panelC}{" + "\n".join(cmds) + "}")

# D: candidates (unique points), alpha line, selected rule
tmin, tmax, rmin_, rmax_ = 11.0, 56.0, 0.04, 0.32
pts = sorted({(round(c["t"]), round(c["risk"], 2)) for c in cand if c["risk"] <= rmax_ - 0.01})
ya = lin(ALPHA, rmin_, rmax_, YB, YT)
cmds = [f"\\fill[warn!11] ({f2(XL - 0.08)},{f2(ya)}) rectangle ({f2(XR + 0.08)},{f2(YT + 0.07)});",
        f"\\draw[warn,line width=0.7pt,dash pattern=on 2.4pt off 1.4pt] ({f2(XL - 0.08)},{f2(ya)}) -- ({f2(XR + 0.08)},{f2(ya)});"]
for tt, rr in pts:
    col = "ours" if rr <= ALPHA else "gry"
    cmds.append(f"\\fill[{col}] ({f2(lin(tt, tmin, tmax, XL, XR))},{f2(lin(rr, rmin_, rmax_, YB, YT))}) circle (0.42mm);")
sx, sy = lin(sel["t"], tmin, tmax, XL, XR), lin(sel["risk"], rmin_, rmax_, YB, YT)
cmds.append(f"\\draw[warn,line width=0.7pt] ({f2(sx)},{f2(sy)}) circle (1.15mm);")
out.append("\\newcommand{\\panelD}{" + "\n".join(cmds) + "}")
out.append(f"\\newcommand{{\\panelDalpha}}{{{f2(ya)}}}\\newcommand{{\\panelDsx}}{{{f2(sx)}}}\\newcommand{{\\panelDsy}}{{{f2(sy)}}}")

# E: online panel in absolute coordinates (box 13.75..18.10 x 1.38..3.15)
EXL, EXR, EYB, EYT = 13.88, 17.50, 1.45, 3.08
band = sel["band"]; bx = lin(GRID[:-1], 12, 66, EXL, EXR)
up = lin(band[:-1, 1], 0, 1, EYB, EYT); dn = lin(band[:-1, 0], 0, 1, EYB, EYT)
jt = sel["j"][ex]; tau = GRID[jt]
px = lin(GRID[:jt + 1], 12, 66, EXL, EXR); py = lin(P[ex, :jt + 1], 0, 1, EYB, EYT)
cmds = [f"\\fill[ours!13] {poly(bx, up)} -- {poly(bx[::-1], dn[::-1])} -- cycle;",
        f"\\draw[oursd,line width=0.6pt] {poly(bx, up)};", f"\\draw[oursd,line width=0.6pt] {poly(bx, dn)};",
        f"\\draw[gry,line width=0.5pt,dash pattern=on 0.6pt off 1.6pt] ({f2(EXL - 0.03)},{f2(lin(sel['ps'], 0, 1, EYB, EYT))}) -- ({f2(EXR)},{f2(lin(sel['ps'], 0, 1, EYB, EYT))});",
        f"\\draw[ink,line width=0.8pt] {poly(px, py)};",
        f"\\fill[warn] ({f2(px[-1])},{f2(py[-1])}) circle (0.65mm);"]
out.append("\\newcommand{\\panelE}{" + "\n".join(cmds) + "}")
out.append(f"\\newcommand{{\\panelEtx}}{{{f2(px[-1])}}}\\newcommand{{\\panelEty}}{{{f2(py[-1])}}}"
           f"\\newcommand{{\\panelEbu}}{{{f2(up[-1])}}}\\newcommand{{\\panelEbd}}{{{f2(dn[-1])}}}"
           f"\\newcommand{{\\panelEps}}{{{f2(lin(sel['ps'], 0, 1, EYB, EYT))}}}")
io.open("fig_overview_data.tex", "w", encoding="utf-8", newline="\n").write("\n".join(out) + "\n")
print("wrote fig_overview_data.tex")
