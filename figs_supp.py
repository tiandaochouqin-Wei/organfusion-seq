#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stage 3O: all seven Supplementary figures (S1-S7), one script, one style.

Before this, S1-S4 came from regen_supp_figs.py and S5-S7 were assembled in LaTeX from
single panels of figs_ieee.py, each included at its own scale factor -- so the lettering
on the page ranged from about 5.5 pt (S5, S6) to about 12 pt (S7). Here every figure is
drawn at its final size with the same 8 pt Arial and palette as the main-text figures
(figs_tkde.py) and is included at natural size, so 8 pt in this file is 8 pt on the page.
S5 and S6 are now single composite figures with panel letters instead of LaTeX-tiled PDFs.

The computations are unchanged from the two older scripts (kept for the frozen TCYB
version); the built-in prints let the numbers be checked against the captions.
Needs openpyxl for S7:  python figs_supp.py
After any change run:   py check_text_line_overlap.py   (must end with TOTAL: 0)
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import DATA_DIR  # $ORGANFUSION_DATA, or ./datasets (see README)
import os, sys, csv, numpy as np
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import panel_labels
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle
from scipy.stats import pearsonr, spearmanr, gaussian_kde
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

OUT = "figures_supp"; os.makedirs(OUT, exist_ok=True)
F = 8.0
mpl.rcParams.update({
    "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
    "mathtext.fontset": "custom", "mathtext.rm": "Arial", "mathtext.it": "Arial:italic", "mathtext.bf": "Arial:bold",
    "mathtext.cal": "Arial:italic",
    "font.size": F, "axes.labelsize": F, "axes.titlesize": F, "xtick.labelsize": F, "ytick.labelsize": F,
    "legend.fontsize": F, "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.7,
    "xtick.major.width": 0.7, "ytick.major.width": 0.7, "xtick.major.size": 2.6, "ytick.major.size": 2.6,
    "xtick.major.pad": 2.2, "ytick.major.pad": 2.2, "axes.labelpad": 2.5, "legend.frameon": False,
    "lines.linewidth": 1.2, "lines.markersize": 3.6, "savefig.dpi": 300, "hatch.linewidth": 0.5,
    "axes.edgecolor": "#2B2B2B", "axes.labelcolor": "#2B2B2B", "text.color": "#2B2B2B",
    "xtick.color": "#2B2B2B", "ytick.color": "#2B2B2B",
})
OURS = "#0072BD"; OURS_D = "#00406A"; CAL = "#7E2F8E"; ECO = "#77AC30"
FIX = "#555555"; GRY = "#A6A6A6"; LGRY = "#D9D9D9"; RED = "#A2142F"; INK = "#2B2B2B"
HALO = dict(facecolor="white", edgecolor="none", pad=0.9, alpha=0.96)
LEG = dict(frameon=True, facecolor="white", edgecolor="none", framealpha=0.94, borderpad=0.2)


def rows(fn):
    with open(os.path.join("posterior", fn), encoding="utf-8") as f:
        return list(csv.DictReader(f))


def plabel(ax, s, dx=None, dy=None):
    """IEEE style (Stage 3Q-4): "(a)" centered under the panel, placed by panel_labels.place() in save();
    dx, dy are the old top-left offsets and are no longer used."""
    panel_labels.register(ax, s)


def save(fig, name):
    panel_labels.place(fig)
    fig.canvas.draw()
    tb = fig.get_tightbbox(fig.canvas.get_renderer()); w, h = fig.get_size_inches()
    over = (-tb.x0, -tb.y0, tb.x1 - w, tb.y1 - h)
    if max(over) > 0.004:
        print(f"  !! {name}: content outside the canvas by (L,B,R,T) = "
              + ", ".join(f"{72 * max(o, 0):.1f}pt" for o in over))
    fig.savefig(os.path.join(OUT, name + ".pdf")); fig.savefig(os.path.join(OUT, name + ".png"))
    plt.close(fig); print(f"wrote {name}  ({w * 72.27:.0f} x {h * 72.27:.0f} pt)")


# ------------------------------------------------------------------ data
dg = np.load("posterior/posterior_data_grouped.npz", allow_pickle=True)
P = dg["P_group"]; groups = dg["groups"]
d0 = np.load("posterior/posterior_data.npz", allow_pickle=True)
y = d0["y"].astype(int); GRID = d0["GRID"].astype(float); xg = d0["xgrid"]
drug = d0["drug"]; conc = d0["conc"].astype(float); N, T = P.shape
assert np.array_equal(d0["y"], dg["y"]), "organoid ordering mismatch between posterior npz files"
P_sh = np.load("posterior/posterior_data_shared_strat.npz", allow_pickle=True)["P_group"]
DRUGS = ["Stauro", "Ner", "Lap"]; SHORT = {"Stauro": "Stau.", "Ner": "Ner.", "Lap": "Lap."}
print(f"N={N} (leave-condition-out posterior)")


def controller(Pm, m, ps=0.5):
    tt = np.empty(len(Pm)); dd = np.empty(len(Pm), int); ex = np.zeros(len(Pm), bool)
    for i in range(len(Pm)):
        e = np.where(np.abs(Pm[i] - ps) >= m)[0]
        j = e[0] if len(e) else T - 1
        tt[i] = GRID[j]; dd[i] = int(Pm[i, j] >= ps); ex[i] = len(e) > 0
    return tt, dd, ex


# =============================================================== S1: cost sensitivity
ms = np.linspace(0.0, 0.48, 49)
ct = np.array([controller(P, m)[0].mean() for m in ms]); ce = np.array([(controller(P, m)[1] != y).mean() for m in ms])
fe = np.array([((P[:, j] >= 0.5).astype(int) != y).mean() for j in range(T)])
lams = np.logspace(np.log10(0.001), np.log10(0.05), 16)
Jc = np.array([(ce + l * ct).min() for l in lams]); Jf = np.array([(fe + l * GRID).min() for l in lams])
Tc = np.array([ct[np.argmin(ce + l * ct)] for l in lams])
fig = plt.figure(figsize=(5.5, 2.15))
gs = GridSpec(1, 2, figure=fig, left=0.095, right=0.905, top=0.90, bottom=0.215, wspace=0.42)
ax = fig.add_subplot(gs[0, 0])
ax.plot(lams, Jf, "--s", color=FIX, ms=3.0, lw=1.0, label="best fixed horizon")
ax.plot(lams, Jc, "-o", color=OURS, ms=3.0, lw=1.2, label="controller")
ax.set_xscale("log"); ax.set_xlabel(r"delay weight $\lambda$ (per h)"); ax.set_ylabel(r"optimal total cost $J^\star$")
ax.legend(loc="upper left", bbox_to_anchor=(-0.02, 1.03), handlelength=2.0, **LEG)
plabel(ax, "a", dx=-0.24)
ax = fig.add_subplot(gs[0, 1])
ax.plot(lams, 100 * (Jf - Jc) / Jf, "-o", color=OURS, ms=3.0, lw=1.2)
ax.axhline(0, color=INK, lw=0.7); ax.set_xscale("log")
ax.set_xlabel(r"delay weight $\lambda$ (per h)"); ax.set_ylabel("cost reduction (%)")
ax2 = ax.twinx(); ax2.plot(lams, Tc, ":", color=FIX, lw=1.3); ax2.set_ylabel("mean stop (h), dotted", color=FIX)
ax2.spines["top"].set_visible(False); ax2.spines["right"].set_visible(True); ax2.tick_params(axis="y", colors=FIX)
plabel(ax, "b", dx=-0.24)
save(fig, "figS1_cost")
print("  S1 rel. reduction (%):", {round(float(l), 4): round(float(100 * (a - b) / a), 1) for l, a, b in zip(lams, Jf, Jc)})

# =============================================================== S2: early-endpoint relation
j24 = int(np.where(GRID == 24)[0][0]); a = xg[:, j24]; b = xg[:, -1]
r_p = pearsonr(a, b)[0]; r_s = spearmanr(a, b)[0]
fig = plt.figure(figsize=(3.1, 2.65))
ax = fig.add_axes([0.150, 0.165, 0.825, 0.805])
ax.scatter(a[y == 0], b[y == 0], s=4, facecolors="white", edgecolors=INK, linewidths=0.4, label="resistant")
ax.scatter(a[y == 1], b[y == 1], s=4, color=OURS, linewidths=0, alpha=0.75, label="sensitive")
ax.axhline(0.9, color=FIX, ls=":", lw=0.8)
ax.text(a.max() * 0.99, 1.0, r"$\theta$ = 0.9", ha="right", va="bottom", fontsize=F, color=FIX, bbox=HALO)
ax.set_xlabel(r"$x(24)$, normalized mass at 24 h"); ax.set_ylabel(r"$x(72)$, endpoint")
# headroom above the highest point (3.5), so the statistics sit clear of every marker: a
# marker beside "0.65" once read as a stray degree sign
ax.set_ylim(-0.15, 4.95); ax.set_yticks([0, 1, 2, 3, 4])
ax.text(0.03, 0.975, f"Pearson $r$ = {r_p:.2f}\nSpearman $\\rho$ = {r_s:.2f}\n$n$ = {N:,}", transform=ax.transAxes,
        va="top", fontsize=F, linespacing=1.15)
ax.legend(loc="lower right", handlelength=1.0, markerscale=2.2, handletextpad=0.3, **LEG)
save(fig, "figS2_endpoint")
print(f"  S2 pearson={r_p:.3f} spearman={r_s:.3f}")

# =============================================================== S3: controller behavior by drug
tt, dd, ex = controller(P, 0.30)
fig = plt.figure(figsize=(5.9, 2.25))
gs = GridSpec(1, 2, figure=fig, width_ratios=[1.45, 0.72], left=0.070, right=0.800, top=0.905, bottom=0.205, wspace=0.34)
ax = fig.add_subplot(gs[0, 0])
RIDGE = [("Lap", CAL), ("Stauro", OURS), ("Ner", OURS_D)]
yt = []
for k, (dg_, c) in enumerate(RIDGE):
    tdg = tt[drug == dg_]; frac = np.array([(tdg == g).mean() for g in GRID])
    base = k * 1.0; yt.append(base + 0.20)
    ax.fill_between(GRID, base, base + frac * 2.6, step="mid", color=c, alpha=0.38, lw=0)
    ax.step(GRID, base + frac * 2.6, where="mid", color=c, lw=0.9)
    ax.text(75.5, base + 0.60, f"mean {tdg.mean():.1f} h", fontsize=F, color=c, ha="right", va="center")
ax.set_yticks(yt); ax.set_yticklabels([SHORT[g_] for g_, _ in RIDGE])
ax.tick_params(axis="y", length=0); ax.spines["left"].set_visible(False)
ax.set_xlim(10, 76); ax.set_ylim(-0.05, 2.95); ax.set_xticks([12, 24, 36, 48, 60, 72])
ax.set_xlabel("stopping time (h)")
plabel(ax, "a", dx=-0.10)
ax = fig.add_subplot(gs[0, 1])
comp = []
for dg_ in DRUGS:
    mk = drug == dg_
    comp.append([(ex[mk] & (dd[mk] == 1)).mean(), (ex[mk] & (dd[mk] == 0)).mean(), (~ex[mk]).mean()])
comp = np.array(comp) * 100
xs = np.arange(3)
ax.bar(xs, comp[:, 0], color=OURS, edgecolor=INK, lw=0.6, width=0.62, label="prioritise")
ax.bar(xs, comp[:, 1], bottom=comp[:, 0], color="white", edgecolor=INK, lw=0.6, width=0.62, label="exclude")
ax.bar(xs, comp[:, 2], bottom=comp[:, 0] + comp[:, 1], color=LGRY, edgecolor=INK, lw=0.6, width=0.62, hatch="/////", label="defer")
ax.set_xticks(xs); ax.set_xticklabels([SHORT[d_] for d_ in DRUGS]); ax.set_ylabel("% of organoids")
ax.set_yticks([0, 50, 100])
ax.legend(loc="center left", bbox_to_anchor=(1.03, 0.5), handlelength=1.2, handleheight=0.9, labelspacing=0.5)
plabel(ax, "b", dx=-0.42)
save(fig, "figS3_bydrug")
print("  S3 per-drug mean stop (h):", {g_: round(float(tt[drug == g_].mean()), 1) for g_ in DRUGS},
      " median:", {g_: float(np.median(tt[drug == g_])) for g_ in DRUGS}, " composition %:", np.round(comp, 1).tolist())

# =============================================================== S4: window information
wi = rows("table3_window_info.csv")
fig = plt.figure(figsize=(7.14, 2.20))
gs = GridSpec(1, 3, figure=fig, left=0.062, right=0.990, top=0.905, bottom=0.205, wspace=0.36)
ax = fig.add_subplot(gs[0, 0])
De = [float(r["D_end"]) for r in wi]; Iw = [float(r["I_W"]) for r in wi]
ax.bar(xs, De, color=LGRY, edgecolor=INK, lw=0.6, width=0.6, label=r"$D_\mathrm{end}$")
ax.bar(xs, Iw, bottom=De, color=OURS, edgecolor=INK, lw=0.6, width=0.6, label=r"$I_W$")
ax.set_xticks(xs); ax.set_xticklabels([SHORT[r["drug"]] for r in wi]); ax.set_ylabel("separability")
ax.set_ylim(0, 3.05); ax.set_yticks([0, 1, 2, 3])
ax.legend(loc="upper right", bbox_to_anchor=(1.02, 1.03), handlelength=1.1, ncol=2, columnspacing=1.0, **LEG)
plabel(ax, "a", dx=-0.21)
# (b) the measured gain is 2-3 orders below the bound, so a bar would be invisible:
# bound as a capped whisker from 0, measured value as a marked point with its number
ax = fig.add_subplot(gs[0, 1])
bd = [float(r["bound_sqrtIW_2sqrtpi"]) for r in wi]; da = [float(r["measured_dAUC"]) for r in wi]
ax.vlines(xs, 0, bd, color=LGRY, lw=5.0, zorder=1)
for i, v in enumerate(bd): ax.plot([i - 0.20, i + 0.20], [v, v], color=INK, lw=0.9, zorder=2)
ax.scatter(xs, da, s=22, color=OURS_D, edgecolor="white", lw=0.5, zorder=3)
ax.axhline(0, color=INK, lw=0.7)
for i, v in enumerate(da): ax.text(i, v - 0.012, f"{v:+.3f}", fontsize=F, va="top", ha="center", color=OURS_D)
ax.set_xlim(-0.55, 2.55); ax.set_ylim(-0.048, 0.215); ax.set_yticks([0, 0.1, 0.2])
ax.set_xticks(xs); ax.set_xticklabels([SHORT[r["drug"]] for r in wi]); ax.set_ylabel("AUC gain")
ax.plot([], [], color=LGRY, lw=5.0, label="bound"); ax.scatter([], [], s=22, color=OURS_D, label="measured")
ax.legend(loc="upper right", bbox_to_anchor=(1.02, 1.03), handlelength=1.1, ncol=2, columnspacing=1.0, **LEG)
plabel(ax, "b", dx=-0.23)
# (c) endpoint-only vs trajectory posterior by horizon (leave-condition-out GroupKFold)
ax = fig.add_subplot(gs[0, 2])
doh = np.array([np.eye(3)[DRUGS.index(x)] for x in drug]); lc = np.log1p(conc)[:, None]
auc_end, auc_traj = [], []
for j in range(T):
    X = np.column_stack([xg[:, j], doh, lc])
    pe = cross_val_predict(make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)), X, y,
                           cv=GroupKFold(5).split(X, y, groups), method="predict_proba")[:, 1]
    auc_end.append(roc_auc_score(y, pe)); auc_traj.append(roc_auc_score(y, P[:, j]))
ax.plot(GRID, auc_end, "--s", color=FIX, ms=3.0, lw=1.0, label="endpoint $x(T)$")
ax.plot(GRID, auc_traj, "-o", color=OURS, ms=3.0, lw=1.2, label="trajectory prefix")
ax.set_xlabel("horizon $T$ (h)"); ax.set_ylabel("AUC"); ax.set_xticks([12, 24, 36, 48, 60, 72]); ax.set_yticks([0.8, 0.9, 1.0])
ax.legend(loc="lower right", bbox_to_anchor=(1.02, -0.02), handlelength=2.0, **LEG)
plabel(ax, "c", dx=-0.23)
save(fig, "figS4_window_info")
print("  S4 AUC endpoint-only vs trajectory:", [(int(t), round(a_, 3), round(b_, 3)) for t, a_, b_ in zip(GRID, auc_end, auc_traj)])
print("  S4 max |AUC gap|:", round(float(np.max(np.abs(np.array(auc_end) - np.array(auc_traj)))), 4))

# =============================================================== S5: the optimal-stopping band
Bbin = np.linspace(0, 1, 51); bc = (Bbin[:-1] + Bbin[1:]) / 2; nbk = len(bc)
rho_bc = np.minimum(1 - bc, bc)


def kern(Pm):
    Ks = []
    for j in range(T - 1):
        Kk = np.zeros((nbk, nbk))
        a_ = np.clip(np.digitize(Pm[:, j], Bbin) - 1, 0, nbk - 1); b_ = np.clip(np.digitize(Pm[:, j + 1], Bbin) - 1, 0, nbk - 1)
        for u, v in zip(a_, b_): Kk[u, v] += 1
        Kk += 1e-3; Kk /= Kk.sum(1, keepdims=True); Ks.append(Kk)
    return Ks


def band_of(Pm, lam):
    Ks = kern(Pm); V = rho_bc.copy(); out = {}
    for j in range(T - 2, -1, -1):
        cont = lam * (GRID[j + 1] - GRID[j]) + Ks[j] @ V
        cs = np.where(cont < rho_bc - 1e-9)[0]
        if len(cs): out[GRID[j]] = (bc[cs.min()], bc[cs.max()])
        V = np.minimum(rho_bc, cont)
    return dict(sorted(out.items()))


fig = plt.figure(figsize=(7.14, 2.05))
gs = GridSpec(1, 4, figure=fig, left=0.062, right=0.992, top=0.895, bottom=0.215, wspace=0.50)
# (a) the band UNDER the regularity of Prop. S4 (an illustration; this cohort fails it)
ax = fig.add_subplot(gs[0, 0])
tt_ = np.linspace(12, 72, 40)
a_ill = 0.5 - 0.42 * (1 - (tt_ - 12) / 60) ** 0.9; b_ill = 0.5 + 0.42 * (1 - (tt_ - 12) / 60) ** 0.9
ax.fill_between(tt_, a_ill, b_ill, color=OURS, alpha=0.28, lw=0)
ax.plot(tt_, b_ill, "-", color=OURS_D, lw=1.0); ax.plot(tt_, a_ill, "-", color=OURS_D, lw=1.0)
ax.plot([12, 72], [0.5, 0.5], ":", color=FIX, lw=0.8)
ax.text(74.5, 0.5, r"$p^\star$", fontsize=F, color=FIX, ha="left", va="center")
ax.text(26, 0.655, "continue", fontsize=F, color=OURS_D, ha="center", va="center")
ax.text(50, 0.84, r"$b_t$", fontsize=F, color=OURS_D, ha="center", va="center")
ax.text(50, 0.16, r"$a_t$", fontsize=F, color=OURS_D, ha="center", va="center")
ax.set_xlabel("time (h)"); ax.set_ylabel(r"band $(a_t, b_t)$"); ax.set_ylim(0, 1); ax.set_xlim(8, 84)
ax.set_xticks([12, 36, 60]); ax.set_yticks([0, 0.5, 1])
plabel(ax, "a", dx=-0.36)
# (b) the EMPIRICAL band: an interval at every horizon, but not monotone
ax = fig.add_subplot(gs[0, 1])
bd_ = band_of(P_sh, 0.01); ts = np.array(list(bd_))
lo = np.array([bd_[t][0] for t in ts]); hi = np.array([bd_[t][1] for t in ts])
ax.fill_between(ts, lo, hi, step="mid", color=OURS, alpha=0.28, lw=0)
ax.step(ts, hi, where="mid", color=OURS_D, lw=1.0); ax.step(ts, lo, where="mid", color=OURS_D, lw=1.0)
ax.axhline(0.5, color=FIX, ls=":", lw=0.8)
ax.set_xlabel("time (h)"); ax.set_ylabel(r"band $(a_t, b_t)$"); ax.set_ylim(0, 1); ax.set_xticks([12, 36, 60]); ax.set_yticks([0, 0.5, 1])
plabel(ax, "b", dx=-0.36)
# (c) value curve
ax = fig.add_subplot(gs[0, 2])
Ks = kern(P_sh); V = rho_bc.copy()
for j in range(T - 2, 5, -1):
    V = np.minimum(rho_bc, 0.01 * (GRID[j + 1] - GRID[j]) + Ks[j] @ V)
ax.plot(bc, rho_bc, "-", color=INK, lw=1.1, label=r"stop cost $\rho$")
ax.plot(bc, V, "--", color=OURS, lw=1.2, label="continue")
ax.set_xlabel(r"posterior $p$"); ax.set_ylabel("cost / value")
ax.set_ylim(0, 0.80); ax.set_yticks([0, 0.2, 0.4]); ax.set_xticks([0, 0.5, 1]); ax.set_xticklabels(["0", "0.5", "1"])
ax.legend(loc="upper left", bbox_to_anchor=(-0.04, 1.05), handlelength=1.9, labelspacing=0.15, **LEG)
plabel(ax, "c", dx=-0.36)
# (d) width vs lambda
ax = fig.add_subplot(gs[0, 3])
for lam, st, c in [(0.004, "-o", OURS), (0.02, "--s", FIX)]:
    bdl = band_of(P_sh, lam); tl = np.array(list(bdl))
    w_ = np.array([bdl[t][1] - bdl[t][0] for t in tl])
    ax.plot(tl, w_, st, color=c, ms=3.0, lw=1.1, label=rf"$\lambda$ = {lam}")
ax.set_xlabel("time (h)"); ax.set_ylabel(r"width $b_t - a_t$"); ax.set_ylim(0, 1.52); ax.set_yticks([0, 0.5, 1]); ax.set_xticks([12, 36, 60])
ax.legend(loc="upper left", bbox_to_anchor=(-0.04, 1.05), handlelength=1.9, labelspacing=0.15, **LEG)
plabel(ax, "d", dx=-0.36)
save(fig, "figS5_band")

# =============================================================== S6: motivation
fig = plt.figure(figsize=(7.14, 2.30))
gs = GridSpec(1, 3, figure=fig, left=0.068, right=0.990, top=0.905, bottom=0.195, wspace=0.34)
# (a) clique diagram from the audited pairwise DeLong table: each bar is a maximal run of
#     consecutive-rank models with EVERY pairwise p > 0.05 (computed, not hand-picked)
bt = {r["model"]: r for r in rows("table2_fixed_time_baselines.csv") if r["horizon_h"] == "24"}
pw = {}
for r in rows("table_auc_pairwise_delong.csv"):
    if r["horizon_h"] != "24": continue
    p_ = float(r["delong_p"]); pw[(r["model_a"], r["model_b"])] = p_; pw[(r["model_b"], r["model_a"])] = p_
order = sorted(bt, key=lambda m: float(bt[m]["AUC"]))
vals = np.array([float(bt[m]["AUC"]) for m in order])
SH = {"near-Bayes (LDA)": "LDA", "trajectory encoder": "enc.", "SVM-RBF": "SVM"}
n = len(order)
sig = lambda i, j: pw[(order[i], order[j])] <= 0.05
cliques = [(i, j) for i in range(n) for j in range(i + 1, n)
           if all(not sig(a_, c_) for a_ in range(i, j + 1) for c_ in range(a_ + 1, j + 1))]
cliques = [c for c in cliques if not any(c != q and q[0] <= c[0] and c[1] <= q[1] for q in cliques)]
print("  S6a cliques (mutually indistinguishable, p>0.05 throughout):", [[order[k] for k in range(c[0], c[1] + 1)] for c in cliques],
      "| AUC spread", round(float(vals.max() - vals.min()), 3))
ax = fig.add_subplot(gs[0, 0])
xs7 = np.arange(n)
yfloor = vals.min() - 0.006 - 0.0064 * len(cliques)
for i, (lo_, hi_) in enumerate(cliques):
    yy = vals.min() - 0.006 - 0.0064 * (len(cliques) - i)
    ax.plot([xs7[lo_], xs7[hi_]], [yy, yy], "-", color=OURS_D, lw=2.6, solid_capstyle="butt", zorder=2)
ax.vlines(xs7, yfloor, vals, color=LGRY, lw=0.9, zorder=1)
ax.scatter(xs7, vals, s=24, color=OURS_D, edgecolor="white", lw=0.5, zorder=3)
ax.set_ylim(yfloor - 0.004, 0.868); ax.set_ylabel("AUC at 24 h"); ax.set_yticks([0.80, 0.82, 0.84, 0.86])
ax.set_xlim(-0.6, n - 0.4); ax.set_xticks(xs7); ax.set_xticklabels([SH.get(m, m) for m in order])
plabel(ax, "a", dx=-0.25)
# (b) stopping time by drug as a raincloud
ax = fig.add_subplot(gs[0, 1])
rng_b = np.random.default_rng(2)
cols3 = {"Stauro": OURS, "Ner": OURS_D, "Lap": CAL}
for i, g_ in enumerate(["Stauro", "Ner", "Lap"], 1):
    v = tt[drug == g_]; c = cols3[g_]
    kde = gaussian_kde(v, bw_method=0.3)
    yy = np.linspace(v.min() - 2, v.max() + 2, 150); dd_ = kde(yy); dd_ = dd_ / dd_.max() * 0.30
    ax.fill_betweenx(yy, i - dd_, i, color=c, alpha=0.35, lw=0); ax.plot(i - dd_, yy, color=c, lw=0.7)
    q1, q2, q3 = np.percentile(v, [25, 50, 75])
    ax.add_patch(Rectangle((i + 0.02, q1), 0.07, q3 - q1, facecolor="white", edgecolor=INK, lw=0.6, zorder=3))
    ax.plot([i + 0.02, i + 0.09], [q2, q2], color=INK, lw=1.0, zorder=4)
    ax.scatter(i + rng_b.uniform(0.13, 0.34, len(v)), v, s=2.2, color=c, alpha=0.4, lw=0, zorder=2)
    ax.plot(i + 0.235, v.mean(), "D", color=INK, ms=3.4, zorder=5)
    ax.text(i + 0.02, 84.5, f"{v.mean():.1f} h", ha="center", va="center", fontsize=F, color=c, fontweight="bold")
ax.set_xticks([1, 2, 3]); ax.set_xticklabels([SHORT[g_] for g_ in ["Stauro", "Ner", "Lap"]])
ax.set_xlim(0.50, 3.50); ax.set_ylim(6, 91); ax.set_yticks([12, 36, 60]); ax.set_ylabel("stopping time (h)")
plabel(ax, "b", dx=-0.22)
# (c) example posterior paths
ax = fig.add_subplot(gs[0, 2])
rng = np.random.default_rng(3)
sens = rng.choice(np.where(y == 1)[0], 2, replace=False); res = rng.choice(np.where(y == 0)[0], 2, replace=False)
for i in sens: ax.plot(GRID, P[i], "-o", color=OURS, ms=2.6, lw=1.1)
for i in res: ax.plot(GRID, P[i], "--s", color=INK, ms=2.6, lw=1.0, mfc="white")
ax.axhline(0.5, color=FIX, ls=":", lw=0.8)
ax.set_xlabel("time (h)"); ax.set_ylabel(r"posterior $\hat p_t$"); ax.set_ylim(0, 1); ax.set_xticks([12, 24, 36, 48, 60, 72]); ax.set_yticks([0, 0.5, 1])
ax.plot([], [], "-o", color=OURS, ms=2.6, label="sensitive"); ax.plot([], [], "--s", color=INK, ms=2.6, mfc="white", label="resistant")
ax.legend(loc="lower left", bbox_to_anchor=(-0.02, -0.03), handlelength=2.0, **LEG)
plabel(ax, "c", dx=-0.22)
save(fig, "figS6_motivation")

# =============================================================== S7: independent pancreatic cohort
import openpyxl
wb = openpyxl.load_workbook(os.path.join(DATA_DIR, "Deben_NOGR_SupplementaryData1.xlsx"), read_only=True, data_only=True)
rws = list(wb["Sheet1"].iter_rows(values_only=True))
colm = {h: i for i, h in enumerate(list(rws[0]))}


def num(r, h):
    try: return float(r[colm[h]])
    except Exception: return np.nan


dg_d = np.array([str(r[colm["Conditions"]]) for r in rws[1:]])
nogr = np.array([num(r, "NOGR") for r in rws[1:]]); rv = np.array([num(r, "RV") for r in rws[1:]])
okm = np.isfinite(nogr) & np.isfinite(rv); dg_d, nogr, rv = dg_d[okm], nogr[okm], rv[okm]
names, rs = [], []
for g_ in sorted(set(dg_d.tolist())):
    mk = dg_d == g_
    if mk.sum() > 10 and np.std(nogr[mk]) > 0 and np.std(rv[mk]) > 0:
        names.append(g_); rs.append(abs(pearsonr(nogr[mk], rv[mk])[0]))
o = np.argsort(rs); names = [names[i] for i in o]; rs = [rs[i] for i in o]
fig = plt.figure(figsize=(3.3, 1.95))
ax = fig.add_axes([0.265, 0.215, 0.700, 0.745])
yy7 = np.arange(len(rs)); cols7 = [GRY if v < 0.5 else OURS for v in rs]
ax.hlines(yy7, 0, rs, color=cols7, lw=2.6)
ax.scatter(rs, yy7, s=26, color=cols7, edgecolor=INK, lw=0.5, zorder=3)
ax.axvline(0.5, color=FIX, ls=":", lw=0.8)
ax.set_yticks(yy7); ax.set_yticklabels(names); ax.tick_params(axis="y", length=0)
ax.set_xlim(0, 1); ax.set_ylim(-0.6, len(rs) - 0.4); ax.set_xticks([0, 0.5, 1]); ax.set_xticklabels(["0", "0.5", "1"])
ax.set_xlabel(r"per-drug $|r|$ (NOGR, viability)")
save(fig, "figS7_pancreatic")
print("  S7 per-drug |r|:", {n_: round(float(v), 3) for n_, v in zip(names, rs)})


# =============================================================== S8: UCR benchmark, all 40 two-class datasets
# (Stage 3P) reads etsc_ucr_binary40.py / ucr40_analysis.py outputs; draws nothing that is not in those tables.
from matplotlib.colors import LinearSegmentedColormap, Normalize
LOSS = "#D95319"; TIE = "#E4E4E4"; TOL8 = 1e-3
ub = [r for r in rows("table_etsc_ucr_binary40.csv") if r["posterior"] == "raw"]
rk8 = rows("table_ucr40_ranks.csv")
NAME8 = {"const_margin(train-selected)": "constant margin (ours)", "bayes_dp_band(train-kernel)": "Bellman band (ours)",
         "CALIMERA": "CALIMERA", "EconomyGamma": "EconomyGamma", "fixed(train-selected)": "fixed horizon",
         "TEASER-cost": "TEASER, cost criterion", "TEASER": "TEASER"}
COL8 = {"const_margin(train-selected)": OURS, "bayes_dp_band(train-kernel)": OURS_D, "CALIMERA": CAL, "EconomyGamma": ECO,
        "fixed(train-selected)": FIX, "TEASER-cost": RED, "TEASER": RED}
rank8 = {p_: {r["method"]: float(r["mean_rank"]) for r in rk8 if r["posterior"] == p_} for p_ in ("raw", "platt")}
cd8 = float(rk8[0]["nemenyi_cd_0.05"])
order8 = sorted(NAME8, key=lambda m_: rank8["raw"][m_])
dsets8 = sorted({r["dataset"] for r in ub})
lam8 = {d_: sorted({float(r["lambda"]) for r in ub if r["dataset"] == d_}) for d_ in dsets8}
J8 = {(r["dataset"], r["method"], float(r["lambda"])): float(r["J_method"]) for r in ub}
gain8 = {(d_, m_): np.array([J8[d_, "fixed(train-selected)", l_] - J8[d_, m_, l_] for l_ in lam8[d_]])
         for d_ in dsets8 for m_ in NAME8}                       # > 0: beats the like-for-like fixed horizon
wtl8 = {m_: (sum(int((gain8[d_, m_] > TOL8).sum()) for d_ in dsets8),
             sum(int((np.abs(gain8[d_, m_]) <= TOL8).sum()) for d_ in dsets8),
             sum(int((gain8[d_, m_] < -TOL8).sum()) for d_ in dsets8)) for m_ in NAME8}

fig = plt.figure(figsize=(516.0 / 72.27, 4.55))
# ---- a: mean rank with the Nemenyi interval (two rules differ at 0.05 iff their bars do not overlap)
ax = fig.add_axes([0.215, 0.672, 0.315, 0.285])
for i, m_ in enumerate(order8):
    r1, r2 = rank8["raw"][m_], rank8["platt"][m_]
    ax.hlines(i - 0.14, r1 - cd8 / 2, r1 + cd8 / 2, color=COL8[m_], lw=2.4)
    ax.plot([r1], [i - 0.14], "o", color=COL8[m_], ms=4.6, mec=INK, mew=0.5, zorder=3)
    ax.hlines(i + 0.20, r2 - cd8 / 2, r2 + cd8 / 2, color=GRY, lw=1.1)
    ax.plot([r2], [i + 0.20], "o", color="white", ms=3.8, mec=FIX, mew=0.8, zorder=3)
ax.set_yticks(range(len(order8))); ax.set_yticklabels([NAME8[m_] for m_ in order8]); ax.tick_params(axis="y", length=0)
ax.get_yticklabels()[order8.index("const_margin(train-selected)")].set_fontweight("bold")
ax.set_ylim(len(order8) - 0.45, -0.6); ax.set_xlim(1, 7.2); ax.set_xticks(range(1, 8))
ax.set_xlabel(r"mean rank over 40 datasets (1 = lowest cost)")
ax.plot([], [], "o", color=FIX, ms=4.6, mec=INK, mew=0.5, label="MiniRocket posterior")
ax.plot([], [], "o", color="white", ms=3.8, mec=FIX, mew=0.8, label="after Platt scaling")
ax.legend(loc="lower left", bbox_to_anchor=(-0.02, -0.02), handletextpad=0.2, labelspacing=0.25, borderpad=0.0)
plabel(ax, "a", dx=-0.66, dy=0.985)
# ---- b: delay weights won / tied / lost against the fixed horizon chosen the same way (on training data)
ax = fig.add_axes([0.605, 0.672, 0.300, 0.285])
for i, m_ in enumerate(order8):
    if m_ == "fixed(train-selected)":
        ax.text(4, i, "reference", va="center", ha="left", fontsize=F, color=FIX, fontstyle="italic"); continue
    w_, t_, l_ = wtl8[m_]
    ax.barh(i, w_, height=0.62, color=COL8[m_], lw=0)
    ax.barh(i, t_, left=w_, height=0.62, color=TIE, lw=0)
    ax.barh(i, l_, left=w_ + t_, height=0.62, color=LOSS, hatch="xxxxx", edgecolor="white", lw=0)
    ax.text(326, i, f"{w_}/{t_}/{l_}", va="center", ha="left", fontsize=F, color=INK)
ax.set_ylim(len(order8) - 0.45, -0.6); ax.set_xlim(0, 320); ax.set_yticks([]); ax.spines["left"].set_visible(False)
ax.set_xticks([0, 80, 160, 240, 320]); ax.set_xlabel(r"delay weights (40 datasets $\times$ 8): win | tie | loss")
plabel(ax, "b", dx=-0.07, dy=0.985)
# ---- c: per dataset, wins minus losses over the eight delay weights
strip8 = [m_ for m_ in order8 if m_ != "fixed(train-selected)"]
dorder = sorted(dsets8, key=lambda d_: -np.mean(gain8[d_, "const_margin(train-selected)"]))
S8 = np.array([[int((gain8[d_, m_] > TOL8).sum()) - int((gain8[d_, m_] < -TOL8).sum()) for d_ in dorder] for m_ in strip8])
CM8 = LinearSegmentedColormap.from_list("wl", [LOSS, "#F6C9B4", "#FFFFFF", "#A8CEE8", OURS_D])
ax = fig.add_axes([0.215, 0.385, 0.690, 0.200])
ax.imshow(S8, cmap=CM8, norm=Normalize(-8, 8), aspect="auto", interpolation="nearest")
for (r_i, c_i), v_ in np.ndenumerate(S8):                  # hatched where losses outnumber wins
    if v_ < 0:
        ax.add_patch(Rectangle((c_i - 0.5, r_i - 0.5), 1, 1, fill=False, hatch="xxxx", edgecolor="white", lw=0))
ax.set_xticks(np.arange(-0.5, len(dorder)), minor=True); ax.set_yticks(np.arange(-0.5, len(strip8)), minor=True)
ax.grid(which="minor", color="white", lw=0.8); ax.tick_params(which="minor", length=0)
ax.set_yticks(range(len(strip8))); ax.set_yticklabels([NAME8[m_] for m_ in strip8])
ax.get_yticklabels()[strip8.index("const_margin(train-selected)")].set_fontweight("bold")
ax.set_xticks(range(len(dorder))); ax.set_xticklabels(dorder, rotation=90)
ax.tick_params(length=0, pad=2.5)
for sp in ax.spines.values(): sp.set_visible(False)
cax = fig.add_axes([0.922, 0.385, 0.012, 0.200])
cb = fig.colorbar(mpl.cm.ScalarMappable(Normalize(-8, 8), CM8), cax=cax, ticks=[-8, -4, 0, 4, 8])
cb.ax.tick_params(length=1.8, width=0.5); cb.outline.set_linewidth(0.5); cb.set_label("wins $-$ losses", labelpad=2)
plabel(ax, "c", dx=-0.301, dy=1.02)
save(fig, "figS8_ucr40")
print(f"  S8: Nemenyi CD {cd8:.3f}; raw mean ranks", {NAME8[m_]: rank8['raw'][m_] for m_ in order8})
print("  S8: W/T/L vs like-for-like fixed horizon", {NAME8[m_]: wtl8[m_] for m_ in order8 if m_ != 'fixed(train-selected)'})
print("  S8: datasets with W>L / L>W for the margin rule:", int((S8[strip8.index('const_margin(train-selected)')] > 0).sum()),
      int((S8[strip8.index('const_margin(train-selected)')] < 0).sum()))
