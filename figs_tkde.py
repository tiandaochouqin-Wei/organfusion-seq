#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stage 3O: main-text figures for the TKDE version, rebuilt to IEEE's lettering rule.

Why this version exists. The Stage-3G..3M composites packed four or five panels into one
double-column figure, which forced the lettering down to a 5.4 pt median (two thirds of it
under 6 pt) -- below the 8 pt that IEEE Transactions guides ask for at final printed size.
Here every figure is drawn at exactly \\textwidth (517.935 pt) and saved WITHOUT a tight
bounding box, so it is included 1:1 and the 8 pt set below is 8 pt on the page. To afford
that, the three composites became six figures, in-panel titles moved to the captions, and
labels that only repeated the caption were dropped. Series are also separated by line
style / marker / hatch, not color alone, so the figures survive grayscale printing.

Figure contracts (claim -> evidence -> data):
Fig 2  RQ1   a error-time frontier | b posterior raster to the stopping time | c survival by drug
             <- posterior_data_grouped.npz, table_matched_ops_grouped.csv,
                table_seq_baselines_frontier.csv, table_etsc_baselines_lambda.csv
Fig 3  RQ1   time-to-error matrix: rules x target errors, color = hours behind the fastest
             <- same + table_extra_baselines.csv + Bellman CV band recomputed here
Fig 4  RQ2-3 a shift matrix (hours saved per held-out stratum x margin) with AUC strip
             <- table_shift_matrix.csv | b regime maps: share of alpha left under Thm 3 / Thm 4
Fig 5  RQ3   a raincloud of per-split test risk | b price vs per-condition coverage
             <- table_constrained_control_crc.csv
Fig 6  RQ4-5 a reliability at the stopping time | b gamma path vs the 2L*ECE_stop term |
             c |believed - realized| vs L*ECE_stop in four domains
             <- table_stage3a_calib_regret.csv + recomputed
Fig 7  RQ5   win/tie/loss barcode per delay weight
             <- table_crossdomain_realized.csv, table_etsc_ucr_strong.csv, organoid audit
Metric directions: error, time, risk, cost, excess, ECE -> lower is better; hours saved, wins -> higher.
After any change run:  py check_text_line_overlap.py   (must end with TOTAL: 0)
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
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Rectangle, Patch
from scipy.stats import gaussian_kde
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans

OUT = "figures_tkde"; os.makedirs(OUT, exist_ok=True)
POST = "posterior"
F = 8.0                                    # the one lettering size; nothing is drawn smaller
mpl.rcParams.update({
    "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
    "mathtext.fontset": "custom", "mathtext.rm": "Arial", "mathtext.it": "Arial:italic",
    "mathtext.bf": "Arial:bold",
    "font.size": F, "axes.labelsize": F, "axes.titlesize": F, "xtick.labelsize": F,
    "ytick.labelsize": F, "legend.fontsize": F,
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.7,
    "xtick.major.width": 0.7, "ytick.major.width": 0.7, "xtick.major.size": 2.6,
    "ytick.major.size": 2.6, "xtick.major.pad": 2.2, "ytick.major.pad": 2.2, "axes.labelpad": 2.5,
    "legend.frameon": False, "axes.grid": False,
    "lines.linewidth": 1.2, "lines.markersize": 3.6, "savefig.dpi": 300,
    "axes.edgecolor": "#2B2B2B", "axes.labelcolor": "#2B2B2B", "text.color": "#2B2B2B",
    "xtick.color": "#2B2B2B", "ytick.color": "#2B2B2B", "hatch.linewidth": 0.5,
})
# MATLAB default order (what most IEEE readers see): 1 blue = ours, 4 purple / 5 green =
# published triggers, 7 dark red = failure/violation, 2 orange = loss; grays = classical/fixed.
OURS = "#0072BD"; OURS_D = "#00406A"; CAL = "#7E2F8E"; ECO = "#77AC30"
FIX = "#555555"; GRY = "#A6A6A6"; LGRY = "#D9D9D9"; RED = "#A2142F"; AMB = "#EDB120"; INK = "#2B2B2B"
LOSS = "#D95319"; TIE = "#E4E4E4"; NA = "#F3F3F3"; OURS_PALE = "#A8CEE8"
W2 = 517.935 / 72.27                       # \textwidth of IEEEtran [journal,compsoc], inches
CM_BEHIND = LinearSegmentedColormap.from_list("behind", ["#FFFFFF", "#FBDCC6", "#F2A074", "#D95319", "#7E2E0B"])
CM_SAVED = LinearSegmentedColormap.from_list("saved", ["#F2F7FB", "#BBD9EC", "#5BA3D0", "#0072BD", "#00406A"])
CM_POST = LinearSegmentedColormap.from_list("post", ["#8A3109", "#D95319", "#F6CFB2", "#FFFFFF", "#BBD9EC", "#0072BD", "#00406A"])
CM_SHARE = LinearSegmentedColormap.from_list("share", ["#FFFFFF", "#CFE3F2", "#6FA8D4", "#0072BD", "#00406A"])
# gamma ramp: every dot is CertStop (Bellman), so a blue ramp (gamma = 0.9 -> u = 0.5 -> OURS exactly); purple/amber/red
# are reserved for CALIMERA / loss / violation elsewhere
CM_GAMMA = LinearSegmentedColormap.from_list("gamma", ["#6FA8D4", OURS, OURS_D])
HALO = dict(facecolor="white", edgecolor="none", pad=0.9, alpha=0.96)


def rows(fn):
    with open(os.path.join(POST, fn), encoding="utf-8") as f:
        return [r for r in csv.DictReader(f)]


def plabel(ax, s, dx=None, dy=None):
    """IEEE style (Stage 3Q-4): "(a)" centered under the panel, placed by panel_labels.place() in save();
    dx, dy are the old top-left offsets and are no longer used."""
    panel_labels.register(ax, s)


def save(fig, name):
    """Save at the exact figure size (no tight bbox), so \\includegraphics[width=\\textwidth]
    is 1:1 and the lettering keeps its size; warn if anything pokes outside the canvas."""
    panel_labels.place(fig)
    fig.canvas.draw()
    tb = fig.get_tightbbox(fig.canvas.get_renderer())
    w, h = fig.get_size_inches()
    over = (-tb.x0, -tb.y0, tb.x1 - w, tb.y1 - h)
    if max(over) > 0.004:
        print(f"  !! {name}: content outside the canvas by (L,B,R,T) = "
              + ", ".join(f"{72 * max(o, 0):.1f}pt" for o in over))
    for ext in ("pdf", "svg", "png"):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"))
    plt.close(fig); print(f"wrote {name}  ({w * 72.27:.1f} x {h * 72.27:.1f} pt)")


# ------------------------------------------------------------------ shared data / rules
d = np.load(os.path.join(POST, "posterior_data_grouped.npz"), allow_pickle=True)
P = d["P_group"]; y = d["y"].astype(int); G = d["GRID"].astype(float); groups = d["groups"]; drug = d["drug"]
N, T = P.shape


def margin_rule(Pm, Gm, m, ps=0.5):
    ex = np.abs(Pm - ps) >= m
    j = np.where(ex.any(1), ex.argmax(1), Pm.shape[1] - 1)
    return Gm[j], (Pm[np.arange(len(Pm)), j] >= ps).astype(int), Pm[np.arange(len(Pm)), j]


def envelope(t, e):
    o = np.argsort(t); t = np.asarray(t)[o]; e = np.asarray(e)[o]
    return t, np.minimum.accumulate(e)


def time_at(t, e, target):
    t2, e2 = envelope(t, e)
    return np.nan if (target < e2.min() or target > e2.max()) else float(np.interp(-target, -e2, t2))


def t_reach(t, e, target):
    """time at which a rule's frontier first reaches `target`; inf if it never does."""
    t2, e2 = envelope(t, e)
    if target < e2.min() - 1e-12: return np.inf
    if target >= e2.max(): return float(t2[0])
    return float(np.interp(-target, -e2, t2))


LAMS = np.logspace(-4, -0.7, 30)
folds = list(GroupKFold(5).split(P, y, groups))


def bellman_cv(nb):
    Bb = np.linspace(0, 1, nb + 1); bc = (Bb[:-1] + Bb[1:]) / 2; rho = np.minimum(1 - bc, bc)
    kern = []
    for tr, te in folds:
        Ks = []
        for j in range(T - 1):
            K = np.zeros((nb, nb))
            a = np.clip(np.digitize(P[tr, j], Bb) - 1, 0, nb - 1); b = np.clip(np.digitize(P[tr, j + 1], Bb) - 1, 0, nb - 1)
            np.add.at(K, (a, b), 1); K += 0.05 / nb; K /= K.sum(1, keepdims=True); Ks.append(K)
        kern.append(Ks)
    ts, es = [], []
    for lam in LAMS:
        tau = np.empty(N); dec = np.empty(N, int)
        for (tr, te), Ks in zip(folds, kern):
            V = rho.copy(); ab = np.full((T, 2), 0.5)
            for j in range(T - 2, -1, -1):
                cont = lam * (G[j + 1] - G[j]) + Ks[j] @ V
                cs = np.where(cont < rho - 1e-9)[0]
                if len(cs): ab[j] = [bc[cs.min()], bc[cs.max()]]
                V = np.minimum(rho, cont)
            Pt = P[te]; inside = (Pt[:, :T - 1] > ab[:T - 1, 0]) & (Pt[:, :T - 1] < ab[:T - 1, 1])
            j = np.where((~inside).any(1), (~inside).argmax(1), T - 1)
            tau[te] = G[j]; dec[te] = (Pt[np.arange(len(te)), j] >= 0.5).astype(int)
        ts.append(tau.mean()); es.append((dec != y).mean())
    return np.array(ts), np.array(es)


print("computing frontiers ...")
MS = np.linspace(0.0, 0.48, 49)
cm_t, cm_e = [], []
for m in MS:
    t_, d_, _ = margin_rule(P, G, m); cm_t.append(t_.mean()); cm_e.append((d_ != y).mean())
cm_t, cm_e = np.array(cm_t), np.array(cm_e)
fx_e = np.array([((P[:, j] >= 0.5) != y).mean() for j in range(T)])
bins_list = [25, 50, 100, 200, 400]
bell = {nb: bellman_cv(nb) for nb in bins_list}
floor = {nb: envelope(*bell[nb])[1].min() for nb in bins_list}
print("  Bellman CV floors:", {k: round(v, 4) for k, v in floor.items()})
fr = rows("table_seq_baselines_frontier.csv")
my_t = np.array([float(r["mean_time_h"]) for r in fr if r["policy"] == "myopic_voi"])
my_e = np.array([float(r["decision_error"]) for r in fr if r["policy"] == "myopic_voi"])
et = rows("table_etsc_baselines_lambda.csv")
ext = {nm: (np.array([float(r["mean_time_h"]) for r in et if r["method"] == nm]),
            np.array([float(r["decision_error"]) for r in et if r["method"] == nm]))
       for nm in ("CALIMERA", "EconomyGamma", "TEASER", "TEASER-cost")}
ops = rows("table_matched_ops_grouped.csv")
xb = {r["policy"]: r for r in rows("table_extra_baselines.csv")}

# =============================================================== FIG 2 (RQ1): frontier | raster | survival
fig = plt.figure(figsize=(W2, 2.95))
gs = GridSpec(1, 3, figure=fig, width_ratios=[1.38, 1.0, 1.02], left=0.068, right=0.992, top=0.840,
              bottom=0.155, wspace=0.42)
# ---- a: error-time frontier
ax = fig.add_subplot(gs[0, 0])
eg = np.linspace(0.04, 0.27, 120)
t_o = np.array([t_reach(cm_t, cm_e, e_) for e_ in eg]); t_f = np.array([t_reach(G, fx_e, e_) for e_ in eg])
okb = np.isfinite(t_o) & np.isfinite(t_f)
ax.fill_betweenx(eg[okb], t_o[okb], t_f[okb], color=OURS, alpha=0.10, lw=0, zorder=1)
ax.plot(G, fx_e, "-s", color=FIX, ms=3.0, lw=1.1, zorder=3, label="fixed horizon")
ax.plot(my_t, my_e, ":", color=GRY, lw=1.4, zorder=2, label="myopic one-step")
ax.plot(ext["EconomyGamma"][0], ext["EconomyGamma"][1], "-D", color=ECO, ms=2.8, lw=1.0, zorder=3, label="EconomyGamma")
ax.plot(ext["CALIMERA"][0], ext["CALIMERA"][1], "-^", color=CAL, ms=3.2, lw=1.0, zorder=3, label="CALIMERA")
bt, be = envelope(*bell[400])
ax.plot(bt, be, "--", color=OURS_D, lw=1.2, zorder=4, label="CertStop (Bellman)")
o = np.argsort(cm_t); ax.plot(cm_t[o], cm_e[o], "-", color=OURS, lw=1.9, zorder=5, label="CertStop (margin)")
opt = np.array([float(r["t_ctrl"]) for r in ops]); ope = np.array([float(r["err"]) for r in ops])
ax.errorbar(opt, ope,
            xerr=[opt - np.array([float(r["t_ctrl_lo95_condCB"]) for r in ops]),
                  np.array([float(r["t_ctrl_hi95_condCB"]) for r in ops]) - opt],
            yerr=[ope - np.array([float(r["err_lo95_condCB"]) for r in ops]),
                  np.array([float(r["err_hi95_condCB"]) for r in ops]) - ope],
            fmt="o", color=OURS, ms=3.0, mfc="white", mew=0.9, elinewidth=0.6, capsize=0, zorder=6)
tx, te_ = ext["TEASER"]
ax.plot([tx.mean()], [te_.mean()], "x", color=RED, ms=6.5, mew=1.5, zorder=7)
tcx, tce = ext["TEASER-cost"]                       # ml_edm's cost criterion: at most five settings of v
tc = sorted({(round(a_, 2), round(b_, 4)) for a_, b_ in zip(tcx, tce)} - {(round(tx.mean(), 2), round(te_.mean(), 4))})
ax.plot([a_ for a_, _ in tc], [b_ for _, b_ in tc], "x", color=RED, ms=4.2, mew=1.0, zorder=7)
print("  TEASER (published criterion):", round(tx.mean(), 2), round(te_.mean(), 4), " cost criterion adds:", tc)
ax.annotate("TEASER", (tx.mean(), te_.mean()), xytext=(18.0, 0.307), fontsize=F, color=RED, va="center",
            arrowprops=dict(arrowstyle="-", color=RED, lw=0.6, shrinkA=1.5, shrinkB=4))
ax.plot([my_t.max()], [my_e.min()], "o", color=GRY, ms=3.0)
e6, t6, f6 = 0.123, 30.0, 48.6
ax.annotate("", xy=(t6, e6), xytext=(f6, e6), arrowprops=dict(arrowstyle="<->", color=INK, lw=0.8))
# sits ON the dimension arrow with a halo: between the frontiers every free strip is narrower
# than an 8 pt label, and the fixed-horizon curve cuts through anything placed above or below
ax.text(40.6, e6, "18.6 h", ha="center", va="center", fontsize=F, bbox=HALO, zorder=8)
ax.legend(loc="upper right", bbox_to_anchor=(1.02, 1.03), handlelength=1.5, labelspacing=0.22,
          handletextpad=0.45, borderpad=0.25, frameon=True, facecolor="white", edgecolor="none", framealpha=0.96)
ax.set_xlim(10, 78); ax.set_ylim(0, 0.32)
ax.set_xticks([10, 20, 30, 40, 50, 60, 70]); ax.set_yticks([0, 0.1, 0.2, 0.3])
ax.set_xlabel("mean observation time (h)"); ax.set_ylabel("decision error")
plabel(ax, "a", dx=-0.215, dy=1.03)

# ---- b: posterior raster until the OP6 rule (margin 0.30) stops; wrong calls marked in a strip at the right
ax = fig.add_subplot(gs[0, 1])
MOP6 = 0.30
t6s, d6s, p6s = margin_rule(P, G, MOP6)
js = np.searchsorted(G, t6s)
edges = np.r_[G[0] - 1.5, (G[:-1] + G[1:]) / 2, G[-1] + 3.0]
GAP = 40
row0 = 0; Z = []; stair = []; ticks = []; nsens = 0; wrong_rows = []
for cls, nm in ((1, "sensitive"), (0, "resistant")):
    idx = np.where(y == cls)[0]
    idx = idx[np.lexsort((-np.abs(P[idx, js[idx]] - 0.5), js[idx]))]
    for r_, i in enumerate(idx):
        z = P[i].copy(); z[js[i] + 1:] = np.nan; Z.append(z)
        if d6s[i] != y[i]: wrong_rows.append(row0 + r_)
    stair.append((np.arange(row0, row0 + len(idx)), edges[js[idx] + 1]))
    ticks.append((row0 + len(idx) / 2, f"{nm} ({len(idx)})"))
    row0 += len(idx)
    if cls == 1:
        nsens = len(idx); Z += [np.full(T, np.nan)] * GAP; row0 += GAP
Z = np.ma.masked_invalid(np.array(Z))
cmp = CM_POST.copy(); cmp.set_bad("#EBEBEB")
ax.pcolormesh(edges, np.arange(Z.shape[0] + 1), Z, cmap=cmp, vmin=0, vmax=1, rasterized=True, shading="flat")
ax.axhspan(nsens, nsens + GAP, color="white", lw=0, zorder=2)
for rr, xe in stair:
    ax.step(xe, rr, where="post", color=INK, lw=0.7, zorder=3)
ax.set_ylim(Z.shape[0], 0); ax.set_xlim(edges[0], edges[-1])
ax.set_yticks([t[0] for t in ticks]); ax.set_yticklabels([t[1] for t in ticks], rotation=90, va="center")
ax.tick_params(axis="y", length=0, pad=1.5)
for sp in ("left", "top", "right"): ax.spines[sp].set_visible(False)
ax.set_xticks([12, 24, 36, 48, 60, 72]); ax.set_xlabel("time (h)")
ax.text(73.5, 0.36 * nsens, "not observed", fontsize=F, ha="right", va="center", color="#6A6A6A", zorder=4)
# strip of wrong calls, aligned row by row with the raster
sax = ax.inset_axes([1.02, 0.0, 0.045, 1.0])
sax.set_ylim(Z.shape[0], 0); sax.set_xlim(0, 1)
sax.add_patch(Rectangle((0, 0), 1, nsens, color="#F4F4F4", lw=0))
sax.add_patch(Rectangle((0, nsens + GAP), 1, Z.shape[0] - nsens - GAP, color="#F4F4F4", lw=0))
sax.hlines(wrong_rows, 0, 1, color=RED, lw=0.45)
sax.axis("off")
sax.text(1.35, nsens + GAP + 0.5 * (Z.shape[0] - nsens - GAP), "wrong calls", fontsize=F, ha="left", va="center",
         rotation=90, color=RED)
cax = ax.inset_axes([0.36, 1.115, 0.60, 0.05])
cb = fig.colorbar(mpl.cm.ScalarMappable(Normalize(0, 1), CM_POST), cax=cax, orientation="horizontal",
                  ticks=[0, 0.2, 0.5, 0.8, 1])
cb.ax.tick_params(length=1.6, width=0.5, pad=1.2); cb.outline.set_linewidth(0.5)
cb.ax.set_xticklabels(["0", "0.2", "0.5", "0.8", "1"])
for v_ in (0.2, 0.8):
    cb.ax.axvline(v_, color=INK, lw=0.7)
cb.ax.text(0.1, 1.25, "stop", transform=cb.ax.transAxes, ha="center", va="bottom", fontsize=F)
cb.ax.text(0.9, 1.25, "stop", transform=cb.ax.transAxes, ha="center", va="bottom", fontsize=F)
ax.text(0.33, 1.14, r"$\hat p_t$", transform=ax.transAxes, ha="right", va="center", fontsize=F)
plabel(ax, "b", dx=-0.16, dy=1.03)

# ---- c: what has been decided by each time, and how well (stacked shares), against the fixed horizon
ax = fig.add_subplot(gs[0, 2])
xs = np.linspace(6, 74, 1361)
dec = t6s[None, :] <= xs[:, None]                                   # [x, unit] decided by time x
ok = (d6s == y)
s_sens = (dec & ok & (d6s == 1)).mean(1); s_res = (dec & ok & (d6s == 0)).mean(1); s_wrong = (dec & ~ok).mean(1)
s_dec = s_sens + s_res + s_wrong
polys = ax.stackplot(xs, s_sens, s_res, s_wrong, 1 - s_dec, colors=[OURS, "#F2B48F", RED, "#E9E9E9"], lw=0)
polys[2].set_hatch("/////"); polys[2].set_edgecolor("white")
late = xs >= f6                                                     # still observed beyond the fixed horizon
ax.fill_between(xs[late], s_dec[late], 1, facecolor="none", edgecolor=LOSS, hatch="\\\\\\\\", lw=0, zorder=3)
ax.plot(xs, s_dec, "-", color=INK, lw=0.8, zorder=4)
ax.plot([6, f6, f6, 74], [0, 0, 1, 1], "--", color=FIX, lw=1.0, zorder=5)
ax.text(f6 - 1.0, 0.975, f"fixed horizon\n{f6:.1f} h", fontsize=F, color=FIX, ha="right", va="top",
        linespacing=1.05, zorder=6)
ax.text(62, 0.20, "correct:\nsensitive", fontsize=F, color="white", ha="center", va="center", linespacing=1.05, zorder=6)
ax.text(62, 0.585, "correct:\nresistant", fontsize=F, color=INK, ha="center", va="center", linespacing=1.05, zorder=6)
iw = np.searchsorted(xs, 64.0)
ax.text(64, s_sens[iw] + s_res[iw] + 0.5 * s_wrong[iw], "wrong", fontsize=F, color=INK, ha="center", va="center",
        bbox=dict(facecolor="white", edgecolor="none", pad=0.6, alpha=0.95), zorder=6)
ax.text(9.0, 0.84, "still\nobserved", fontsize=F, color="#555555", ha="left", va="center", linespacing=1.05, zorder=6)
ax.set_xlim(6, 74); ax.set_ylim(0, 1.0); ax.set_xticks([12, 24, 36, 48, 60, 72]); ax.set_yticks([0, 0.5, 1])
ax.set_xlabel("time (h)"); ax.set_ylabel("share of organoids")
plabel(ax, "c", dx=-0.27, dy=1.03)
print(f"  OP6: mean tau {t6s.mean():.2f} h (fixed {f6} h, saving {f6 - t6s.mean():.1f} h), "
      f"wrong {(~ok).mean():.3f}, stop<72h {np.mean(t6s < 72):.3f}; wrong-call rows {len(wrong_rows)}")
save(fig, "Fig2_rq1")

# =============================================================== FIG 3 (RQ1): time-to-error matrix
TG = [0.03, 0.04, 0.06, 0.08, 0.12, 0.16, 0.20, 0.24]
NR = np.nan


def xb_row(key):
    r = xb[key]; mn = float(r["min_achievable_err"]); out = []
    for tg in TG:
        k = f"time_at_err_{tg:.2f}"
        if k in r and r[k] != "": out.append(float(r[k]))
        elif tg < mn: out.append(np.inf)
        else: out.append(NR)                                   # reachable but not recorded
    return out


mrows = [("fixed horizon", FIX, [t_reach(G, fx_e, tg) for tg in TG]),
         ("static + wait", FIX, xb_row("static_selective_wait")),
         ("RL (tabular Q)", FIX, xb_row("rl_qlearning_stopping")),
         ("myopic one-step", FIX, [t_reach(my_t, my_e, tg) for tg in TG]),
         ("TEASER", RED, [np.inf] * len(TG)),
         ("TEASER, cost criterion", RED, [t_reach(*ext["TEASER-cost"], tg) for tg in TG]),
         ("EconomyGamma", ECO, [t_reach(*ext["EconomyGamma"], tg) for tg in TG]),
         ("CALIMERA", CAL, [t_reach(*ext["CALIMERA"], tg) for tg in TG])]
for nb in (25, 50, 400):                    # the three that show the floor falling with the bins
    mrows.append((f"CertStop (Bellman, {nb} bins)", OURS_D, [t_reach(*bell[nb], tg) for tg in TG]))
mrows.append(("CertStop (margin)", OURS, [t_reach(cm_t, cm_e, tg) for tg in TG]))
M = np.array([r[2] for r in mrows], float)
fin = np.where(np.isfinite(M), M, np.nan)
behind = fin - np.nanmin(fin, 0)
nR, nC = M.shape
gap_after = {3: 0.38, 7: 0.38}              # visual gaps: classical | published | ours
ypos = []; yc = 0.0
for i in range(nR):
    ypos.append(yc); yc += 1.0 + gap_after.get(i, 0.0)
ypos = np.array(ypos)
fig = plt.figure(figsize=(W2, 2.50))
ax = fig.add_axes([0.218, 0.03, 0.668, 0.815])
norm_b = Normalize(0, 20)
for i in range(nR):
    for j in range(nC):
        v = M[i, j]; x0, y0 = j, ypos[i]
        if np.isinf(v):
            ax.add_patch(Rectangle((x0 + 0.03, y0 + 0.06), 0.94, 0.88, facecolor=NA, edgecolor="#C9C9C9", hatch="/////", lw=0.0))
            ax.text(x0 + 0.5, y0 + 0.5, "\u221e", ha="center", va="center", fontsize=9.5, color="#8E8E8E")
        elif np.isnan(v):
            ax.add_patch(Rectangle((x0 + 0.03, y0 + 0.06), 0.94, 0.88, facecolor="white", edgecolor="#DADADA", lw=0.5))
            ax.text(x0 + 0.5, y0 + 0.5, "n/r", ha="center", va="center", fontsize=F, color="#9A9A9A")
        else:
            b_ = behind[i, j]
            ax.add_patch(Rectangle((x0 + 0.03, y0 + 0.06), 0.94, 0.88, facecolor=CM_BEHIND(norm_b(b_)), lw=0))
            ax.text(x0 + 0.5, y0 + 0.5, f"{v:.1f}", ha="center", va="center", fontsize=F,
                    color="white" if b_ > 11 else INK, fontweight="bold" if b_ <= 1.0 else "normal")
ax.set_xlim(0, nC); ax.set_ylim(ypos[-1] + 1.0, 0)
ax.set_yticks(ypos + 0.5); ax.set_yticklabels([r[0] for r in mrows])
for tl, r in zip(ax.get_yticklabels(), mrows): tl.set_color(r[1])
ax.get_yticklabels()[-1].set_fontweight("bold")
ax.xaxis.tick_top(); ax.xaxis.set_label_position("top")
ax.set_xticks(np.arange(nC) + 0.5); ax.set_xticklabels([f"{t:.2f}" for t in TG])
ax.tick_params(length=0, pad=2)
for sp in ax.spines.values(): sp.set_visible(False)
ax.set_xlabel("target decision error (cells: hours to reach it)", labelpad=3)
cax = fig.add_axes([0.902, 0.10, 0.016, 0.66])
cb = fig.colorbar(mpl.cm.ScalarMappable(norm_b, CM_BEHIND), cax=cax, ticks=[0, 5, 10, 15, 20])
cb.ax.tick_params(length=1.8, width=0.5); cb.outline.set_linewidth(0.5)
cb.ax.set_yticklabels(["0", "5", "10", "15", "\u226520"])
cb.set_label("h behind the fastest", labelpad=2)
print("  Fig3 time-to-error matrix:")
for r in mrows:
    print(f"    {r[0]:<24}", " ".join(("  inf" if np.isinf(v) else ("  n/r" if np.isnan(v) else f"{v:5.1f}")) for v in r[2]))
save(fig, "Fig3_matrix")

# =============================================================== FIG 4 (RQ2, RQ3): shift matrix | regime maps
sm = rows("table_shift_matrix.csv")
settings = list(dict.fromkeys(r["setting"] for r in sm))
grp_of = lambda s: "cell line" if s.startswith("cross-cell") else ("drug" if s.startswith("leave-drug") else "dose")
CODE = {"cross-cell MCF-7->BT-474": r"M$\rightarrow$B", "cross-cell BT-474->MCF-7": r"B$\rightarrow$M",
        "leave-drug Lap": "Lap.", "leave-drug Ner": "Ner.", "leave-drug Stauro": "Stau.",
        "leave-conc 0.1uM": "0.1", "leave-conc 1.0uM": "1", "leave-conc 10.0uM": "10"}
cols_g = {"cell line": OURS_D, "drug": OURS, "dose": FIX}
H = np.full((4, len(settings)), np.nan); auc = []
for j, s in enumerate(settings):
    rr = [r for r in sm if r["setting"] == s]; auc.append(float(rr[0]["auc_24h"]))
    for r in rr:
        H[int(r["op"]) - 1, j] = float(r["hours_saved"]) if r["hours_saved"] else np.nan
fig = plt.figure(figsize=(W2, 2.42))
gs = GridSpec(1, 2, figure=fig, width_ratios=[1.75, 1.0], left=0.090, right=0.922, top=0.815, bottom=0.215, wspace=0.42)
sub = GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[0, 0], height_ratios=[0.55, 1.0], hspace=0.10)
axa = fig.add_subplot(sub[0]); ax = fig.add_subplot(sub[1])
norm_s = Normalize(0, 30)
for j in range(len(settings)):
    for k in range(4):
        v = H[k, j]
        if np.isnan(v):
            ax.add_patch(Rectangle((j + 0.03, k + 0.05), 0.94, 0.90, facecolor=NA, edgecolor="#C9C9C9", hatch="/////", lw=0))
            ax.text(j + 0.5, k + 0.5, "n/a", ha="center", va="center", fontsize=F, color="#8A8A8A")
        else:
            ax.add_patch(Rectangle((j + 0.03, k + 0.05), 0.94, 0.90, facecolor=CM_SAVED(norm_s(v)), lw=0))
            ax.text(j + 0.5, k + 0.5, f"{v:.1f}", ha="center", va="center", fontsize=F, color="white" if v > 15 else INK)
ax.set_xlim(0, len(settings)); ax.set_ylim(4, 0)
ax.set_yticks(np.arange(4) + 0.5); ax.set_yticklabels([f"{m_:.2f}" for m_ in np.linspace(0.10, 0.42, 4)])
ax.set_ylabel(r"band margin $\omega$")
ax.set_xticks(np.arange(len(settings)) + 0.5); ax.set_xticklabels([CODE[s] for s in settings])
for tl, s in zip(ax.get_xticklabels(), settings): tl.set_color(cols_g[grp_of(s)])
ax.tick_params(length=0, pad=2)
for sp in ax.spines.values(): sp.set_visible(False)
for xs_ in (2, 5): ax.axvline(xs_, color="white", lw=2.4)
for g_, x_ in (("cell line", 1.0), ("drug", 3.5), ("dose (µM)", 6.5)):
    ax.text(x_, 4.86, "held-out " + g_, ha="center", va="top", fontsize=F, color=cols_g[g_.split(" (")[0]], fontweight="bold")
cax = ax.inset_axes([1.022, 0.0, 0.022, 1.0])
cb = fig.colorbar(mpl.cm.ScalarMappable(norm_s, CM_SAVED), cax=cax, ticks=[0, 10, 20, 30])
cb.ax.tick_params(length=1.8, width=0.5); cb.outline.set_linewidth(0.5)
cb.set_label("hours saved", labelpad=2)
for j, (s, a_) in enumerate(zip(settings, auc)):
    axa.vlines(j + 0.5, 0.79, a_, color=LGRY, lw=1.8)
    axa.plot(j + 0.5, a_, "o", color=cols_g[grp_of(s)], ms=4.0)
    axa.text(j + 0.5, a_ + 0.007, f"{a_:.2f}", fontsize=F, va="bottom", ha="center", color=INK)
axa.set_xlim(0, len(settings)); axa.set_ylim(0.79, 0.935); axa.set_yticks([0.80, 0.90])
axa.set_xticks([]); axa.spines["bottom"].set_visible(False)
axa.set_ylabel("AUC")
plabel(axa, "a", dx=-0.165, dy=1.06)
print(f"  Fig4a: hours saved {np.nanmin(H):.1f}..{np.nanmax(H):.1f} over {np.isfinite(H).sum()} matched cells; "
      f"all positive: {bool((H[np.isfinite(H)] > 0).all())}; at widest margin {np.nanmin(H[3]):.1f}..{np.nanmax(H[3]):.1f}")

# ---- b: regime maps (analytic): share of alpha left after each construction's correction
subb = GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[0, 1], wspace=0.13)
NG = 52; DEL = 0.10
ng = np.logspace(np.log10(3), np.log10(5000), 320); ag = np.linspace(0.02, 0.40, 240)
NNg, AAg = np.meshgrid(ng, ag)
share_h = 1 - np.sqrt(np.log(2 * NG / DEL) / (2 * NNg)) / AAg
share_c = 1 - (1 - AAg) / (NNg * AAg)
n_unit = float(np.mean([len(np.where(groups == g_)[0]) for g_ in np.unique(groups)])) * 9
maps = []
for k, (shr, ttl) in enumerate(((share_h, "Hoeffding\n(Thm. 3)"), (share_c, "condition CRC\n(Thm. 4)"))):
    axr = fig.add_subplot(subb[0, k]); maps.append(axr)
    cmr = CM_SHARE.copy(); cmr.set_bad(NA)
    axr.pcolormesh(ng, ag, np.ma.masked_less(shr, 0.0), cmap=cmr, vmin=0, vmax=1, shading="auto", rasterized=True)
    axr.contourf(ng, ag, (shr < 0).astype(float), levels=[0.5, 1.5], colors="none", hatches=["/////"])
    axr.contour(ng, ag, shr, levels=[0.0], colors=INK, linewidths=0.9)
    axr.contour(ng, ag, shr, levels=[0.5], colors=INK, linewidths=0.55, linestyles=":")
    axr.set_xscale("log"); axr.set_xlim(ng[0], ng[-1]); axr.set_ylim(ag[0], ag[-1])
    axr.set_xticks([10, 100, 1000]); axr.set_xticklabels(["10", "100", "1k"])
    axr.xaxis.set_minor_locator(mpl.ticker.NullLocator())
    axr.axvline(9, color=RED, lw=0.8, ls="--")
    for sp in ("top", "right"): axr.spines[sp].set_visible(True)
    axr.set_title(ttl, fontsize=F, pad=3, linespacing=1.05)
    axr.set_yticks([0.1, 0.2, 0.3, 0.4])
    if k == 0:
        axr.set_ylabel(r"risk target $\alpha$")
        axr.plot([n_unit] * 2, [0.15, 0.20], "*", color="white", mec=INK, mew=0.6, ms=8.0, zorder=5)
        axr.plot([9, 9], [0.15, 0.20], "X", color=RED, mec="white", mew=0.4, ms=5.6, zorder=5)
    else:
        axr.set_yticklabels([])
        axr.plot([9, 9, 9], [0.10, 0.15, 0.20], "o", color=OURS, mec="white", mew=0.5, ms=5.0, zorder=5)
cax = maps[1].inset_axes([1.09, 0.0, 0.09, 1.0])
cb = fig.colorbar(mpl.cm.ScalarMappable(Normalize(0, 1), CM_SHARE), cax=cax, ticks=[0, 0.5, 1])
cb.ax.tick_params(length=1.8, width=0.5); cb.outline.set_linewidth(0.5)
cb.ax.set_yticklabels(["0", "0.5", "1"]); cb.set_label(r"share of $\alpha$ left", labelpad=2)
p0 = maps[0].get_position(); p1 = maps[1].get_position()
fig.text((p0.x0 + p1.x1) / 2, 0.050, "calibration units ($n$ or $m$)", ha="center", va="bottom", fontsize=F)
plabel(maps[0], "b", dx=-0.60, dy=1.21)
print(f"  Fig4b: unit-level n ~ {n_unit:.0f}; U_n there {np.sqrt(np.log(2*NG/DEL)/(2*n_unit)):.3f}; "
      f"Hoeffding at m=9: {np.sqrt(np.log(2*NG/DEL)/18):.3f}; CRC deflated target at m=9, a=0.2: {0.2 - 0.8/9:.3f}")
save(fig, "Fig4_shift")

# =============================================================== FIG 5 (RQ3): raincloud | price vs coverage
crc = []
with open(os.path.join(POST, "table_constrained_control_crc.csv"), encoding="utf-8") as f:
    rd = csv.reader(f); hdr = next(rd)
    for r in rd:
        if not r or r[0].startswith("#"): break
        crc.append(dict(zip(hdr, r)))
summ = {}
with open(os.path.join(POST, "table_constrained_control_crc.csv"), encoding="utf-8") as f:
    txt = f.read().split("# summary")[1].strip().splitlines()
    sh = txt[0].split(",")
    for l in txt[1:]:
        if not l.strip(): break                 # a "# control" block follows the summary
        r = dict(zip(sh, l.split(",")))
        summ[(float(r["alpha"]), r["family"], r["selection"])] = r

fig = plt.figure(figsize=(W2, 2.80))
gs = GridSpec(1, 2, figure=fig, width_ratios=[1.45, 1.0], left=0.075, right=0.988, top=0.925, bottom=0.215, wspace=0.27)
# ---- a: raincloud of the per-split test risk
ax = fig.add_subplot(gs[0, 0])
arms = [("no-slack", "none", GRY), ("ltt-organoid", "unit\nLTT", AMB), ("hoeffding-organoid", "unit\nThm. 3", OURS_D),
        ("crc-condition", "cond.\nThm. 4", OURS)]
rng = np.random.default_rng(1)
xt, xl = [], []; xpos = 0.0
for ai, a in enumerate((0.15, 0.20)):
    x0 = xpos
    for sel, nm, c in arms:
        rr = [r for r in crc if r["family"] == "margin" and r["selection"] == sel and abs(float(r["alpha"]) - a) < 1e-9]
        risk = np.array([max(float(r["test_FP_pooled"]), float(r["test_FN_pooled"])) for r in rr])
        viol = risk > a
        kde = gaussian_kde(risk, bw_method=0.45)
        yy_ = np.linspace(risk.min() - 0.012, risk.max() + 0.012, 200); dd_ = kde(yy_); dd_ = dd_ / dd_.max() * 0.34
        ax.fill_betweenx(yy_, xpos - dd_, xpos, color=c, alpha=0.38, lw=0)
        ax.plot(xpos - dd_, yy_, color=c, lw=0.7)
        q1, q2, q3 = np.percentile(risk, [25, 50, 75])
        ax.add_patch(Rectangle((xpos + 0.03, q1), 0.07, q3 - q1, facecolor="white", edgecolor=INK, lw=0.6, zorder=3))
        ax.plot([xpos + 0.03, xpos + 0.10], [q2, q2], color=INK, lw=1.0, zorder=4)
        jit = rng.uniform(0.15, 0.40, len(risk))
        ax.scatter(xpos + jit[~viol], risk[~viol], s=6, color=c, edgecolor="none", alpha=0.85, zorder=3)
        ax.scatter(xpos + jit[viol], risk[viol], s=13, color=RED, marker="X", lw=0.2, zorder=4)
        tm = np.mean([float(r["test_mean_time_h"]) for r in rr])
        ax.text(xpos + 0.05, 0.318, f"{(~viol).sum()}/{len(risk)}", ha="center", va="center", fontsize=F,
                color=RED if viol.any() else INK, fontweight="bold")
        ax.text(xpos + 0.05, 0.296, f"{tm:.1f} h", ha="center", va="center", fontsize=F, color=FIX)
        xt.append(xpos + 0.05); xl.append(nm); xpos += 1.0
    ax.hlines(a, x0 - 0.45, xpos - 0.52, color=RED, lw=1.0, ls="--", zorder=2)
    ax.text(xpos - 0.54, a + 0.004, rf"$\alpha$ = {a:.2f}", color=RED, fontsize=F, va="bottom", ha="right")
    if ai == 0:
        ax.axvline(xpos - 0.5, color=LGRY, lw=0.7); xpos += 0.15
ax.set_xticks(xt); ax.set_xticklabels(xl, linespacing=1.05)
ax.set_xlim(-0.55, xpos - 0.45); ax.set_ylim(0.0, 0.335); ax.set_yticks([0, 0.1, 0.2, 0.3])
ax.tick_params(axis="x", length=0, pad=3)
ax.set_ylabel("test risk  max(FP, FN)"); ax.set_xlabel("certificate used to select the rule")
plabel(ax, "a", dx=-0.125, dy=1.01)

# ---- b: price vs per-condition coverage (numbers on the condition rule: alpha)
ax = fig.add_subplot(gs[0, 1])
AL = [0.10, 0.15, 0.20, 0.25]
series = [("margin", "no-slack", "no slack", GRY, "o", ":"),
          ("margin", "hoeffding-organoid", "unit cert.", OURS_D, "s", "--"),
          ("fixed", "crc-condition", "fixed $T$, cond.", FIX, "^", "-."),
          ("margin", "crc-condition", "condition rule", OURS, "o", "-")]
for fam, sel, nm, c, mk, ls in series:
    xx_ = np.array([float(summ[(a, fam, sel)]["mean_time_h"]) for a in AL])
    yy_ = np.array([float(summ[(a, fam, sel)]["frac_conditions_within_alpha"]) for a in AL])
    lo_ = np.array([float(summ[(a, fam, sel)]["frac_ci_lo"]) for a in AL])
    hi_ = np.array([float(summ[(a, fam, sel)]["frac_ci_hi"]) for a in AL])
    ax.errorbar(xx_, yy_, yerr=[yy_ - lo_, hi_ - yy_], ls=ls, marker=mk, color=c, lw=1.0, ms=4.2 if mk != "^" else 4.6,
                mfc=c if sel != "no-slack" else "white", mew=0.9, elinewidth=0.6, capsize=0, zorder=3, label=nm)
    if sel == "crc-condition" and fam == "margin":
        spots = {0.25: (11.6, 0.805, "center"), 0.20: (20.6, 0.875, "center"), 0.15: (35.8, 1.03, "center"), 0.10: (72.0, 1.03, "center")}
        for a, x_, y_ in zip(AL, xx_, yy_):
            sx, sy, ha_ = spots[a]
            ax.annotate(f"{a:.2f}", (x_, y_), xytext=(sx, sy), fontsize=F, color=OURS_D, ha=ha_, va="bottom", bbox=HALO,
                        arrowprops=dict(arrowstyle="-", lw=0.5, color=OURS_D, shrinkA=0.5, shrinkB=3.0) if a in (0.25, 0.20) else None)
ax.set_xlim(7, 79); ax.set_ylim(0.33, 1.10); ax.set_xticks([20, 40, 60]); ax.set_yticks([0.4, 0.6, 0.8, 1.0])
ax.set_xlabel("mean decision time (h)"); ax.set_ylabel(r"conditions with risk $\leq\alpha$")
ax.legend(loc="lower right", bbox_to_anchor=(1.02, -0.02), handlelength=2.0, labelspacing=0.2, handletextpad=0.5,
          borderpad=0.2, frameon=True, facecolor="white", edgecolor="none", framealpha=0.94)
plabel(ax, "b", dx=-0.21, dy=1.01)
save(fig, "Fig5_cert")

# =============================================================== FIG 6 (RQ4-RQ5): reliability | gamma path | Corollary 1
cr = rows("table_stage3a_calib_regret.csv")


def ece_bin(p, yy, nb=10):
    bb = np.linspace(0, 1, nb + 1); e = 0.0
    for i in range(nb):
        mk = (p >= bb[i]) & ((p < bb[i + 1]) if i < nb - 1 else (p <= bb[i + 1]))
        if mk.any(): e += mk.mean() * abs(yy[mk].mean() - p[mk].mean())
    return e


def ece_stop(ps, ts, yy):
    return sum((ts == t).mean() * ece_bin(ps[ts == t], yy[ts == t]) for t in np.unique(ts))


# Stage-3A rule (stage3a_calib_regret.py): in-sample 50-bin kernel, lambda = 0.004, gamma-scaled logits
def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6); return np.log(p / (1 - p))


Bb51 = np.linspace(0, 1, 51); bc51 = (Bb51[:-1] + Bb51[1:]) / 2; rho51 = np.minimum(1 - bc51, bc51)


def dp_band_3a(Pm, lam=0.004):
    nb = len(bc51); Ks = []
    for j in range(T - 1):
        K = np.zeros((nb, nb))
        a = np.clip(np.digitize(Pm[:, j], Bb51) - 1, 0, nb - 1); b = np.clip(np.digitize(Pm[:, j + 1], Bb51) - 1, 0, nb - 1)
        np.add.at(K, (a, b), 1); K += 1e-3; K /= K.sum(1, keepdims=True); Ks.append(K)
    V = rho51.copy(); ab = np.full((T, 2), 0.5)
    for j in range(T - 2, -1, -1):
        cont = lam * (G[j + 1] - G[j]) + Ks[j] @ V
        cs = np.where(cont < rho51 - 1e-9)[0]
        if len(cs): ab[j] = [bc51[cs.min()], bc51[cs.max()]]
        V = np.minimum(rho51, cont)
    inside = (Pm[:, :T - 1] > ab[:T - 1, 0]) & (Pm[:, :T - 1] < ab[:T - 1, 1])
    j = np.where((~inside).any(1), (~inside).argmax(1), T - 1)
    return G[j], (Pm[np.arange(len(Pm)), j] >= 0.5).astype(int), Pm[np.arange(len(Pm)), j]


GSHOW = [0.5, 0.9, 3.0]


def gcol(g):
    g = np.asarray(g, float)
    u = np.where(g <= 0.9, 0.5 * np.log(g / 0.5) / np.log(0.9 / 0.5), 0.5 + 0.5 * np.log(g / 0.9) / np.log(4.0 / 0.9))
    return CM_GAMMA(np.clip(u, 0, 1))


rel = {}
for g_ in GSHOW:
    Pg = 1 / (1 + np.exp(-g_ * logit(P)))
    tt_, dd_, ps_ = dp_band_3a(Pg)
    ref = [r for r in cr if r["rule"] == "dp_band" and abs(float(r["gamma"]) - g_) < 1e-9][0]
    es_ = ece_stop(ps_, tt_, y)
    print(f"  Fig6a gamma={g_}: ECE_stop {es_:.4f} (CSV {float(ref['ece_stop_strat']):.4f}), mean tau {tt_.mean():.1f} h")
    rel[g_] = (ps_, tt_, float(ref["ece_stop_strat"]), Pg[:, 4])

print("computing Corollary-1 check ...")


def cor1_points(Pm, yy, Gm, cfp, cfn, ms):
    pst = cfp / (cfp + cfn); L = max(cfp, cfn); out = []
    for m in ms:
        t_, d_, p_ = margin_rule(Pm, Gm, m, pst)
        bel = np.minimum(cfp * (1 - p_), cfn * p_).mean()
        rea = (cfp * ((d_ == 1) & (yy == 0)) + cfn * ((d_ == 0) & (yy == 1))).mean()
        out.append(("margin", L * ece_stop(p_, t_, yy), abs(rea - bel), rea - bel))
    for j in range(Pm.shape[1]):
        p_ = Pm[:, j]; d_ = (p_ >= pst).astype(int)
        bel = np.minimum(cfp * (1 - p_), cfn * p_).mean()
        rea = (cfp * ((d_ == 1) & (yy == 0)) + cfn * ((d_ == 0) & (yy == 1))).mean()
        out.append(("fixed", L * ece_bin(p_, yy), abs(rea - bel), rea - bel))
    return out


DDIR = os.path.join(DATA_DIR, "cmapss")


def cmapss_posterior(subset, grid, regimes):
    raw = np.loadtxt(os.path.join(DDIR, f"train_{subset}.txt"))
    eng = raw[:, 0].astype(int); cyc = raw[:, 1].astype(int); opset = raw[:, 2:5]; S = raw[:, 5:26].copy()
    if regimes > 1:
        lab = KMeans(n_clusters=regimes, n_init=5, random_state=0).fit(opset).labels_
        for r_ in range(regimes):
            m = lab == r_; mu = S[m].mean(0); sd = S[m].std(0); sd[sd < 1e-8] = 1.0; S[m] = (S[m] - mu) / sd
    S = S[:, np.where(S.var(0) > 1e-6)[0]]
    engines = np.unique(eng); maxcyc = {e: cyc[eng == e].max() for e in engines}
    med = np.median(list(maxcyc.values())); ya = np.array([int(maxcyc[e] <= med) for e in engines])
    Pm = np.zeros((len(engines), len(grid)))
    for j, Tq in enumerate(grid):
        X, yy, keep = [], [], []
        for i, e in enumerate(engines):
            m = (eng == e) & (cyc <= Tq)
            if m.sum() < 2: continue
            Sp = S[m]; tp = cyc[m]
            X.append(sum(([Sp[-1, k], (Sp[-1, k] - Sp[0, k]) / max(tp[-1] - tp[0], 1)] for k in range(Sp.shape[1])), []))
            yy.append(ya[i]); keep.append(i)
        X = np.array(X); yy = np.array(yy); keep = np.array(keep)
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
        Pm[keep, j] = cross_val_predict(clf, X, yy, cv=GroupKFold(5).split(X, yy, keep), method="predict_proba")[:, 1]
    return Pm, ya, np.array(grid, float)


ph = np.load(os.path.join(POST, "physionet_posterior_data.npz"))
dom_pts = {
    "organoids": cor1_points(P, y, G, 1.0, 1.0, np.linspace(0, 0.48, 25)),
    "sepsis": cor1_points(ph["P"], ph["y"].astype(int), ph["GRID"].astype(float), 1.0, 9.0, np.linspace(0, 0.15, 25)),
}
for sub_, reg in (("FD001", 1), ("FD004", 6)):
    Pm, ya, Gm = cmapss_posterior(sub_, [20, 30, 40, 50, 60, 70, 80], reg)
    dom_pts[sub_] = cor1_points(Pm, ya, Gm, 1.0, 3.0, np.linspace(0, 0.35, 25))
ntot = 0; nok = 0
for k_, v in dom_pts.items():
    bnd = np.array([p[1] for p in v]); gap = np.array([p[2] for p in v])
    ntot += len(v); nok += int((gap <= bnd + 1e-12).sum())
    print(f"  {k_}: gap <= L*ECE_stop for {np.mean(gap <= bnd + 1e-12):.2f} of {len(v)} policies; max gap {gap.max():.4f}")
print(f"  Corollary 1 holds on {nok}/{ntot} policies")
# Stage 3Q: with ECE bins that have an edge at p* (organoids p*=0.5, sepsis p*=0.1) the sample version of
# Corollary 1 holds by the triangle inequality; on turbofan (p*=0.25) one bin straddles p* and the check is a
# genuine test. What the panel is for is the ratio: how much of the bound the believed-realized gap uses.
for k_, v in dom_pts.items():
    bnd = np.array([p_[1] for p_ in v]); gap = np.array([p_[2] for p_ in v]); r_ = gap / np.maximum(bnd, 1e-12)
    print(f"  {k_}: gap/(L*ECE_stop) median {np.median(r_):.3f}, max {r_.max():.3f}, "
          f"share above 0.5 {np.mean(r_ > 0.5):.2f}, above 0.1 {np.mean(r_ > 0.1):.2f}  (n={len(v)})")

# organoid realized-cost audit (same protocol as crossdomain_realized.py, halves split by CONDITION)
print("organoid realized-cost audit ...")
LAM_O = np.geomspace(5e-4, 5e-2, 20); MSO = np.linspace(0, 0.48, 49)
mar = [margin_rule(P, G, m) for m in MSO]
fix = [(np.full(N, G[j]), (P[:, j] >= 0.5).astype(int), P[:, j]) for j in range(T)]
gss = list(GroupShuffleSplit(n_splits=20, test_size=0.5, random_state=0).split(P, y, groups))
org_gb, org_gh = [], []
for lam in LAM_O:
    Rm = np.array([lam * t_ + (d_ != y) for t_, d_, _ in mar]); Rf = np.array([lam * t_ + (d_ != y) for t_, d_, _ in fix])
    Bm = np.array([lam * t_ + np.minimum(1 - p_, p_) for t_, _, p_ in mar]); Bf = np.array([lam * t_ + np.minimum(1 - p_, p_) for t_, _, p_ in fix])
    org_gb.append(Bf.mean(1).min() - Bm.mean(1).min())
    org_gh.append(np.mean([Rf[int(np.argmin(Rf[:, a].mean(1))), b].mean() - Rm[int(np.argmin(Rm[:, a].mean(1))), b].mean() for a, b in gss]))
with open(os.path.join(POST, "table_organoid_realized_audit.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f); w.writerow(["lambda", "gain_heldout_mean"])
    for lam, g_ in zip(LAM_O, org_gh): w.writerow([round(float(lam), 6), round(float(g_), 5)])

fig = plt.figure(figsize=(W2, 2.90))
gs = GridSpec(1, 3, figure=fig, width_ratios=[1.0, 1.0, 1.12], left=0.075, right=0.978, top=0.925, bottom=0.165, wspace=0.43)
# ---- a: reliability at the stopping time; line style and marker differ per gamma (grayscale-safe)
sub = GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[0, 0], height_ratios=[1.0, 0.24], hspace=0.07)
ax = fig.add_subplot(sub[0]); axh = fig.add_subplot(sub[1], sharex=ax)
ax.plot([0, 1], [0, 1], color=INK, lw=0.7, ls=(0, (1, 1.6)))
bb = np.linspace(0, 1, 11)
GST = {0.5: ((0, (4, 1.6)), "s"), 0.9: ("-", "o"), 3.0: ((0, (5, 1.5, 1, 1.5)), "^")}
for g_ in GSHOW:
    ps_, tt_, es_, _ = rel[g_]
    xm, ym, nn = [], [], []
    for i in range(10):
        mk = (ps_ >= bb[i]) & ((ps_ < bb[i + 1]) if i < 9 else (ps_ <= bb[i + 1]))
        if mk.sum() >= 15: xm.append(ps_[mk].mean()); ym.append(y[mk].mean()); nn.append(mk.sum())
    c = gcol(g_); ls_, mk_ = GST[g_]
    ax.plot(xm, ym, ls=ls_, color=c, lw=1.2 if g_ != 0.9 else 1.8, zorder=3 if g_ == 0.9 else 2)
    ax.scatter(xm, ym, s=np.sqrt(nn) * 1.15, marker=mk_, color=c, edgecolor="white", lw=0.35, zorder=4)
    ax.plot([], [], ls=ls_, marker=mk_, color=c, lw=1.2, ms=4.0, mec="white", mew=0.35, label=rf"$\gamma$ = {g_:g}")
ax.set_xlim(0, 1); ax.set_ylim(0, 1)
ax.set_ylabel("observed frequency"); ax.set_yticks([0, 0.5, 1])
plt.setp(ax.get_xticklabels(), visible=False)
ax.legend(loc="lower right", bbox_to_anchor=(1.03, -0.03), handlelength=2.1, handletextpad=0.4, labelspacing=0.18,
          borderpad=0.2, frameon=True, facecolor="white", edgecolor="none", framealpha=0.92)
ps9, _, _, p24 = rel[0.9]
hb = np.linspace(0, 1, 21)
axh.hist(p24, bins=hb, density=True, histtype="step", color=FIX, lw=0.9)
axh.hist(ps9, bins=hb, density=True, color=OURS, alpha=0.55, lw=0)
axh.set_yticks([]); axh.spines["left"].set_visible(False)
axh.set_xlabel(r"predicted $\hat p_\tau$"); axh.set_xticks([0, 0.5, 1]); axh.set_xticklabels(["0", "0.5", "1"])
plabel(ax, "a", dx=-0.30, dy=1.01)

# ---- b: gamma path, excess vs ECE_stop with the 2L*ECE_stop term (an estimate, not a certified bound; ring: gamma = 0.9)
ax = fig.add_subplot(gs[0, 1])
xx = np.linspace(0, 0.26, 10)
ax.fill_between(xx, 2 * xx, 0.5, color=RED, alpha=0.07, lw=0)
ax.plot(xx, 2 * xx, "-", color=INK, lw=1.0)
ax.text(0.047, 0.080, r"$2L\,\mathrm{ECE}_\mathrm{stop}$", fontsize=F, va="center", ha="left")
for rule in ("margin_0.30", "dp_band"):
    rr = sorted([r for r in cr if r["rule"] == rule], key=lambda r: float(r["gamma"]))
    xs_ = np.array([float(r["ece_stop_strat"]) for r in rr]); ys_ = np.array([float(r["excess"]) for r in rr])
    gs_ = np.array([float(r["gamma"]) for r in rr])
    ax.plot(xs_, ys_, "-", color=LGRY if rule != "dp_band" else GRY, lw=0.8, zorder=2)
    if rule == "dp_band":
        ax.scatter(xs_, ys_, c=gcol(gs_), s=17, edgecolor="white", lw=0.35, zorder=4)
        k9 = int(np.argmin(np.abs(gs_ - 0.9)))
        ax.scatter([xs_[k9]], [ys_[k9]], s=62, facecolor="none", edgecolor=INK, lw=0.9, zorder=5)
        ax.text(xs_[-1] + 0.006, ys_[-1], r"$\gamma$=4", fontsize=F, va="center", ha="left", color=INK)
        k5 = int(np.argmin(np.abs(gs_ - 0.5)))
        ax.annotate(r"$\gamma$=0.5", (xs_[k5], ys_[k5]), xytext=(0.062, 0.066), fontsize=F, color=INK, ha="center", va="center",
                    arrowprops=dict(arrowstyle="-", lw=0.5, color=INK, shrinkA=1.0, shrinkB=3.0))
    else:
        ax.scatter(xs_, ys_, marker="s", s=11, facecolor="white", edgecolor=FIX, lw=0.6, zorder=3)
ax.set_xlim(0, 0.26); ax.set_ylim(-0.003, 0.14); ax.set_xticks([0, 0.1, 0.2]); ax.set_yticks([0, 0.05, 0.10])
ax.set_xlabel(r"$\mathrm{ECE}_\mathrm{stop}$"); ax.set_ylabel("excess cost")
ax.scatter([], [], s=17, color=OURS, label="CertStop (Bellman)")
ax.scatter([], [], s=11, marker="s", facecolor="white", edgecolor=FIX, lw=0.6, label="CertStop (margin)")
ax.legend(loc="upper right", bbox_to_anchor=(1.03, 0.90), handletextpad=0.2, borderpad=0.2, labelspacing=0.18,
          frameon=True, facecolor="white", edgecolor="none", framealpha=0.92)
plabel(ax, "b", dx=-0.33, dy=1.01)

# ---- c: Corollary 1 across four domains (filled: margin rules; open: fixed horizons)
ax = fig.add_subplot(gs[0, 2])
# domain hues carry no meaning elsewhere in the paper (purple/green/red = CALIMERA/EconomyGamma/violation); markers differ too
dstyle = {"organoids": (OURS, "o"), "sepsis": ("#8C564B", "D"), "FD004": ("#CC79A7", "^"), "FD001": ("#2A8C8C", "s")}
FLOOR = 3e-4
lim = [FLOOR, 1.0]
ax.fill_between(lim, lim, [1, 1], color=RED, alpha=0.07, lw=0)
for fct in (0.1, 0.01):
    ax.plot(lim, [l_ * fct for l_ in lim], ":", color=GRY, lw=0.7)
for k_ in ("organoids", "sepsis", "FD001", "FD004"):
    v = dom_pts[k_]; c, mk = dstyle[k_]
    for kind in ("margin", "fixed"):
        pts = [(p[1], p[2]) for p in v if p[0] == kind]
        xb_ = np.maximum([p[0] for p in pts], FLOOR); yb_ = np.maximum([p[1] for p in pts], FLOOR)
        ax.scatter(xb_, yb_, s=13 if kind == "margin" else 17, marker=mk, color=c if kind == "margin" else "white",
                   edgecolor=c, lw=0.8, alpha=0.88, label=k_ if kind == "margin" else None, zorder=3)
ax.plot(lim, lim, "-", color=INK, lw=1.0)
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(*lim); ax.set_ylim(*lim)
ax.set_xticks([1e-3, 1e-2, 1e-1, 1]); ax.set_yticks([1e-3, 1e-2, 1e-1, 1])
ax.text(0.33, 0.33 * 0.1, "10%", fontsize=F, color=FIX, ha="center", va="center", rotation=49, bbox=HALO)
ax.text(0.33, 0.33 * 0.01, "1%", fontsize=F, color=FIX, ha="center", va="center", rotation=49, bbox=HALO)
ax.text(4.2e-4, 7.0e-3, "violated", fontsize=F, color=RED, ha="left", va="center", style="italic")
ax.set_xlabel(r"$L\cdot\mathrm{ECE}_\mathrm{stop}$"); ax.set_ylabel(r"|believed $-$ realized cost|")
ax.legend(loc="upper left", bbox_to_anchor=(-0.02, 1.03), handletextpad=0.1, labelspacing=0.18, borderpad=0.2,
          frameon=True, facecolor="white", edgecolor="none", framealpha=0.92)
plabel(ax, "c", dx=-0.30, dy=1.01)
save(fig, "Fig6_calib")

# =============================================================== FIG 7 (RQ5): domains barcode | UCR-40 CD diagram | per-dataset strip
# Stage 3Q: the old Figs 7 and 8 merged into one RQ5 composite. The old right block (four hand-picked UCR sets) is
# superseded by the 40-dataset benchmark: its ranks are drawn as a Demsar-style critical-difference diagram whose
# cliques come from pairwise Wilcoxon signed-rank tests with Holm correction over all 21 pairs (not Nemenyi), and a
# matrix panel shows the outcome on every dataset (names in Supplementary Fig. S8).
from scipy.stats import wilcoxon, rankdata
TOL = 1e-3
cd = rows("table_crossdomain_realized.csv")


def lam_series(dom, key):
    rr = sorted([r for r in cd if r["domain"] == dom], key=lambda r: float(r["lambda"]))
    return np.array([float(r[key]) for r in rr])


left = [("organoids", [("believed", np.array(org_gb), "b"), ("realized", np.array(org_gh), OURS)]),
        ("turbofan FD004", [("believed", lam_series("C-MAPSS-FD004", "gain_believed_oracle"), "b"),
                            ("realized", lam_series("C-MAPSS-FD004", "gain_heldout_mean"), OURS)]),
        ("turbofan FD001", [("believed", lam_series("C-MAPSS-FD001", "gain_believed_oracle"), "b"),
                            ("realized", lam_series("C-MAPSS-FD001", "gain_heldout_mean"), OURS)]),
        ("sepsis", [("believed", lam_series("PhysioNet2019", "gain_believed_oracle"), "b"),
                    ("realized", lam_series("PhysioNet2019", "gain_heldout_mean"), OURS)])]
LABCOL = {"believed": "#7A7A7A", "realized": OURS_D}

# ---- UCR-40: mean realized cost over the eight delay weights, ranks, pairwise Wilcoxon-Holm cliques
ub = [r for r in rows("table_etsc_ucr_binary40.csv") if r["posterior"] == "raw"]
M7 = ["const_margin(train-selected)", "CALIMERA", "EconomyGamma", "bayes_dp_band(train-kernel)",
      "fixed(train-selected)", "TEASER-cost", "TEASER"]
NM7 = {"const_margin(train-selected)": "CertStop (margin)", "bayes_dp_band(train-kernel)": "CertStop (Bellman)",
       "CALIMERA": "CALIMERA", "EconomyGamma": "EconomyGamma", "fixed(train-selected)": "fixed horizon",
       "TEASER-cost": "TEASER (cost)", "TEASER": "TEASER"}
COL7 = {"const_margin(train-selected)": OURS, "bayes_dp_band(train-kernel)": OURS_D, "CALIMERA": CAL,
        "EconomyGamma": ECO, "fixed(train-selected)": FIX, "TEASER-cost": RED, "TEASER": RED}
dsets7 = sorted({r["dataset"] for r in ub})
J7 = {(r["dataset"], r["method"], float(r["lambda"])): float(r["J_method"]) for r in ub}
lam7 = {d_: sorted({float(r["lambda"]) for r in ub if r["dataset"] == d_}) for d_ in dsets7}
Mc = np.array([[np.mean([J7[d_, m_, l_] for l_ in lam7[d_]]) for m_ in M7] for d_ in dsets7])      # [40, 7]
avg7 = np.array([rankdata(np.round(r_, 6)) for r_ in Mc]).mean(0)
pairs7 = [(i, j) for i in range(len(M7)) for j in range(i + 1, len(M7))]
pv7 = [1.0 if np.allclose(Mc[:, i] - Mc[:, j], 0) else wilcoxon(Mc[:, i] - Mc[:, j], zero_method="wilcox").pvalue
       for i, j in pairs7]
PADJ = {}; run_ = 0.0
for k_, ix in enumerate(np.argsort(pv7)):                     # Holm step-down over all 21 pairs
    run_ = max(run_, (len(pv7) - k_) * pv7[ix]); i, j = pairs7[ix]; PADJ[i, j] = PADJ[j, i] = min(1.0, run_)
ord7 = list(np.argsort(avg7))
cliq7 = []
for a_ in range(len(ord7)):
    b_ = a_
    while b_ + 1 < len(ord7) and all(PADJ[ord7[x], ord7[b_ + 1]] > 0.05 for x in range(a_, b_ + 1)):
        b_ += 1
    if b_ > a_ and not any(c0 <= a_ and b_ <= c1 for c0, c1 in cliq7):
        cliq7.append((a_, b_))
print("  Fig7b mean ranks:", {NM7[M7[i]]: round(float(avg7[i]), 3) for i in ord7})
print("  Fig7b Wilcoxon-Holm cliques:", [[NM7[M7[ord7[x]]] for x in range(c0, c1 + 1)] for c0, c1 in cliq7])
print("  Fig7b Holm p (margin vs others):", {NM7[M7[j]]: round(PADJ[0, j], 4) for j in range(1, len(M7))})
print("  Fig7b Holm p (fixed vs others):", {NM7[M7[j]]: round(PADJ[4, j], 4) for j in range(len(M7)) if j != 4})
# per-dataset wins minus losses against the fixed horizon chosen on training data (like Supplementary Fig. S8c)
strip7 = [m_ for m_ in [M7[i] for i in ord7] if m_ != "fixed(train-selected)"]
gain7 = {(d_, m_): np.array([J7[d_, "fixed(train-selected)", l_] - J7[d_, m_, l_] for l_ in lam7[d_]])
         for d_ in dsets7 for m_ in M7}
dord7 = sorted(dsets7, key=lambda d_: -np.mean(gain7[d_, "const_margin(train-selected)"]))
S7 = np.array([[int((gain7[d_, m_] > TOL).sum()) - int((gain7[d_, m_] < -TOL).sum()) for d_ in dord7] for m_ in strip7])
CMWL = LinearSegmentedColormap.from_list("wl", [LOSS, "#F6C9B4", "#FFFFFF", "#A8CEE8", OURS_D])

fig = plt.figure(figsize=(W2, 3.05))
# ---- a: domains, believed vs realized win/tie/loss barcode
ax = fig.add_axes([0.088, 0.035, 0.300, 0.905])
yc = 0.0; yt, ytl = [], []
print("  Fig7a W/T/L (TOL=1e-3):")
for gi, (gname, rows_) in enumerate(left):
    ax.text(0.0, yc + 0.5, gname, fontsize=F, ha="left", va="center", color=INK,
            fontweight="bold" if gname == "organoids" else "normal")
    if gi == 0: ax.text(1.03, yc + 0.5, "W/T/L", fontsize=F, ha="left", va="center", color=INK)
    yc += 1.0
    for (sub_nm, g_arr, col) in rows_:
        n_ = len(g_arr); xe = np.linspace(0, 1, n_ + 1)
        w_ = int((g_arr > TOL).sum()); l_ = int((g_arr < -TOL).sum()); t_ = n_ - w_ - l_
        believed = isinstance(col, str) and col == "b"
        for k, gv in enumerate(g_arr):
            kw = dict(lw=0)
            if gv > TOL:
                kw.update(facecolor=OURS_PALE, hatch="/////", edgecolor="white") if believed else kw.update(facecolor=col)
            elif gv < -TOL:
                kw.update(facecolor=LOSS, hatch="xxxxx", edgecolor="white")
            else:
                kw.update(facecolor=TIE)
            ax.add_patch(Rectangle((xe[k] + 0.0025, yc + 0.09), (xe[1] - xe[0]) - 0.005, 0.82, **kw))
        ax.text(1.03, yc + 0.5, f"{w_}/{t_}/{l_}", fontsize=F, va="center", ha="left",
                color="#7A7A7A" if believed else INK, fontweight="normal" if believed else "bold")
        yt.append(yc + 0.5); ytl.append(sub_nm)
        print(f"    {gname:<16} {sub_nm:<10} {w_}/{t_}/{l_}  (n={n_})")
        yc += 1.0
    yc += 0.36
yend = yc - 0.36
ax.annotate("", xy=(1.0, yend + 0.50), xytext=(0.0, yend + 0.50), arrowprops=dict(arrowstyle="-|>", color=FIX, lw=0.7))
ax.text(0.5, yend + 0.80, r"delay weight $\lambda$:  cheap $\rightarrow$ costly", fontsize=F, ha="center", va="top", color=FIX)
hand = [Patch(facecolor=OURS, label="win"), Patch(facecolor=OURS_PALE, hatch="/////", edgecolor="white", label="believed win"),
        Patch(facecolor=TIE, label="tie"), Patch(facecolor=LOSS, hatch="xxxxx", edgecolor="white", label="loss")]
ax.legend(handles=hand, loc="lower left", bbox_to_anchor=(-0.02, -0.02), ncol=2, handlelength=1.3, handleheight=0.85,
          columnspacing=1.0, handletextpad=0.4, labelspacing=0.3, borderpad=0.0)
YMAX7 = yend + 3.55
ax.set_xlim(0, 1); ax.set_ylim(YMAX7, -0.1)
ax.set_yticks(yt); ax.set_yticklabels(ytl)
for tl in ax.get_yticklabels(): tl.set_color(LABCOL[tl.get_text()])
ax.tick_params(axis="y", length=0, pad=2); ax.set_xticks([])
for sp in ax.spines.values(): sp.set_visible(False)
plabel(ax, "a", dx=-0.27, dy=0.985)

# ---- b: critical-difference diagram, 40 UCR datasets
ax = fig.add_axes([0.505, 0.515, 0.485, 0.425])
R0, R1 = 2.0, 7.0                              # every mean rank lies in [2.46, 6.23]
ax.plot([R0, R1], [0, 0], color=INK, lw=0.8)
for rr_ in range(2, 8):
    ax.plot([rr_, rr_], [0, 0.10], color=INK, lw=0.7)
    ax.text(rr_, 0.17, str(rr_), ha="center", va="bottom", fontsize=F)
ax.text((R0 + R1) / 2, 0.62, "mean rank over 40 datasets (1 = lowest cost)", ha="center", va="bottom", fontsize=F)
half = (len(ord7) + 1) // 2
for pos, i in enumerate(ord7):
    if pos < half:
        yl = -0.62 - 0.40 * pos; xl = R0 - 0.35; ha = "right"
    else:
        yl = -0.62 - 0.40 * (len(ord7) - 1 - pos); xl = R1 + 0.35; ha = "left"
    ax.plot([avg7[i], avg7[i], xl], [0, yl, yl], color=COL7[M7[i]], lw=0.9, solid_joinstyle="miter")
    r2 = f"{np.floor(avg7[i] * 100 + 0.5) / 100:.2f}"            # half-up, as in the text (249/40 = 6.225 -> 6.23)
    lab = f"{NM7[M7[i]]} {r2}" if ha == "left" else f"{r2} {NM7[M7[i]]}"
    ax.text(xl + (0.08 if ha == "left" else -0.08), yl, lab, ha=ha, va="center", fontsize=F, color=INK,
            fontweight="bold" if M7[i] == "const_margin(train-selected)" else "normal")
for ci, (c0, c1) in enumerate(cliq7):
    xs_ = [avg7[ord7[c0]] - 0.04, avg7[ord7[c1]] + 0.04]; yb = -0.20 - 0.13 * ci
    ax.plot(xs_, [yb, yb], color=INK, lw=2.6, solid_capstyle="butt")
ax.set_xlim(R0 - 4.2, R1 + 4.2); ax.set_ylim(-0.62 - 0.40 * (half - 1) - 0.25, 0.95)
ax.axis("off")
plabel(ax, "b", dx=-0.02, dy=0.94)

# ---- c: every dataset, wins minus losses against the fixed horizon chosen on training data
ax = fig.add_axes([0.637, 0.115, 0.318, 0.300])
ax.imshow(S7, cmap=CMWL, norm=Normalize(-8, 8), aspect="auto", interpolation="nearest")
for (r_i, c_i), v_ in np.ndenumerate(S7):                  # hatched where losses outnumber wins: reads in grayscale
    if v_ < 0:
        ax.add_patch(Rectangle((c_i - 0.5, r_i - 0.5), 1, 1, fill=False, hatch="xxxx", edgecolor="white", lw=0))
ax.set_xticks(np.arange(-0.5, len(dord7)), minor=True); ax.set_yticks(np.arange(-0.5, len(strip7)), minor=True)
ax.grid(which="minor", color="white", lw=0.6); ax.tick_params(which="minor", length=0)
ax.set_yticks(range(len(strip7))); ax.set_yticklabels([NM7[m_] for m_ in strip7])
ax.get_yticklabels()[strip7.index("const_margin(train-selected)")].set_fontweight("bold")
ax.set_xticks([]); ax.tick_params(length=0, pad=2)
ax.set_xlabel("40 datasets, sorted by the margin rule's gain", labelpad=3)
for sp in ax.spines.values(): sp.set_visible(False)
cax = fig.add_axes([0.962, 0.115, 0.010, 0.300])
cb = fig.colorbar(mpl.cm.ScalarMappable(Normalize(-8, 8), CMWL), cax=cax, ticks=[-8, 0, 8])
cb.ax.tick_params(length=1.8, width=0.5); cb.outline.set_linewidth(0.5)
plabel(ax, "c", dx=-0.42, dy=1.05)
print("  Fig7c datasets with W>L / L>W (margin rule):", int((S7[strip7.index('const_margin(train-selected)')] > 0).sum()),
      int((S7[strip7.index('const_margin(train-selected)')] < 0).sum()))
save(fig, "Fig7_rq5")
