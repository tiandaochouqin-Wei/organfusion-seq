#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fig.3 motivation (3 panels, REAL; slimmed per reviewer): (a) fixed-time
baselines cluster @24h; (b) earliest-confident-time by drug (violin); (c) example
posterior paths + band. The early-endpoint hexbin moved to supplementary (Fig.S2)."""
import csv, numpy as np, matplotlib as mpl; mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
plt.rcParams.update({"font.family":"serif","font.serif":["Times New Roman","DejaVu Serif"],
  "mathtext.fontset":"stix","font.size":7.5,"axes.labelsize":7.5,"xtick.labelsize":6.6,
  "ytick.labelsize":6.6,"axes.linewidth":0.8,"axes.spines.top":False,"axes.spines.right":False,
  "legend.frameon":False,"axes.grid":True,"grid.alpha":0.18,"grid.linewidth":0.5,"axes.axisbelow":True,
  "svg.fonttype":"none","pdf.fonttype":42})
PRI="#2166AC";PRI_D="#15406E";ROSE="#C0567E";GRAY="#8C8C8C";GOLD="#C99A2E";LB="#9EC5E0";INK="#222"
d=np.load("posterior/posterior_data.npz",allow_pickle=True); P=d["P"];xg=d["xgrid"];y=d["y"];G=d["GRID"].astype(float);drug=d["drug"]
def lab(ax,s): ax.text(0.5,-0.34,s,transform=ax.transAxes,ha="center",va="top",fontsize=8.5,fontweight="bold")
fig=plt.figure(figsize=(7.1,3.0)); gs=GridSpec(1,3,figure=fig,wspace=0.42)
# (a) baselines cluster @24h
r2=[r for r in csv.DictReader(open("posterior/table2_fixed_time_baselines.csv",encoding="utf-8-sig")) if int(float(r["horizon_h"]))==24]
order=["LR","RF","HGB","SVM-RBF","kNN","trajectory encoder","near-Bayes (LDA)"]
a24={r["model"]:float(r["AUC"]) for r in r2}; vals=[a24[m] for m in order]
ax=fig.add_subplot(gs[0,0]); ax.bar(range(len(order)),vals,color=PRI,edgecolor="white",lw=0.5)
ax.axhline(np.mean(vals),ls="--",color=ROSE,lw=1.0); ax.text(len(order)-1,np.mean(vals)+0.004,f"mean {np.mean(vals):.3f}",color=ROSE,fontsize=5.6,ha="right")
ax.set_xticks(range(len(order))); ax.set_xticklabels([m.replace(" (LDA)","").replace("trajectory encoder","traj-enc") for m in order],rotation=32,ha="right",fontsize=5.6)
ax.set_ylabel("AUC @ 24 h");ax.set_ylim(0.74,0.90);lab(ax,"(a) fixed-time baselines cluster")
# (b) earliest-confident-time by drug (first |p-0.5|>=0.30)
m0=0.30; tct=np.empty(len(y))
for i in range(len(y)):
    ex=np.where(np.abs(P[i]-0.5)>=m0)[0]; tct[i]=G[ex[0]] if len(ex) else G[-1]
ax=fig.add_subplot(gs[0,1]); ordg=["Stauro","Ner","Lap"]; cols=[PRI,ROSE,GOLD]; data=[tct[drug==g] for g in ordg]
parts=ax.violinplot(data,positions=[1,2,3],showextrema=False,widths=0.85)
for pc,c in zip(parts['bodies'],cols): pc.set_facecolor(c);pc.set_alpha(0.3);pc.set_edgecolor(c)
bp=ax.boxplot(data,positions=[1,2,3],widths=0.2,showfliers=False,patch_artist=True)
for i,c in enumerate(cols):
    bp['boxes'][i].set(facecolor="white",edgecolor=c,lw=1.0)
    for el in('whiskers','caps','medians'):
        for art in bp[el]: art.set_color(c)
ax.set_xticks([1,2,3]); ax.set_xticklabels([f"{g}\n{np.median(tct[drug==g]):.0f}h" for g in ordg],fontsize=6.3)
ax.set_ylabel("earliest confident time (h)");ax.set_ylim(6,74);lab(ax,"(b) time-to-decision is heterogeneous")
# (c) example posterior paths + band (over a faint real-path ensemble)
ax=fig.add_subplot(gs[0,2])
_rng=np.random.default_rng(0)
for _i in _rng.choice(len(y),60,replace=False): ax.plot(G,P[_i],"-",color=GRAY,lw=0.35,alpha=0.20)
ax.fill_between(G,0.5-m0,0.5+m0,color=LB,alpha=0.25);ax.axhline(0.5,ls=":",color=GRAY,lw=0.7)
cs=[i for i in range(len(y)) if y[i]==1 and (P[i]>=0.5+m0).any()]
cr=[i for i in range(len(y)) if y[i]==0 and (P[i]<=0.5-m0).any()]
cl=[i for i in range(len(y)) if not (np.abs(P[i]-0.5)>=m0)[:6].any()]
for i,c,nm in [(cs[0],PRI,"sensitive"),(cr[0],ROSE,"resistant"),(cl[0] if cl else 0,GOLD,"late")]:
    ex=np.where(np.abs(P[i]-0.5)>=m0)[0]; j=ex[0] if len(ex) else len(G)-1
    ax.plot(G[:j+1],P[i,:j+1],"-",color=c,lw=1.6,label=nm);ax.plot(G[j],P[i,j],"o",color=c,ms=4)
ax.legend(loc="lower right",fontsize=5.2,handlelength=1.1,handletextpad=0.4,labelspacing=0.2,borderpad=0.25)
ax.set_xlabel("time (h)");ax.set_ylabel("$\\hat p_t$");ax.set_ylim(0,1);lab(ax,"(c) example posterior paths")
for _a in fig.axes: _a.set_box_aspect(1)
for e in ("pdf","png","svg"): fig.savefig(f"figures_regen/fig3_motivation.{e}",dpi=600,bbox_inches="tight")
print("wrote fig3_motivation (3 panels)  24h baseline AUC spread %.3f"%(max(vals)-min(vals)))
