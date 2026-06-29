#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fig.6 robustness (3 panels, Ocean Dusk / serif): (a) anytime certificate
U_n~1/sqrt(n) with a coverage-failure inset; (b) robustness heatmap shift x OP
(hours saved; non-dominated cells marked); (c) external pancreatic per-drug
early-indicator r. Reads E10_results.json + tcyb_final_figure_data/shift_results.csv."""
import csv, json, numpy as np, matplotlib as mpl; mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
plt.rcParams.update({"font.family":"serif","font.serif":["Times New Roman","DejaVu Serif"],
  "mathtext.fontset":"stix","font.size":7.5,"axes.labelsize":7.5,"xtick.labelsize":6.6,
  "ytick.labelsize":6.6,"axes.linewidth":0.8,"axes.spines.top":False,"axes.spines.right":False,
  "legend.frameon":False,"axes.grid":True,"grid.alpha":0.18,"grid.linewidth":0.5,"axes.axisbelow":True,
  "svg.fonttype":"none","pdf.fonttype":42})
PRI="#2166AC";ROSE="#C0567E";GRAY="#8C8C8C";GOLD="#C99A2E";INK="#222"
def lab(ax,s): ax.text(0.5,-0.34,s,transform=ax.transAxes,ha="center",va="top",fontsize=8,fontweight="bold")
B=json.load(open("E10_results.json"))["B"]; n=[b[0] for b in B]; Un=[b[1] for b in B]; fr=[b[2] for b in B]
rows=list(csv.DictReader(open("tcyb_final_figure_data/shift_results.csv",encoding="utf-8")))
SH=["MCF7_to_BT474","BT474_to_MCF7","leave_Staurosporine","leave_Neratinib","leave_Lapatinib","leave_0.1uM","leave_1.0uM","leave_10.0uM"]
SHL=["MCF7$\\to$BT","BT$\\to$MCF7","$-$Stauro","$-$Ner","$-$Lap","$-$0.1$\\mu$M","$-$1$\\mu$M","$-$10$\\mu$M"]
OPS=["OP%d"%k for k in range(1,10)]
HS=np.full((8,9),np.nan); DM=np.zeros((8,9),int)
for r in rows:
    i=SH.index(r["shift_setting"]); j=OPS.index(r["operating_point"]); HS[i,j]=float(r["hours_saved"]); DM[i,j]=int(r["dominated"])
exd=["Oxalipl.","5-FU","5-FU/LV","Pacli.","Gemcit.","SN38"]; exr=[0.074,0.200,0.315,0.817,0.888,0.889]

fig=plt.figure(figsize=(7.1,2.7)); gs=GridSpec(1,3,figure=fig,wspace=0.5); main=[]
# (a) certificate width + coverage-failure inset
ax=fig.add_subplot(gs[0,0]); main.append(ax)
c=float(np.mean(np.array(Un)*np.sqrt(n))); ng=np.linspace(min(n),max(n),100)
ax.plot(ng,c/np.sqrt(ng),"--",color=GRAY,lw=1.0,label=r"$\propto 1/\sqrt{n}$"); ax.plot(n,Un,"-o",color=PRI,ms=3.4,lw=1.5,label="$U_n$")
ax.set_xscale("log"); ax.set_xlabel("calibration size $n$"); ax.set_ylabel("certificate width $U_n$"); ax.legend(loc="lower left",fontsize=5.8)
axin=ax.inset_axes([0.60,0.58,0.37,0.35])
axin.bar(range(len(n)),fr,color=PRI,edgecolor="white",lw=0.4); axin.axhline(0.10,ls="--",color=ROSE,lw=0.8)
axin.set_ylim(0,0.13); axin.set_xticks([]); axin.tick_params(labelsize=3.4); axin.set_title("coverage failure",fontsize=4.4)
axin.text(0.5,0.60,"$0$ at all $n$\n($\\delta{=}0.1$)",transform=axin.transAxes,ha="center",fontsize=4.0,color=ROSE)
lab(ax,"(a) certificate $U_n\\sim 1/\\sqrt{n}$")
# (b) robustness heatmap (shift x OP, hours saved; non-dominated marked)
ax=fig.add_subplot(gs[0,1]); main.append(ax); ax.grid(False)
ax.imshow(np.clip(HS,0,None),cmap="YlGnBu",aspect="auto",vmin=0)
ax.set_xticks(range(9)); ax.set_xticklabels([str(k) for k in range(1,10)],fontsize=5.0)
ax.set_yticks(range(8)); ax.set_yticklabels(SHL,fontsize=5.0); ax.set_xlabel("operating point")
for i in range(8):
    for j in range(9):
        if DM[i,j]: ax.text(j,i,"%.0f"%HS[i,j],ha="center",va="center",fontsize=4.0,color=INK)
        else: ax.text(j,i,"$\\times$",ha="center",va="center",fontsize=5.0,color=ROSE)
lab(ax,"(b) robustness (h saved)")
# (c) external pancreatic per-drug r
ax=fig.add_subplot(gs[0,2]); main.append(ax)
cols=[ROSE if r<0.5 else PRI for r in exr]
ax.bar(range(len(exr)),exr,color=cols,edgecolor="white",lw=0.5); ax.axhline(0.5,ls="--",color=GRAY,lw=0.8)
ax.set_xticks(range(len(exr))); ax.set_xticklabels(exd,rotation=35,ha="right",fontsize=5.2); ax.set_ylabel("Pearson $r$"); ax.set_ylim(0,1)
lab(ax,"(c) external pancreatic")
for _a in main: _a.set_box_aspect(1)
for e in ("pdf","png","svg"): fig.savefig(f"figures_regen/fig8_robustness.{e}",dpi=600,bbox_inches="tight")
print("wrote fig8_robustness (3 panels: certificate+inset, robustness heatmap, external)")
