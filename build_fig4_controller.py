#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""STEP (Fig.4): optimal-stopping controller theory & empirics from the posterior.
(a) theoretical tightening band; (b) empirical band from backward DP; (c) value curve
V_t(p) = min(stop-cost, continue-value), intersections = a_t,b_t; (d) band width vs lambda."""
import numpy as np, matplotlib as mpl; mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
plt.rcParams.update({"font.family":"serif","font.serif":["Times New Roman","DejaVu Serif"],
  "mathtext.fontset":"stix","font.size":7.5,"axes.labelsize":7.5,"xtick.labelsize":6.6,
  "ytick.labelsize":6.6,"axes.linewidth":0.8,"axes.spines.top":False,"axes.spines.right":False,
  "legend.frameon":False,"axes.grid":True,"grid.alpha":0.18,"grid.linewidth":0.5,"axes.axisbelow":True,
  "svg.fonttype":"none","pdf.fonttype":42})
PRI="#2166AC";PRI_D="#15406E";ROSE="#C0567E";GRAY="#8C8C8C";GOLD="#C99A2E";TEAL="#2A9D8F";LB="#9EC5E0";INK="#222"
d=np.load("posterior/posterior_data.npz",allow_pickle=True); P=d["P"]; G=d["GRID"].astype(float); y=d["y"]
Bbin=np.linspace(0,1,51); bc=(Bbin[:-1]+Bbin[1:])/2; nb=len(bc); rho=np.minimum(1-bc,bc)
def kernel(pj,pj1):
    K=np.zeros((nb,nb)); bi=np.clip(np.digitize(pj,Bbin)-1,0,nb-1); bj=np.clip(np.digitize(pj1,Bbin)-1,0,nb-1)
    for a,b in zip(bi,bj): K[a,b]+=1
    K+=1e-3; K/=K.sum(1,keepdims=True); return K
Ks=[kernel(P[:,j],P[:,j+1]) for j in range(len(G)-1)]
def dp(lam):
    V=rho.copy(); bands={}; vc={}
    for j in range(len(G)-2,-1,-1):
        cont=lam*(G[j+1]-G[j])+Ks[j]@V
        cs=np.where(cont<rho-1e-9)[0]
        if len(cs): bands[G[j]]=(bc[cs.min()],bc[cs.max()])
        vc[G[j]]=(rho.copy(),cont.copy()); V=np.minimum(rho,cont)
    return bands,vc

fig=plt.figure(figsize=(7.1,2.35)); gs=GridSpec(1,4,figure=fig,wspace=0.48)
# (a) theoretical band
ax=fig.add_subplot(gs[0,0]); t=np.linspace(6,72,100); ps=0.5; w=0.42*((72-t)/66)+0.04
ax.fill_between(t,ps+w,1,color=PRI,alpha=0.08,lw=0);ax.fill_between(t,ps-w,ps+w,color=LB,alpha=0.25,lw=0);ax.fill_between(t,0,ps-w,color=ROSE,alpha=0.08,lw=0)
ax.plot(t,ps+w,"-",color=PRI,lw=1.2);ax.plot(t,ps-w,"-",color=ROSE,lw=1.2);ax.axhline(ps,ls=":",color=GRAY,lw=0.8)
ax.text(70,0.93,"confident sensitive",fontsize=5.6,color=PRI,ha="right");ax.text(70,0.05,"confident resistant",fontsize=5.6,color=ROSE,ha="right")
ax.text(40,0.52,"continuation $(a_t,b_t)$",fontsize=5.8,color=INK,ha="center")
ax.set_xlabel("time (h)");ax.set_ylabel("posterior $\\hat p_t$");ax.set_ylim(0,1)
ax.text(0.5,-0.34,"(a) theoretical band",transform=ax.transAxes,ha="center",va="top",fontsize=8,fontweight="bold")
# (b) empirical band (lambda=0.02)
ax=fig.add_subplot(gs[0,1]); bands,_=dp(0.02); ts=sorted(bands); a=[bands[tt][0] for tt in ts]; b=[bands[tt][1] for tt in ts]
ax.fill_between(ts,a,b,color=LB,alpha=0.35,lw=0);ax.plot(ts,a,"-o",color=ROSE,ms=2.5,lw=1.1,label="$a_t$ (lower)");ax.plot(ts,b,"-o",color=PRI,ms=2.5,lw=1.1,label="$b_t$ (upper)")
ax.axhline(0.5,ls=":",color=GRAY,lw=0.8);ax.set_xlabel("time (h)");ax.set_ylabel("empirical $(a_t,b_t)$");ax.set_ylim(0,1);ax.legend(loc="lower right",fontsize=5.6,handlelength=1.1,handletextpad=0.4,labelspacing=0.2,borderpad=0.25)
ax.text(0.5,-0.34,"(b) empirical band ($\\lambda{=}0.02$)",transform=ax.transAxes,ha="center",va="top",fontsize=8,fontweight="bold")
# (c) value curve at T=24
ax=fig.add_subplot(gs[0,2]); bands,vc=dp(0.02); T0=24.0
stop,cont=vc[T0]
ax.plot(bc,stop,"-",color=GRAY,lw=1.4,label="stop cost $\\rho(p)$");ax.plot(bc,cont,"-",color=PRI,lw=1.4,label="continue value")
cross=np.where(cont<stop)[0]
if len(cross):
    a_t,b_t=bc[cross.min()],bc[cross.max()]
    ax.axvline(a_t,ls="--",color=ROSE,lw=0.8);ax.axvline(b_t,ls="--",color=PRI,lw=0.8)
    ax.text(a_t,0.46,"$a_t$",color=ROSE,fontsize=6.5,ha="center");ax.text(b_t,0.46,"$b_t$",color=PRI,fontsize=6.5,ha="center")
ax.set_xlabel("posterior $p$");ax.set_ylabel("cost / value at $T{=}24$h");ax.legend(loc="lower center",fontsize=5.4,handlelength=1.0,handletextpad=0.35,labelspacing=0.18,borderpad=0.25)
ax.text(0.5,-0.34,"(c) value curve $V_t(p)$",transform=ax.transAxes,ha="center",va="top",fontsize=8,fontweight="bold")
# (d) band width vs time for several lambda
ax=fig.add_subplot(gs[0,3]); cols={0.004:ROSE,0.02:PRI,0.05:GOLD}
for lam,c in cols.items():
    bands,_=dp(lam); ts=sorted(bands)
    if len(ts)<2: continue  # empty continuation region -> no curve, so no legend entry
    wd=[bands[tt][1]-bands[tt][0] for tt in ts]; ax.plot(ts,wd,"-o",color=c,ms=2.6,lw=1.3,label=f"$\\lambda$={lam}")
ax.set_xlabel("time (h)");ax.set_ylabel("band width $b_t-a_t$");ax.set_ylim(0,0.92);ax.legend(loc="center",bbox_to_anchor=(0.5,0.33),fontsize=5.4,handlelength=1.0,handletextpad=0.35,labelspacing=0.18,borderpad=0.25)
ax.text(0.5,-0.34,"(d) width vs $\\lambda$",transform=ax.transAxes,ha="center",va="top",fontsize=8,fontweight="bold")
for _a in fig.axes: _a.set_box_aspect(1)
for e in ("pdf","png","svg"): fig.savefig(f"figures_regen/fig4_controller_theory.{e}",dpi=600,bbox_inches="tight")
print("wrote fig4_controller_theory")
