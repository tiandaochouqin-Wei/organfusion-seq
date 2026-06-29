#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Supplementary figures holding the panels removed from the slimmed main figures
(per reviewer): S1 cost sensitivity (was Fig.1e); S2 early-endpoint relation
(was Fig.3b); S3 controller behaviour by drug (was Fig.5d/e/f). All REAL data."""
import csv, numpy as np, matplotlib as mpl; mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
plt.rcParams.update({"font.family":"serif","font.serif":["Times New Roman","DejaVu Serif"],
  "mathtext.fontset":"stix","font.size":7.5,"axes.labelsize":7.5,"xtick.labelsize":6.6,
  "ytick.labelsize":6.6,"axes.linewidth":0.8,"axes.spines.top":False,"axes.spines.right":False,
  "legend.frameon":False,"axes.grid":True,"grid.alpha":0.18,"grid.linewidth":0.5,"axes.axisbelow":True,
  "svg.fonttype":"none","pdf.fonttype":42})
PRI="#2166AC";PRI_D="#15406E";ROSE="#C0567E";RED="#D55E4D";GRAY="#8C8C8C";GOLD="#C99A2E";LB="#9EC5E0";INK="#222"
import os; os.makedirs("figures_supp",exist_ok=True)
d=np.load("posterior/posterior_data.npz",allow_pickle=True)
P=d["P"];xg=d["xgrid"];y=d["y"];G=d["GRID"].astype(float);drug=d["drug"];N=len(y)
def lab(ax,s): ax.text(0.5,-0.32,s,transform=ax.transAxes,ha="center",va="top",fontsize=8.5,fontweight="bold")
def controller(m):
    times=np.empty(N); dec=np.empty(N,int); exited=np.zeros(N,bool)
    for i in range(N):
        ex=np.where(np.abs(P[i]-0.5)>=m)[0]; j=ex[0] if len(ex) else len(G)-1
        times[i]=G[j]; dec[i]=int(P[i,j]>=0.5); exited[i]=len(ex)>0
    return times,dec,exited

# ---------- S1 : cost sensitivity vs delay weight ----------
rows=list(csv.DictReader(open("figures_cs/data/cost.csv",encoding="utf-8-sig")))
lam=[float(r["lambda"]) for r in rows]; cc=[float(r["controller"]) for r in rows]; cf=[float(r["fixed"]) for r in rows]
fig=plt.figure(figsize=(3.4,2.6)); ax=fig.add_subplot(111)
ax.plot(lam,cf,"-o",color=GRAY,ms=3,lw=1.4,label="best fixed-time")
ax.plot(lam,cc,"-o",color=PRI,ms=3,lw=1.8,label="controller")
ax.set_xlabel("delay weight $\\lambda$ (per h)");ax.set_ylabel("optimal total cost")
ax.legend(loc="upper left",fontsize=6.6)
fig.savefig("figures_supp/figS1_cost.pdf",dpi=600,bbox_inches="tight")
fig.savefig("figures_supp/figS1_cost.png",dpi=300,bbox_inches="tight"); plt.close(fig)

# ---------- S2 : early-endpoint relation ----------
x24=xg[:,4]; endr=xg[:,12]; r=np.corrcoef(x24,endr)[0,1]
fig=plt.figure(figsize=(3.4,2.6)); ax=fig.add_subplot(111)
hb=ax.hexbin(x24,endr,gridsize=32,cmap="YlOrRd",mincnt=1,linewidths=0.1)
m,b=np.polyfit(x24,endr,1); xs=np.linspace(x24.min(),x24.max(),40); ax.plot(xs,m*xs+b,"--",color=PRI_D,lw=1.2)
ax.axhline(0.9,ls=":",color=GRAY,lw=0.8); ax.text(0.04,0.93,f"$r$={r:.3f}\n$n$={len(y)}",transform=ax.transAxes,va="top",fontsize=6.4,color=INK)
ax.set_xlabel("normalized mass $x(24)$");ax.set_ylabel("end ratio $x(72)/x(6)$")
fig.savefig("figures_supp/figS2_endpoint.pdf",dpi=600,bbox_inches="tight")
fig.savefig("figures_supp/figS2_endpoint.png",dpi=300,bbox_inches="tight"); plt.close(fig)

# ---------- S3 : controller behaviour by drug (3 panels) ----------
fig=plt.figure(figsize=(7.1,2.5)); gs=GridSpec(1,3,figure=fig,wspace=0.42)
t30,dec30,ex30=controller(0.30); order=["Stauro","Ner","Lap"]; cols=[PRI,ROSE,GOLD]
# (a) stopping-time distribution by drug
axD=fig.add_subplot(gs[0,0]); data=[t30[drug==g] for g in order]
parts=axD.violinplot(data,positions=[1,2,3],showextrema=False,widths=0.85)
for pc,c in zip(parts['bodies'],cols): pc.set_facecolor(c);pc.set_alpha(0.3);pc.set_edgecolor(c)
bp=axD.boxplot(data,positions=[1,2,3],widths=0.2,showfliers=False,patch_artist=True)
for i,c in enumerate(cols):
    bp['boxes'][i].set(facecolor="white",edgecolor=c,lw=1.0)
    for el in('whiskers','caps','medians'):
        for art in bp[el]: art.set_color(c)
axD.set_xticks([1,2,3]);axD.set_xticklabels([f"{g}\n{np.mean(t30[drug==g]):.0f}h" for g in order],fontsize=6)
axD.set_ylabel("stop time (h)");axD.set_ylim(6,74);lab(axD,"(a) stopping time by drug")
# (b) decision composition by drug
axE=fig.add_subplot(gs[0,1]); sens=[];res=[];defr=[]
for g in order:
    msk=drug==g; ex=ex30[msk]; dc=dec30[msk]
    sens.append((ex&(dc==1)).mean());res.append((ex&(dc==0)).mean());defr.append((~ex).mean())
x=np.arange(3)
axE.bar(x,sens,0.6,color=ROSE,label="sensitive");axE.bar(x,res,0.6,bottom=sens,color=PRI,label="resistant")
axE.bar(x,defr,0.6,bottom=np.array(sens)+np.array(res),color=GRAY,label="defer")
axE.set_xticks(x);axE.set_xticklabels(order,fontsize=6.5);axE.set_ylabel("fraction");axE.legend(loc="upper right",fontsize=5.4,ncol=1);lab(axE,"(b) decision composition")
# (c) representative posterior paths + band
axF=fig.add_subplot(gs[0,2]); m=0.30
cand_s=[i for i in range(N) if y[i]==1 and (P[i]>=0.5+m).any()]
cand_r=[i for i in range(N) if y[i]==0 and (P[i]<=0.5-m).any()]
cand_l=[i for i in range(N) if not (np.abs(P[i]-0.5)>=m)[:6].any()]
sel=[(cand_s[0],PRI),(cand_r[0],ROSE),(cand_l[0] if cand_l else 0,GRAY)]
axF.fill_between(G,0.5-m,0.5+m,color=LB,alpha=0.25); axF.axhline(0.5,ls=":",color=GRAY,lw=0.7)
for i,c in sel:
    ex=np.where(np.abs(P[i]-0.5)>=m)[0]; j=ex[0] if len(ex) else len(G)-1
    axF.plot(G[:j+1],P[i,:j+1],"-",color=c,lw=1.3); axF.plot(G[j],P[i,j],"o",color=c,ms=4)
axF.set_xlabel("time (h)");axF.set_ylabel("$\\hat p_t$");axF.set_ylim(0,1);lab(axF,"(c) example posterior paths")
fig.savefig("figures_supp/figS3_bydrug.pdf",dpi=600,bbox_inches="tight")
fig.savefig("figures_supp/figS3_bydrug.png",dpi=300,bbox_inches="tight"); plt.close(fig)
print("wrote figS1_cost, figS2_endpoint, figS3_bydrug ; x24-endpoint r=%.3f"%r)
