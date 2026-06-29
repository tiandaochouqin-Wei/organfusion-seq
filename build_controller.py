#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""STEP 3: optimal-stopping controller outputs from the unified posterior.
Fig.5 (6 panels): frontier, hours saved, PPV/NPV vs coverage, stopping-time
distribution by drug, decision composition, representative posterior paths.
Table V: matched operating points. REAL data from posterior_data.npz."""
import os, csv, numpy as np, matplotlib as mpl; mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
plt.rcParams.update({"font.family":"serif","font.serif":["Times New Roman","DejaVu Serif"],
  "mathtext.fontset":"stix","font.size":7.5,"axes.labelsize":7.5,"xtick.labelsize":6.6,
  "ytick.labelsize":6.6,"axes.linewidth":0.8,"axes.spines.top":False,"axes.spines.right":False,
  "legend.frameon":False,"axes.grid":True,"grid.alpha":0.18,"grid.linewidth":0.5,
  "axes.axisbelow":True,"svg.fonttype":"none","pdf.fonttype":42})
PRI="#2166AC";PRI_D="#15406E";ROSE="#C0567E";RED="#D55E4D";GRAY="#8C8C8C";GOLD="#C99A2E";TEAL="#2A9D8F";LB="#9EC5E0";INK="#222"
d=np.load("posterior/posterior_data.npz",allow_pickle=True)
P=d["P"];y=d["y"];G=d["GRID"].astype(float);drug=d["drug"];N=len(y)
def lab(ax,s): ax.text(0.5,-0.30,s,transform=ax.transAxes,ha="center",va="top",fontsize=9,fontweight="bold")

def controller(m):
    """constant half-width band around 0.5; stop at first exit, else horizon."""
    times=np.empty(N); dec=np.empty(N,int); exited=np.zeros(N,bool)
    for i in range(N):
        ex=np.where(np.abs(P[i]-0.5)>=m)[0]
        j=ex[0] if len(ex) else len(G)-1
        times[i]=G[j]; dec[i]=int(P[i,j]>=0.5); exited[i]=len(ex)>0
    return times,dec,exited
# frontier
ms=np.linspace(0.0,0.48,49); cf=[(controller(m)[0].mean(),(controller(m)[1]!=y).mean()) for m in ms]
ct=np.array([c[0] for c in cf]); ce=np.array([c[1] for c in cf])
ft=G.copy(); fe=np.array([ (np.array(P[:,j]>=0.5,int)!=y).mean() for j in range(len(G))])

# bootstrap 95% CI for the controller frontier at the 9 operating points
mvals9=np.linspace(0.10,0.42,9); np.random.seed(0); mt_=[];me_=[];lot=[];hit=[];loe=[];hie=[]
for m in mvals9:
    tt,dd,ee=controller(m); bt=np.empty(200);be=np.empty(200)
    for b in range(200):
        bi=np.random.randint(0,N,N); bt[b]=tt[bi].mean(); be[b]=(dd[bi]!=y[bi]).mean()
    mt_.append(tt.mean()); me_.append((dd!=y).mean())
    lot.append(np.percentile(bt,2.5));hit.append(np.percentile(bt,97.5))
    loe.append(np.percentile(be,2.5));hie.append(np.percentile(be,97.5))
mt_=np.array(mt_);me_=np.array(me_)
xerr=np.array([mt_-np.array(lot),np.array(hit)-mt_]); yerr=np.array([me_-np.array(loe),np.array(hie)-me_])

fig=plt.figure(figsize=(7.1,2.55)); gs=GridSpec(1,4,figure=fig,wspace=0.52)
# (a) frontier + bootstrap CI
axA=fig.add_subplot(gs[0,0])
axA.plot(ft,fe,"-o",color=GRAY,ms=2.4,lw=1.3,label="fixed-time")
axA.errorbar(mt_,me_,xerr=xerr,yerr=yerr,fmt="-o",color=PRI,ms=2.4,lw=1.5,capsize=1.4,elinewidth=0.7,label="controller")
axA.set_xlabel("mean observation time (h)");axA.set_ylabel("decision error");axA.legend(loc="upper right",fontsize=5.8);lab(axA,"(a) frontier $+$ 95\\% CI")
# (b) hours saved at matched error
errs=np.linspace(0.12,0.24,9); hs=[(np.interp(-e,-fe,ft)-np.interp(-e,-ce,ct)) for e in errs]
axB=fig.add_subplot(gs[0,1]); axB.bar(range(len(errs)),hs,color=PRI,edgecolor="white",lw=0.5)
axB.axhline(0,color=GRAY,lw=0.7); axB.set_xticks(range(len(errs))); axB.set_xticklabels([f"{e:.2f}" for e in errs],rotation=40,ha="right",fontsize=5.4)
axB.set_xlabel("target error");axB.set_ylabel("hours saved");lab(axB,"(b) hours saved")
# (c) PPV/NPV vs coverage
cov=[];ppv=[];npv=[]
for m in np.linspace(0.0,0.45,30):
    t,dec,ex=controller(m)
    if ex.sum()<20: continue
    cov.append(ex.mean()); pos=ex&(dec==1); neg=ex&(dec==0)
    ppv.append((y[pos]==1).mean() if pos.sum() else np.nan); npv.append((y[neg]==0).mean() if neg.sum() else np.nan)
axC=fig.add_subplot(gs[0,2]); o=np.argsort(cov); cov=np.array(cov)[o];ppv=np.array(ppv)[o];npv=np.array(npv)[o]
axC.plot(cov,ppv,"-o",color=PRI,ms=2.2,lw=1.2,label="PPV");axC.plot(cov,npv,"-o",color=ROSE,ms=2.2,lw=1.2,label="NPV")
axC.set_xlabel("coverage");axC.set_ylabel("PPV / NPV");axC.set_ylim(0.6,1.0);axC.legend(loc="lower left",fontsize=5.8);lab(axC,"(c) reliability vs coverage")
# (d) sample-level stopping raster at OP6 (m=0.30)
axD=fig.add_subplot(gs[0,3]); axD.grid(False); tt,dd,ee=controller(0.30)
order=np.argsort(tt,kind="stable"); ts=tt[order];ds=dd[order];es=ee[order];ys=np.arange(N)
axD.fill_betweenx(ys,6,ts,color=LB,alpha=0.6,lw=0)
code=np.where(es,np.where(ds==1,0,1),2)
for cv,c,nm in [(0,PRI,"sensitive"),(1,ROSE,"resistant"),(2,GRAY,"defer")]:
    mk=code==cv
    if mk.any(): axD.scatter(ts[mk],ys[mk],s=0.5,c=c,marker="s",linewidths=0,label=nm)
axD.set_xlabel("time (h)");axD.set_ylabel("organoids (sorted by stop)");axD.set_xlim(6,72);axD.set_ylim(0,N)
axD.legend(loc="lower right",fontsize=4.8,markerscale=6,handletextpad=0.2,borderpad=0.2)
lab(axD,"(d) sample-level raster")
for _a in fig.axes: _a.set_box_aspect(1)
for e in ("pdf","png","svg"): fig.savefig(f"figures_regen/fig5_controller.{e}",dpi=600,bbox_inches="tight")
plt.close(fig)

# Table V: matched operating points (9 controller margins)
mvals=np.linspace(0.10,0.42,9)
with open("posterior/table5_matched_ops.csv","w",newline="",encoding="utf-8") as f:
    w=csv.writer(f); w.writerow(["OP","band_margin","controller_error","controller_mean_time_h","matched_fixed_time_h","hours_saved","coverage","PPV","NPV"])
    for k,m in enumerate(mvals,1):
        t,dec,ex=controller(m); err=(dec!=y).mean(); mt=t.mean(); ftm=np.interp(-err,-fe,ft)
        pos=ex&(dec==1);neg=ex&(dec==0)
        w.writerow([f"OP{k}",round(m,3),round(err,4),round(mt,2),round(ftm,2),round(ftm-mt,2),round(ex.mean(),3),
                    round((y[pos]==1).mean() if pos.sum() else float('nan'),3),round((y[neg]==0).mean() if neg.sum() else float('nan'),3)])
print("wrote fig5_controller + table5_matched_ops.csv ; frontier controller error range %.3f-%.3f"%(ce.min(),ce.max()))
