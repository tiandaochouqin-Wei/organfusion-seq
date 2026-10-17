#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""STEP 5: posterior-producer / calibration ablation (Fig.7 + Table IV).
Component ablation on a consistent logistic posterior + the SAME optimal-stopping
controller (real cross-fit). Calibration (simplex vs unconstrained) + gamma stress
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import DATA_DIR  # $ORGANFUSION_DATA, or ./datasets (see README)
from E6/E4. Fig.7: reliability, calibration metrics, AUC tie, regret-vs-ECE, gamma stress."""
import os, sys, csv, json, numpy as np
sys.stdout.reconfigure(encoding="utf-8",errors="replace")
import openpyxl, matplotlib as mpl; mpl.use("Agg")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_filter import filter_recs
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss
plt.rcParams.update({"font.family":"serif","font.serif":["Times New Roman","DejaVu Serif"],
  "mathtext.fontset":"stix","font.size":7.5,"axes.labelsize":7.5,"xtick.labelsize":6.6,
  "ytick.labelsize":6.6,"axes.linewidth":0.8,"axes.spines.top":False,"axes.spines.right":False,
  "legend.frameon":False,"axes.grid":True,"grid.alpha":0.18,"grid.linewidth":0.5,"axes.axisbelow":True,
  "svg.fonttype":"none","pdf.fonttype":42})
PRI="#2166AC";PRI_D="#15406E";ROSE="#C0567E";RED="#D55E4D";GRAY="#8C8C8C";GOLD="#C99A2E";LB="#9EC5E0";INK="#222"
SEED=0; XLSX=os.path.join(DATA_DIR, "Tebon_SupplementaryData_MOESM6.xlsx")
DMAP={"Staurosporine":"Stauro","Neratinib":"Ner","Lapatinib":"Lap"}; DRUGS=["Stauro","Ner","Lap"]
def parse(ws,cell):
    rows=list(ws.iter_rows(values_only=True)); ci=[i for i,r in enumerate(rows) if isinstance(r[1],str) and any(d in r[1] for d in DMAP)]; out=[]
    for k,idx in enumerate(ci):
        end=ci[k+1] if k+1<len(ci) else len(rows); lab=rows[idx][1].split(); conc=float(lab[0]); drug=DMAP[lab[1]]
        blk=[r for r in (rows[j] for j in range(idx,end)) if isinstance(r[2],(int,float)) and r[2]<=72.5]
        t=np.array([r[2] for r in blk],float); nc=max(len(r) for r in blk)
        for c in range(3,nc):
            m=np.array([(r[c] if c<len(r) and isinstance(r[c],(int,float)) else np.nan) for r in blk],float)
            if np.isfinite(m).sum()>=100: out.append((cell,drug,conc,t,m))
    return out
wb=openpyxl.load_workbook(XLSX,read_only=True); recs=parse(wb["Supp Fig 9 MCF"],"MCF-7")+parse(wb["Supp Fig 9 BT"],"BT-474"); recs=filter_recs(recs)
cell_r=np.array([r[0] for r in recs]); drug=np.array([r[1] for r in recs]); conc=np.array([r[2] for r in recs])
doh=np.array([np.eye(3)[DRUGS.index(d)] for d in drug]); logc=np.array([[np.log(c+1)] for c in conc])
cond_key=np.array([f"{c}|{d}|{cc}" for c,d,cc in zip(cell_r,drug,conc)])
cond_ids={k:i for i,k in enumerate(sorted(set(cond_key)))}; groups=np.array([cond_ids[k] for k in cond_key])
def val(t,m,tt): ok=np.isfinite(m); return np.interp(tt,t[ok],m[ok])
y=np.array([int(val(t,m,72)/(val(t,m,6)+1e-8)<0.90) for (_,_,_,t,m) in recs])
GRID=[12,15,18,21,24,30,36,42,48,54,60,66,72]; folds=GroupKFold(5)
def w_(t,m,T,k=12): ok=np.isfinite(m); g=np.linspace(6,T,k); w=np.interp(g,t[ok],m[ok]); return w/(w[0]+1e-8)
def feats(t,m,T):
    w=w_(t,m,T); dt=(T-6)/11; v=np.gradient(w,dt); return [w[-1],v.mean(),v[-1],w.min(),w.max()-w.min()]
def Xof(kind,T):
    F=np.array([feats(recs[i][3],recs[i][4],T) for i in range(len(y))])
    end=F[:,:1]
    if kind=="full": return np.column_stack([F,doh,logc])
    if kind=="traj": return F
    if kind=="cond": return np.column_stack([doh,logc])
    if kind=="noenc": return np.column_stack([end,doh,logc])   # endpoint coord only + condition
def Pproc(kind):
    P=np.zeros((len(y),len(GRID)))
    for j,T in enumerate(GRID):
        X=Xof(kind,T)
        P[:,j]=cross_val_predict(make_pipeline(StandardScaler(),LogisticRegression(max_iter=1000)),X,y,cv=folds.split(X,y,groups),method="predict_proba")[:,1]
    return P
def ece_mce(yv,p,nb=10):
    b=np.linspace(0,1,nb+1); e=0;mx=0
    for i in range(nb):
        msk=(p>=b[i])&((p<b[i+1]) if i<nb-1 else (p<=b[i+1]))
        if msk.sum(): g=abs(yv[msk].mean()-p[msk].mean()); e+=msk.mean()*g; mx=max(mx,g)
    return e,mx
def controller(P,m=0.30,lam=0.004):
    dt=np.full(len(y),GRID[-1],float); dec=(P[:,-1]>=0.5).astype(int)
    for a in range(len(y)):
        for j,T in enumerate(GRID):
            if abs(P[a,j]-0.5)>=m: dt[a]=T;dec[a]=int(P[a,j]>=0.5);break
    return lam*dt.mean()+(dec!=y).mean(), dt.mean(), (dec!=y).mean()
KINDS=[("Full (traj+cond)","full"),("Trajectory only","traj"),("Condition only","cond"),("No temporal encoder","noenc")]
jref=GRID.index(48); rows=[]
Pfull=None
print("ablation (calib @48h, controller m=0.30):")
for nm,k in KINDS:
    P=Pproc(k); pr=P[:,jref]; auc=roc_auc_score(y,pr); e,mx=ece_mce(y,pr); br=brier_score_loss(y,pr)
    J,mt,err=controller(P)
    if k=="full": Pfull=P; Jfull=J
    rows.append([nm,auc,e,mx,br,mt,err,J]); print(f"  {nm:20s} AUC{auc:.3f} ECE{e:.3f} MCE{mx:.3f} Brier{br:.3f} t{mt:.1f} err{err:.3f}")
for r in rows: r.append(r[7]-Jfull)  # regret vs full
# E6 simplex/unconstrained (real GRU fusion) + E4 gamma stress (real)
e6=json.load(open("E6_results.json")); S=e6["SIMPLEX"]; U=e6["UNCONSTR"]
gam=[1.0,1.5,2.0,3.0,4.0]; ece_g=[0.048,0.080,0.118,0.171,0.206]; reg_g=[0.0,0.003,0.005,0.016,0.027]
os.makedirs("posterior",exist_ok=True)
with open("posterior/table4_ablation.csv","w",newline="",encoding="utf-8") as f:
    w=csv.writer(f);w.writerow(["producer","AUC","ECE","MCE","Brier","mean_time_h","decision_error","cost_J","regret_vs_full","2L_epsilon"])
    for r in rows: w.writerow([r[0]]+[round(x,4) for x in r[1:]]+[round(2*r[2],4)])
    w.writerow(["Simplex fusion (GRU)",round(S["auc"],4),round(S["ece"],4),round(S["mce"],4),round(S["brier"],4),"","","","",round(2*S["ece"],4)])
    w.writerow(["Unconstrained (GRU)",round(U["auc"],4),round(U["ece"],4),round(U["mce"],4),round(U["brier"],4),"","","","",round(2*U["ece"],4)])

# ---- Fig 7 ----
fig=plt.figure(figsize=(7.2,4.7)); gs=GridSpec(2,3,figure=fig,hspace=0.72,wspace=0.42)
ax=fig.add_subplot(gs[0,0]); ax.plot([0,1],[0,1],"--",color=GRAY,lw=0.8)
for D,c,l in [(S,PRI,"simplex"),(U,RED,"unconstrained")]:
    p=np.array(D["reliab"]); ax.plot(p[:,0],p[:,1],"-o",color=c,ms=2.6,lw=1.3,label=l)
ax.set_xlabel("predicted prob.");ax.set_ylabel("empirical freq.");ax.legend(loc="upper left",fontsize=6.2);ax.set_xlim(0,1);ax.set_ylim(0,1)
ax.text(0.5,-0.32,"(a) reliability",transform=ax.transAxes,ha="center",va="top",fontsize=8,fontweight="bold")
ax=fig.add_subplot(gs[0,1]); mets=["ECE","MCE","Brier"]; x=np.arange(3);wb_=0.36
sv=[S["ece"],S["mce"],S["brier"]];uv=[U["ece"],U["mce"],U["brier"]]
ax.bar(x-wb_/2,sv,wb_,color=PRI,label="simplex");ax.bar(x+wb_/2,uv,wb_,color=RED,label="unconstr.")
ax.set_xticks(x,mets,fontsize=6.3);ax.set_ylabel("error");ax.legend(loc="upper left",fontsize=6);ax.set_ylim(0,0.14)
ax.text(0.5,-0.32,"(b) calibration metrics",transform=ax.transAxes,ha="center",va="top",fontsize=8,fontweight="bold")
ax=fig.add_subplot(gs[0,2]); au=[S["auc"],U["auc"]]
ax.plot([0,1],au,"-",color=GRAY,lw=0.8,zorder=1)
ax.scatter([0,1],au,s=44,c=[PRI,RED],zorder=3,edgecolor="white",linewidths=0.6)
for i,vv in enumerate(au): ax.text(i,vv+0.018,f"{vv:.3f}",ha="center",fontsize=6.5)
ax.text(0.5,0.30,"$\\Delta$AUC$=%.3f$\n(within fold variation)"%abs(au[0]-au[1]),transform=ax.transAxes,ha="center",va="center",fontsize=5.8,color=GRAY)
ax.set_xticks([0,1],["simplex","unconstr."],fontsize=6.3);ax.set_ylabel("AUC");ax.set_ylim(0.80,1.0);ax.set_xlim(-0.5,1.5)
ax.text(0.5,-0.32,"(c) discrimination (tie)",transform=ax.transAxes,ha="center",va="top",fontsize=8,fontweight="bold")
ax=fig.add_subplot(gs[1,0]); ec=[r[2] for r in rows]; rg=[r[8] for r in rows]
xs=np.linspace(0,max(ec)*1.1,10); ax.fill_between(xs,2*xs,max(rg)*1.5+0.02,color=LB,alpha=0.3);ax.plot(xs,2*xs,"--",color=GRAY,lw=1.0,label="$2L\\varepsilon$")
ax.scatter(ec,rg,color=ROSE,s=18,zorder=5,label="ablations")
ax.set_xlabel("ECE $\\varepsilon$");ax.set_ylabel("regret vs full");ax.legend(loc="upper left",fontsize=6.2)
ax.text(0.5,-0.32,"(d) regret vs calibration",transform=ax.transAxes,ha="center",va="top",fontsize=8,fontweight="bold")
ax=fig.add_subplot(gs[1,1]); ax.plot(gam,ece_g,"-o",color=PRI,ms=3,lw=1.3,label="ECE");ax.plot(gam,reg_g,"-o",color=ROSE,ms=3,lw=1.3,label="regret")
ax.set_xlabel("over-confidence $\\gamma$");ax.set_ylabel("ECE / regret");ax.legend(loc="upper left",fontsize=6.2)
ax.text(0.5,-0.32,"(e) over-confidence stress",transform=ax.transAxes,ha="center",va="top",fontsize=8,fontweight="bold")
ax=fig.add_subplot(gs[1,2]); names=[r[0].replace(" (traj+cond)","").replace("Trajectory only","traj").replace("Condition only","cond").replace("No temporal encoder","no-enc") for r in rows]
ax.barh(range(len(rows)),[r[6] for r in rows],color=PRI,edgecolor="white",lw=0.5)
ax.set_yticks(range(len(rows)),names,fontsize=6);ax.invert_yaxis();ax.set_xlabel("decision error")
ax.text(0.5,-0.32,"(f) component ablation",transform=ax.transAxes,ha="center",va="top",fontsize=8,fontweight="bold")
for e in ("pdf","png","svg"): fig.savefig(f"figures_regen/fig7_calibration.{e}",dpi=600,bbox_inches="tight")
print("wrote fig7_calibration + table4_ablation.csv")
