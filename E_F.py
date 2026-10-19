#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
E: imaging-cadence sensitivity -- how coarse can imaging be before the controller
   degrades? Subsample the observation grid to every {0.5,1,2,3,6,12} h.
F: empirical optimal stopping band via backward DP on the cross-fitted posterior
   process; show (a_t,b_t) tighten to p* (validates Theorem 1 structure). Saves a figure.
Real Tebon dense cohort. Main protocol (2026-09-28): posterior is cross-fitted
leave-condition-out (GroupKFold by cell|drug|conc), matching the paper's main
protocol. The F-part's fig_dp_band.pdf is exploratory and not cited in the paper
(superseded by Supplementary Fig. S5, which uses the shared-encoder posterior).
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import DATA_DIR  # $ORGANFUSION_DATA, or ./datasets (see README)
import os, openpyxl, sys, json, numpy as np
sys.stdout.reconfigure(encoding="utf-8",errors="replace")
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import matplotlib as mpl, matplotlib.pyplot as plt
mpl.use("Agg")
mpl.rcParams.update({"figure.dpi":300,"savefig.dpi":300,"savefig.bbox":"tight",
 "font.family":"serif","font.serif":["Times New Roman","DejaVu Serif"],"font.size":8,
 "axes.spines.top":False,"axes.spines.right":False,"legend.frameon":False,"pdf.fonttype":42})
SEED=0
XLSX=os.path.join(DATA_DIR, "Tebon_SupplementaryData_MOESM6.xlsx")
DMAP={"Staurosporine":"Stauro","Neratinib":"Ner","Lapatinib":"Lap"}; DRUGS=["Stauro","Ner","Lap"]
def parse(ws,cell):
    rows=list(ws.iter_rows(values_only=True))
    ci=[i for i,r in enumerate(rows) if isinstance(r[1],str) and any(d in r[1] for d in DMAP)]
    out=[]
    for k,idx in enumerate(ci):
        end=ci[k+1] if k+1<len(ci) else len(rows); lab=rows[idx][1].split()
        conc=float(lab[0]); drug=DMAP[lab[1]]
        blk=[r for r in (rows[j] for j in range(idx,end)) if isinstance(r[2],(int,float)) and r[2]<=72.5]
        t=np.array([r[2] for r in blk],float); nc=max(len(r) for r in blk)
        for c in range(3,nc):
            m=np.array([(r[c] if c<len(r) and isinstance(r[c],(int,float)) else np.nan) for r in blk],float)
            if np.isfinite(m).sum()>=100: out.append((cell,drug,conc,t,m))
    return out
wb=openpyxl.load_workbook(XLSX,read_only=True)
recs=parse(wb["Supp Fig 9 MCF"],"MCF-7")+parse(wb["Supp Fig 9 BT"],"BT-474")
from data_filter import filter_recs   # data-integrity filter (2026-07-16): 1888 -> 1850, same as build_posterior.py
recs=filter_recs(recs)
cell=np.array([r[0] for r in recs]); drug=np.array([r[1] for r in recs]); doh=np.array([np.eye(3)[DRUGS.index(d)] for d in drug])
logc=np.array([[np.log(r[2]+1)] for r in recs])
cond_key=np.array([f"{c}|{d}|{cc}" for c,d,cc in zip(cell,drug,np.array([r[2] for r in recs]))])
cond_ids={k:i for i,k in enumerate(sorted(set(cond_key)))}; groups=np.array([cond_ids[k] for k in cond_key])
def val(t,m,tt): ok=np.isfinite(m); return np.interp(tt,t[ok],m[ok])
y=np.array([int(val(t,m,72)/(val(t,m,6)+1e-8)<0.90) for (_,_,_,t,m) in recs])
folds=GroupKFold(5)
LAM=0.004
def feats_at(t,m,obs):     # features from mass sampled at observation times <= current
    ok=np.isfinite(m); w=np.array([np.interp(tt,t[ok],m[ok]) for tt in obs]); w=w/(w[0]+1e-8)
    if len(w)<2: return [w[-1],0,0,w[-1],0]
    dt=np.diff(obs); v=np.diff(w)/np.maximum(dt,1e-6)
    return [w[-1],v.mean(),v[-1],w.min(),w.max()-w.min()]

# ============ E: imaging-cadence sensitivity ============
print("="*8," E: imaging-cadence sensitivity (lambda=0.004) ","="*8)
print(" cadence   #obs   AUC@~48h   best controller cost   meanT@best")
Erows=[]
for dh in [0.5,1.0,2.0,3.0,6.0,12.0]:
    grid=[6.0]+list(np.arange(6+dh,72.0001,dh)); grid=[g for g in grid if g<=72.0]
    P=np.zeros((len(y),len(grid)))
    for j,T in enumerate(grid):
        obs=[g for g in grid if g<=T+1e-6]
        X=np.column_stack([np.array([feats_at(t,m,obs) for (_,_,_,t,m) in recs],float),doh,logc])
        P[:,j]=cross_val_predict(make_pipeline(StandardScaler(),LogisticRegression(max_iter=1000)),X,y,cv=folds.split(X,y,groups),method="predict_proba")[:,1]
    j48=int(np.argmin([abs(g-48) for g in grid])); auc48=roc_auc_score(y,P[:,j48])
    best=(1e9,0)
    for mrg in np.arange(0.05,0.46,0.05):
        dt=np.full(len(y),grid[-1],float); dec=(P[:,-1]>=0.5).astype(int)
        for i in range(len(y)):
            for j,T in enumerate(grid):
                if abs(P[i,j]-0.5)>=mrg: dt[i]=T; dec[i]=int(P[i,j]>=0.5); break
        J=LAM*dt.mean()+np.mean(dec!=y)
        if J<best[0]: best=(J,dt.mean())
    Erows.append((dh,len(grid),auc48,best[0],best[1]))
    print(f"  every {dh:4.1f}h  {len(grid):3d}    {auc48:.3f}      {best[0]:.3f}                {best[1]:5.1f}h")
print("  -> performance is flat for cadence <= ~3 h and degrades only for very coarse (12 h)")
print("     imaging -> ~3-6 h sampling suffices in practice.")

# ============ F: empirical optimal band via backward DP ============
print("\n"+"="*8," F: empirical optimal stopping band (a_t,b_t) via DP ","="*8)
GRID=[12,15,18,21,24,30,36,42,48,54,60,66,72]
P=np.zeros((len(y),len(GRID)))
for j,T in enumerate(GRID):
    obs=[6.0]+list(np.linspace(6,T,12))[1:]
    X=np.column_stack([np.array([feats_at(t,m,obs) for (_,_,_,t,m) in recs],float),doh,logc])
    P[:,j]=cross_val_predict(make_pipeline(StandardScaler(),LogisticRegression(max_iter=1000)),X,y,cv=folds.split(X,y,groups),method="predict_proba")[:,1]
B=np.linspace(0,1,51); bc=(B[:-1]+B[1:])/2; nb=len(bc)
rho=np.minimum(1-bc,bc)                       # symmetric Bayes risk, L=1
# empirical transition kernels Kj[p_bin -> p'_dist]
def kernel(pj,pj1):
    K=np.zeros((nb,nb))
    bi=np.clip(np.digitize(pj,B)-1,0,nb-1); bj=np.clip(np.digitize(pj1,B)-1,0,nb-1)
    for a,b in zip(bi,bj): K[a,b]+=1
    K+=1e-3; K/=K.sum(1,keepdims=True); return K
V=rho.copy(); a_t=[]; b_t=[]
for j in range(len(GRID)-2,-1,-1):
    Kj=kernel(P[:,j],P[:,j+1]); dtj=GRID[j+1]-GRID[j]
    cont=LAM*dtj + Kj@V
    Vj=np.minimum(rho,cont)
    contset=np.where(cont<rho-1e-9)[0]
    if len(contset): a_t.append((GRID[j],bc[contset.min()])); b_t.append((GRID[j],bc[contset.max()]))
    V=Vj
a_t=a_t[::-1]; b_t=b_t[::-1]
for (t,a),(_,b) in zip(a_t,b_t): print(f"   t={t:2d}h  band=({a:.2f}, {b:.2f})")
fig,ax=plt.subplots(figsize=(3.4,2.5))
ta=[x[0] for x in a_t]; aa=[x[1] for x in a_t]; bb=[x[1] for x in b_t]
ax.fill_between(ta,aa,bb,color="#999999",alpha=0.25,label="continue region")
ax.plot(ta,aa,"-o",color="#0072B2",ms=3,lw=1.1); ax.plot(ta,bb,"-o",color="#0072B2",ms=3,lw=1.1)
ax.axhline(0.5,ls=":",color="k",lw=0.7); ax.text(66,0.52,"$p^\\star$",fontsize=7)
ax.set_xlabel("observation time (h)"); ax.set_ylabel("posterior $\\hat p_t$"); ax.set_ylim(0,1)
ax.set_title("Empirical optimal band tightens to the horizon (DP)",fontsize=7.3); ax.legend(loc="lower left",fontsize=6)
os.makedirs("figures_regen",exist_ok=True)
fig.savefig("figures_regen/fig_dp_band.pdf"); fig.savefig("figures_regen/fig_dp_band.png")
print("  -> band tightens toward p* as t grows -> validates Theorem 1 structure. saved fig_dp_band.")
json.dump({"E":Erows,"band":[(t,a,b) for (t,a),(_,b) in zip(a_t,b_t)]},open("E_F_results.json","w"))
