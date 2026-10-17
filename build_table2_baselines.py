#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""STEP 2: fixed-time discrimination baselines (Table II + Fig.3a).
Cross-fit several models on [trajectory-prefix feats + condition] at T=24h and 48h;
report AUC, balanced accuracy, ECE, Brier. Shows the models CLUSTER (saturation)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import DATA_DIR  # $ORGANFUSION_DATA, or ./datasets (see README)
import os, sys, csv, numpy as np
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import openpyxl, matplotlib as mpl; mpl.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, brier_score_loss

SEED=0; XLSX=os.path.join(DATA_DIR, "Tebon_SupplementaryData_MOESM6.xlsx")
DMAP={"Staurosporine":"Stauro","Neratinib":"Ner","Lapatinib":"Lap"}; DRUGS=["Stauro","Ner","Lap"]
def parse(ws,cell):
    rows=list(ws.iter_rows(values_only=True))
    ci=[i for i,r in enumerate(rows) if isinstance(r[1],str) and any(d in r[1] for d in DMAP)]
    out=[]
    for k,idx in enumerate(ci):
        end=ci[k+1] if k+1<len(ci) else len(rows); lab=rows[idx][1].split(); conc=float(lab[0]); drug=DMAP[lab[1]]
        blk=[r for r in (rows[j] for j in range(idx,end)) if isinstance(r[2],(int,float)) and r[2]<=72.5]
        t=np.array([r[2] for r in blk],float); nc=max(len(r) for r in blk)
        for c in range(3,nc):
            m=np.array([(r[c] if c<len(r) and isinstance(r[c],(int,float)) else np.nan) for r in blk],float)
            if np.isfinite(m).sum()>=100: out.append((cell,drug,conc,t,m))
    return out
wb=openpyxl.load_workbook(XLSX,read_only=True); recs=parse(wb["Supp Fig 9 MCF"],"MCF-7")+parse(wb["Supp Fig 9 BT"],"BT-474")
# data-integrity filter (2026-07-16): match build_posterior.py -- drop ratio>5x tracks
def _rmax(t,m):
    ok=np.isfinite(m)
    if ok.sum()<2: return np.inf
    b=np.interp(6,t[ok],m[ok]); return np.inf if b<=1e-3 else float(np.nanmax(m[ok])/b)
_n0=len(recs); recs=[r for r in recs if _rmax(r[3],r[4])<=5.0]; print(f"integrity filter: kept {len(recs)}/{_n0}")
cell_r=np.array([r[0] for r in recs]); drug=np.array([r[1] for r in recs])
doh=np.array([np.eye(3)[DRUGS.index(d)] for d in drug]); logc=np.array([[np.log(r[2]+1)] for r in recs])
cond_key=np.array([f"{c}|{d}|{cc}" for c,d,cc in zip(cell_r,drug,[r[2] for r in recs])])
cond_ids={k:i for i,k in enumerate(sorted(set(cond_key)))}; groups=np.array([cond_ids[k] for k in cond_key])
def val(t,m,tt): ok=np.isfinite(m); return np.interp(tt,t[ok],m[ok])
y=np.array([int(val(t,m,72)/(val(t,m,6)+1e-8)<0.90) for (_,_,_,t,m) in recs])
folds=GroupKFold(5)
def feats5(t,m,T,k=12):
    ok=np.isfinite(m); g=np.linspace(6,T,k); w=np.interp(g,t[ok],m[ok]); w=w/(w[0]+1e-8)
    dt=(T-6)/(k-1); v=np.gradient(w,dt); return [w[-1],v.mean(),v[-1],w.min(),w.max()-w.min()]
def traj(t,m,T,k=12):
    ok=np.isfinite(m); g=np.linspace(6,T,k); w=np.interp(g,t[ok],m[ok]); return list(w/(w[0]+1e-8))
def ece(yv,p,nb=10):
    b=np.linspace(0,1,nb+1); e=0
    for i in range(nb):
        msk=(p>=b[i])&(p<b[i+1] if i<nb-1 else p<=b[i+1])
        if msk.sum(): e+=msk.mean()*abs(yv[msk].mean()-p[msk].mean())
    return e
MODELS={
 "LR":lambda:make_pipeline(StandardScaler(),LogisticRegression(max_iter=1000)),
 "RF":lambda:RandomForestClassifier(n_estimators=300,random_state=SEED,n_jobs=-1),
 "HGB":lambda:HistGradientBoostingClassifier(random_state=SEED),
 "SVM-RBF":lambda:make_pipeline(StandardScaler(),SVC(probability=True,random_state=SEED)),
 "kNN":lambda:make_pipeline(StandardScaler(),KNeighborsClassifier(15)),
 "near-Bayes (LDA)":lambda:make_pipeline(StandardScaler(),LinearDiscriminantAnalysis()),
}
rows_out=[]
for T in (24,48):
    F5=np.column_stack([np.array([feats5(t,m,T) for (_,_,_,t,m) in recs],float),doh,logc])
    FT=np.column_stack([np.array([traj(t,m,T) for (_,_,_,t,m) in recs],float),doh,logc])
    for name,mk in MODELS.items():
        p=cross_val_predict(mk(),F5,y,cv=folds.split(F5,y,groups),method="predict_proba")[:,1]
        rows_out.append((name,T,roc_auc_score(y,p),balanced_accuracy_score(y,p>0.5),ece(y,p),brier_score_loss(y,p)))
        print(f"T={T} {name:18s} AUC={rows_out[-1][2]:.3f} bAcc={rows_out[-1][3]:.3f} ECE={rows_out[-1][4]:.3f} Brier={rows_out[-1][5]:.3f}")
    # trajectory encoder = logistic on dense resampled trajectory
    p=cross_val_predict(make_pipeline(StandardScaler(),LogisticRegression(max_iter=1000)),FT,y,cv=folds.split(FT,y,groups),method="predict_proba")[:,1]
    rows_out.append(("trajectory encoder",T,roc_auc_score(y,p),balanced_accuracy_score(y,p>0.5),ece(y,p),brier_score_loss(y,p)))
    print(f"T={T} trajectory encoder AUC={rows_out[-1][2]:.3f}")

os.makedirs("posterior",exist_ok=True)
with open("posterior/table2_fixed_time_baselines.csv","w",newline="",encoding="utf-8") as f:
    w=csv.writer(f); w.writerow(["model","horizon_h","AUC","balanced_accuracy","ECE","Brier"])
    for r in rows_out: w.writerow([r[0],r[1]]+[round(x,4) for x in r[2:]])

# Fig 3a: 24h model cluster (AUC bar, all near same level)
order=["LR","RF","HGB","SVM-RBF","kNN","trajectory encoder","near-Bayes (LDA)"]
a24={r[0]:r[2] for r in rows_out if r[1]==24}
fig,ax=plt.subplots(figsize=(3.6,2.5))
vals=[a24[m] for m in order]
ax.bar(range(len(order)),vals,color="#2166AC",edgecolor="white",lw=0.5)
ax.axhline(np.mean(vals),ls="--",color="#C0567E",lw=1.0)
ax.text(len(order)-1,np.mean(vals)+0.004,f"mean {np.mean(vals):.3f}",color="#C0567E",fontsize=6,ha="right")
for i,v in enumerate(vals): ax.text(i,v+0.003,f"{v:.3f}",ha="center",fontsize=5.4,color="#333")
ax.set_xticks(range(len(order))); ax.set_xticklabels(order,rotation=30,ha="right",fontsize=6)
ax.set_ylabel("AUC @ 24 h"); ax.set_ylim(0.78,0.90)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
plt.rcParams["font.family"]="serif"; plt.rcParams["font.serif"]=["Times New Roman","DejaVu Serif"]
fig.tight_layout();
for e in ("pdf","png","svg"): fig.savefig(f"figures_regen/fig3a_baselines.{e}",dpi=600,bbox_inches="tight")
print("wrote table2_fixed_time_baselines.csv + fig3a_baselines  (AUC spread @24h = %.3f)"%(max(vals)-min(vals)))
