#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""STEP 1 (per fig.docx): build the unified cross-fitted posterior file that all
downstream figures/tables derive from. Reuses the REAL pipeline (F2_band parse/feats/
5-fold cross_val_predict). Adds cell_line. Dumps:
  posterior/unified_posterior.csv  long: organoid_id,drug,concentration,cell_line,time,p_hat_t,true_label,x_t
  posterior/posterior_data.npz     P(N,G),xgrid(N,G),y,drug,conc,cell,GRID
  posterior/auc_by_horizon.csv     horizon,auc  (cross-fitted trajectory-up-to-T AUC, honest accrual)
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import DATA_DIR  # $ORGANFUSION_DATA, or ./datasets (see README)
import os, sys, csv, numpy as np
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import openpyxl
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

SEED=0; XLSX=os.path.join(DATA_DIR, "Tebon_SupplementaryData_MOESM6.xlsx")
DMAP={"Staurosporine":"Stauro","Neratinib":"Ner","Lapatinib":"Lap"}; DRUGS=["Stauro","Ner","Lap"]
OUT="posterior"; os.makedirs(OUT,exist_ok=True)

def parse(ws, cell):
    rows=list(ws.iter_rows(values_only=True))
    ci=[i for i,r in enumerate(rows) if isinstance(r[1],str) and any(d in r[1] for d in DMAP)]
    out=[]
    for k,idx in enumerate(ci):
        end=ci[k+1] if k+1<len(ci) else len(rows); lab=rows[idx][1].split(); conc=float(lab[0]); drug=DMAP[lab[1]]
        blk=[r for r in (rows[j] for j in range(idx,end)) if isinstance(r[2],(int,float)) and r[2]<=72.5]
        t=np.array([r[2] for r in blk],float); nc=max(len(r) for r in blk)
        for c in range(3,nc):
            m=np.array([(r[c] if c<len(r) and isinstance(r[c],(int,float)) else np.nan) for r in blk],float)
            if np.isfinite(m).sum()>=100: out.append((drug,conc,cell,t,m))
    return out

print("parsing Tebon xlsx ...")
wb=openpyxl.load_workbook(XLSX,read_only=True)
recs=parse(wb["Supp Fig 9 MCF"],"MCF-7")+parse(wb["Supp Fig 9 BT"],"BT-474")
# ---- data-integrity filter (added 2026-07-16) --------------------------------
# A handful of source columns (concentrated in Lapatinib 10uM) contain corrupted
# mass values: e.g. one organoid's raw mass equals the TIMESTAMP (mass=24 at 24h,
# mass=72 at 72h), others spike to 500-860x the 6h baseline. Normalized dry-mass
# ratio cannot physically exceed a few-fold; ratios in the hundreds are source
# corruption, not biology (the independent 4,066-cohort export of the SAME
# conditions has none of these). We drop any organoid whose normalized mass ratio
# exceeds 5x at ANY horizon -- an extremely lenient ceiling (5-fold growth) that
# removes only the corrupted tracks and matches the clean 4,066 cohort's range.
def _ratio_max(t,m):
    ok=np.isfinite(m)
    if ok.sum()<2: return np.inf
    base=np.interp(6,t[ok],m[ok])
    if base<=1e-3: return np.inf
    return float(np.nanmax(m[ok])/base)
_n0=len(recs)
recs=[r for r in recs if _ratio_max(r[3],r[4])<=5.0]
print(f"data-integrity filter: dropped {_n0-len(recs)} corrupted organoids (ratio>5x); kept {len(recs)}")
N=len(recs); print("organoids:",N)
drug=np.array([r[0] for r in recs]); conc=np.array([r[1] for r in recs]); cell=np.array([r[2] for r in recs])
doh=np.array([np.eye(3)[DRUGS.index(d)] for d in drug]); logc=np.array([[np.log(c+1)] for c in conc])
def val(t,m,tt): ok=np.isfinite(m); return np.interp(tt,t[ok],m[ok])
y=np.array([int(val(t,m,72)/(val(t,m,6)+1e-8)<0.90) for (_,_,_,t,m) in recs])
GRID=[12,15,18,21,24,30,36,42,48,54,60,66,72]
folds=StratifiedKFold(5,shuffle=True,random_state=SEED)
def feats(t,m,T,k=12):
    ok=np.isfinite(m); g=np.linspace(6,T,k); w=np.interp(g,t[ok],m[ok]); w=w/(w[0]+1e-8)
    dt=(T-6)/(k-1); v=np.gradient(w,dt); return [w[-1],v.mean(),v[-1],w.min(),w.max()-w.min()]

P=np.zeros((N,len(GRID))); xgrid=np.zeros((N,len(GRID)))
auc_h=[]
for j,T in enumerate(GRID):
    X=np.column_stack([np.array([feats(t,m,T) for (_,_,_,t,m) in recs],float),doh,logc])
    P[:,j]=cross_val_predict(make_pipeline(StandardScaler(),LogisticRegression(max_iter=1000)),X,y,cv=folds,method="predict_proba")[:,1]
    xgrid[:,j]=np.array([val(t,m,T)/(val(t,m,6)+1e-8) for (_,_,_,t,m) in recs])
    a=roc_auc_score(y,P[:,j]); auc_h.append((T,a)); print(f"  T={T:>2}h  cross-fitted AUC={a:.3f}")

# dumps
np.savez(os.path.join(OUT,"posterior_data.npz"),P=P,xgrid=xgrid,y=y,drug=drug,conc=conc,cell=cell,GRID=np.array(GRID))
with open(os.path.join(OUT,"unified_posterior.csv"),"w",newline="",encoding="utf-8") as f:
    w=csv.writer(f); w.writerow(["organoid_id","drug","concentration","cell_line","time","p_hat_t","true_label","x_t"])
    for i in range(N):
        for j,T in enumerate(GRID):
            w.writerow([i,drug[i],conc[i],cell[i],T,round(float(P[i,j]),5),int(y[i]),round(float(xgrid[i,j]),5)])
with open(os.path.join(OUT,"auc_by_horizon.csv"),"w",newline="",encoding="utf-8") as f:
    w=csv.writer(f); w.writerow(["horizon","auc"]); [w.writerow([T,round(a,5)]) for T,a in auc_h]
print("label balance: sensitive(1)=%d resistant(0)=%d"%(int(y.sum()),int((y==0).sum())))
import collections
print("per-drug:",dict(collections.Counter(drug)),"per-cell:",dict(collections.Counter(cell)))
print("wrote posterior/unified_posterior.csv, posterior_data.npz, auc_by_horizon.csv")
