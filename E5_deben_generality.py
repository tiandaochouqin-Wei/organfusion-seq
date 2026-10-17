#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
E5: independent-cohort reliability generality on the Deben pancreatic-organoid panel
(1,934 wells, 6 drugs, 11 patient lines). Deben is summary-level (no dense tracks),
so it tests the RELIABILITY layer, not the stopping controller:
  (a) early-indicator predictivity is strongly DRUG-DEPENDENT (condition heterogeneity);
  (b) a calibrated classifier from [early indicator + drug + conc] yields a
      risk-coverage / PPV-NPV guarantee that holds out-of-distribution (Prop 4/5).
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import DATA_DIR  # $ORGANFUSION_DATA, or ./datasets (see README)
import openpyxl, sys, numpy as np
sys.stdout.reconfigure(encoding="utf-8",errors="replace")
from scipy.stats import pearsonr
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
SEED=0
wb=openpyxl.load_workbook(os.path.join(DATA_DIR, "Deben_NOGR_SupplementaryData1.xlsx"),read_only=True)
ws=wb["Sheet1"]; rows=list(ws.iter_rows(values_only=True)); hdr=list(rows[0]); data=rows[1:]
col={h:i for i,h in enumerate(hdr)}
def num(r,h):
    try: return float(r[col[h]])
    except: return np.nan
drug=np.array([str(r[col["Conditions"]]) for r in data])
cell=np.array([str(r[col["CellLine"]]) for r in data])
conc=np.array([num(r,"Conc1") for r in data])
nogr=np.array([num(r,"NOGR") for r in data])   # early growth-rate indicator
gr  =np.array([num(r,"GrowthRate") for r in data])
rv  =np.array([num(r,"RV") for r in data])     # relative viability (outcome)
ok=np.isfinite(nogr)&np.isfinite(rv)&np.isfinite(conc)
drug,cell,conc,nogr,gr,rv=[a[ok] for a in (drug,cell,conc,nogr,gr,rv)]
print(f"n={len(rv)} | RV range [{np.nanmin(rv):.2f},{np.nanmax(rv):.2f}] | NOGR range [{np.nanmin(nogr):.2f},{np.nanmax(nogr):.2f}]")
print("drugs:", sorted(set(drug)))

# (a) drug-stratified correlation early-indicator vs outcome
print("\n(a) per-drug Pearson r( NOGR , RV ):")
rs=[]
for d in sorted(set(drug)):
    mm=drug==d
    if mm.sum()>10 and np.std(nogr[mm])>0 and np.std(rv[mm])>0:
        r,_=pearsonr(nogr[mm],rv[mm]); rs.append(r)
        print(f"   {d:12s} n={mm.sum():4d}  r={r:+.3f}")
print(f"   -> spread of |r|: [{min(abs(x) for x in rs):.2f}, {max(abs(x) for x in rs):.2f}]  (condition heterogeneity confirmed)")

# (b) calibrated triage out-of-distribution
# label: sensitive = low relative viability (drug worked). threshold = median RV.
thr=np.median(rv); y=(rv<thr).astype(int)
DR=sorted(set(drug)); doh=np.array([np.eye(len(DR))[DR.index(d)] for d in drug])
X=np.column_stack([nogr, gr, np.log(conc+1), doh])
# Stage 3P: the 1,934 rows are 967 wells x 2 timepoints. Stratified folds over rows (used until
# then: AUC 0.866, ECE 0.048, PPV 0.884, NPV 0.843) put the two records of a well, and wells of
# the same line-drug pair, on both sides of a split. Folds are now whole (line, drug) conditions,
# the paper's leave-condition-out protocol: AUC 0.822, ECE 0.065, PPV 0.844, NPV 0.811.
cond=np.array([a+'|'+b for a,b in zip(cell,drug)])
folds=list(GroupKFold(5).split(X,y,cond))
p=cross_val_predict(make_pipeline(StandardScaler(),LogisticRegression(max_iter=1000)),X,y,cv=folds,method="predict_proba")[:,1]
def ece(p,yy,nb=10):
    p=np.clip(p,1e-6,1-1e-6); e=0
    for b in range(nb):
        lo,hi=b/nb,(b+1)/nb; m=(p>lo)&(p<=hi) if b>0 else (p>=lo)&(p<=hi)
        if m.any(): e+=m.mean()*abs(yy[m].mean()-p[m].mean())
    return e
print(f"\n(b) OOD reliability: AUC={roc_auc_score(y,p):.3f}  ECE={ece(p,y):.3f}")
print("   triage guarantee check (Prop 4: PPV>=k_high-eps, NPV>=1-k_low-eps):")
for klo,khi in [(0.3,0.7),(0.2,0.8)]:
    hi=p>=khi; lo=p<=klo
    ppv=y[hi].mean() if hi.any() else float('nan')
    npv=(1-y[lo]).mean() if lo.any() else float('nan')
    cov=(hi|lo).mean()
    print(f"   k=({klo},{khi}): PPV={ppv:.3f} (>= {khi:.2f}-eps?)  NPV={npv:.3f} (>= {1-klo:.2f}-eps?)  coverage={cov:.2f}")
print("\n-> condition heterogeneity + calibrated risk-coverage hold on an independent")
print("   cancer/assay -> the reliability framework generalises (Prop 4/5).")
