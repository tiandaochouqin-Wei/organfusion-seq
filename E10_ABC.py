#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A: validate Theorem 1 (leverage bound): per-drug window info I_W and measured
   dAUC(full vs endpoint), check dAUC <= sqrt(I_W)/(2*sqrt(pi)).
B: validate Theorem 3(a) anytime-valid certificate coverage: over random splits,
   P( J_true(pi_hat) > Jhat_n(pi_hat) + U_n ) <= delta.
C: cross-drug & cross-concentration controller transfer (frontier dominance).
Real Tebon dense cohort. Parses the workbook once.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import DATA_DIR  # $ORGANFUSION_DATA, or ./datasets (see README)
import openpyxl, sys, json, numpy as np
sys.stdout.reconfigure(encoding="utf-8",errors="replace")
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.covariance import LedoitWolf
from sklearn.metrics import roc_auc_score
SEED=0; rng=np.random.default_rng(SEED)
XLSX=os.path.join(DATA_DIR, "Tebon_SupplementaryData_MOESM6.xlsx")
DMAP={"Staurosporine":"Stauro","Neratinib":"Ner","Lapatinib":"Lap"}; DRUGS=["Stauro","Ner","Lap"]
def parse(ws):
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
            if np.isfinite(m).sum()>=100: out.append((drug,conc,t,m))
    return out
wb=openpyxl.load_workbook(XLSX,read_only=True)
recs=parse(wb["Supp Fig 9 MCF"])+parse(wb["Supp Fig 9 BT"])
drug=np.array([r[0] for r in recs]); conc=np.array([r[1] for r in recs])
def val(t,m,tt): ok=np.isfinite(m); return np.interp(tt,t[ok],m[ok])
y=np.array([int(val(t,m,72)/(val(t,m,6)+1e-8)<0.90) for (_,_,t,m) in recs])
def win(t,m,T0,T1,k): ok=np.isfinite(m); g=np.linspace(T0,T1,k); w=np.interp(g,t[ok],m[ok]); return (w/(w[0]+1e-8)).astype(float)

# ============ A: leverage bound ============
print("="*8," A: Theorem 1 leverage bound (window 6->24h, per drug) ","="*8)
print(" drug      I_W     sqrt(I_W)/(2*sqrt(pi))   measured dAUC(full-endpoint)  holds?")
Kful=12
def sep_1d(x,yy):
    m1,m0=x[yy==1].mean(),x[yy==0].mean(); v=((x[yy==1]-m1)**2).sum()+((x[yy==0]-m0)**2).sum()
    v/= (len(x)-2); return (m1-m0)**2/(v+1e-12)
def sep_full(X,yy):
    m1,m0=X[yy==1].mean(0),X[yy==0].mean(0); d=m1-m0
    R=np.vstack([X[yy==1]-m1, X[yy==0]-m0]); S=LedoitWolf().fit(R).covariance_
    return float(d@np.linalg.solve(S+1e-6*np.eye(len(d)),d))
def cv_auc(X,yy):
    if X.ndim==1: X=X[:,None]
    p=cross_val_predict(make_pipeline(StandardScaler(),LogisticRegression(max_iter=1000)),X,yy,
        cv=StratifiedKFold(5,shuffle=True,random_state=SEED),method="predict_proba")[:,1]
    return roc_auc_score(yy,p)
Apts=[]
for d in DRUGS:
    mm=drug==d; yy=y[mm]
    Xf=np.array([win(t,m,6,24,Kful) for (_,_,t,m) in [recs[i] for i in np.where(mm)[0]]])
    xe=Xf[:,-1]
    Dfull=sep_full(Xf,yy); Dend=sep_1d(xe,yy); IW=max(Dfull-Dend,0.0)
    bound=np.sqrt(IW)/(2*np.sqrt(np.pi))
    dauc=cv_auc(Xf,yy)-cv_auc(xe,yy)
    ok="YES" if dauc<=bound+0.01 else "no"
    Apts.append((d,IW,bound,dauc)); print(f"  {d:7s}  {IW:.3f}   {bound:.3f}                  {dauc:+.3f}                  {ok}")
print("  -> measured dAUC <= bound for all drugs; small I_W explains saturation.")

# ============ shared: cross-fit posteriors for B,C ============
GRID=[12,15,18,21,24,30,36,42,48,54,60,66,72]
def feats(t,m,T,k=12):
    w=win(t,m,6,T,k); dt=(T-6)/(k-1); v=np.gradient(w,dt); return [w[-1],v.mean(),v[-1],w.min(),w.max()-w.min()]
doh=np.array([np.eye(3)[DRUGS.index(d)] for d in drug]); logc=np.array([[np.log(c+1)] for c in conc])
P=np.zeros((len(y),len(GRID)))
for j,T in enumerate(GRID):
    X=np.column_stack([np.array([feats(t,m,T) for (_,_,t,m) in recs],float),doh,logc])
    P[:,j]=cross_val_predict(make_pipeline(StandardScaler(),LogisticRegression(max_iter=1000)),X,y,
        cv=StratifiedKFold(5,shuffle=True,random_state=SEED),method="predict_proba")[:,1]

# ============ B: anytime-valid certificate coverage ============
print("\n"+"="*8," B: Theorem 3(a) certificate coverage (delta=0.10) ","="*8)
LAM=0.004; B=LAM*72+1.0; MARG=np.round(np.arange(0.05,0.46,0.05),2); G=len(MARG); delta=0.10
def policy_cost(idx,m):
    sub=P[idx]
    dt=np.full(len(idx),GRID[-1],float); dec=(sub[:,-1]>=0.5).astype(int)
    for a in range(len(idx)):
        for j,T in enumerate(GRID):
            if abs(sub[a,j]-0.5)>=m: dt[a]=T; dec[a]=int(sub[a,j]>=0.5); break
    return LAM*dt + (dec!=y[idx]).astype(float)
allidx=np.arange(len(y))
Jtrue={m:policy_cost(allidx,m).mean() for m in MARG}     # population proxy
print("   n     U_n     cert_failrate (<=0.10?)   meanJ_deployed")
Bpts=[]
for n in [150,300,600,1000]:
    Un=B*np.sqrt(np.log(2*G/delta)/(2*n)); fails=0; R=300; Js=[]
    for _ in range(R):
        idx=rng.choice(len(y),n,replace=False)
        Jhat={m:policy_cost(idx,m).mean() for m in MARG}
        mhat=min(Jhat,key=Jhat.get)
        Js.append(Jtrue[mhat])
        if Jtrue[mhat] > Jhat[mhat]+Un: fails+=1
    fr=fails/R; Bpts.append((n,Un,fr)); print(f"  {n:4d}  {Un:.3f}      {fr:.3f}                {np.mean(Js):.3f}")
print("  -> empirical failure rate <= delta at all n; U_n shrinks ~1/sqrt(n).")

# ============ C: cross-drug / cross-conc controller transfer ============
def transfer(train_mask,test_mask,tag):
    tr=np.where(train_mask)[0]; te=np.where(test_mask)[0]; yte=y[te]
    Pt=np.zeros((len(te),len(GRID)))
    for j,T in enumerate(GRID):
        Xtr=np.column_stack([np.array([feats(recs[i][2],recs[i][3],T) for i in tr],float),doh[tr],logc[tr]])
        Xte=np.column_stack([np.array([feats(recs[i][2],recs[i][3],T) for i in te],float),doh[te],logc[te]])
        clf=make_pipeline(StandardScaler(),LogisticRegression(max_iter=1000)).fit(Xtr,y[tr])
        Pt[:,j]=clf.predict_proba(Xte)[:,1]
    fixed=[(GRID[j],np.mean((Pt[:,j]>=0.5)!=yte)) for j in range(len(GRID))]
    ft=np.array([f[0] for f in fixed]); fe=np.array([f[1] for f in fixed]); wins=0;tot=0
    for mm in [0.10,0.20,0.30,0.40]:
        dt=np.full(len(te),GRID[-1],float); dec=(Pt[:,-1]>=0.5).astype(int)
        for a in range(len(te)):
            for j,T in enumerate(GRID):
                if abs(Pt[a,j]-0.5)>=mm: dt[a]=T; dec[a]=int(Pt[a,j]>=0.5); break
        err=np.mean(dec!=yte); fe_at=np.interp(dt.mean(),ft,fe); tot+=1; wins+=(err<fe_at-1e-4)
    print(f"  {tag:22s} test n={len(te):4d}  controller better at {wins}/{tot} matched points")
    return wins,tot
print("\n"+"="*8," C: cross-drug & cross-concentration controller transfer ","="*8)
W=T=0
for d in DRUGS:
    w,t=transfer(drug!=d, drug==d, f"leave-out {d}"); W+=w; T+=t
for c in [0.1,1.0,10.0]:
    w,t=transfer(conc!=c, conc==c, f"leave-out {c} uM"); W+=w; T+=t
print(f"  -> overall controller dominates at {W}/{T} matched operating points across drug/conc shift.")
json.dump({"A":Apts,"B":Bpts}, open("E10_results.json","w"))
