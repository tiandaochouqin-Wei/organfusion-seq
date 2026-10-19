#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""STEP 4: window-information decomposition (Table S7; data behind Fig. S4).
Reuses E10's separability definitions. Per-drug and per-(drug,conc):
D_end, D_full, I_W=D_full-D_end, bound=sqrt(I_W)/(2 sqrt pi), measured dAUC.
Diagnostic figure fig6_window_info: (a) D_end/I_W decomposition, (b) measured vs bound, (c) stratified I_W heatmap,
(d) endpoint-only vs trajectory AUC by horizon. REAL data."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import DATA_DIR  # $ORGANFUSION_DATA, or ./datasets (see README)
import os, sys, csv, numpy as np
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
from sklearn.covariance import LedoitWolf
from sklearn.metrics import roc_auc_score
plt.rcParams.update({"font.family":"serif","font.serif":["Times New Roman","DejaVu Serif"],
  "mathtext.fontset":"stix","font.size":7.5,"axes.labelsize":7.5,"xtick.labelsize":6.6,
  "ytick.labelsize":6.6,"axes.linewidth":0.8,"axes.spines.top":False,"axes.spines.right":False,
  "legend.frameon":False,"axes.grid":True,"grid.alpha":0.18,"grid.linewidth":0.5,"axes.axisbelow":True,
  "svg.fonttype":"none","pdf.fonttype":42})
PRI="#2166AC";PRI_D="#15406E";ROSE="#C0567E";GRAY="#8C8C8C";GOLD="#C99A2E";LB="#9EC5E0";INK="#222"
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
wb=openpyxl.load_workbook(XLSX,read_only=True); recs=parse(wb["Supp Fig 9 MCF"],"MCF-7")+parse(wb["Supp Fig 9 BT"],"BT-474"); recs=filter_recs(recs)
cell=np.array([r[0] for r in recs]); drug=np.array([r[1] for r in recs]); conc=np.array([r[2] for r in recs])
cond_key=np.array([f"{c}|{d}|{cc}" for c,d,cc in zip(cell,drug,conc)])
cond_ids={k:i for i,k in enumerate(sorted(set(cond_key)))}; groups_all=np.array([cond_ids[k] for k in cond_key])
def val(t,m,tt): ok=np.isfinite(m); return np.interp(tt,t[ok],m[ok])
y=np.array([int(val(t,m,72)/(val(t,m,6)+1e-8)<0.90) for (_,_,_,t,m) in recs])
def win(t,m,T0,T1,k): ok=np.isfinite(m); g=np.linspace(T0,T1,k); w=np.interp(g,t[ok],m[ok]); return (w/(w[0]+1e-8)).astype(float)
def sep_1d(x,yy):
    m1,m0=x[yy==1].mean(),x[yy==0].mean(); v=(((x[yy==1]-m1)**2).sum()+((x[yy==0]-m0)**2).sum())/(len(x)-2); return (m1-m0)**2/(v+1e-12)
def sep_full(X,yy):
    m1,m0=X[yy==1].mean(0),X[yy==0].mean(0); d=m1-m0
    R=np.vstack([X[yy==1]-m1,X[yy==0]-m0]); S=LedoitWolf().fit(R).covariance_
    return float(d@np.linalg.solve(S+1e-6*np.eye(len(d)),d))
def cv_auc(X,yy,grp):
    if X.ndim==1: X=X[:,None]
    n_grp=len(set(grp.tolist())); nfold=min(5,n_grp)
    p=cross_val_predict(make_pipeline(StandardScaler(),LogisticRegression(max_iter=1000)),X,yy,
        cv=GroupKFold(nfold).split(X,yy,grp),method="predict_proba")[:,1]; return roc_auc_score(yy,p)
Kful=12
def decomp(mask):
    yy=y[mask]; grp=groups_all[mask]; Xf=np.array([win(recs[i][3],recs[i][4],6,24,Kful) for i in np.where(mask)[0]]); xe=Xf[:,-1]
    Df=sep_full(Xf,yy); De=sep_1d(xe,yy); IW=max(Df-De,0.0); bd=np.sqrt(IW)/(2*np.sqrt(np.pi))
    dauc=cv_auc(Xf,yy,grp)-cv_auc(xe,yy,grp); return De,Df,IW,bd,dauc

# per-drug
DR={};
print("drug   D_end  D_full  I_W    bound  dAUC")
for d in DRUGS:
    De,Df,IW,bd,da=decomp(drug==d); DR[d]=(De,Df,IW,bd,da)
    print(f"{d:6s} {De:.3f}  {Df:.3f}  {IW:.3f}  {bd:.3f}  {da:+.3f}")
# per (drug,conc) I_W
concs=[0.1,1.0,10.0]; IWm=np.full((3,3),np.nan)
for i,d in enumerate(DRUGS):
    for j,c in enumerate(concs):
        mk=(drug==d)&(conc==c)
        if mk.sum()>=40: IWm[i,j]=decomp(mk)[2]
# endpoint-only vs trajectory AUC by horizon (from full cohort)
GRID=[12,15,18,21,24,36,48,60,72]
endA=[];trajA=[]
for T in GRID:
    Xe=np.array([win(recs[i][3],recs[i][4],6,T,Kful)[-1] for i in range(len(y))])  # endpoint coord at T
    Xt=np.array([win(recs[i][3],recs[i][4],6,T,Kful) for i in range(len(y))])      # full window
    endA.append(cv_auc(Xe,y,groups_all)); trajA.append(cv_auc(Xt,y,groups_all))

# ---- diagnostic figure fig6_window_info ----
fig=plt.figure(figsize=(7.0,4.7)); gs=GridSpec(2,2,figure=fig,hspace=0.62,wspace=0.36)
ax=fig.add_subplot(gs[0,0]); x=np.arange(3)
De=[DR[d][0] for d in DRUGS]; IW=[DR[d][2] for d in DRUGS]
ax.bar(x,De,0.6,color=GRAY,label="$\\mathcal{D}_{\\rm end}$")
ax.bar(x,IW,0.6,bottom=De,color=PRI,label="$\\mathcal{I}_W$")
ax.set_xticks(x);ax.set_xticklabels(DRUGS);ax.set_ylabel("separability $\\mathcal{D}$");ax.legend(loc="upper right",fontsize=6.2)
ax.set_title("$\\mathcal{D}_{\\rm full}=\\mathcal{D}_{\\rm end}+\\mathcal{I}_W$",fontsize=7.5,pad=3);ax.text(0.5,-0.30,"(a)",transform=ax.transAxes,ha="center",va="top",fontsize=9,fontweight="bold")
ax=fig.add_subplot(gs[0,1]); bd=[DR[d][3] for d in DRUGS]; da=[DR[d][4] for d in DRUGS]
ax.bar(x,bd,0.5,color=LB,edgecolor=PRI_D,lw=0.6,label="bound $\\sqrt{\\mathcal{I}_W}/(2\\sqrt{\\pi})$")
ax.plot(x,da,"D",color=ROSE,ms=6,label="measured $\\Delta$AUC",zorder=5);ax.axhline(0,color=GRAY,lw=0.7)
for xi,v in zip(x,da): ax.annotate(f"{v:+.3f}",(xi,v),(0,-10 if v<0 else 6),textcoords="offset points",ha="center",fontsize=5.6,color=ROSE)
ax.set_xticks(x);ax.set_xticklabels(DRUGS);ax.set_ylabel("AUC gain");ax.set_ylim(-0.17,0.20);ax.legend(loc="lower left",fontsize=5.8)
ax.text(0.5,-0.30,"(b)",transform=ax.transAxes,ha="center",va="top",fontsize=9,fontweight="bold")
ax=fig.add_subplot(gs[1,0]); im=ax.imshow(IWm,cmap="YlOrRd",aspect="auto",vmin=0)
ax.set_xticks(range(3),[f"{c:g}$\\mu$M" for c in concs],fontsize=6.3);ax.set_yticks(range(3),DRUGS,fontsize=6.3)
for i in range(3):
    for j in range(3):
        if np.isfinite(IWm[i,j]): ax.text(j,i,f"{IWm[i,j]:.2f}",ha="center",va="center",fontsize=6,color="white" if IWm[i,j]>0.15 else INK)
ax.set_title("stratified $\\mathcal{I}_W$",fontsize=7.5,pad=3);ax.grid(False);ax.text(0.5,-0.32,"(c)",transform=ax.transAxes,ha="center",va="top",fontsize=9,fontweight="bold")
ax=fig.add_subplot(gs[1,1])
ax.plot(GRID,endA,"-o",color=GRAY,ms=3,lw=1.4,label="endpoint-only")
ax.plot(GRID,trajA,"-o",color=PRI,ms=3,lw=1.6,label="trajectory")
ax.set_xlabel("horizon $T$ (h)");ax.set_ylabel("AUC");ax.legend(loc="lower right",fontsize=6.3)
ax.text(0.5,-0.30,"(d)",transform=ax.transAxes,ha="center",va="top",fontsize=9,fontweight="bold")
os.makedirs("figures_regen",exist_ok=True)
for e in ("pdf","png","svg"): fig.savefig(f"figures_regen/fig6_window_info.{e}",dpi=600,bbox_inches="tight")
plt.close(fig)
# ---- Table S7 full ----
with open("posterior/table3_window_info.csv","w",newline="",encoding="utf-8") as f:
    w=csv.writer(f);w.writerow(["drug","D_end","D_full","I_W","bound_sqrtIW_2sqrtpi","measured_dAUC","satisfied"])
    for d in DRUGS:
        De,Df,IW,bd,da=DR[d]; w.writerow([d,round(De,4),round(Df,4),round(IW,4),round(bd,4),round(da,4),"Yes" if da<=bd+0.01 else "No"])
print("wrote fig6_window_info + table3_window_info.csv")
print("endpoint vs trajectory AUC@24h: %.3f vs %.3f (gain %.3f)"%(endA[GRID.index(24)],trajA[GRID.index(24)],trajA[GRID.index(24)]-endA[GRID.index(24)]))
