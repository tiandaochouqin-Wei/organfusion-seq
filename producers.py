#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared library for the shared-prefix-encoder experiments.

Everything downstream (chain recompute, fusion ablation, extra stopping baselines,
deep estimator swap) imports from here, so every comparison uses ONE implementation
of the encoder, the pooling, the calibrators and the metrics. The whole point of
these experiments is fair comparison; re-implementing per script is how unfair
comparisons get in.

Design constraint that matters: the posterior must be CAUSAL. p_t may depend only on
x_{1:t}. Every pooling variant below aggregates over the prefix {h_1..h_t} only
(cumulative mean/max, causally-masked attention), never the whole sequence.
"""
import numpy as np, torch, torch.nn as nn
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss

DEV = "cpu"
DRUGS = ["Stauro", "Ner", "Lap"]


def set_seed(s=0):
    torch.manual_seed(s); np.random.seed(s)


def load_data(npz="posterior/posterior_data.npz"):
    d = np.load(npz, allow_pickle=True)
    xgrid = d["xgrid"]; y = d["y"].astype(np.float32); GRID = d["GRID"].astype(float)
    drug = d["drug"]; conc = d["conc"].astype(float); cell = d["cell"]
    N, T = xgrid.shape
    tnorm = np.tile((GRID / 72.0)[None, :], (N, 1))
    inc = np.concatenate([np.zeros((N, 1)), np.diff(xgrid, axis=1)], axis=1)
    SEQ = np.stack([xgrid, tnorm, inc], axis=-1).astype(np.float32)
    doh = np.array([np.eye(3)[DRUGS.index(x)] for x in drug], dtype=np.float32)
    COND = np.concatenate([doh, np.log1p(conc)[:, None].astype(np.float32)], axis=1)
    cond_key = np.array([f"{c}|{dr}|{cc}" for c, dr, cc in zip(cell, drug, conc)])
    uniq = {k: i for i, k in enumerate(sorted(set(cond_key)))}
    groups = np.array([uniq[k] for k in cond_key])
    return dict(SEQ=SEQ, COND=COND, y=y, GRID=GRID, groups=groups, drug=drug,
                conc=conc, cell=cell, xgrid=xgrid, N=N, T=T,
                P_legacy=d["P"] if "P" in d else None)


# --------------------------------------------------------------------------- #
# Causal encoders: each maps [B,T,3] -> H[B,T,D] with H[:,t] depending on x_{1:t}
# --------------------------------------------------------------------------- #
class GRUEnc(nn.Module):
    def __init__(self, d=32):
        super().__init__(); self.rnn = nn.GRU(3, d, batch_first=True); self.d = d
    def forward(self, x): return self.rnn(x)[0]


class TCNEnc(nn.Module):
    """Dilated CAUSAL conv stack (left-padding only, so no future leakage)."""
    def __init__(self, d=32, layers=3):
        super().__init__()
        self.convs = nn.ModuleList()
        cin = 3
        self.pads = []
        for i in range(layers):
            dil = 2 ** i
            self.convs.append(nn.Conv1d(cin, d, kernel_size=2, dilation=dil))
            self.pads.append(dil)          # left pad = (k-1)*dilation = dilation
            cin = d
        self.d = d
    def forward(self, x):
        h = x.transpose(1, 2)                       # [B,3,T]
        for conv, p in zip(self.convs, self.pads):
            h = torch.relu(conv(nn.functional.pad(h, (p, 0))))   # causal left-pad
        return h.transpose(1, 2)                    # [B,T,d]


class TransformerEnc(nn.Module):
    """Small CAUSALLY-MASKED transformer encoder."""
    def __init__(self, d=32, heads=4, layers=2):
        super().__init__()
        self.inp = nn.Linear(3, d)
        layer = nn.TransformerEncoderLayer(d_model=d, nhead=heads, dim_feedforward=2 * d,
                                           batch_first=True, dropout=0.0)
        self.enc = nn.TransformerEncoder(layer, num_layers=layers)
        self.d = d
    def forward(self, x):
        T = x.shape[1]
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)  # block future
        return self.enc(self.inp(x), mask=mask)


ENCODERS = {"GRU": GRUEnc, "TCN": TCNEnc, "Transformer": TransformerEnc}


# --------------------------------------------------------------------------- #
# Causal prefix pooling variants (the reviewer's Sec. 6 fusion ablation)
# --------------------------------------------------------------------------- #
def causal_pool(H, mode, score=None):
    """H:[B,T,D] -> G:[B,T,D] where G[:,t] aggregates ONLY H[:, :t+1].
    score:[B,T] optional attention logits (condition-driven)."""
    B, T, D = H.shape
    if mode == "last":
        return H
    if mode == "mean":
        return H.cumsum(1) / torch.arange(1, T + 1, device=H.device).view(1, T, 1)
    if mode == "max":
        return H.cummax(1).values
    if mode in ("attn_simplex", "attn_unconstrained"):
        # causal mask: position t may attend to j<=t
        idx = torch.arange(T, device=H.device)
        m = (idx.view(1, T, 1) >= idx.view(1, 1, T))          # [1,T,T] True where j<=t
        s = score.unsqueeze(1).expand(B, T, T)                 # [B,T(query),T(key)]
        if mode == "attn_simplex":
            s = s.masked_fill(~m, float("-inf"))
            a = torch.softmax(s, dim=-1)                       # simplex: a>=0, sum=1
        else:
            a = s.masked_fill(~m, 0.0)                         # free weights, no simplex
        return torch.bmm(a, H)                                 # [B,T,D]
    raise ValueError(mode)


class Producer(nn.Module):
    """encoder -> causal prefix pooling -> [g_t ; condition embedding] -> shared head.
    One theta for every prefix length (the reviewer's h_t = f_theta(x_{1:t}, z))."""
    def __init__(self, enc="GRU", pool="last", d=32, d_cond=8):
        super().__init__()
        self.enc = ENCODERS[enc](d); self.pool = pool
        self.cond = nn.Linear(4, d_cond)
        self.score = nn.Linear(d + d_cond, 1)      # condition-driven attention logits
        self.head = nn.Linear(d + d_cond, 1)
    def forward(self, x, c):
        H = self.enc(x)                                        # [B,T,D]
        e = torch.tanh(self.cond(c))                           # [B,dc]
        ee = e.unsqueeze(1).expand(-1, H.shape[1], -1)
        sc = None
        if self.pool.startswith("attn"):
            sc = self.score(torch.cat([H, ee], -1)).squeeze(-1)  # [B,T]
        G = causal_pool(H, self.pool, sc)
        return self.head(torch.cat([G, ee], -1)).squeeze(-1)   # [B,T] logits


def fit_producer(D, tr, enc="GRU", pool="last", epochs=300, lr=0.01, seed=0):
    set_seed(seed)
    m = Producer(enc, pool).to(DEV)
    opt = torch.optim.Adam(m.parameters(), lr=lr); lf = nn.BCEWithLogitsLoss()
    xs = torch.tensor(D["SEQ"][tr]); cs = torch.tensor(D["COND"][tr])
    ys = torch.tensor(D["y"][tr]).unsqueeze(1).expand(-1, D["T"])
    m.train()
    for _ in range(epochs):
        opt.zero_grad(); lf(m(xs, cs), ys).backward(); opt.step()
    m.eval(); return m


def logits_of(m, D, idx):
    with torch.no_grad():
        return m(torch.tensor(D["SEQ"][idx]), torch.tensor(D["COND"][idx])).cpu().numpy()


# --------------------------------------------------------------------------- #
# Calibrators: fit on a calibration split ONLY, then applied elsewhere
# --------------------------------------------------------------------------- #
def cal_none(lo, yv):
    return (lambda L: 1 / (1 + np.exp(-L))), "none"

def cal_temperature(lo, yv):
    l = torch.tensor(lo.ravel(), dtype=torch.float32)
    t = torch.tensor(np.repeat(yv, lo.shape[1]), dtype=torch.float32)
    logT = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([logT], lr=0.1, max_iter=60); lf = nn.BCEWithLogitsLoss()
    def cl():
        opt.zero_grad(); z = lf(l / torch.exp(logT), t); z.backward(); return z
    opt.step(cl)
    Tv = float(torch.exp(logT).detach())
    return (lambda L: 1 / (1 + np.exp(-L / Tv))), f"temp(T={Tv:.2f})"

def cal_isotonic(lo, yv):
    p = 1 / (1 + np.exp(-lo.ravel()))
    ir = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(p, np.repeat(yv, lo.shape[1]))
    return (lambda L: ir.predict(np.clip(1 / (1 + np.exp(-L)), 0, 1).ravel()).reshape(L.shape)), "isotonic"

def cal_beta(lo, yv):
    p = np.clip(1 / (1 + np.exp(-lo.ravel())), 1e-6, 1 - 1e-6)
    X = np.column_stack([np.log(p), -np.log(1 - p)])
    lr = LogisticRegression(max_iter=1000).fit(X, np.repeat(yv, lo.shape[1]))
    def f(L):
        q = np.clip(1 / (1 + np.exp(-L)), 1e-6, 1 - 1e-6)
        Xt = np.column_stack([np.log(q.ravel()), -np.log(1 - q.ravel())])
        return lr.predict_proba(Xt)[:, 1].reshape(L.shape)
    return f, "beta"

CALIBRATORS = {"none": cal_none, "temperature": cal_temperature,
               "isotonic": cal_isotonic, "beta": cal_beta}


# --------------------------------------------------------------------------- #
# Metrics (the reviewer's Sec. 15 list)
# --------------------------------------------------------------------------- #
def ece_mce(p, yv, nb=10):
    b = np.linspace(0, 1, nb + 1); e = 0.0; m = 0.0
    for i in range(nb):
        msk = (p >= b[i]) & (p < b[i + 1]) if i < nb - 1 else (p >= b[i]) & (p <= b[i + 1])
        if msk.sum():
            gap = abs(yv[msk].mean() - p[msk].mean())
            e += msk.mean() * gap; m = max(m, gap)
    return e, m

def cal_slope_intercept(p, yv):
    """Regress y on logit(p): y ~ sigma(a + b*logit(p)). Perfect calibration => b=1, a=0."""
    q = np.clip(p, 1e-6, 1 - 1e-6); z = np.log(q / (1 - q)).reshape(-1, 1)
    if len(set(yv.tolist())) < 2: return np.nan, np.nan
    lr = LogisticRegression(max_iter=1000).fit(z, yv)
    return float(lr.coef_[0][0]), float(lr.intercept_[0])

def controller(P, GRID, m, p_star=0.5):
    N, T = P.shape
    times = np.empty(N); dec = np.empty(N, int)
    for i in range(N):
        ex = np.where(np.abs(P[i] - p_star) >= m)[0]
        j = ex[0] if len(ex) else T - 1
        times[i] = GRID[j]; dec[i] = int(P[i, j] >= p_star)
    return times, dec

def full_metrics(P, y, GRID, lam=0.004, m_grid=None):
    """Prediction-layer + decision-layer + control-layer metrics at 24h and at the
    lambda-optimal band."""
    j24 = int(np.where(GRID == 24)[0][0])
    p24 = P[:, j24]
    auc = roc_auc_score(y, p24); br = brier_score_loss(y, p24)
    e, mx = ece_mce(p24, y); sl, ic = cal_slope_intercept(p24, y)
    m_grid = np.linspace(0.02, 0.45, 30) if m_grid is None else m_grid
    best = None
    for m in m_grid:
        t_, d_ = controller(P, GRID, m)
        J = (d_ != y).mean() + lam * t_.mean()
        if best is None or J < best[0]: best = (J, m, (d_ != y).mean(), t_.mean())
    J, m_star, err, tbar = best
    return dict(auc=auc, brier=br, ece=e, mce=mx, slope=sl, intercept=ic,
                m_star=m_star, dec_err=err, mean_t=tbar, control_cost=J)
