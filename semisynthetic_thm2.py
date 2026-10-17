#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stage 3R: Theorem 2 checked with every quantity known exactly (semi-synthetic Bayesian model).

On real data kappa, eta and Delta_info are not observable: ECE_stop estimates kappa only under
self-sufficiency, and eta and Delta_info need the true posterior. Here the posterior is known:
  y ~ Bernoulli(1/2); observations x_1..x_N i.i.d. N(+mu, 1) if y = 1 and N(-mu, 1) if y = 0, so the true
  log-odds l_k is a random walk with Gaussian increments and p_k = sigmoid(l_k) is a Markov posterior
  (Assumption 1 holds). Costs c_FP = 1, c_FN = 3 (L = 3, p* = 1/4), delay lambda per step.
Block A -- miscalibration only: the deployed estimate is p_hat = sigmoid(gamma * l), a strictly monotone
  recalibration, so G = F, q = p and Delta_info = 0. The deployed rule is the exact recursion of
  Theorem 1 on p_hat with its true kernel (eta = 0 up to the grid). Everything is computed by dynamic
  programming on a fine log-odds grid: V, J_dep(tau_hat), kappa(tau_hat), kappa(tau*), and
  kappa_bar = sup_tau kappa(tau) (an optimal stopping problem that maximises E|p_hat - p|).
  A Monte Carlo sample of 20,000 units gives the stopped-posterior ECE that the experiments use as the
  estimate of kappa.
Block B -- information deficit: the estimator only sees the SIGN of each observation, so G is coarser
  than F; Delta_info = V_q - V > 0 and does not change under any recalibration of the sign-based estimate.
Output: posterior/table_semisynthetic_thm2.csv
"""
import os, csv, sys
import numpy as np
from scipy.stats import norm
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
OUT = "posterior"
MU, N, LAM = 0.5, 12, 0.006
CFP, CFN = 1.0, 3.0; L = max(CFP, CFN); PSTAR = CFP / (CFP + CFN)
sig = lambda z: 1 / (1 + np.exp(-z))
rho = lambda p: np.minimum(CFP * (1 - p), CFN * p)

# ---- log-odds grid and the exact one-step kernel of the true posterior
G_ = np.linspace(-14, 14, 2241); dl = G_[1] - G_[0]
m, v = 2 * MU ** 2, 4 * MU ** 2                         # LLR increment: N(+-m, v) given y
P = sig(G_)
K = P[:, None] * norm.pdf(G_[None, :], G_[:, None] + m, np.sqrt(v)) + \
    (1 - P[:, None]) * norm.pdf(G_[None, :], G_[:, None] - m, np.sqrt(v))
K /= K.sum(1, keepdims=True)
i0 = int(np.argmin(np.abs(G_)))                        # prior log-odds 0


def solve(stop_cost):
    """optimal stopping of E[lambda*tau + stop_cost(l_tau)]; returns stop sets S[k] (bool over grid)."""
    V = stop_cost.copy(); S = [None] * (N + 1); S[N] = np.ones_like(V, bool)
    for k in range(N - 1, -1, -1):
        cont = LAM + K @ V
        S[k] = stop_cost <= cont; V = np.minimum(stop_cost, cont)
    return S, V


def evaluate(S, terminal, step=LAM):
    """E[step*tau + terminal(l_tau)] for the policy S (stop at the first k with S[k])."""
    W = terminal.copy()
    for k in range(N - 1, -1, -1):
        W = np.where(S[k], terminal, step + K @ W)
    return W[i0]


S_star, Vtrue = solve(rho(P)); V = Vtrue[i0]                    # full-information optimum
kappa_star = {}
rows = []
print(f"true value V = {V:.4f}  (L = {L}, p* = {PSTAR}, lambda = {LAM}/step, N = {N})")
print(f"{'gamma':>6} {'J_dep-V':>8} {'L[k(that)+k(t*)]':>17} {'2L kbar':>8} {'ratio':>6} {'kappa(that)':>11} "
      f"{'ECE_stop(n=20k)':>15} {'kappa_hat(n=20k)':>16}")
rng = np.random.default_rng(0)
NS = 20000
ys = rng.integers(0, 2, NS); xs = rng.normal(np.where(ys[:, None] == 1, MU, -MU), 1.0, (NS, N))
lpath = np.concatenate([np.zeros((NS, 1)), np.cumsum(2 * MU * xs, axis=1)], axis=1)   # exact log-odds paths
for gamma in (0.4, 0.6, 0.8, 1.0, 1.25, 1.6, 2.5):
    Ph = sig(gamma * G_)
    S_hat, _ = solve(rho(Ph))                                     # Theorem 1 run on p_hat: eta = 0
    dec_cost = np.where(Ph >= PSTAR, CFP * (1 - P), CFN * P)      # realised decision cost given l
    J_dep = evaluate(S_hat, dec_cost)
    gap = np.abs(Ph - P)
    k_hat = evaluate(S_hat, gap, step=0.0)
    k_star = evaluate(S_star, gap, step=0.0)
    Vk = gap.copy()                                               # kappa_bar: maximise E|p_hat - p| at a stop
    for k in range(N - 1, -1, -1): Vk = np.maximum(gap, K @ Vk)
    kbar = Vk[i0]
    # Monte Carlo: stop each simulated unit by S_hat, then the stopped-posterior ECE (10 bins, by horizon)
    idx = np.clip(np.round((lpath - G_[0]) / dl).astype(int), 0, len(G_) - 1)
    stop = np.array([S_hat[k][idx[:, k]] for k in range(N + 1)]).T
    tau = stop.argmax(1)
    ph_s = sig(gamma * lpath[np.arange(NS), tau]); p_s = sig(lpath[np.arange(NS), tau])
    ece = 0.0
    for t in np.unique(tau):
        mk = tau == t; pp = ph_s[mk]; yy = ys[mk]; e = 0.0
        for b in range(10):
            sel = (pp >= b / 10) & ((pp < (b + 1) / 10) if b < 9 else (pp <= 1))
            if sel.any(): e += sel.mean() * abs(yy[sel].mean() - pp[sel].mean())
        ece += mk.mean() * e
    k_mc = float(np.mean(np.abs(ph_s - p_s)))
    bound = L * (k_hat + k_star)
    rows.append(dict(block="A miscalibration", gamma=gamma, excess=J_dep - V, bound_i=bound, bound_2Lkbar=2 * L * kbar,
                     kappa_hat=k_hat, kappa_star=k_star, kappa_bar=kbar, eta=0.0, delta_info=0.0,
                     ece_stop_mc=ece, kappa_mc=k_mc))
    print(f"{gamma:>6.2f} {J_dep - V:>8.4f} {bound:>17.4f} {2 * L * kbar:>8.4f} "
          f"{(J_dep - V) / bound if bound > 0 else float('nan'):>6.2f} {k_hat:>11.4f} {ece:>15.4f} {k_mc:>16.4f}")

# ---- Block B: the estimator only sees signs -> information deficit
a = norm.cdf(MU)                                                   # Pr(x > 0 | y = 1)
cnt = np.arange(N + 1)
# sign-based posterior after k signs with s positives: log-odds = (2s - k) * log(a/(1-a)); state (k, s)
def sign_value(gamma, recal=True):
    """value V_q of the best G-stopping rule (gamma irrelevant: recalibration does not change G), and the
    realised cost of the Theorem-1 recursion run on the miscalibrated sign-based estimate."""
    w = np.log(a / (1 - a))
    q = lambda k, s: sig((2 * s - k) * w)                          # calibrated sign posterior
    ph = lambda k, s: sig(gamma * (2 * s - k) * w)
    def nxt(k, s, f):                                              # expectation over the next sign
        qq = q(k, s); pu = qq * a + (1 - qq) * (1 - a)
        return pu * f(k + 1, s + 1) + (1 - pu) * f(k + 1, s)
    from functools import lru_cache
    @lru_cache(None)
    def Vq(k, s):
        st = rho(q(k, s))
        return st if k == N else min(st, LAM + nxt(k, s, Vq))
    @lru_cache(None)
    def Vh(k, s):                                                  # believed recursion on p_hat
        st = rho(ph(k, s))
        return st if k == N else min(st, LAM + nxt(k, s, Vh))
    @lru_cache(None)
    def Jd(k, s):                                                  # realised cost of that rule
        stop_now = k == N or rho(ph(k, s)) <= LAM + nxt(k, s, Vh)
        if stop_now:
            qq = q(k, s); return CFP * (1 - qq) if ph(k, s) >= PSTAR else CFN * qq
        return LAM + nxt(k, s, Jd)
    return Vq(0, 0), Jd(0, 0)

for gamma in (0.5, 1.0, 2.0):
    Vq, Jd = sign_value(gamma)
    rows.append(dict(block="B sign-only (information deficit)", gamma=gamma, excess=Jd - V, bound_i=np.nan,
                     bound_2Lkbar=np.nan, kappa_hat=np.nan, kappa_star=np.nan, kappa_bar=np.nan, eta=0.0,
                     delta_info=Vq - V, ece_stop_mc=np.nan, kappa_mc=np.nan))
    print(f"[B] gamma={gamma:.1f}: Delta_info = V_q - V = {Vq - V:.4f} (same for every gamma); "
          f"J_dep - V = {Jd - V:.4f}; J_dep - V_q = {Jd - Vq:.4f}")

with open(os.path.join(OUT, "table_semisynthetic_thm2.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader()
    for r in rows: w.writerow({k: (round(v, 6) if isinstance(v, float) else v) for k, v in r.items()})
print("wrote table_semisynthetic_thm2.csv")
