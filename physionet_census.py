#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Quick census of PhysioNet Challenge 2019 files already on disk (partial
download is fine -- random file order, representative sample) to inform grid
spacing / stay-length cutoff / core-vital coverage before building the full
protocol. Prints distributional stats only, writes nothing."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import DATA_DIR  # $ORGANFUSION_DATA, or ./datasets (see README)
import glob, sys, numpy as np
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

CORE = ["HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp"]

files = sorted(glob.glob(os.path.join(DATA_DIR, "physionet_challenge2019/training_set*/*.psv")))
print(f"files on disk: {len(files)}")
if not files:
    sys.exit(0)

with open(files[0], encoding="utf-8") as f:
    header = f.readline().strip().split("|")
col = {c: i for i, c in enumerate(header)}
core_idx = {c: col[c] for c in CORE}
sep_idx = col["SepsisLabel"]; iculos_idx = col["ICULOS"]

stay_lens = []; sepsis_flags = []; coverage = {c: [] for c in CORE}
for fp in files:
    with open(fp, encoding="utf-8") as f:
        lines = f.read().splitlines()[1:]
    if not lines:
        continue
    rows = [ln.split("|") for ln in lines]
    T = len(rows)
    stay_lens.append(T)
    sep = any(r[sep_idx] == "1" for r in rows)
    sepsis_flags.append(sep)
    for c in CORE:
        idx = core_idx[c]
        n_real = sum(1 for r in rows if r[idx] != "NaN")
        coverage[c].append(n_real / T)

stay_lens = np.array(stay_lens); sepsis_flags = np.array(sepsis_flags)
print(f"\nstay length (hours): min={stay_lens.min()} p10={np.percentile(stay_lens,10):.0f} "
      f"median={np.median(stay_lens):.0f} p90={np.percentile(stay_lens,90):.0f} max={stay_lens.max()}")
print(f"sepsis prevalence (ever positive): {sepsis_flags.mean():.4f} ({sepsis_flags.sum()}/{len(sepsis_flags)})")
print("\ncore vital coverage (fraction of hours with a real reading, mean across patients):")
for c in CORE:
    cov = np.array(coverage[c])
    print(f"  {c:6s} mean_coverage={cov.mean():.3f}  frac_patients_with_>=1_reading={ (cov>0).mean():.3f}")

for cutoff in [8, 12, 24, 36, 48]:
    frac = (stay_lens >= cutoff).mean()
    print(f"fraction of patients with stay >= {cutoff:>3}h: {frac:.3f}  (n={ (stay_lens>=cutoff).sum() })")
