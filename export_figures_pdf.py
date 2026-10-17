#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Collect every figure of the manuscript and the supplement as one vector PDF per figure,
named by its number, into figures_submission/ (what a journal upload expects).

Since Stage 3O every figure is a single file drawn at its final size (main text:
figs_tkde.py + fig_overview_tkde.tex; supplement: figs_supp.py), so this is a plain copy
plus a report. The report is the submission check: page size, smallest lettering, and
whether any lettering is under IEEE's 8 pt floor (sub/superscripts excepted).

Run after the figures have been rebuilt.
"""
import os, shutil, sys, collections
import pymupdf as fitz
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

OUT = "figures_submission"
if os.path.isdir(OUT): shutil.rmtree(OUT)
os.makedirs(OUT)

FIGS = [
    ("Fig1", "figures_tkde/fig_overview_tkde.pdf"),
    ("Fig2", "figures_tkde/Fig2_rq1.pdf"),
    ("Fig3", "figures_tkde/Fig3_matrix.pdf"),
    ("Fig4", "figures_tkde/Fig4_shift.pdf"),
    ("Fig5", "figures_tkde/Fig5_cert.pdf"),
    ("Fig6", "figures_tkde/Fig6_calib.pdf"),
    ("Fig7", "figures_tkde/Fig7_rq5.pdf"),
    
    ("FigS1", "figures_supp/figS1_cost.pdf"),
    ("FigS2", "figures_supp/figS2_endpoint.pdf"),
    ("FigS3", "figures_supp/figS3_bydrug.pdf"),
    ("FigS4", "figures_supp/figS4_window_info.pdf"),
    ("FigS5", "figures_supp/figS5_band.pdf"),
    ("FigS6", "figures_supp/figS6_motivation.pdf"),
    ("FigS7", "figures_supp/figS7_pancreatic.pdf"),
    ("FigS8", "figures_supp/figS8_ucr40.pdf"),
]
SUBSCRIPT = 6.5          # spans at or under this size next to 8 pt text are sub/superscripts

print(f"{'figure':<7} {'kB':>6}  {'mm (w x h)':>15}  {'lettering pt':>22}  under 8 pt")
bad = 0
for name, src in FIGS:
    dst = os.path.join(OUT, name + ".pdf")
    shutil.copyfile(src, dst)
    p = fitz.open(dst)[0]
    sizes = [round(s["size"], 1) for b in p.get_text("dict")["blocks"] for l in b.get("lines", [])
             for s in l.get("spans", []) if s["text"].strip()]
    c = collections.Counter(sizes)
    body = [s for s in sizes if s > SUBSCRIPT]
    small = sum(1 for s in body if s < 7.95)
    bad += small
    print(f"{name:<7} {os.path.getsize(dst)/1024:6.1f}  {p.rect.width*25.4/72:6.1f} x {p.rect.height*25.4/72:5.1f}  "
          f"body {min(body):.1f}-{max(body):.1f}, sub {min(sizes):.1f}  {small if small else 'none'}")
print(f"\n{len(FIGS)} figures written to {OUT}/;  lettering under 8 pt (excluding sub/superscripts): {bad}")
