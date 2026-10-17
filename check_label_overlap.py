#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stage 3Q-3: panel labels "(a)", "(b)", ... must not touch any other lettering in the figure.

check_text_line_overlap.py checks text against strokes only; a wider label could still sit on a tick
label or a title. This compares the bounding box of every panel label with every other text span on
the page (glyph boxes shrunk by 0.4 pt so that merely adjacent text is not reported).
Usage: python check_label_overlap.py   -> prints offending pairs and TOTAL.
"""
import glob, re, sys
import pymupdf as fitz
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
SHRINK = 0.4
files = ["figures_tkde/fig_overview_tkde.pdf"] + sorted(glob.glob("figures_tkde/Fig*.pdf")) + sorted(glob.glob("figures_supp/figS*.pdf"))
total = 0
for fn in files:
    pg = fitz.open(fn)[0]
    spans = [(s["text"].strip(), fitz.Rect(s["bbox"])) for b in pg.get_text("dict")["blocks"]
             for l in b.get("lines", []) for s in l.get("spans", []) if s["text"].strip()]
    labels = [(t, r) for t, r in spans if re.fullmatch(r"\([a-h]\)", t)]
    hits = []
    for t, r in labels:
        rr = fitz.Rect(r.x0 + SHRINK, r.y0 + SHRINK, r.x1 - SHRINK, r.y1 - SHRINK)
        for t2, r2 in spans:
            if t2 == t and r2 == r: continue
            r2s = fitz.Rect(r2.x0 + SHRINK, r2.y0 + SHRINK, r2.x1 - SHRINK, r2.y1 - SHRINK)
            if rr.intersects(r2s): hits.append((t, t2))
    total += len(hits)
    print(f"{fn}: {len(labels)} panel labels" + ("" if not hits else f", OVERLAPS: {hits}"))
print(f"TOTAL: {total}")
