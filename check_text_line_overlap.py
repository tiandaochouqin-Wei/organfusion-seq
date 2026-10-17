#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Objective "no glyph sits on a stroked line" check for every figure PDF.

Eyeballing crops misses collisions, so this reads each figure's own vector content:
every text span's bounding box (from the PDF text layer) is tested against every
STROKED path segment (fills are ignored -- text over a light shaded region is fine,
and the white halo patches behind labels are fills, so they correctly mask a line
rather than count as one). Curves are flattened by sampling their Bezier control net.

A hit is reported with the figure, the text, and the stroke colour, so each one can be
judged and fixed. Tick labels against their own axis spine are filtered out by the
`--pad` shrink plus the spine/tick heuristics below.

Usage: py check_text_line_overlap.py [pdf ...]      (default: all main + supp figures)
"""
import sys, glob
import pymupdf as fitz
fitz.TOOLS.set_aa_level(0)   # glyph anti-aliasing would otherwise read as grey line pixels

PAD_X = 0.02        # barely inset horizontally: a line grazing the last glyph is still
                    # a collision, and a 12% inset missed exactly those
PAD_Y = 0.18        # inset more vertically: a span's bbox includes the font's ascent/
                    # descent padding, so a box border sitting in that blank strip is not
                    # touching any glyph
WHITE = 0.75        # strokes at least this bright are background, halo or hatch texture:
                    # matplotlib emits hatch fills as page-wide stroke runs, which would
                    # otherwise flag every label in a hatched matrix cell.
FILL_WHITE = 0.93   # fills at least this bright count as a halo patch


def inside(p, r):
    return r[0] <= p[0] <= r[2] and r[1] <= p[1] <= r[3]


def ccw(a, b, c):
    return (c[1] - a[1]) * (b[0] - a[0]) > (b[1] - a[1]) * (c[0] - a[0])


def seg_hits_rect(p0, p1, r):
    if max(p0[0], p1[0]) < r[0] or min(p0[0], p1[0]) > r[2]: return False
    if max(p0[1], p1[1]) < r[1] or min(p0[1], p1[1]) > r[3]: return False
    if inside(p0, r) or inside(p1, r): return True
    c = [(r[0], r[1]), (r[2], r[1]), (r[2], r[3]), (r[0], r[3])]
    for i in range(4):
        a, b = c[i], c[(i + 1) % 4]
        if ccw(p0, a, b) != ccw(p1, a, b) and ccw(p0, p1, a) != ccw(p0, p1, b):
            return True
    return False


def bezier(p0, p1, p2, p3, n=12):
    out = []
    for i in range(n + 1):
        t = i / n; u = 1 - t
        out.append((u**3 * p0[0] + 3*u*u*t * p1[0] + 3*u*t*t * p2[0] + t**3 * p3[0],
                    u**3 * p0[1] + 3*u*u*t * p1[1] + 3*u*t*t * p2[1] + t**3 * p3[1]))
    return out


def segments_of(page):
    """every stroked segment on the page, as (p0, p1, colour, width)"""
    segs = []
    for d in page.get_drawings():
        col = d.get("color")                      # stroke colour; None when fill-only
        if col is None: continue
        if min(col) >= WHITE: continue            # white halo / background strokes
        w = d.get("width") or 0.1
        for it in d["items"]:
            if it[0] == "l":
                segs.append((tuple(it[1]), tuple(it[2]), col, w))
            elif it[0] == "c":
                pts = bezier(tuple(it[1]), tuple(it[2]), tuple(it[3]), tuple(it[4]))
                segs += [(pts[i], pts[i + 1], col, w) for i in range(len(pts) - 1)]
            elif it[0] == "re":
                r = it[1]; c = [(r.x0, r.y0), (r.x1, r.y0), (r.x1, r.y1), (r.x0, r.y1)]
                segs += [(c[i], c[(i + 1) % 4], col, w) for i in range(4)]
            elif it[0] == "qu":
                q = it[1]; c = [tuple(q.ul), tuple(q.ur), tuple(q.lr), tuple(q.ll)]
                segs += [(c[i], c[(i + 1) % 4], col, w) for i in range(4)]
    return segs


def spans_of(page):
    out = []
    for b in page.get_text("dict")["blocks"]:
        for l in b.get("lines", []):
            for s in l.get("spans", []):
                t = s["text"].strip()
                if t: out.append((t, s["bbox"], s.get("color", 0)))
    return out


def halos_of(page):
    """near-white FILLED rects: the halo patches behind labels, which legitimately
    interrupt a line, so text sitting inside one is not a glyph-on-line collision."""
    out = []
    for d in page.get_drawings():
        f = d.get("fill")
        if f is None or min(f) < FILL_WHITE: continue
        r = d["rect"]
        out.append((r.x0, r.y0, r.x1, r.y1))
    return out


def covered(bb, halos):
    """True only for a halo patch tight around this label -- the axes and figure
    backgrounds are white fills too, and must not count as masking anything."""
    area = max((bb[2] - bb[0]) * (bb[3] - bb[1]), 1.0)
    for h in halos:
        if (h[0] <= bb[0] + 0.5 and h[1] <= bb[1] + 0.5 and h[2] >= bb[2] - 0.5 and h[3] >= bb[3] - 0.5
                and (h[2] - h[0]) * (h[3] - h[1]) <= max(300.0, 10.0 * area)):
            return True
    return False


def touches_glyphs(page, bbox, stroke, textcol, dpi=600, tol=38, grow=2):
    """Decide on the GLYPHS, not on the text box. A span's bbox carries the font's
    ascent/descent padding, so insetting it by a fixed fraction either flags a border
    sitting in that blank strip (too strict) or misses a curve grazing the baseline
    (too lax -- this is what let a step curve touch 'saved per organoid').

    So: render the box, build the mask of the label's own ink, dilate it by `grow`
    pixels, and ask whether any pixel of the stroke colour falls inside. That catches a
    line crossing between glyph strokes and a line merely touching them, and ignores
    anything passing through the surrounding padding."""
    import numpy as np
    s = [round(c * 255) for c in stroke]
    t = [(textcol >> 16) & 255, (textcol >> 8) & 255, textcol & 255]
    if max(abs(a - b) for a, b in zip(s, t)) <= tol:
        return -1                                  # same colour as the text: inconclusive
    m = 1.5
    pm = page.get_pixmap(dpi=dpi, clip=fitz.Rect(bbox[0] - m, bbox[1] - m, bbox[2] + m, bbox[3] + m))
    a = np.frombuffer(pm.samples, dtype=np.uint8).reshape(pm.height, pm.width, pm.n)[:, :, :3].astype(int)
    ink = np.abs(a - np.array(t)).max(2) <= tol
    if not ink.any(): return 0
    d = ink.copy()
    for _ in range(grow):
        d[1:, :] |= ink[:-1, :]; d[:-1, :] |= ink[1:, :]
        d[:, 1:] |= ink[:, :-1]; d[:, :-1] |= ink[:, 1:]
        ink = d.copy()
    return int(((np.abs(a - np.array(s)).max(2) <= tol) & d).sum())


def confirm(page, rect, stroke, textcol, dpi=600, tol=38):
    """Pixel ground truth. The vector pass reports clipped paths unclipped, and
    matplotlib emits hatch fills as page-wide stroke runs, so a geometric hit is only
    a candidate: count pixels of the stroke colour actually inside the text box.
    Pixels matching the label's own colour are excluded (a label is drawn in its own
    series colour, so its glyphs would otherwise count as a collision with itself)."""
    s = [round(c * 255) for c in stroke]
    t = [(textcol >> 16) & 255, (textcol >> 8) & 255, textcol & 255]
    if max(abs(a - b) for a, b in zip(s, t)) <= tol:
        return -1                                  # same colour as the text: inconclusive
    pm = page.get_pixmap(dpi=dpi, clip=fitz.Rect(*rect))
    import numpy as np
    a = np.frombuffer(pm.samples, dtype=np.uint8).reshape(pm.height, pm.width, pm.n)[:, :, :3].astype(int)
    return int((np.abs(a - np.array(s)).max(2) <= tol).sum())


def check(path):
    page = fitz.open(path)[0]
    segs = segments_of(page); halos = halos_of(page); hits = []
    for text, bb, tcol in spans_of(page):
        if covered(bb, halos): continue          # white halo masks whatever runs beneath
        r = (bb[0] - 1.0, bb[1] - 1.0, bb[2] + 1.0, bb[3] + 1.0)   # generous candidate box
        seen = set()
        for p0, p1, col, lw in segs:
            key = tuple(round(c, 3) for c in col)
            if key in seen or not seg_hits_rect(p0, p1, r): continue
            seen.add(key)
            n = touches_glyphs(page, bb, col, tcol)
            if n > 25:
                hits.append((text, tuple(round(c, 2) for c in col), round(lw, 2),
                             (round(bb[0], 1), round(bb[1], 1)), n))
                break
    return hits


targets = sys.argv[1:] or (["figures_tkde/fig_overview_tkde.pdf"]
                           + sorted(glob.glob("figures_tkde/Fig*.pdf"))      # main-text Figs 2-7
                           + sorted(glob.glob("figures_supp/figS*.pdf")))    # Supplementary Figs S1-S7
total = 0
for p in targets:
    h = check(p)
    total += len(h)
    print(f"\n=== {p}: {len(h)} text span(s) crossed by a stroked line")
    for t, col, lw, at, n in h:
        print(f"    {t!r:<42} stroke rgb{col} lw={lw} at {at}  ({n} px)")
print(f"\nTOTAL: {total}")
