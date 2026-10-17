# -*- coding: utf-8 -*-
"""IEEE-style panel labels: "(a)", "(b)", ... centred BELOW each panel (Stage 3Q-4).

IEEE Transactions set the part label of a multi-part figure under the part, like a subfigure
caption. Our figures are composites drawn at final size, so the labels are drawn inside the figure:
  register(ax, "a")   -- ax is the main axes of panel (a);
  place(fig)          -- call once before saving.
place() groups each labelled axes with the unlabelled axes stacked under it in the same column
(e.g. a histogram under a reliability diagram) and with unlabelled axes beside it in the same row
(e.g. several small maps that form one panel; colourbars excluded), puts the label under the lowest
lettering of that group, aligns labels that end up at about the same height, and, if a label would
fall below the canvas, makes the figure taller at the bottom (axes keep their size in inches).
"""
import matplotlib.pyplot as plt

_REG = []
FS = 9.0          # label size (pt): the figure lettering is 8 pt
PAD = 2.0         # gap between the panel's lowest lettering and the label (pt)
SNAP = 0.12       # labels within this many inches of each other are put on one line


def register(ax, s):
    _REG.append((ax.figure, ax, s))


def _xov(a, b):
    lo, hi = max(a.x0, b.x0), min(a.x1, b.x1)
    return max(0.0, hi - lo) / max(1e-9, min(a.width, b.width))


def _cover(a, b):
    """share of a's width that b spans."""
    lo, hi = max(a.x0, b.x0), min(a.x1, b.x1)
    return max(0.0, hi - lo) / max(1e-9, a.width)


def _yov(a, b):
    lo, hi = max(a.y0, b.y0), min(a.y1, b.y1)
    return max(0.0, hi - lo) / max(1e-9, min(a.height, b.height))


def _groups(fig, regs):
    lab = [ax for ax, _ in regs]
    out = []
    for A, s in regs:
        pa = A.get_position(); g = [A]
        for B in fig.axes:
            if B in lab or not B.get_visible(): continue
            pb = B.get_position()
            below = _cover(pa, pb) > 0.5 and pb.y1 <= pa.y0 + 1e-6 and not any(
                L is not A and _xov(L.get_position(), pb) > 0.5 and pb.y1 <= L.get_position().y0 + 1e-6
                and L.get_position().y1 <= pa.y0 + 1e-6 for L in lab)
            beside = _yov(pa, pb) > 0.5 and pb.width > 0.06 and pb.x0 >= pa.x1 - 1e-6 and not any(
                L is not A and _yov(L.get_position(), pb) > 0.5 and L.get_position().x0 >= pa.x1 - 1e-6
                and L.get_position().x1 <= pb.x0 + 1e-6 for L in lab)
            if below or beside: g.append(B)
        out.append((A, s, g))
    return out


def _layout(fig, groups):
    """label top (inches from the bottom) and centre (figure fraction) for every group."""
    fig.canvas.draw(); r = fig.canvas.get_renderer(); dpi = fig.dpi
    res = []
    for A, s, g in groups:
        bottom = min(a.get_tightbbox(r).y0 for a in g) / dpi            # inches
        x0 = min(a.get_position().x0 for a in g); x1 = max(a.get_position().x1 for a in g)
        top_in = max(a.get_position().y1 for a in g) * fig.get_size_inches()[1]
        for t in fig.texts:                                               # shared labels drawn with fig.text
            e = t.get_window_extent(r)
            fx0, fx1 = e.x0 / (fig.get_size_inches()[0] * dpi), e.x1 / (fig.get_size_inches()[0] * dpi)
            if min(fx1, x1) - max(fx0, x0) > 0 and e.y1 / dpi < top_in:
                bottom = min(bottom, e.y0 / dpi)
        res.append([A, s, (x0 + x1) / 2, bottom - PAD / 72.0])
    res.sort(key=lambda t: t[3])                                       # snap nearby label heights
    for i in range(len(res)):
        for j in range(i):
            if abs(res[i][3] - res[j][3]) < SNAP: res[i][3] = res[j][3]
    return res


def place(fig):
    regs = [(ax, s) for f, ax, s in _REG if f is fig]
    if not regs: return
    groups = _groups(fig, regs)
    lay = _layout(fig, groups)
    W, H = fig.get_size_inches()
    need = max(FS * 1.25 / 72.0 - top for *_, top in lay) + 1.0 / 72.0  # label height below its top
    if need > 0:                                                        # grow the canvas at the bottom
        for a in fig.axes:
            p = a.get_position()
            a.set_position([p.x0, (p.y0 * H + need) / (H + need), p.width, p.height * H / (H + need)])
        for t in fig.texts:
            x, y = t.get_position(); t.set_position((x, (y * H + need) / (H + need)))
        fig.set_size_inches(W, H + need)
        lay = _layout(fig, groups)
    H2 = fig.get_size_inches()[1]
    for A, s, xc, top in lay:
        fig.text(xc, top / H2, f"({s})", ha="center", va="top", fontsize=FS)
    for i in range(len(_REG) - 1, -1, -1):
        if _REG[i][0] is fig: _REG.pop(i)
