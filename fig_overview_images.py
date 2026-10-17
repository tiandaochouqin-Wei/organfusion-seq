r"""Organoid image tiles for Fig. 1 (Stage 3AD).

Crops phase images of single organoids from one HSLCI field of the Tebon et al. cohort (BioImage Archive
S-BIAD616, CC0; MCF-7, 0.1 uM lapatinib, well B4, position 1) and writes the tiles that fig_overview_tkde.tex
places in the diagram:
  figures_tkde/fig1_org_s06.png, fig1_org_s72.png  organoid that loses mass   (x(72)/x(6) = 0.36, sensitive)
  figures_tkde/fig1_org_r06.png, fig1_org_r72.png  organoid that gains mass   (x(72)/x(6) = 1.26, resistant)
  figures_tkde/fig1_org_new.png                    a third organoid at 6 h (the 'new unit' of the online lane)
Phase is background-corrected (coarse median background); mass is the integrated corrected phase in a disc
around the organoid. Stack frame 0 is experiment time 4.28 h (the source tracks run 6.0-72.04 h and the
stack spans 67.76 h), so 6 h is frame 10 and 72 h the last frame. All tiles share one window size and one
colour scale, so sizes and intensities are comparable.

Data: download PhaseImageStack_2021_07_24_MCF-7_B4_Pos1.mat (357 MB) from
  https://ftp.ebi.ac.uk/biostudies/fire/S-BIAD/616/S-BIAD616/Files/
  2021_07_24_MCF7_print_stauro_lapatinib_neratinib_tx_test_BW_processed_stacks/B4_Pos1/
into <data dir>/S-BIAD616_samples/B4_Pos1/, then:  python -I fig_overview_images.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import DATA_DIR  # $ORGANFUSION_DATA, or ./datasets (see README)
import os
import numpy as np, scipy.io as sio
from scipy import ndimage as ndi
from datetime import datetime
from PIL import Image
import matplotlib.cm as cm

HERE = os.path.dirname(os.path.abspath(__file__))
STACK = os.path.join(DATA_DIR, "S-BIAD616_samples", "B4_Pos1",
                     "PhaseImageStack_2021_07_24_MCF-7_B4_Pos1.mat")
OUT = os.path.join(HERE, "figures_tkde")
OFFSET_H = 4.28
HALF = 32                                    # tile = 64 x 64 px of the 514 x 514 field
UP = 4                                       # upsampled for print (bicubic), 256 px per tile
ORGANOIDS = {"s": (426, 355), "r": (102, 196), "new": (420, 285)}   # (x, y) centres at 6 h

m = sio.loadmat(STACK); S = m["phaseImageStack"].astype(np.float64)
ts = [str(x).strip() for x in m["acquisitionTimes"]]
t0 = datetime.strptime(ts[0], "%d-%b-%Y %H:%M:%S")
hrs = np.array([(datetime.strptime(s, "%d-%b-%Y %H:%M:%S") - t0).total_seconds() / 3600 for s in ts]) + OFFSET_H
k6, k72 = int(np.argmin(np.abs(hrs - 6))), int(np.argmin(np.abs(hrs - 72)))


def corrected(k):
    f = S[:, :, k]
    sm = ndi.median_filter(f[::4, ::4], size=21)
    bg = ndi.zoom(sm, (f.shape[0] / sm.shape[0], f.shape[1] / sm.shape[1]), order=1)
    return bg - f                            # organoids lie below the coarse background in these stacks


s6, s72 = corrected(k6), corrected(k72)
vmax = np.percentile(s6, 99.8)


def mass(s, cx, cy, R):
    yy, xx = np.ogrid[:s.shape[0], :s.shape[1]]
    w = np.where((yy - cy) ** 2 + (xx - cx) ** 2 <= (R + 8) ** 2, s.clip(0), 0)
    cy2, cx2 = ndi.center_of_mass(w)
    return s[(yy - cy2) ** 2 + (xx - cx2) ** 2 <= R ** 2].clip(0).sum()


def tile(s, cx, cy, name):
    c = s[cy - HALF:cy + HALF, cx - HALF:cx + HALF].clip(0, vmax) / vmax
    rgb = (cm.magma(c)[:, :, :3] * 255).astype(np.uint8)
    Image.fromarray(rgb).resize((2 * HALF * UP, 2 * HALF * UP), Image.BICUBIC).save(os.path.join(OUT, name))


for key, (cx, cy) in ORGANOIDS.items():
    if key == "new":
        tile(s6, cx, cy, "fig1_org_new.png"); continue
    r = mass(s72, cx, cy, 34) / mass(s6, cx, cy, 34)
    print(f"organoid {key}: centre ({cx},{cy})  x(72)/x(6) = {r:.2f}")
    tile(s6, cx, cy, f"fig1_org_{key}06.png"); tile(s72, cx, cy, f"fig1_org_{key}72.png")
print(f"frames {k6} ({hrs[k6]:.2f} h) and {k72} ({hrs[k72]:.2f} h); tiles written to {OUT}")
