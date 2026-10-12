"""Redraw Figure 10 (risk-offset zones) from the saved Task 2 results, without rerunning Task 2.

    python redraw_fig10.py

Reads OUT_DIR/task2_multimodel_1deg.nc and writes OUT_DIR/figures/fig10_risk_offset_zones.png/.pdf
(same plotting code as Task2_VPD_Evaporative_Demand.ipynb).
"""
import os
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pyrosai as P

OUT_DIR = "./pyrosai_output"
if os.path.exists("local_config.py"):
    cfg = {}
    exec(open("local_config.py").read(), cfg)
    OUT_DIR = cfg.get("OUT_DIR", OUT_DIR)
FIG_DIR = f"{OUT_DIR}/figures"
os.makedirs(FIG_DIR, exist_ok=True)
FIG_DPI = 300
G6 = ["G6-1.5K-SAI", "G6-1.5K-HiLLA"]
mm = xr.open_dataset(f"{OUT_DIR}/task2_multimodel_1deg.nc").load()
models = [m.strip() for m in mm.attrs.get("models", "").split(",") if m.strip()]
mm_land = P.land_mask(mm.lon, mm.lat)

import cartopy.crs as ccrs
fracs = {P.SCEN_LABEL[s]: P.area_fractions(mm[f"offset_class_{s}"], mm_land) for s in G6}
fr = pd.DataFrame(fracs)
shown = [c for c in fr.index if (fr.loc[c] >= 0.0005).any()]     # classes with no land area are left out
fr_s = fr.loc[shown]
# layout: maps stacked in the left column, class legend directly under them, bars on the right
fig = plt.figure(figsize=(11, 7.4))
gs = fig.add_gridspec(3, 2, width_ratios=[1.55, 1], height_ratios=[1, 1, 0.30], wspace=0.08, hspace=0.12,
                      left=0.02, right=0.97, top=0.95, bottom=0.03)
for i, s in enumerate(G6):
    ax = fig.add_subplot(gs[i, 0], projection=ccrs.Robinson())
    P.class_map_panel(ax, mm[f"offset_class_{s}"], f"({'ab'[i]}) {P.SCEN_LABEL[s]} − SSP2-4.5 ({'MMM' if len(models) > 1 else models[0]})", mm_land)
axl = fig.add_subplot(gs[2, 0]); axl.axis("off")
handles = [mpatches.Patch(facecolor=P.OFFSET_CLASSES[k][1], hatch="////" if k == 3 else None, edgecolor="white",
                          label=f"{k}: {P.OFFSET_CLASSES[k][0]}") for k in shown]
axl.legend(handles=handles, loc="upper center", ncol=1, fontsize=8.5, frameon=False, handlelength=2.2)
axb = fig.add_subplot(gs[0:2, 1])
y = np.arange(len(fr_s))
for k, col in enumerate(fr_s.columns):
    yy = y + (k - 0.5) * 0.4
    axb.barh(yy, fr_s[col] * 100, height=0.38, color=[P.OFFSET_CLASSES[c][1] for c in fr_s.index],
             hatch=None if k == 0 else "..", edgecolor="white", label=col)
    for yv, v in zip(yy, fr_s[col] * 100):
        axb.text(v + 1, yv, f"{v:.0f}%", va="center", fontsize=8, color="#333333")
axb.set_yticks(y, [f"class {c}" for c in fr_s.index]); axb.invert_yaxis()
axb.set_xlim(0, max(10, float(fr_s.max().max()) * 100 * 1.18))
axb.set_xlabel("% of land area"); P.style_axes(axb)
axb.set_title("(c) Land area per class", fontsize=10)
axb.legend(fontsize=8, frameon=False, loc="lower right")
P.save_figure(fig, f"{FIG_DIR}/fig10_risk_offset_zones", FIG_DPI); 
print((fr * 100).round(1).rename(index={k: v[0] for k, v in P.OFFSET_CLASSES.items()}))
print(f"wrote {FIG_DIR}/fig10_risk_offset_zones.png and .pdf")
