"""Redraw Figure 4 (regional change in P95 days; report Figure 1) from saved Task 1 results.

    python redraw_fig4.py [path/to/task1_AR6_regional_stats.csv]

Default input: OUT_DIR/task1_AR6_regional_stats.csv. Writes OUT_DIR/figures/fig4_regional_delta_p95_days.png/.pdf
(same plotting code as Task1_FWI_P95_Extremes.ipynb).
"""
import os
import sys
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import pyrosai as P

OUT_DIR = "./pyrosai_output"
if os.path.exists("local_config.py"):
    cfg = {}
    exec(open("local_config.py").read(), cfg)
    OUT_DIR = cfg.get("OUT_DIR", OUT_DIR)
csv = sys.argv[1] if len(sys.argv) > 1 else f"{OUT_DIR}/task1_AR6_regional_stats.csv"
FIG_DIR = f"{OUT_DIR}/figures"
os.makedirs(FIG_DIR, exist_ok=True)
reg_df = pd.read_csv(csv)
models = [m for m in P.MODEL_MARKERS if m in set(reg_df.model)] + sorted(set(reg_df.model) - set(P.MODEL_MARKERS))
P.regional_bars(reg_df, "days_p95", ["d_ssp245_minus_target", "d_SAI_minus_ssp245", "d_HiLLA_minus_ssp245"],
                "Δ days/yr with FWI > P95", f"{FIG_DIR}/fig4_regional_delta_p95_days", models, P.FOCUS_REGIONS, 300)
print(f"wrote {FIG_DIR}/fig4_regional_delta_p95_days.png and .pdf (models: {', '.join(models)})")
