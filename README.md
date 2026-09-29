# Pyro SAI: Tasks 1 & 2 pipeline (FWI P95 extremes, VPD)

Put everything in one folder on the Reflective Hub and run the notebooks in this order:

| # | notebook | what it does | run time (full CESM, 3 members) |
|---|---|---|---|
| 0 | `00_Data_Discovery.ipynb` | lists the S3 buckets, finds **daily** files, writes `pyrosai_catalog.json`, a report and a small sample | minutes |
| – | `Regrid_E3SM_daily.ipynb` | *(only for E3SM)* ne30pg2 -> 1° regridding of daily variables, output to Zarr | long, one-off |
| 1 | `Task1_FWI_P95_Extremes.ipynb` | daily FWI, P95 thresholds, extreme days/spells/seasonality, significance, AR6 stats, driver attribution | hours (cached, resumable) |
| 2 | `Task2_VPD_Evaporative_Demand.ipynb` | VPD, P95-VPD days, T/RH decomposition, risk-offset zones | < 1 h (reuses Task 1 inputs) |

`pyrosai.py` holds all shared code. The notebooks contain configuration, calls and figures only.

**Before running Task 1:** set `WORK_ROOT` to `s3://reflective-persistent-prod/<username>/pyrosai` in both task notebooks (or a local folder for a quick test with `BBOX`). Then review `pyrosai_catalog.json`, or at least the inventory table.

**Environment:** do **not** run the old `pip install "zarr<3" "dask<2025.1"` line. It broke `distributed` (see the old notebook's pip errors) and would break the Icechunk/Earthmover stores. The current Hub image is fine. Note the image tag you use, for reproducibility.

**Testing without cloud data:** `bash tests/run_tests.sh` builds a synthetic archive with CESM-style and UKESM-style naming, units and calendars, then runs all three notebooks end to end.
