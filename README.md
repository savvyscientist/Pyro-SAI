# Pyro SAI: Tasks 1 & 2 pipeline (FWI P95 extremes, VPD)

| # | notebook | what it does |
|---|---|---|
| 0 | `00_Data_Discovery.ipynb` | lists the Hub buckets, finds daily files, writes `pyrosai_catalog.json` |
| 1 | `Task1_FWI_P95_Extremes.ipynb` | daily FWI, P95 extremes, seasonality, significance, AR6 stats, driver attribution |
| 2 | `Task2_VPD_Evaporative_Demand.ipynb` | VPD, P95-VPD days, T/RH decomposition, risk-offset zones |
| – | `Regrid_E3SM_daily.ipynb` | (E3SM only) ne30pg2 to 1° regridding |

`pyrosai.py` holds all shared code (data-quality safeguards included).

## Your settings: `local_config.py`
Copy `local_config_example.py` to `local_config.py` and edit it (dry run vs full run, `WORK_ROOT`, models...).
Task 1/2 apply it after their CONFIG cell, so updating the notebooks never resets your settings. Commit it.

## Running and sharing results
```bash
nohup bash run_headless.sh Task1_FWI_P95_Extremes Task2_VPD_Evaporative_Demand > runs.log 2>&1 &
tail -f runs.log                       # progress; safe to close the browser
git add runs/ local_config.py && git commit -m "run outputs" && git push
```
Each run writes `runs/<date>_<notebook>/` with `summary.txt` (all outputs and errors), `log.txt`,
the executed notebook, figures and regional CSVs. Large outputs (`*.zarr`, `*.nc`, work folders) are git-ignored.

## Extra data: UKESM daily tasmax
`python prep_ukesm_tasmax.py` adds UKESM1-1 daily tasmax from A. Duffey's `daily_Tmaxmin` files
(identifies the max variable, matches each file to its ensemble member, copies tasmax to your
persistent bucket, writes `pyrosai_catalog_extra.json`, which `load_catalog` merges in). Run it when
no other heavy job is running (each file is ~6 GB, read whole). `--check` only reports.

## Tests
`bash tests/run_tests.sh` runs the whole chain on a synthetic archive (no cloud access needed).
