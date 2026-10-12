# Pyro SAI: fire weather under stratospheric aerosol injection

Analysis code for the Pyro SAI project, which asks how stratospheric aerosol injection (SAI) would change
wildfire risk by altering fire-weather drivers (temperature, humidity, wind and rainfall), and how two
GeoMIP G6-1.5K strategies differ: subtropical (G6-1.5K-SAI) and high-latitude (G6-1.5K-HiLLA) injection,
both designed to hold global warming at 1.5 K under SSP2-4.5.

This release covers the fire-weather analysis (daily Fire Weather Index and vapor pressure deficit) for the
mid-project report. Fire-impact simulations with the pyrE fire model will follow in a later release.

## Data

Daily output of the GeoMIP G6-1.5K experiments hosted on the Reflective Cloud Hub:

| Model | Experiments | Members |
|---|---|---|
| CESM2-WACCM | G6-1.5K-SAI, G6-1.5K-HiLLA (Hub); SSP2-4.5 (CMIP6 archive on AWS, `s3://cmip6-pds`) | 3 per experiment |
| MIROC-ES2H | G6-1.5K-SAI, G6-1.5K-HiLLA, SSP2-4.5 | 3 per experiment |
| UKESM1-1 | G6-1.5K-SAI, G6-1.5K-HiLLA, SSP2-4.5 | 3 (2 for SSP2-4.5) |

Periods: 1.5 K target climate = SSP2-4.5 2020–2039; assessment period = 2064–2083.
Known data issues and how they are handled are documented in `pyrosai.py` (`apply_known_issues`) and in
the discovery report (`results/discovery/`).

## Methods (summary)

- **FWI**: Canadian Forest Fire Weather Index System (Van Wagner 1987) as implemented in
  [xclim](https://xclim.readthedocs.io) (`cffwis_indices`), from daily temperature (daily maximum where
  available, otherwise daily mean, consistently within each model), daily-mean relative humidity, 10 m wind
  speed and daily precipitation; one spin-up year discarded.
- **Extremes**: days per year above the local 95th percentile (P95) of daily FWI in the target climate,
  pooled over the SSP2-4.5 members of each model.
- **VPD**: Sonntag (1990) saturation vapor pressure; change decomposed into temperature and humidity terms.
- **Risk-offset classes**: signs of the fire-season changes in VPD, precipitation and FWI.
- **Statistics**: Welch t-test with Benjamini–Hochberg false-discovery-rate control (α = 0.1);
  multi-model robustness = at least 2/3 of models agree on the sign and at least 1/2 are significant;
  IPCC AR6 land regions; multi-model means on a common 1° grid.

## Repository contents

| Path | Contents |
|---|---|
| `pyrosai.py` | shared library: data discovery and loading, data-quality checks, FWI, VPD, statistics, plotting |
| `00_Data_Discovery.ipynb` | inventories the Hub buckets and writes `pyrosai_catalog.json` |
| `Task1_FWI_P95_Extremes.ipynb` | daily FWI, P95 extremes, seasonality, significance, AR6 regional statistics |
| `Task2_VPD_Evaporative_Demand.ipynb` | VPD, P95-VPD days, temperature/humidity decomposition, risk-offset classes |
| `Check_CESM_tasmax.ipynb` | check of CESM2-WACCM daily maximum temperature |
| `pyrosai_catalog.json`, `pyrosai_catalog_extra.json` | file catalog produced by discovery, plus hand-added entries |
| `redraw_fig4.py`, `redraw_fig10.py` | redraw the two report figures from saved results |
| `run_headless.sh`, `summarize_run.py` | run notebooks in the background and summarise their outputs |
| `tests/` | end-to-end test on a synthetic archive (no cloud access needed) |
| `results/` | executed notebooks, regional statistics (CSV) and figures (PNG, 300 dpi, and PDF) |
| `results/report_figures/` | Figures 1 and 2 of the mid-project report |

## Running

Requirements (versions used): Python 3.12, numpy 2.0, pandas 2.2, xarray 2025.1, dask 2025.1,
xclim 0.54, regionmask 0.13, cartopy, matplotlib, s3fs/fsspec, zarr 2.18.

```bash
cp local_config_example.py local_config.py      # edit WORK_ROOT, OUT_DIR, models, members
nohup bash run_headless.sh Task1_FWI_P95_Extremes Task2_VPD_Evaporative_Demand > runs.log 2>&1 &
```

Each run writes `runs/<date>_<notebook>/` with the executed notebook, a text summary, figures and regional
CSVs. Intermediate data (`*.zarr`, `*.nc`) go to `WORK_ROOT` and `OUT_DIR` and are not tracked.
To test the pipeline without cloud access: `bash tests/run_tests.sh`.

## Citation

See `CITATION.cff`. A Zenodo DOI will be added with the first archived release.

## Acknowledgements

Funded by Reflective. Data access through the Reflective Cloud Hub; we thank the GeoMIP modelling groups
and the Reflective team.

Code development was assisted by an AI coding assistant (Claude, Anthropic). The author designed the
analysis and reviewed and validated all methods and results.

## License

BSD 3-Clause (see `LICENSE`).
