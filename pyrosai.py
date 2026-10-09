"""
pyrosai.py - shared utilities for Pyro SAI, Tasks 1 (FWI / P95 extremes) and 2 (VPD).

Designed for the Reflective Cloud Hub (30 GB / 4 CPU default profile):
  * catalog-driven loading of daily data (S3 NetCDF, Zarr, Earthmover/Arraylake)
  * harmonisation of model-specific names, units, coordinates and time stamps
    (CESM2-WACCM / E3SM CAM-style names, CMOR names, UKESM oddities, MIROC)
  * optional "staging" of daily inputs to time-contiguous Zarr (read once, reuse
    for FWI, VPD and later pyrE runs)
  * FWI via xclim (spatially chunked, time contiguous -> memory bounded)
  * VPD (Sonntag 1990 saturation vapour pressure, same as xclim default)
  * P95 thresholds, exceedance days, spell length, seasonal cycle
  * Welch t-test + Benjamini-Hochberg FDR, multi-model agreement
  * AR6 regional (land-only, area-weighted) statistics
  * delta-method driver attribution (T / RH / wind / precip) for FWI and VPD
  * map / bar / seasonal-cycle plotting helpers

Author: Pyro SAI project (K. Mezuman), 2026.
"""
from __future__ import annotations

import json
import os
import re
import warnings
from collections import OrderedDict

import numpy as np
import pandas as pd
import xarray as xr

# -----------------------------------------------------------------------------
# Project-wide defaults (override in notebook CONFIG cells)
# -----------------------------------------------------------------------------
TARGET_PERIOD = (2020, 2039)   # SSP2-4.5 period that defines the G6-1.5K temperature target
ASSESS_PERIOD = (2064, 2083)   # assessment window (2083 not 2084: UKESM is missing Dec 2084)
SPINUP_YEARS = 1               # FWI moisture-code spin-up year, computed then discarded

SCENARIOS = ["ssp245", "G6-1.5K-SAI", "G6-1.5K-HiLLA"]
SCEN_LABEL = {
    "target": "SSP2-4.5 2020-2039 (1.5K target)",
    "ssp245": "SSP2-4.5",
    "G6-1.5K-SAI": "G6-1.5K-SAI",
    "G6-1.5K-HiLLA": "G6-1.5K-HiLLA",
}
# Reference series are deliberately neutral (black/gray); experiments get hues.
SCEN_STYLE = {
    "target": dict(color="#222222", ls="--", marker=None),
    "ssp245": dict(color="#8c8c8c", ls="-", marker="o"),
    "G6-1.5K-SAI": dict(color="#7b3294", ls="-", marker="s"),
    "G6-1.5K-HiLLA": dict(color="#1b9e77", ls="-", marker="^"),
}

# 14 fire-relevant AR6 land regions (edit to match the list in the proposal/atlas)
FOCUS_REGIONS = ["NWN", "NEN", "WSB", "ESB", "WNA", "MED", "NSA", "SAM",
                 "NES", "WAF", "ESAF", "SEA", "NAU", "EAU"]

# Canonical variable -> list of names used by different models / archives
ALIASES = OrderedDict(
    tasmax=["tasmax", "TREFHTMX", "TREFMXAV", "TMAX", "tmax", "t2max"],
    tas=["tas", "TREFHT", "T2", "t2m", "SurfT"],
    hursmin=["hursmin", "RHREFHTMN", "RHREFHTMIN", "hurs_min"],
    hurs=["hurs", "RHREFHT", "rh", "RH2M", "relhum"],
    sfcWind=["sfcWind", "U10", "WSPDSRFAV", "wind10", "si10"],
    pr=["pr", "PRECT", "precip", "PRECTOT"],
    huss=["huss", "QREFHT", "q2m"],
    ps=["ps", "PS", "sp"],
    uas=["uas", "UAS"],
    vas=["vas", "VAS"],
)
TARGET_UNITS = {"tasmax": "K", "tas": "K", "hurs": "%", "hursmin": "%", "sfcWind": "m s-1",
                "pr": "kg m-2 s-1", "huss": "1", "ps": "Pa", "uas": "m s-1", "vas": "m s-1"}

# Monthly climatological ratio mean(sfcWind)/|mean-vector wind| per model, applied when
# daily wind has to be derived from daily-mean uas/vas (which underestimates mean speed).
# Filled by prepare_wind_corrections().
WIND_UV_RATIO: dict = {}

# Custom openers (e.g. paste the Earthmover example code into a function and
# register it:  pyrosai.CUSTOM_OPENERS["earthmover"] = my_open_fn ).
CUSTOM_OPENERS: dict = {}


warnings.filterwarnings("ignore", message=".*Compilation requested for previously compiled.*")
warnings.filterwarnings("ignore", message=".*Consolidated metadata is currently not part.*")
warnings.filterwarnings("ignore", message=".*specified chunks separate the stored chunks.*")
warnings.filterwarnings("ignore", message=".*invalid value encountered in divide.*")


def log(msg):
    """Print, and (headless runs) append a time-stamped copy to $PYROSAI_PROGRESS, so progress
    is visible while nbconvert holds back the notebook outputs until the end."""
    print(msg, flush=True)
    f = os.environ.get("PYROSAI_PROGRESS")
    if f:
        try:
            import datetime
            with open(f, "a") as fh:
                fh.write(f"{datetime.datetime.now():%Y-%m-%d %H:%M} {msg}\n")
        except OSError:
            pass


# -----------------------------------------------------------------------------
# Catalog
# -----------------------------------------------------------------------------
def load_catalog(path):
    """Catalog JSON: {model: {scenario: {member: {canonical_var: entry}}}}.

    entry = {"kind": "netcdf" | "zarr" | "arraylake" | <custom>,
             "paths": [...glob patterns or files...]   (netcdf)
             "store": "...", "group": "..."             (zarr / arraylake)
             "var": "TREFHTMX"                           (optional; else aliases searched)
             "storage_options": {...}}                   (optional, fsspec)
    """
    with open(path) as f:
        cat = json.load(f)
    # hand-added entries (e.g. files the discovery notebook cannot classify) live next to the
    # catalog as <name>_extra.json and are merged in; they never overwrite existing variables
    extra = os.path.splitext(path)[0] + "_extra.json"
    if os.path.exists(extra):
        with open(extra) as f:
            add = json.load(f)
        n = 0
        for m, scens in add.items():
            if m.startswith("_"):
                continue
            for s, mems in scens.items():
                for mem, vs in mems.items():
                    tgt = cat.setdefault(m, {}).setdefault(s, {}).setdefault(mem, {})
                    for v, e in vs.items():
                        if v not in tgt:
                            tgt[v] = e; n += 1
        log(f"[catalog] merged {n} extra entr{'y' if n == 1 else 'ies'} from {os.path.basename(extra)}")
    return cat


def catalog_inventory(catalog, needed=("tasmax", "hurs", "sfcWind", "pr", "tas")):
    """Table of which model/scenario/member has which canonical variables."""
    rows = []
    for model, scens in catalog.items():
        if model.startswith("_"):
            continue
        for scen, mems in scens.items():
            for mem, vars_ in mems.items():
                row = dict(model=model, scenario=scen, member=mem)
                for v in list(needed) + ["hursmin", "huss", "ps"]:
                    row[v] = "Y" if v in vars_ else ""
                rows.append(row)
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Discovery: classify archive paths and auto-build a catalog
# -----------------------------------------------------------------------------
KNOWN_MODELS = ["CESM2-WACCM", "UKESM1-1", "UKESM1-0-LL", "MIROC-ES2H", "E3SMv3", "E3SM",
                "MPI-ESM1-2-LR", "MPI-ESM1-2-HR", "GISS-E2-1-G", "IPSL-CM6A-LR", "CNRM-ESM2-1"]
DAY_TOKENS = {"day", "DAY", "Day", "daily", "Daily", "h1", "day_mean"}
SUBDAILY_TOKENS = {"3hr", "6hr", "1hr", "E3hr", "CF3hr", "E1hr", "6hrPlev", "6hrLev", "3hrPt",
                   "1hrPt", "hourly", "3h", "6h", "1h", "ap8"}
MODEL_ALIASES = {"E3SM": "E3SMv3", "UKESM1-1-LL": "UKESM1-1"}
# Output of non-atmosphere model components (land, ocean, sea ice, coupler). Their files can
# carry the same variable names (e.g. CLM TREFMXAV/U10, h1 = MONTHLY in CLM) and must never be
# mixed into atmospheric daily series.
NON_ATMOS_TOKENS = {"LDAY", "LMON", "ODAY", "OMON", "IDAY", "IMON", "Lmon", "Omon", "SImon", "SIday",
                    "Oday", "clm2", "pop", "cice", "elm", "mpaso", "mpassi", "cpl", "mosart", "rtm"}


def is_non_atmos(path):
    toks = set(re.split(r"[/._\-]", path))
    return bool(toks & NON_ATMOS_TOKENS)
MON_TOKENS = {"Amon", "AMON", "Mon", "mon", "h0", "monthly", "Emon", "Lmon", "LMON", "OMON", "IMON"}
_ALIAS_TO_CANON = {a: c for c, al in ALIASES.items() for a in al}


def classify_path(path):
    """Heuristically classify an archive path -> model, scenario, member, var, freq."""
    p = path.replace("\\", "/")
    segs = [s for s in p.split("/") if s]
    base = segs[-1]
    toks_base = re.split(r"[._\-]", base)
    toks_dirs = set(segs[:-1])
    all_toks = set(toks_base) | toks_dirs | set(t for s in segs[:-1] for t in re.split(r"[._\-]", s))

    model = next((s for s in segs for m in KNOWN_MODELS if s.lower() == m.lower()), None)
    if model is None:
        model = next((m for m in KNOWN_MODELS if m.lower() in p.lower()), segs[1] if len(segs) > 1 else "?")

    def scen_of(s):
        s = s.lower()
        if "baseline" in s:
            return "ssp245"
        if "hilla" in s or re.search(r"ds27[5]|ds28[67]", s):     # UKESM HiLLA suites ds275/ds286/ds287
            return "G6-1.5K-HiLLA"
        if re.search(r"di189|di86[45]", s):                     # UKESM SAI suites di189/di864/di865
            return "G6-1.5K-SAI"
        if re.search(r"g6.{0,12}(sai|sulfur)|[._\-]sai[._\-/]|g6sulfur", s):
            return "G6-1.5K-SAI"
        if re.search(r"ssp2-?4\.?5|ssp245", s):
            return "ssp245"
        return None
    # most specific segment wins (E3SM keeps 'v3.LR.ssp245_0201' baselines inside G6-1.5K-HiLLA/)
    scenario = next((sc for sc in (scen_of(sg) for sg in reversed(segs)) if sc), "?")

    member = None
    for pat, fmt in [(r"(r\d+i\d+p\d+f\d+)", "{}"), (r"/(r\d+)/", "{}"),
                     (r"\.0*(\d{1,3})\.cam\.", "r{}"), (r"\.0*(\d{1,3})\.(?:eam|elm|clm2)\.", "r{}"),
                     (r"_r0*(\d+)(?:[._-][^/_]*)*\.nc4?$", "r{}"), (r"[._](0101|0151|0201|0251|0301)[._/]", "e3sm{}")]:
        mm = re.search(pat, p)
        if mm:
            member = fmt.format(mm.group(1))
            break
    if member and member.startswith("e3sm"):
        member = {"e3sm0101": "r1", "e3sm0151": "r2", "e3sm0201": "r3",
                  "e3sm0251": "r4", "e3sm0301": "r5"}[member]
    member = member or "?"

    var = None
    for t in toks_base:            # filename tokens first
        if t in _ALIAS_TO_CANON:
            var = _ALIAS_TO_CANON[t]
            break
    if var is None:
        for s in reversed(segs[:-1]):
            if s in _ALIAS_TO_CANON:
                var = _ALIAS_TO_CANON[s]
                break
    if all_toks & SUBDAILY_TOKENS:
        freq = "subdaily"
    elif all_toks & DAY_TOKENS:
        freq = "day"
    elif all_toks & MON_TOKENS:
        freq = "mon"
    else:
        freq = "unknown"
    model = MODEL_ALIASES.get(model, model)
    yrs = re.search(r"(?<!\d)(\d{4})(?:\d{2}){1,4}[-_](\d{4})(?:\d{2}){1,4}(?!\d)", base)
    y0, y1 = (int(yrs.group(1)), int(yrs.group(2))) if yrs else (np.nan, np.nan)
    return dict(path=path, model=model, scenario=scenario, member=member, var=var,
                var_name=next((t for t in toks_base if t in _ALIAS_TO_CANON), var), freq=freq,
                y0=y0, y1=y1, component="other" if is_non_atmos(p) else "atm")


def list_archive(roots, suffixes=(".nc", ".nc4"), storage_options=None):
    """Recursively list files under roots (local or s3://). Returns DataFrame of classifications."""
    import fsspec
    rows = []
    for root in roots:
        fs, fp = fsspec.core.url_to_fs(root, **(storage_options or {}))
        proto = fs.protocol[0] if isinstance(fs.protocol, (tuple, list)) else fs.protocol
        try:
            found = fs.find(fp)
        except Exception as e:
            log(f"  could not list {root}: {e}")
            continue
        for f in found:
            if f.endswith(suffixes):
                full = f if proto in ("file", "local") else fs.unstrip_protocol(f)
                rows.append(classify_path(full))
    return pd.DataFrame(rows)


def probe_frequency(path, storage_options=None):
    """Median time step (days) of one file; to resolve 'unknown' frequency."""
    try:
        ds = harmonize_coords(open_entry({"kind": "netcdf", "paths": [path], "storage_options": storage_options}))
        if "time" not in ds.dims:
            return np.nan
        t = ds.indexes["time"]
        dt = np.median(np.diff(np.array([x.toordinal() if hasattr(x, "toordinal") else
                                         pd.Timestamp(x).toordinal() for x in t[:50]])))
        return float(dt)
    except Exception as e:
        log(f"  probe failed for {path}: {e}")
        return np.nan


def build_catalog(listing, freq="day", scenarios=SCENARIOS):
    """Catalog dict from a list_archive() DataFrame (daily files only by default)."""
    df = listing[(listing.freq == freq) & listing["var"].notnull() & listing.scenario.isin(scenarios)]
    if "component" in df:
        df = df[df.component == "atm"]
    else:
        df = df[~df.path.map(is_non_atmos)]
    cat = {}
    for (model, scen, mem, var), g in df.groupby(["model", "scenario", "member", "var"]):
        cat.setdefault(model, {}).setdefault(scen, {}).setdefault(mem, {})[var] = {
            "kind": "netcdf", "paths": sorted(g.path.tolist()), "var": g.var_name.iloc[0]}
    return cat


def save_catalog(cat, path):
    with open(path, "w") as f:
        json.dump(cat, f, indent=1)


# -----------------------------------------------------------------------------
# Opening sources
# -----------------------------------------------------------------------------
def _expand_paths(paths, storage_options=None):
    import fsspec
    out = []
    for p in paths:
        fs, fp = fsspec.core.url_to_fs(p, **(storage_options or {}))
        if any(c in p for c in "*?["):
            found = sorted(fs.glob(fp))
        else:
            found = [fp]
        proto = fs.protocol[0] if isinstance(fs.protocol, (tuple, list)) else fs.protocol
        for f in found:
            out.append(f if proto in ("file", "local") else fs.unstrip_protocol(f))
    if not out:
        raise FileNotFoundError(f"No files matched {paths}")
    return out


def _keep_vars(ds, keep):
    keep = [v for v in ds.variables if v in keep or v in ds.dims or v in ds.coords]
    return ds[[v for v in keep if v in ds.data_vars]] if any(v in ds.data_vars for v in keep) else ds


def _per_file_midpoints(ds):
    """Within ONE file: label each step with the midpoint of its time bounds and drop the
    bounds. Bounds that were left undecoded (missing 'bounds' link) are decoded with the time
    axis' own units/calendar; if that is impossible the bounds are just dropped."""
    if "time" not in ds.coords:
        return ds
    bname = ds["time"].attrs.get("bounds")
    if bname not in ds.variables:
        bname = next((b for b in ("time_bnds", "time_bounds", "time_bnd") if b in ds.variables), None)
    if bname is None:
        return ds
    b = ds[bname]
    t_is_num = np.issubdtype(ds["time"].dtype, np.number)
    try:
        if np.issubdtype(b.dtype, np.number) and not t_is_num:
            import cftime
            enc = ds["time"].encoding
            units = b.attrs.get("units") or enc.get("units")
            cal = b.attrs.get("calendar") or enc.get("calendar", "standard")
            vals = cftime.num2date(b.values, units, cal, only_use_cftime_datetimes=True)
            if np.issubdtype(ds["time"].dtype, np.datetime64):
                vals = np.array([np.datetime64(v.isoformat()) for v in vals.ravel()]).reshape(vals.shape)
            b = xr.DataArray(vals, dims=b.dims)
        if not np.issubdtype(b.dtype, np.number) and not t_is_num:
            bdim = [d for d in b.dims if d != "time"][0]
            lo, hi = b.isel({bdim: 0}).values, b.isel({bdim: 1}).values
            mid = np.array([l + (h - l) / 2 for l, h in zip(lo, hi)])
            ds = ds.assign_coords(time=("time", mid, ds["time"].attrs))
    except Exception as e:
        log(f"  [time] could not use time bounds in {str(ds.encoding.get('source', '?')).split('/')[-1]}: "
            f"{type(e).__name__}")
    return ds.drop_vars(bname)


# CF standard names of the canonical variables (tas/tasmax share one; files hold a single field)
STANDARD_NAMES = {"tasmax": "air_temperature", "tas": "air_temperature", "hurs": "relative_humidity",
                  "hursmin": "relative_humidity", "huss": "specific_humidity",
                  "pr": "precipitation_flux", "sfcWind": "wind_speed", "uas": "eastward_wind",
                  "vas": "northward_wind", "ps": "surface_air_pressure"}
for _c, _names in ALIASES.items():
    for _a in _names:
        STANDARD_NAMES.setdefault(_a, STANDARD_NAMES.get(_c))
_RENAMED_SEEN: set = set()


def _note_renamed(ds, name, var_candidates):
    key = (name, tuple(var_candidates))
    if key not in _RENAMED_SEEN:
        _RENAMED_SEEN.add(key)
        log(f"  [vars] using '{name}' (standard_name={ds[name].attrs.get('standard_name')!r}, "
            f"cell_methods={ds[name].attrs.get('cell_methods')!r}) "
            f"for {list(var_candidates)[:1]}: no variable with an expected name in "
            f"{str(ds.encoding.get('source', '?')).split('/')[-1]}")


def _open_netcdf(entry, var_candidates):
    """Open (possibly many, possibly overlapping) NetCDF files lazily.

    Remote files: h5netcdf (netCDF-4/HDF5) -> scipy (netCDF-3 classic/64-bit offset) ->
    download to a local cache and use netCDF4 (handles CDF-5, e.g. E3SM output).
    Cache dir: $PYROSAI_NC_CACHE (default <tmp>/pyrosai_nc_cache).
    """
    import fsspec
    import tempfile
    so = entry.get("storage_options") or {}
    files = _expand_paths(entry["paths"], so)
    names = set(var_candidates)
    bounds = {"time_bnds", "time_bounds", "time_bnd"}
    canon = next((c for c, names in ALIASES.items() if var_candidates and var_candidates[0] in names), None)

    def pre(ds):
        dv = [v for v in ds.data_vars if v in names]
        if not dv:
            # files named by variable but with a different internal name (e.g. UKESM output
            # saved as 'air_temperature'): use the standard_name, else the file's only field
            std = {STANDARD_NAMES.get(c) for c in var_candidates} - {None}
            dv = [v for v in ds.data_vars if ds[v].attrs.get("standard_name") in std]
            if len(dv) > 1:
                # several fields of one quantity (UKESM 'tas' files hold daily max, min and mean
                # air_temperature): pick by cell_methods
                want = {"tasmax": "maximum", "hursmin": "minimum"}.get(canon, "mean")
                pick = [v for v in dv if f"time: {want}" in str(ds[v].attrs.get("cell_methods", ""))]
                dv = pick if len(pick) == 1 else dv
            fields = [v for v in ds.data_vars if ds[v].ndim >= 3]
            if not dv and len(fields) == 1:
                dv = fields
            if dv:
                _note_renamed(ds, dv[0], var_candidates)
            else:
                log(f"  [vars] no usable variable for {list(var_candidates)[:1]} in "
                    f"{str(ds.encoding.get('source', '?')).split('/')[-1]}: {list(ds.data_vars)}")
        # time bounds are kept only alongside a data variable (an iris 'time_bnds' alone used to
        # be mistaken for the field and the real data dropped)
        ds = ds[dv + [b for b in ds.data_vars if b in bounds]] if dv else ds[[]]
        if "t" in ds.dims and "time" not in ds.dims:
            ds = ds.rename(t="time")
        ds = _per_file_midpoints(ds)
        if "time" in ds.coords and np.issubdtype(ds["time"].dtype, np.number):
            src = ds.encoding.get("source", "?")
            try:
                ds = xr.decode_cf(ds, use_cftime=True)
            except Exception:
                pass
            if np.issubdtype(ds["time"].dtype, np.number):
                log(f"  [time] skipping file with undecodable time axis: {str(src).split('/')[-1]} "
                    f"(units={ds['time'].attrs.get('units')!r})")
                ds = ds.isel(time=slice(0, 0))
        return ds

    # nested concat along time; overlaps/duplicates are removed later in normalize_time
    kw = dict(combine="nested", concat_dim="time", preprocess=pre, chunks={"time": 365},
              data_vars="minimal", coords="minimal", compat="override", join="override")
    local = all(not re.match(r"^[a-z0-9]+://", f) or f.startswith("file://") for f in files)
    if local:
        return xr.open_mfdataset(files, **kw)
    fs, _ = fsspec.core.url_to_fs(files[0], **so)
    errs = []
    for engine in ("h5netcdf", "scipy"):
        try:
            return xr.open_mfdataset([fs.open(f, "rb") for f in files], engine=engine, **kw)
        except Exception as e:
            errs.append(f"{engine}: {type(e).__name__}: {e}")
    cache = os.environ.get("PYROSAI_NC_CACHE", os.path.join(tempfile.gettempdir(), "pyrosai_nc_cache"))
    proto = files[0].split("://")[0]
    try:
        loc = [fsspec.open_local(f"simplecache::{f}", simplecache={"cache_storage": cache},
                                 **({proto: so} if so else {})) for f in files]
        return xr.open_mfdataset(loc, engine="netcdf4", **kw)
    except Exception as e:
        errs.append(f"netcdf4 (cached download): {type(e).__name__}: {e}")
    raise IOError("could not open " + files[0] + "\n  " + "\n  ".join(errs))


def _open_zarr(entry):
    so = entry.get("storage_options")
    return xr.open_zarr(entry["store"], group=entry.get("group"), storage_options=so,
                        consolidated=entry.get("consolidated", None), chunks={})


def _open_arraylake(entry):
    """Open an Earthmover/Arraylake (Icechunk) repo. Adapt to the Hub example in
    /shared/Code_example/Earthmover_zarr_GeoMIP if the API differs."""
    from arraylake import Client
    client = Client()
    repo = client.get_repo(entry["repo"])
    session = repo.readonly_session(entry.get("branch", "main"))
    return xr.open_zarr(session.store, group=entry.get("group"), consolidated=False, chunks={})


def open_entry(entry, var_candidates=()):
    kind = entry.get("kind", "netcdf")
    if kind in CUSTOM_OPENERS:
        return CUSTOM_OPENERS[kind](entry)
    if kind == "netcdf":
        return _open_netcdf(entry, var_candidates)
    if kind == "zarr":
        return _open_zarr(entry)
    if kind == "arraylake":
        return _open_arraylake(entry)
    raise ValueError(f"Unknown catalog entry kind: {kind}")


# -----------------------------------------------------------------------------
# Harmonisation
# -----------------------------------------------------------------------------
def harmonize_coords(ds):
    """lat/lon/time names, drop singleton extra dims, lat ascending, lon in [-180,180)."""
    if "t" in ds.dims and "time" in ds.dims and ds.sizes["time"] == 1:
        ds = ds.isel(time=0, drop=True)  # UKESM (Lee et al. zenodo) layout
    ren = {}
    for a, b in [("latitude", "lat"), ("longitude", "lon"), ("t", "time"),
                 ("nav_lat", "lat"), ("nav_lon", "lon"), ("valid_time", "time")]:
        if (a in ds.dims or a in ds.coords) and b not in ds.variables:
            ren[a] = b
    ds = ds.rename(ren)
    for d in list(ds.dims):
        if d not in ("time", "lat", "lon", "bnds", "nbnd", "bnd", "d2") and ds.sizes[d] == 1:
            ds = ds.isel({d: 0}, drop=True)
    if "lat" in ds.dims and ds.lat.size > 1 and float(ds.lat[0]) > float(ds.lat[-1]):
        ds = ds.sortby("lat")
    if "lon" in ds.dims and float(ds.lon.max()) > 180:
        ds = ds.assign_coords(lon=(((ds.lon + 180) % 360) - 180)).sortby("lon")
    if "lat" in ds.coords:
        ds["lat"].attrs.update(units="degrees_north", standard_name="latitude")
    if "lon" in ds.coords:
        ds["lon"].attrs.update(units="degrees_east", standard_name="longitude")
    return ds


def normalize_time(ds):
    """Label each daily value with its calendar day (00:00).

    CESM/E3SM h1 files stamp daily means at the END of the interval
    (Jan 1 mean -> 'Jan 2 00:00'); the time bounds midpoint fixes that.
    """
    bname = ds.time.attrs.get("bounds")
    if bname not in ds.variables:
        bname = next((b for b in ("time_bnds", "time_bounds", "time_bnd") if b in ds.variables), None)
    if bname is not None:
        b = ds[bname].load()
        bdim = [d for d in b.dims if d != "time"][0]
        lo, hi = b.isel({bdim: 0}).values, b.isel({bdim: 1}).values
        mid = np.array([l + (h - l) / 2 for l, h in zip(lo, hi)])
        ds = ds.assign_coords(time=("time", mid, ds.time.attrs))
        ds = ds.drop_vars(bname)
    idx = ds.indexes["time"]
    try:
        idx = idx.floor("D")
    except Exception:
        pass
    if bname is None:
        # no bounds: detect end-of-interval stamping (first stamp on Jan 2 00:00)
        t0 = idx[0]
        if t0.month == 1 and t0.day == 2 and t0.hour == 0 and len(idx) > 300:
            log("  [time] no bounds and series starts on Jan 2 -> assuming end-of-interval "
                "stamps; shifting back 1 day")
            idx = idx - pd.Timedelta(days=1) if isinstance(idx, pd.DatetimeIndex) else \
                xr.CFTimeIndex([t - pd.Timedelta(days=1).to_pytimedelta() for t in idx])
    ds = ds.assign_coords(time=idx)
    dup = ds.get_index("time").duplicated()
    if dup.any():
        log(f"  [time] dropping {int(dup.sum())} duplicated time steps (overlapping files)")
        ds = ds.isel(time=~dup)
    return ds.sortby("time")


def _sample(da, n=31):
    """First n time steps that are not all-NaN (overlapping/mixed files can leave NaN rows)."""
    v = da.isel(time=slice(0, n)).values
    if not np.isfinite(v).any() and da.sizes.get("time", 0) > n:
        v = da.isel(time=slice(n, 5 * n)).values
    return v


def to_target_units(da, canon):
    u = str(da.attrs.get("units", "")).strip()
    ul = u.lower().replace(" ", "").replace("**", "").replace("^", "").replace(".", "")
    s = _sample(da)
    finite = np.isfinite(s).any()
    med = float(np.nanmedian(s)) if finite else np.nan
    if canon in ("tas", "tasmax"):
        # decide from the VALUES (labels are sometimes wrong); K never < 150 for near-surface air
        if finite:
            celsius = med < 150
            if celsius != (ul in ("c", "degc", "celsius", "°c", "deg_c")) and ul:
                log(f"  [units] {canon}: label '{u}' but values (median {med:.1f}) say "
                    f"{'Celsius' if celsius else 'Kelvin'} - using the values")
        else:
            celsius = ul in ("c", "degc", "celsius", "°c", "deg_c")
        if celsius:
            da = da + 273.15
    elif canon in ("hurs", "hursmin"):
        # decide from the VALUES: CAM labels RHREFHT 'fraction' although it is stored in percent
        frac_label = ul in ("1", "fraction", "0-1")
        if finite:
            # median, not max: a few huge values (unmasked fill values, MIROC) made a global
            # fraction field look like percent and gave RH ~0.65 %
            sv = s[np.isfinite(s) & (np.abs(s) < 1e10)]
            md = float(np.median(sv)) if sv.size else np.nan
            is_fraction = md <= 1.5
            if is_fraction != frac_label:
                log(f"  [units] {canon}: label '{u}' but values (median {md:.2f}) are in "
                    f"{'fraction' if is_fraction else 'percent'} - using the values")
        else:
            is_fraction = frac_label
        da = da.where(np.abs(da) < 1e10)      # unmasked fill values -> missing
        if is_fraction:
            da = da * 100.0
        da = da.clip(0, 100)
    elif canon == "pr":
        if ul in ("m/s", "ms-1", "m/sec"):
            da = da * 1000.0
        elif ul in ("mm/day", "mmd-1", "mm/d", "mmday-1", "kgm-2d-1"):
            da = da / 86400.0
        elif ul in ("kgm-2s-1", "kg/m2/s", "kg/m2s", "kgm-2/s", "mm/s", "mms-1"):
            pass
        else:
            warnings.warn(f"pr units '{u}' not recognised; assuming kg m-2 s-1")
        da = da.clip(min=0)
        m = float(np.nanmean(_sample(da)))
        if m > 1e-2:
            warnings.warn(f"pr mean {m:.3g} kg m-2 s-1 looks too large - check units ('{u}')")
    elif canon in ("sfcWind", "uas", "vas"):
        if ul and not any(k in ul for k in ("m/s", "ms-1", "m/sec", "km/h", "kmh-1", "km/hr", "knot")):
            raise ValueError(f"'{canon}' has units '{u}' - not a wind speed; check the file "
                             f"(add it to KNOWN_BAD_VARIABLES)")
        if ul in ("km/h", "kmh-1", "km/hr"):
            da = da / 3.6
        if canon == "sfcWind":
            da = da.clip(min=0)
    elif canon == "huss":
        if ul in ("g/kg", "gkg-1"):
            da = da / 1000.0
    elif canon == "ps":
        if ul in ("hpa", "mb", "mbar") or med < 2000:
            da = da * 100.0
    da.attrs["units"] = TARGET_UNITS[canon]
    return da


def _find_var(ds, canon, explicit=None):
    if explicit and explicit in ds.data_vars:
        return explicit
    for a in ALIASES[canon]:
        if a in ds.data_vars:
            return a
    if len(ds.data_vars) == 1:
        return list(ds.data_vars)[0]
    raise KeyError(f"None of {ALIASES[canon]} (or '{explicit}') in dataset vars {list(ds.data_vars)}")


def _year_slice(ds, years):
    # year-resolution strings work for every calendar (360_day has no Dec 31)
    return ds.sel(time=slice(f"{years[0]:04d}", f"{years[1]:04d}"))


def _subset_bbox(ds, bbox):
    if bbox is None:
        return ds
    lon0, lon1, lat0, lat1 = bbox
    return ds.sel(lat=slice(lat0, lat1), lon=slice(lon0, lon1))


def check_completeness(da, years, label=""):
    yrs = da.time.dt.year.values
    counts = pd.Series(yrs).value_counts().sort_index()
    expected = range(years[0], years[1] + 1)
    missing_years = [y for y in expected if y not in counts.index]
    short = counts[counts < 360]
    if missing_years:
        warnings.warn(f"{label}: missing years {missing_years}")
    if len(short):
        warnings.warn(f"{label}: incomplete years {dict(short)}")
    return not missing_years and not len(short)


def load_variable(entry, canon, years, bbox=None, label=""):
    ds = open_entry(entry, ALIASES.get(canon, []) + ([entry["var"]] if entry.get("var") else []))
    ds = harmonize_coords(ds)
    ds = normalize_time(ds)
    name = _find_var(ds, canon, entry.get("var"))
    da = ds[name]
    da = _year_slice(da.to_dataset(), years)[name]
    da = _subset_bbox(da.to_dataset(), bbox)[name]
    da = to_target_units(da, canon).astype("float32")
    da.name = canon
    da.attrs["source_variable"] = name
    check_completeness(da, years, f"{label}{canon}")
    return da


def _align_grids(das, label="", tol=1e-3):
    """Put every variable on the grid of the first one before merging. An inner join on lat/lon
    silently produced an EMPTY grid for UKESM, whose files do not share one grid (e.g. wind
    on the staggered u/v grid): near-identical grids are snapped, others interpolated
    bilinearly (edges filled from the nearest point)."""
    names = list(das)
    ref = das[names[0]]
    out = {names[0]: ref}
    for k in names[1:]:
        da = das[k]
        if "lat" not in da.dims or "lon" not in da.dims:
            out[k] = da
            continue
        if da.sizes["lat"] == ref.sizes["lat"] and da.sizes["lon"] == ref.sizes["lon"]:
            off = max(float(np.abs(da.lat.values - ref.lat.values).max()),
                      float(np.abs(da.lon.values - ref.lon.values).max()))
            if off < tol:
                out[k] = da.assign_coords(lat=ref.lat, lon=ref.lon) if off > 0 else da
                continue
        log(f"  {label}{k}: grid {da.sizes['lat']}x{da.sizes['lon']} (lat {float(da.lat[0]):.3f}.., "
            f"lon {float(da.lon[0]):.3f}..) differs from {names[0]} grid {ref.sizes['lat']}x"
            f"{ref.sizes['lon']} (lat {float(ref.lat[0]):.3f}.., lon {float(ref.lon[0]):.3f}..) "
            f"-> bilinear interpolation")
        src = da.chunk({"lat": -1, "lon": -1}) if da.chunks else da
        a = src.interp(lat=ref.lat, lon=ref.lon)
        a = a.fillna(src.reindex(lat=ref.lat, lon=ref.lon, method="nearest"))
        out[k] = a.astype("float32").assign_attrs(da.attrs, regridded_to=names[0])
    return out


def _uv_to_centres(u, w, label=""):
    """uas/vas on a staggered (Arakawa C) grid - UKESM: u at cell-edge longitudes, v at
    cell-edge latitudes - have NO common points, so hypot(u, v) came out empty. Interpolate
    both to the cell centres (latitudes of the variable with fewer of them, longitudes of the
    other) before combining."""
    same = (u.sizes["lat"] == w.sizes["lat"] and u.sizes["lon"] == w.sizes["lon"]
            and np.allclose(u.lat, w.lat) and np.allclose(u.lon, w.lon))
    if same:
        return u, w
    a, b = (u, w) if u.sizes["lat"] <= w.sizes["lat"] else (w, u)   # a: centred latitudes
    lat, lon = a.lat, b.lon
    log(f"  {label}uas/vas on staggered grids ({u.sizes['lat']}x{u.sizes['lon']} vs "
        f"{w.sizes['lat']}x{w.sizes['lon']}) -> interpolated to cell centres {lat.size}x{lon.size}")

    def to(da):
        src = da.chunk({"lat": -1, "lon": -1}) if da.chunks else da
        x = src.interp(lat=lat, lon=lon)
        return x.fillna(src.reindex(lat=lat, lon=lon, method="nearest")).astype("float32")
    return to(u), to(w)


def load_inputs(catalog, model, scenario, member, variables, years, bbox=None,
                optional=("hursmin", "tasmax", "tas")):
    """Return Dataset of canonical daily variables for one model/scenario/member."""
    entries = catalog[model][scenario][member]
    label = f"{model}/{scenario}/{member}: "
    out = {}
    for v in variables:
        if v in entries and entries[v].get("kind") == "wind_climatology":
            continue                                   # filled below, once the time axis is known
        if v in entries:
            out[v] = load_variable(entries[v], v, years, bbox, label)
        elif v == "sfcWind" and "uas" in entries and "vas" in entries:
            u = load_variable(entries["uas"], "uas", years, bbox, label)
            w = load_variable(entries["vas"], "vas", years, bbox, label)
            u, w = _uv_to_centres(u, w, label)
            spd = np.hypot(u, w)
            r = WIND_UV_RATIO.get(model)
            if r is not None:
                r = r.reindex(lat=spd.lat, lon=spd.lon, method="nearest")
                spd = spd * r.sel(month=spd.time.dt.month).drop_vars("month")
                log(f"  {label}sfcWind derived from uas/vas, bias-corrected with sfcWind/|uv| ratio")
            else:
                log(f"  {label}sfcWind derived from uas/vas (no correction available)")
            out[v] = spd.astype("float32").rename("sfcWind").assign_attrs(units="m s-1", derived="hypot(uas,vas)")
        elif v in ("hurs",) and all(k in entries for k in ("huss", "ps", "tas")):
            q = load_variable(entries["huss"], "huss", years, bbox, label)
            p = load_variable(entries["ps"], "ps", years, bbox, label)
            t = load_variable(entries["tas"], "tas", years, bbox, label)
            e = q * p / (0.622 + 0.378 * q)
            out[v] = (100 * e / es_sonntag(t)).clip(0, 100).rename("hurs").assign_attrs(units="%")
            log(f"  {label}hurs derived from huss/ps/tas")
        elif v in optional:
            continue
        else:
            raise KeyError(f"{label}required variable '{v}' not in catalog ({list(entries)})")
    n0 = {k: v.sizes["time"] for k, v in out.items()}
    out = _align_grids(out, label)
    ds = xr.merge(list(out.values()), join="inner", compat="override")
    ds.attrs = {}
    if "sfcWind" in variables and entries.get("sfcWind", {}).get("kind") == "wind_climatology":
        if model not in WIND_CLIM:
            raise RuntimeError(f"{model}: run prepare_wind_climatology() before loading inputs")
        wc = WIND_CLIM[model].reindex(lat=ds.lat, lon=ds.lon, method="nearest")
        wc = wc.reindex(dayofyear=np.arange(1, 367), method="nearest")
        ds["sfcWind"] = wc.sel(dayofyear=ds.time.dt.dayofyear).drop_vars("dayofyear").astype("float32").assign_attrs(
            units="m s-1", derived="SSP2-4.5 day-of-year climatology of hypot(uas,vas)")
        log(f"  {label}sfcWind = SSP2-4.5 day-of-year wind climatology")
    if len(set(n0.values())) > 1 or ds.sizes["time"] < max(n0.values()):
        log(f"  {label}time lengths {n0} -> {ds.sizes['time']} common days kept")
    ds.attrs.update(model=model, scenario=scenario, member=member,
                    years=f"{years[0]}-{years[1]}")
    return ds


# -----------------------------------------------------------------------------
# Storage helpers (local paths or s3://reflective-persistent-prod/<user>/...)
# -----------------------------------------------------------------------------
def store_exists(path, storage_options=None):
    import fsspec
    fs, fp = fsspec.core.url_to_fs(path, **(storage_options or {}))
    return any(fs.exists(f"{fp}/{m}") for m in ("zarr.json", ".zgroup", ".zmetadata"))


def write_zarr(ds, path, space_chunk=48, time_chunk=365, storage_options=None):
    """Write with chunks (time_chunk, space_chunk, space_chunk), float32."""
    ds = ds.copy()
    for v in ds.variables:
        ds[v].encoding = {}
    ny, nx = ds.sizes.get("lat", 1), ds.sizes.get("lon", 1)
    enc = {}
    for v in ds.data_vars:
        dims = ds[v].dims
        ch = tuple({"time": min(time_chunk, ds.sizes["time"]), "lat": min(space_chunk, ny),
                    "lon": min(space_chunk, nx)}.get(d, ds.sizes[d]) for d in dims)
        enc[v] = {"chunks": ch, "dtype": "float32"}
    # each dask chunk must cover whole zarr chunks:
    #  - time-contiguous data (e.g. FWI output): dask (all time, sc, sc)
    #  - time-split data (e.g. inputs read file by file): dask (time_chunk, all lat, all lon)
    first = ds[list(ds.data_vars)[0]]
    tch = dict(zip(first.dims, first.chunks)).get("time") if first.chunks else None
    if tch is None or len(tch) == 1:
        ds = ds.chunk({"time": -1, "lat": space_chunk, "lon": space_chunk})
    else:
        ds = ds.chunk({"time": time_chunk, "lat": -1, "lon": -1})
    kw = {}
    try:
        import zarr
        if int(zarr.__version__.split(".")[0]) >= 3:
            kw["zarr_format"] = 2      # readable by both zarr 2.x and 3.x Hub images
    except Exception:
        pass
    ds.to_zarr(path, mode="w", encoding=enc, storage_options=storage_options, **kw)


def open_store(path, space_chunk=48, storage_options=None):
    ds = xr.open_zarr(path, storage_options=storage_options, chunks={})
    return ds.chunk({"time": -1, "lat": space_chunk, "lon": space_chunk})


# -----------------------------------------------------------------------------
# FWI
# -----------------------------------------------------------------------------
_FWI_ORDER = ("DC", "DMC", "FFMC", "ISI", "BUI", "FWI")   # order returned by xclim.cffwis_indices


def _prepend_pseudo_spinup(ds, year):
    """Prepend a copy of `year` relabelled as year-1 (used as FWI spin-up when the model
    output starts at the analysis start year). Feb 29 is dropped if year-1 has none."""
    first = ds.sel(time=str(year))
    keep, times = [], []
    for i, t in enumerate(first.indexes["time"]):
        try:
            times.append(t.replace(year=t.year - 1))
            keep.append(i)
        except ValueError:
            pass
    sp = first.isel(time=keep).assign_coords(time=times)
    return xr.concat([sp, ds], "time")


def compute_fwi(ds, temp="tasmax", rh="hursmin", rh_fallback="hurs", spinup_years=SPINUP_YEARS,
                keep=("FWI", "ISI", "BUI", "DC", "DMC", "FFMC"), space_chunk=48, season_method=None,
                analysis_start=None):
    """Canadian FWI system (xclim) from daily model output.

    Inputs used ("noon-equivalent" proxies, standard for daily GCM output):
      temperature: daily max (tasmax), humidity: daily min RH if available, else daily mean,
      wind: daily mean 10 m speed, precipitation: daily total.
    Spin-up: output before `analysis_start` is discarded. If the data start at
    `analysis_start` (no spin-up year available), the first year is duplicated as a
    pseudo spin-up year. Without `analysis_start`, the first `spinup_years` are dropped.
    """
    import xclim
    from xclim.indices import cffwis_indices

    rhv = rh if rh in ds else rh_fallback
    tv = temp if temp in ds else "tas"
    if tv != temp:
        warnings.warn(f"{temp} not available; FWI uses {tv}")
    y_first = int(ds.time.dt.year.values[0])
    spin_note = f"{spinup_years} year(s) of model data"
    if analysis_start is not None and y_first >= analysis_start:
        log(f"  [spin-up] data start in {y_first}: using a copy of {y_first} as spin-up year")
        ds = _prepend_pseudo_spinup(ds, y_first)
        spin_note = f"pseudo spin-up (copy of {y_first})"
    ds = ds.chunk({"time": -1, "lat": space_chunk, "lon": space_chunk})
    # explicit mm/day avoids flux->rate conversion differences between xclim versions
    pr_mmd = (ds["pr"] * 86400.0).assign_attrs(units="mm/d", standard_name="precipitation_amount")
    with xclim.set_options(data_validation="log", cf_compliance="log"):
        out = cffwis_indices(tas=ds[tv], pr=pr_mmd, sfcWind=ds["sfcWind"], hurs=ds[rhv],
                             lat=ds["lat"].assign_attrs(units="degrees_north"), season_method=season_method)
    named = {k: (getattr(out, k) if hasattr(out, "_fields") else out[i]) for i, k in enumerate(_FWI_ORDER)}
    # xclim < ~0.56: BUI = 0.8*DC*DMC/(DMC+0.4*DC) gives 0/0 = NaN when DMC = DC = 0 (cold/wet
    # high latitudes), which propagates to FWI. Correct value: BUI = 0, and FWI from ISI
    # (Van Wagner 1987: f(D)=2 for BUI=0; B=0.1*ISI*f(D); FWI = exp(2.72*(0.434 ln B)^0.647) if B>1 else B).
    zero = (named["DMC"] + 0.4 * named["DC"]) == 0
    B = 0.1 * named["ISI"] * 2.0
    S = xr.where(B > 1, np.exp(2.72 * (0.434 * np.log(B.where(B > 1, 1.0))) ** 0.647), B)
    named["BUI"] = named["BUI"].where(~zero, 0.0)
    named["FWI"] = named["FWI"].where(~zero, S)
    res = {}
    for k in keep:
        v = named[k].astype("float32").transpose("time", "lat", "lon")
        v.attrs = {"units": "1", "long_name": f"Canadian FWI system: {k}"}
        res[k] = v
    res = xr.Dataset(res)
    start = analysis_start if analysis_start is not None else int(ds.time.dt.year.values[0]) + spinup_years
    res = res.isel(time=(res.time.dt.year >= start).values)
    res.attrs.update(
        fwi_temperature=tv, fwi_humidity=rhv, fwi_wind="sfcWind (daily mean)",
        fwi_precip="pr (daily total)", xclim_version=xclim.__version__,
        spinup=spin_note, season_method=str(season_method),
        **{k: v for k, v in ds.attrs.items() if isinstance(v, (str, int, float))})
    return res


# -----------------------------------------------------------------------------
# VPD
# -----------------------------------------------------------------------------
def es_sonntag(T):
    """Saturation vapour pressure over water [Pa], Sonntag (1990); T in K.
    (Coefficient 16.635794 gives hPa, hence the factor 100; ~2339 Pa at 20 degC.)"""
    return 100.0 * np.exp(-6096.9385 / T + 16.635794 - 2.711193e-2 * T + 1.673952e-5 * T ** 2
                          + 2.433502 * np.log(T))


def vpd_from_rh(T, rh):
    """VPD [kPa] from temperature [K] and relative humidity [%]."""
    v = es_sonntag(T) * (1 - rh / 100.0) / 1000.0
    return v.clip(min=0).rename("vpd").assign_attrs(units="kPa", long_name="vapour pressure deficit")


def vpd_from_huss(T, huss, ps):
    """VPD [kPa] from temperature [K], specific humidity [kg/kg] and surface pressure [Pa]."""
    e = huss * ps / (0.622 + 0.378 * huss)
    v = (es_sonntag(T) - e) / 1000.0
    return v.clip(min=0).rename("vpd").assign_attrs(units="kPa", long_name="vapour pressure deficit")


# -----------------------------------------------------------------------------
# Extremes metrics
# -----------------------------------------------------------------------------
def pooled_quantile(da_list, q=0.95, space_chunk=48):
    """Local quantile pooled over members (and all days) of the reference runs."""
    da = xr.concat([d.drop_vars([c for c in d.coords if c not in ("time", "lat", "lon")])
                    for d in da_list], dim="member", join="override")
    da = da.chunk({"member": -1, "time": -1, "lat": space_chunk, "lon": space_chunk})
    thr = da.quantile(q, dim=["member", "time"]).drop_vars("quantile")
    return thr.astype("float32")


def exceedance_metrics(da, thr, abs_thr=None, spells=True, prefix=""):
    """Lazy per-member metrics for a daily index and a local threshold map.

    Returns Dataset with
      {p}days_p95   (year, lat, lon)  days per year above local threshold
      {p}days_abs   (year, lat, lon)  days per year above absolute threshold (optional)
      {p}spell_max  (year, lat, lon)  longest run of consecutive days above threshold
      {p}exc_freq   (month, lat, lon) fraction of days in month above threshold
      {p}mean_month (month, lat, lon) monthly mean of the index
      {p}mean       (lat, lon)        all-days mean
    """
    exc = da > thr
    valid = thr.notnull()
    out = {}
    out[f"{prefix}days_p95"] = exc.groupby("time.year").sum("time").where(valid)
    if abs_thr is not None:
        out[f"{prefix}days_abs"] = (da > abs_thr).groupby("time.year").sum("time").where(valid)
    if spells:
        from xclim.indices.run_length import longest_run
        sp = longest_run(exc, freq="YS")
        sp = sp.assign_coords(year=("time", sp.time.dt.year.values)).swap_dims(time="year").drop_vars("time")
        out[f"{prefix}spell_max"] = sp.where(valid)
    out[f"{prefix}exc_freq"] = exc.astype("float32").groupby("time.month").mean("time").where(valid)
    out[f"{prefix}mean_month"] = da.groupby("time.month").mean("time")
    out[f"{prefix}mean"] = da.mean("time")
    return xr.Dataset(out)


def stack_members(ds_list, names):
    return xr.concat(ds_list, dim=pd.Index(names, name="member"), join="override")


# -----------------------------------------------------------------------------
# Statistics
# -----------------------------------------------------------------------------
def welch_ttest(a, b, dims=("member", "year")):
    """Two-sided Welch t-test at each grid point; samples pooled over `dims`.
    (Annual values; inter-annual autocorrelation is ignored.)"""
    from scipy import stats
    da_ = [d for d in dims if d in a.dims]
    db_ = [d for d in dims if d in b.dims]
    na = float(np.prod([a.sizes[d] for d in da_]))
    nb = float(np.prod([b.sizes[d] for d in db_]))
    ma, mb = a.mean(da_), b.mean(db_)
    va, vb = a.var(da_, ddof=1), b.var(db_, ddof=1)
    se2 = va / na + vb / nb
    t = (ma - mb) / np.sqrt(se2)
    df = se2 ** 2 / ((va / na) ** 2 / (na - 1) + (vb / nb) ** 2 / (nb - 1))
    p = xr.apply_ufunc(lambda tt, dd: 2 * stats.t.sf(np.abs(tt), dd), t, df, dask="parallelized",
                       output_dtypes=[float])
    p = p.where(se2 > 0, 1.0).where(ma.notnull() & mb.notnull())
    return p.rename("pvalue")


def fdr_significant(p, alpha_fdr=0.10, mask=None):
    """Benjamini-Hochberg false discovery rate control (Wilks 2016: alpha_FDR = 2*alpha_global)."""
    pv = p.values.copy()
    valid = np.isfinite(pv)
    if mask is not None:
        valid &= np.asarray(mask.values, dtype=bool)
    vals = np.sort(pv[valid])
    n = vals.size
    if n == 0:
        return xr.zeros_like(p, dtype=bool)
    below = vals <= alpha_fdr * np.arange(1, n + 1) / n
    pcrit = vals[below].max() if below.any() else -1.0
    sig = np.zeros_like(pv, dtype=bool)
    sig[valid] = pv[valid] <= pcrit
    return xr.DataArray(sig, coords=p.coords, dims=p.dims, name="significant")


def member_sign_agreement(delta_members, dim="member"):
    """Fraction of members agreeing with the sign of the ensemble mean change."""
    s = np.sign(delta_members.mean(dim))
    return (np.sign(delta_members) == s).mean(dim)


# -----------------------------------------------------------------------------
# Grids, masks, regions
# -----------------------------------------------------------------------------
def to_common_grid(da, res=1.0):
    """Bilinear interpolation to a global res x res grid (cell centres). Longitude is treated as
    periodic only for (near-)global grids; no extrapolation beyond the data (regional boxes stay
    regional; cells outside the source grid are NaN)."""
    lat = np.arange(-90 + res / 2, 90, res)
    lon = np.arange(-180 + res / 2, 180, res)
    dt = da.dtype
    if dt == bool:
        da = da.astype("float32")
    dlon = float(np.median(np.diff(da.lon.values))) if da.lon.size > 1 else 360.0
    is_global = (float(da.lon.max() - da.lon.min()) + dlon) >= 359.0
    if is_global:
        left = da.isel(lon=slice(-2, None)).assign_coords(lon=lambda d: d.lon - 360)
        right = da.isel(lon=slice(0, 2)).assign_coords(lon=lambda d: d.lon + 360)
        da = xr.concat([left, da, right], "lon")
        # allow filling only the polar caps beyond the outermost latitude rows
        lat_lo, lat_hi = float(da.lat.min()), float(da.lat.max())
        out = da.interp(lat=lat, lon=lon, method="linear")
        edge = (out.lat < lat_lo) | (out.lat > lat_hi)
        if bool(edge.any()):
            filled = out.ffill("lat").bfill("lat")        # copy outermost row into the polar caps only
            out = out.where(~edge, filled)
    else:
        out = da.interp(lat=lat, lon=lon, method="linear")    # NaN outside the box
    return out > 0.5 if dt == bool else out


_LAND_CACHE = {}


def land_mask(lon, lat, drop_ice_sheets=True):
    """Boolean land mask (Natural Earth 1:110m); drops Antarctica/Greenland by default."""
    import regionmask
    key = (tuple(np.round(lon.values, 4)), tuple(np.round(lat.values, 4)), drop_ice_sheets)
    if key in _LAND_CACHE:
        return _LAND_CACHE[key]
    try:
        land = regionmask.defined_regions.natural_earth_v5_0_0.land_110
        m = land.mask_3D(lon, lat).any("region")
    except Exception as e:  # offline: fall back to union of AR6 land regions
        warnings.warn(f"Natural Earth land mask unavailable ({type(e).__name__}); using AR6 land regions")
        m = regionmask.defined_regions.ar6.land.mask_3D(lon, lat).any("region")
    if drop_ice_sheets:
        ar6 = regionmask.defined_regions.ar6.land
        ice = ar6.mask_3D(lon, lat)
        ice = ice.isel(region=np.isin(ice.abbrevs.values, ["GIC", "EAN", "WAN"])).any("region")
        m = m & ~ice & (lat > -60)
    m = m.drop_vars([c for c in m.coords if c not in ("lat", "lon")])
    _LAND_CACHE[key] = m
    return m


def regional_means(da, regions=None, land_only=True):
    """Area-weighted (cos lat) AR6 land-region means. Output dim 'region' (abbrevs)."""
    import regionmask
    ar6 = regionmask.defined_regions.ar6.land
    m3 = ar6.mask_3D(da.lon, da.lat)
    if regions is not None:
        m3 = m3.isel(region=np.isin(m3.abbrevs.values, list(regions)))
    if land_only:
        m3 = m3 & land_mask(da.lon, da.lat)
    w = (m3 * np.cos(np.deg2rad(da.lat))).astype(float)
    out = da.weighted(w).mean(("lat", "lon"))
    out = out.assign_coords(region=m3.abbrevs.values, region_name=("region", m3.names.values))
    if regions is not None:
        out = out.sel(region=[r for r in regions if r in out.region.values])
    return out


# -----------------------------------------------------------------------------
# Fire season (from the target-period FWI climatology)
# -----------------------------------------------------------------------------
_DAYS_IN_MONTH = xr.DataArray([31, 28.25, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31],
                              dims="month", coords={"month": np.arange(1, 13)})


def fire_season_mask(monthly_clim, n=3):
    """Boolean (month, lat, lon): the n consecutive months with highest mean index."""
    v = monthly_clim.transpose("month", ...).values
    sums = sum(np.roll(v, -k, axis=0) for k in range(n))
    allnan = np.all(~np.isfinite(sums), axis=0)
    start = np.nanargmax(np.where(np.isfinite(sums), sums, -np.inf), axis=0)
    mask = np.zeros_like(v, dtype=bool)
    for k in range(n):
        np.put_along_axis(mask, ((start + k) % 12)[None], True, axis=0)
    mask[:, allnan] = False
    return xr.DataArray(mask, coords=monthly_clim.transpose("month", ...).coords,
                        dims=monthly_clim.transpose("month", ...).dims, name="fire_season")


def seasonal_mean(monthly, season_mask):
    """Day-weighted mean of monthly values over months where season_mask is True."""
    w = _DAYS_IN_MONTH * season_mask
    return (monthly * w).sum("month") / w.sum("month").where(w.sum("month") > 0)


# -----------------------------------------------------------------------------
# Delta-method driver attribution
# -----------------------------------------------------------------------------
DELTA_KIND = {"tasmax": "add", "tas": "add", "hurs": "add", "hursmin": "add",
              "sfcWind": "mul", "pr": "mul"}


def monthly_climatology(da):
    return da.groupby("time.month").mean("time")


def climatological_deltas(clim_scen, clim_base, variables, ratio_limits=(0.2, 5.0)):
    """Monthly deltas: additive for T/RH, multiplicative (bounded) for wind/precip."""
    d = {}
    for v in variables:
        if DELTA_KIND[v] == "add":
            d[v] = clim_scen[v] - clim_base[v]
        else:
            eps = 1e-9 if v == "pr" else 1e-3
            d[v] = ((clim_scen[v] + eps) / (clim_base[v] + eps)).clip(*ratio_limits)
    return xr.Dataset(d)


def apply_deltas(base, deltas, variables):
    """Perturb daily baseline inputs with monthly climatological deltas for `variables`."""
    out = base.copy()
    month = base.time.dt.month
    for v in variables:
        dv = deltas[v].sel(month=month).drop_vars("month")
        x = base[v] + dv if DELTA_KIND[v] == "add" else base[v] * dv
        if v in ("hurs", "hursmin"):
            x = x.clip(0, 100)
        x.attrs = base[v].attrs
        out[v] = x.astype("float32")
    return out


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------
_COAST = None


def coastlines_available():
    global _COAST
    if _COAST is None:
        try:
            import cartopy.io.shapereader as shp
            shp.natural_earth(resolution="110m", category="physical", name="coastline")
            _COAST = True
        except Exception:
            _COAST = False
    return _COAST


def save_figure(fig, path_no_ext, dpi=300):
    """Save a figure as PNG (at `dpi`) and as vector PDF for print-quality use."""
    fig.savefig(f"{path_no_ext}.png", dpi=dpi, bbox_inches="tight")
    fig.savefig(f"{path_no_ext}.pdf", bbox_inches="tight")


def map_axes(nrows, ncols, figsize=None):
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    figsize = figsize or (5.2 * ncols, 3.1 * nrows + 0.6)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False,
                             subplot_kw={"projection": ccrs.Robinson()})
    return fig, axes


def map_panel(ax, da, cmap="RdBu_r", vmin=None, vmax=None, title="", stipple=None,
              cbar_label="", extend="both", levels=None, land=None, cbar=True):
    """Filled map with optional stippling (dots where stipple is True)."""
    import cartopy.crs as ccrs
    import matplotlib.pyplot as plt
    from cartopy.util import add_cyclic_point
    import matplotlib.colors as mcolors

    if land is not None:
        da = da.where(land)
    data, lon = add_cyclic_point(da.values, coord=da.lon.values)
    if levels is None and vmin is not None:
        levels = np.linspace(vmin, vmax, 11)
    norm = mcolors.BoundaryNorm(levels, plt.get_cmap(cmap).N, extend=extend) if levels is not None else None
    pm = ax.pcolormesh(lon, da.lat.values, np.ma.masked_invalid(data), cmap=cmap, norm=norm,
                       transform=ccrs.PlateCarree(), shading="nearest", rasterized=True)
    if stipple is not None:
        st, lon2 = add_cyclic_point(stipple.astype(float).values, coord=stipple.lon.values)
        if np.nanmax(st) > 0.5:
            ax.contourf(lon2, stipple.lat.values, st, levels=[0.5, 1.5], colors="none",
                        hatches=["...."], transform=ccrs.PlateCarree())
    if coastlines_available():
        ax.coastlines(linewidth=0.4, color="#444444")
    ax.set_global()
    ax.set_title(title, fontsize=10)
    if cbar:
        cb = plt.colorbar(pm, ax=ax, orientation="horizontal", pad=0.04, shrink=0.85,
                          aspect=30, extend=extend if norm is None else "neither")
        cb.set_label(cbar_label, fontsize=9)
        cb.ax.tick_params(labelsize=8)
    return pm


def symmetric_limit(da, pct=98, floor=None):
    v = float(np.nanpercentile(np.abs(da.values), pct)) if np.isfinite(da.values).any() else 1.0
    if floor:
        v = max(v, floor)
    # round to a "nice" number
    mag = 10 ** np.floor(np.log10(v)) if v > 0 else 1
    return float(np.ceil(v / mag * 2) / 2 * mag)


def style_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(ls=":", lw=0.5, color="#cccccc")
    ax.set_axisbelow(True)


# -----------------------------------------------------------------------------
# Workspace: caching of staged inputs, daily FWI and per-member metrics
# -----------------------------------------------------------------------------
class IncompleteInputs(Exception):
    """A member's inputs lack whole years of the requested period."""


def coverage_problems(ds, years, min_days=300):
    """Years in `years` (inclusive) with fewer than `min_days` daily values."""
    counts = pd.Series(ds.time.dt.year.values).value_counts()
    return [y for y in range(years[0], years[1] + 1) if counts.get(y, 0) < min_days]


def drop_incomplete(ws, models, members, runs, variables):
    """Stage (or reuse) every model/scenario/member input store and drop members whose inputs
    miss whole years of a run's period (e.g. UKESM SSP2-4.5 r2 humidity/wind lack 7 of the
    target-period years). A member dropped in one run of a scenario is dropped from all runs
    of that scenario, so target and SSP2-4.5 use the same members. Returns the models that
    still have members in every scenario."""
    for m in models:
        for run, (scen, years) in runs.items():
            for mem in list(members[m][scen]):
                try:
                    ws.inputs(m, scen, mem, (years[0] - SPINUP_YEARS, years[1]), variables)
                except IncompleteInputs as e:
                    members[m][scen].remove(mem)
                    log(f"[skip] {m}/{scen}/{mem}: {e} - member dropped from {scen}")
    keep = [m for m in models if all(members[m][s] for s in SCENARIOS)]
    for m in models:
        if m not in keep:
            log(f"[skip] {m}: a scenario has no complete member left")
    return keep


class Workspace:
    """Keeps track of where intermediate products live and (re)uses them.

    root/inputs/<model>/<scenario>/<member>_<y0>-<y1>.zarr   staged canonical daily inputs
    root/fwi/<model>/<scenario>/<member>_<y0>-<y1>.zarr      daily FWI (after spin-up removal)
    """

    def __init__(self, catalog, root, storage_options=None, space_chunk=48, stage_inputs=True,
                 bbox=None):
        self.catalog = catalog
        self.root = root.rstrip("/")
        self.so = storage_options
        self.sc = space_chunk
        self.stage = stage_inputs
        self.bbox = bbox
        self.tag = "" if bbox is None else "_bbox" + "_".join(str(int(b)) for b in bbox)
        self.grids = {}

    def snap(self, model, obj, tol=0.01):
        """Put every dataset of a model on identical lat/lon coordinates (e.g. SSP2-4.5 from
        Pangeo vs G6 from the Hub), so arithmetic between runs never silently drops rows.
        Raises if grids really differ (shape mismatch or offsets > tol degrees)."""
        if model not in self.grids:
            self.grids[model] = (obj.lat.values.copy(), obj.lon.values.copy())
            return obj
        lat, lon = self.grids[model]
        if obj.lat.size != lat.size or obj.lon.size != lon.size:
            raise ValueError(f"{model}: grid {obj.lat.size}x{obj.lon.size} differs from first run "
                             f"{lat.size}x{lon.size} - regrid needed")
        off = max(float(np.abs(obj.lat.values - lat).max()), float(np.abs(obj.lon.values - lon).max()))
        if off > tol:
            raise ValueError(f"{model}: grid offset {off:.4f} deg between runs - regrid needed")
        if off > 0:
            log(f"  [grid] {model}: snapping coordinates (max offset {off:.2e} deg)")
        return obj.assign_coords(lat=("lat", lat, {"units": "degrees_north", "standard_name": "latitude"}),
                                 lon=("lon", lon, {"units": "degrees_east", "standard_name": "longitude"}))

    def models(self, required):
        out = []
        for m, scens in self.catalog.items():
            if m.startswith("_"):
                continue
            if all(s in scens and self.members(m, s, required) for s in SCENARIOS):
                out.append(m)
        return out

    def members(self, model, scen, required, max_members=None):
        mems = self.catalog.get(model, {}).get(scen, {})
        ok = [k for k in sorted(mems) if all(v in mems[k] or (v == "sfcWind" and "uas" in mems[k])
                                             or (v == "hurs" and "huss" in mems[k]) for v in required)]
        return ok[:max_members] if max_members else ok

    def _path(self, kind, model, scen, member, years):
        return f"{self.root}/{kind}/{model}/{scen}/{member}_{years[0]}-{years[1]}{self.tag}.zarr"

    def _drop_if_empty(self, p):
        """Stores staged before the v6.7 grid fix can have an empty (0 x 0) UKESM grid: remove them."""
        if store_exists(p, self.so):
            d = xr.open_zarr(p, storage_options=self.so, chunks={})
            if d.sizes.get("lat", 1) == 0 or d.sizes.get("lon", 1) == 0:
                import fsspec
                fs, fp = fsspec.core.url_to_fs(p, **(self.so or {}))
                fs.rm(fp, recursive=True)
                log(f"  [cache] removed empty-grid store {p}")

    def inputs(self, model, scen, member, years, variables):
        p = self._path("inputs", model, scen, member, years)
        if self.stage:
            self._drop_if_empty(p)
            if not store_exists(p, self.so):
                log(f"  staging inputs -> {p}")
                ds = load_inputs(self.catalog, model, scen, member, variables, years, self.bbox)
                write_zarr(ds, p, self.sc, storage_options=self.so)
            ds = self.snap(model, open_store(p, self.sc, self.so))
            entries = self.catalog[model][scen][member]
            missing = [v for v in variables if v not in ds and _has(entries, v)]
            if missing:
                # variables added to the catalog after staging (e.g. UKESM tasmax): stage them
                # into a sibling store instead of re-staging everything
                px = p[:-len(".zarr")] + "_" + "-".join(sorted(missing)) + ".zarr"
                if not store_exists(px, self.so):
                    log(f"  staging added variable(s) {missing} -> {px}")
                    dx = load_inputs(self.catalog, model, scen, member, missing, years, self.bbox)
                    write_zarr(dx[[v for v in missing if v in dx]], px, self.sc, storage_options=self.so)
                dx = self.snap(model, open_store(px, self.sc, self.so))
                n0 = ds.sizes["time"]
                ds = xr.merge([ds, dx], join="inner", compat="override")
                if ds.sizes["time"] < n0 - 2:
                    raise ValueError(f"{px}: only {ds.sizes['time']} of {n0} days match the staged inputs")
            bad = coverage_problems(ds, (years[0] + SPINUP_YEARS, years[1]))
            if bad:
                raise IncompleteInputs(f"inputs miss or have incomplete years {bad}")
            return ds
        return self.snap(model, load_inputs(self.catalog, model, scen, member, variables, years, self.bbox))

    def fwi(self, model, scen, member, years, input_vars, spinup_years=SPINUP_YEARS,
            keep=("FWI",), **fwi_kw):
        """Daily FWI for `years` (inputs start `spinup_years` earlier)."""
        p = self._path(f"fwi_{fwi_kw.get('temp', 'tasmax')}", model, scen, member, years)
        self._drop_if_empty(p)
        if not store_exists(p, self.so):
            yrs_in = (years[0] - spinup_years, years[1])
            ds = self.inputs(model, scen, member, yrs_in, input_vars)
            log(f"  computing FWI {model}/{scen}/{member} {years[0]}-{years[1]}")
            f = compute_fwi(ds, spinup_years=spinup_years, keep=keep, space_chunk=self.sc,
                            analysis_start=years[0], **fwi_kw)
            write_zarr(f, p, self.sc, storage_options=self.so)
        return self.snap(model, open_store(p, self.sc, self.so))


def cached_netcdf(path, fn, overwrite=False):
    """Compute Dataset via fn() and cache to NetCDF at path (local)."""
    if os.path.exists(path) and not overwrite:
        return xr.open_dataset(path).load()
    ds = fn()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    ds.to_netcdf(path)
    return ds


# -----------------------------------------------------------------------------
# Ensemble / multi-model helpers
# -----------------------------------------------------------------------------
def ensemble_delta(a, b, alpha_fdr=0.10, mask=None, sample_dims=("member", "year")):
    """Change in ensemble mean (a - b) of annual values + Welch p, FDR significance,
    and member sign agreement. a, b: DataArrays with (member, year, lat, lon)."""
    da_ = [d for d in sample_dims if d in a.dims]
    db_ = [d for d in sample_dims if d in b.dims]
    delta = a.mean(da_) - b.mean(db_)
    p = welch_ttest(a, b, sample_dims)
    sig = fdr_significant(p, alpha_fdr, mask)
    agree = member_sign_agreement(a.mean("year") - b.mean(db_)) if "member" in a.dims else None
    return xr.Dataset({"delta": delta, "pvalue": p, "significant": sig,
                       **({"member_agreement": agree} if agree is not None else {})})


def multimodel(fields, sig=None, res=1.0, agree_frac=2 / 3):
    """Multi-model mean on a common grid. Robust = >= agree_frac of models share the sign
    of the mean AND >= half are individually FDR-significant (>=2 models), or
    FDR-significant (single model)."""
    models = list(fields)
    regr = [to_common_grid(fields[m], res) for m in models]
    stack = xr.concat(regr, pd.Index(models, name="model"))
    mmm = stack.mean("model")
    if len(models) == 1:
        robust = to_common_grid(sig[models[0]], res) if sig is not None else None
    else:
        robust = (np.sign(stack) == np.sign(mmm)).mean("model") >= agree_frac
        if sig is not None:   # also require >= half of models individually FDR-significant
            sstack = xr.concat([to_common_grid(sig[m].astype(bool), res) for m in models],
                               pd.Index(models, name="model"))
            robust = robust & (sstack.astype(float).mean("model") >= 0.5)
    return mmm, robust, stack


def to_netcdf_safe(ds, path):
    """Write Dataset to NetCDF, storing boolean masks as int8 (NetCDF has no bool)."""
    ds = ds.copy()
    for k in ds.data_vars:
        if ds[k].dtype == bool:
            ds[k] = ds[k].astype("int8").assign_attrs(ds[k].attrs, flag_meaning="1=True")
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    ds.to_netcdf(path)


# -----------------------------------------------------------------------------
# Task 2 helpers
# -----------------------------------------------------------------------------
def fire_season_annual(monthly, season):
    """Per-year fire-season mean from a monthly time series (time, lat, lon) and a
    (month, lat, lon) boolean season mask. Seasons spanning Dec-Jan are split by
    calendar year (acceptable for 20-year means)."""
    m = monthly.time.dt.month
    days = _DAYS_IN_MONTH.sel(month=m).drop_vars("month")
    w = season.sel(month=m).drop_vars("month").astype(float) * days
    num = (monthly * w).groupby("time.year").sum("time")
    den = w.groupby("time.year").sum("time")
    return num / den.where(den > 0)


OFFSET_CLASSES = OrderedDict([
    (1, ("Consistent mitigation: cooler and wetter (VPD↓, P↑)", "#2166ac")),
    (2, ("Risk offset: FWI falls despite less rain (VPD↓, P↓, FWI↓)", "#2a9d8f")),
    (3, ("Fire-weather paradox: FWI rises with less rain (VPD↓, P↓, FWI↑)", "#e66101")),
    (4, ("Consistent exacerbation (VPD↑, P↓)", "#a50026")),
    (5, ("Higher demand, wetter (VPD↑, P↑)", "#762a83")),
])


def classify_offset(dvpd, dpr, dfwi=None):
    """Risk-offset classes from signs of fire-season changes (see OFFSET_CLASSES).
    Without dfwi, classes 2 and 3 are merged into 2 ('competing: cooling vs drying')."""
    c = xr.full_like(dvpd, np.nan, dtype=float)
    down, wet = dvpd < 0, dpr >= 0
    c = c.where(~(down & wet), 1)
    if dfwi is None:
        c = c.where(~(down & ~wet), 2)
    else:
        c = c.where(~(down & ~wet & (dfwi <= 0)), 2)
        c = c.where(~(down & ~wet & (dfwi > 0)), 3)
    c = c.where(~(~down & ~wet), 4)
    c = c.where(~(~down & wet), 5)
    return c.where(dvpd.notnull() & dpr.notnull()).rename("offset_class")


def class_map_panel(ax, cls, title="", land=None):
    import cartopy.crs as ccrs
    import matplotlib.colors as mcolors
    from cartopy.util import add_cyclic_point
    if land is not None:
        cls = cls.where(land)
    cols = [v[1] for v in OFFSET_CLASSES.values()]
    cmap = mcolors.ListedColormap(cols)
    norm = mcolors.BoundaryNorm(np.arange(0.5, len(cols) + 1), cmap.N)
    data, lon = add_cyclic_point(cls.values, coord=cls.lon.values)
    ax.pcolormesh(lon, cls.lat.values, np.ma.masked_invalid(data), cmap=cmap, norm=norm,
                  transform=ccrs.PlateCarree(), shading="nearest", rasterized=True)
    # secondary encoding for the paradox class (texture, not colour alone)
    par = (data == 3).astype(float)
    if par.max() > 0.5:
        ax.contourf(lon, cls.lat.values, par, levels=[0.5, 1.5], colors="none", hatches=["////"],
                    transform=ccrs.PlateCarree())
    if coastlines_available():
        ax.coastlines(linewidth=0.4, color="#444444")
    ax.set_global()
    ax.set_title(title, fontsize=10)


def area_fractions(cls, land):
    """Land-area fraction (cos-lat weighted) in each offset class."""
    w = np.cos(np.deg2rad(cls.lat)) * land
    tot = float(w.where(cls.notnull()).sum())
    return pd.Series({k: float(w.where(cls == k).sum()) / tot if tot else np.nan for k in OFFSET_CLASSES})


# -----------------------------------------------------------------------------
# Model readiness and wind consistency
# -----------------------------------------------------------------------------
def _has(entries, v):
    return (v in entries or (v == "sfcWind" and "uas" in entries and "vas" in entries)
            or (v == "hurs" and all(k in entries for k in ("huss", "ps", "tas"))))


def readiness_table(catalog, required, max_members=None):
    """Which required variables are missing, per model/scenario/member."""
    rows = []
    for m, scens in catalog.items():
        for s in SCENARIOS:
            mems = scens.get(s, {})
            if not mems:
                rows.append(dict(model=m, scenario=s, member="-", missing="NO DAILY DATA"))
            for mem in sorted(mems)[:max_members] if max_members else sorted(mems):
                miss = [v for v in required if not _has(mems[mem], v)]
                rows.append(dict(model=m, scenario=s, member=mem, missing=", ".join(miss) or "ok"))
    return pd.DataFrame(rows)


# Known data problems: (model, variable) -> reason. These variables are never used.
KNOWN_BAD_VARIABLES = {
    ("E3SMv3", "tasmax"): "E3SMv3 G6 TREFHTMX/TREFHTMN were written from TREFHT (model-version bug; "
                          "Reflective Slack, Oct 2026) - no corrected data will be produced",
    ("MIROC-ES2H", "sfcWind"): "MIROC-ES2H 'sfcWind' files contain a radiative flux (units W/m**2, "
                               "mean ~160-200), not wind speed (found 2026-10-01)",
}
# Models whose FWI uses the SSP2-4.5 day-of-year climatology of hypot(uas, vas) as wind in ALL
# scenarios (because at least one scenario has no usable daily wind). Wind is then not a driver.
WIND_CLIMATOLOGY_MODELS = {"MIROC-ES2H"}
WIND_CLIM: dict = {}


def apply_known_issues(catalog, wind_climatology_models=None):
    """Remove known-bad variables from a catalog and, for wind-climatology models, replace
    every member's sfcWind by a 'wind_climatology' pseudo-entry (filled by
    prepare_wind_climatology). Returns a new catalog."""
    import copy
    wcm = WIND_CLIMATOLOGY_MODELS if wind_climatology_models is None else set(wind_climatology_models)
    cat = copy.deepcopy(catalog)
    # never mix land/ocean/ice-model files into atmospheric series (older catalogs may contain them)
    for m, scens in cat.items():
        if m.startswith("_"):
            continue
        dropped = 0
        for s, mems in scens.items():
            for mem, vs in mems.items():
                for v in list(vs):
                    e = vs[v]
                    if e.get("kind", "netcdf") == "netcdf" and "paths" in e:
                        keep = [p for p in e["paths"] if not is_non_atmos(p)]
                        dropped += len(e["paths"]) - len(keep)
                        if keep:
                            e["paths"] = keep
                        else:
                            vs.pop(v)
        if dropped:
            log(f"[catalog] {m}: ignored {dropped} land/ocean/ice-model file(s) (e.g. CLM LDAY/LMON)")
    for (m, v), why in KNOWN_BAD_VARIABLES.items():
        n = 0
        for s, mems in cat.get(m, {}).items():
            for e in mems.values():
                if v in e and v != "tasmax":          # tasmax is handled by choose_fwi_temp
                    e.pop(v); n += 1
        if n:
            log(f"[known issue] {m}: removed '{v}' from {n} member(s) - {why}")
    for m in wcm:
        for s, mems in cat.get(m, {}).items():
            for e in mems.values():
                e.pop("sfcWind", None)
                e["sfcWind"] = {"kind": "wind_climatology", "model": m}
        if m in cat:
            log(f"[wind] {m}: FWI wind = SSP2-4.5 day-of-year climatology of hypot(uas,vas) in ALL scenarios")
    return cat


def prepare_wind_climatology(ws, models, cache_dir, years=TARGET_PERIOD, window=31):
    """Smoothed day-of-year climatology of hypot(uas, vas) from the first SSP2-4.5 member
    that has uas/vas (target period), for models flagged in the catalog with wind_climatology."""
    for m in models:
        if not any(e.get("sfcWind", {}).get("kind") == "wind_climatology"
                   for mems in ws.catalog.get(m, {}).values() for e in mems.values()):
            continue
        path = os.path.join(cache_dir, f"wind_climatology_{m}{ws.tag}.nc")
        if os.path.exists(path):
            WIND_CLIM[m] = xr.open_dataarray(path).load()
            continue
        donor = next(((mem, e) for mem, e in sorted(ws.catalog[m].get("ssp245", {}).items())
                      if "uas" in e and "vas" in e), None)
        if donor is None:
            raise KeyError(f"{m}: no SSP2-4.5 member with uas/vas for the wind climatology")
        mem, e = donor
        spd = np.hypot(load_variable(e["uas"], "uas", years, ws.bbox), load_variable(e["vas"], "vas", years, ws.bbox))
        clim = spd.groupby("time.dayofyear").mean("time").compute()
        n = clim.sizes["dayofyear"]
        ext = xr.concat([clim.isel(dayofyear=slice(-window, None)), clim, clim.isel(dayofyear=slice(0, window))], "dayofyear")
        ext = ext.assign_coords(dayofyear=np.arange(ext.sizes["dayofyear"]))
        sm = ext.rolling(dayofyear=window, center=True).mean().isel(dayofyear=slice(window, window + n))
        sm = sm.assign_coords(dayofyear=clim.dayofyear.values).astype("float32").rename("wind_climatology")
        sm.attrs.update(units="m s-1", donor=f"{m}/ssp245/{mem} {years}", method=f"hypot(uas,vas) doy mean, {window}-day smoothing")
        os.makedirs(cache_dir, exist_ok=True)
        sm.to_netcdf(path)
        WIND_CLIM[m] = sm
        log(f"[wind] {m}: wind climatology ready (donor {m}/ssp245/{mem}; mean {float(sm.mean()):.2f} m/s)")


def metrics_temp_guard(metrics_dir, temp, patterns=("*_q[0-9]*.nc", "attribution_*.nc")):
    """Task 1 metric caches do not encode which temperature drove the FWI. Remember it in
    <metrics_dir>/fwi_temp.txt and drop the FWI-based caches when it changes (VPD caches stay)."""
    import glob
    os.makedirs(metrics_dir, exist_ok=True)
    marker = os.path.join(metrics_dir, "fwi_temp.txt")
    if os.path.exists(marker):
        old = open(marker).read().strip()
        if old != temp:
            gone = sorted({f for pat in patterns for f in glob.glob(os.path.join(metrics_dir, pat))})
            for f in gone:
                os.remove(f)
            log(f"[cache] {metrics_dir}: FWI temperature {old} -> {temp}; removed {len(gone)} cached file(s)")
    with open(marker, "w") as f:
        f.write(temp)


def tasmax_is_genuine(ws, model, scen, member, ndays=60, min_diff=0.5):
    """Quick check on the first `ndays` that daily tasmax exceeds daily-mean tas on average."""
    e = ws.catalog[model][scen][member]
    if "tas" not in e:
        return True
    try:
        def first(v):
            ds = normalize_time(harmonize_coords(open_entry(e[v], ALIASES[v] + [e[v].get("var", v)])))
            da = ds[_find_var(ds, v, e[v].get("var"))].isel(time=slice(0, ndays))
            return to_target_units(da, v).mean().compute()
        d = float(first("tasmax") - first("tas"))
    except Exception as ex:
        log(f"  [tasmax check] {model}/{scen}/{member}: could not check ({type(ex).__name__})")
        return True
    if d < min_diff:
        log(f"  [tasmax check] {model}/{scen}/{member}: tasmax - tas = {d:.2f} K -> tasmax looks like tas!")
        return False
    return True


def choose_fwi_temp(ws, model, rh="hurs", override=None, check=True):
    """'tasmax' if every scenario has a genuine daily tasmax (>=1 member with all FWI inputs),
    else 'tas'. Using the same variable in every scenario keeps the comparison consistent."""
    if override:
        return override
    if (model, "tasmax") in KNOWN_BAD_VARIABLES:
        log(f"[temp] {model}: using tas - {KNOWN_BAD_VARIABLES[(model, 'tasmax')]}")
        return "tas"
    req = ["tasmax", rh, "sfcWind", "pr"]
    if not all(ws.members(model, s, req) for s in SCENARIOS):
        return "tas"
    if check and not all(tasmax_is_genuine(ws, model, s, ws.members(model, s, req)[0]) for s in SCENARIOS):
        log(f"[temp] {model}: using tas in all scenarios (suspicious tasmax)")
        return "tas"
    return "tasmax"


def prepare_wind_corrections(ws, models, cache_dir, years=TARGET_PERIOD):
    """For models where some members only have uas/vas, derive a monthly climatological
    correction ratio sfcWind / hypot(uas, vas) from a member that has all three
    (preferably SSP2-4.5) and register it in WIND_UV_RATIO."""
    for m in models:
        need = any(("sfcWind" not in e and "uas" in e) for s in SCENARIOS
                   for e in ws.catalog.get(m, {}).get(s, {}).values())
        if not need:
            continue
        donor = None
        for s in ["ssp245"] + SCENARIOS[1:]:
            for mem, e in sorted(ws.catalog[m].get(s, {}).items()):
                if all(k in e for k in ("sfcWind", "uas", "vas")):
                    donor = (s, mem)
                    break
            if donor:
                break
        if donor is None:
            uv_everywhere = all(("sfcWind" not in e) for s in SCENARIOS for e in ws.catalog[m].get(s, {}).values())
            log(f"[wind] {m}: no member has sfcWind+uas+vas; wind from uas/vas "
                + ("in ALL scenarios (consistent; absolute speeds biased low)" if uv_everywhere
                   else "in SOME scenarios only - inconsistent! consider dropping sfcWind entries"))
            continue
        path = os.path.join(cache_dir, f"wind_uv_ratio_{m}{ws.tag}.nc")
        if os.path.exists(path):
            r = xr.open_dataarray(path).load()
        else:
            s, mem = donor
            yrs = years if s == "ssp245" else ASSESS_PERIOD
            e = ws.catalog[m][s][mem]
            sp = load_variable(e["sfcWind"], "sfcWind", yrs, ws.bbox)
            uv = np.hypot(load_variable(e["uas"], "uas", yrs, ws.bbox), load_variable(e["vas"], "vas", yrs, ws.bbox))
            r = (monthly_climatology(sp) / monthly_climatology(uv).where(lambda x: x > 0.1)).clip(0.8, 3.0)
            r = r.fillna(1.0).astype("float32").compute().rename("wind_uv_ratio")
            r.attrs.update(donor=f"{m}/{s}/{mem} {yrs}", meaning="mean(sfcWind)/mean(hypot(uas,vas))")
            os.makedirs(cache_dir, exist_ok=True)
            r.to_netcdf(path)
        med = float(r.median())
        if not (0.95 <= med <= 2.0):
            log(f"[wind] WARNING {m}: sfcWind/|uv| ratio median {med:.2f} is physically implausible "
                f"(expected ~1.1-1.6). NOT applying a correction - run P.wind_diagnostics(ws, '{m}') "
                f"and check variables/units before trusting wind for this model.")
            continue
        WIND_UV_RATIO[m] = r
        log(f"[wind] {m}: uas/vas-derived wind will be scaled by sfcWind/|uv| "
            f"(median {med:.2f}; donor {r.attrs.get('donor')})")


def wind_diagnostics(ws, model, years=(2020, 2021)):
    """Print what the wind variables of a model really contain (units, names, magnitudes,
    day-to-day correlation between sfcWind and hypot(uas, vas)) for each scenario/member 1."""
    rows = []
    for s in SCENARIOS:
        mems = ws.catalog.get(model, {}).get(s, {})
        if not mems:
            continue
        mem = sorted(mems)[0]
        e = mems[mem]
        yrs = years if s == "ssp245" else (ASSESS_PERIOD[0], ASSESS_PERIOD[0] + 1)
        got = {}
        for v in ("sfcWind", "uas", "vas"):
            if v not in e:
                continue
            ds = normalize_time(harmonize_coords(open_entry(e[v], ALIASES[v] + [e[v].get("var", v)])))
            name = _find_var(ds, v, e[v].get("var"))
            da = _year_slice(ds[[name]], yrs)[name]
            if ws.bbox is not None:
                da = _subset_bbox(da.to_dataset(), ws.bbox)[name]
            got[v] = da.load()
            dt = np.diff(np.array([t.toordinal() for t in da.indexes["time"][:10]]))
            rows.append(dict(scenario=s, member=mem, var=v, file_var=name,
                             units=da.attrs.get("units"), long_name=da.attrs.get("long_name", "")[:40],
                             mean=float(da.mean()), mean_abs=float(abs(da).mean()), max=float(da.max()),
                             step_days=float(np.median(dt)) if len(dt) else np.nan,
                             n_days=da.sizes["time"], file=e[v]["paths"][0].split("/")[-1] if "paths" in e[v] else ""))
        if all(k in got for k in ("sfcWind", "uas", "vas")):
            uv = np.hypot(got["uas"], got["vas"])
            sp, uv = xr.align(got["sfcWind"], uv, join="inner")
            corr = float(xr.corr(sp.mean(("lat", "lon")), uv.mean(("lat", "lon"))))
            rows.append(dict(scenario=s, member=mem, var="sfcWind / hypot(uas,vas)",
                             mean=float(sp.mean() / uv.mean()), long_name=f"daily corr {corr:.2f}",
                             n_days=sp.sizes["time"]))
    df = pd.DataFrame(rows)
    with pd.option_context("display.width", 250, "display.max_columns", 20):
        print(df.to_string(index=False))
    return df


def input_sanity(ws, runs, members, variables=("tasmax", "tas", "hurs", "sfcWind", "pr"), ndays=365):
    """Land-mean of each staged input (first member, first `ndays`) per model/run, with
    plausibility flags. Catches unit/label problems before they reach FWI/VPD."""
    ranges = {"tasmax": (240, 320, "K"), "tas": (235, 315, "K"), "hurs": (20, 98, "%"),
              "sfcWind": (0.5, 15, "m/s"), "pr": (0.05, 20, "mm/d")}
    rows = []
    for m in members:
        for run, (scen, years) in runs.items():
            mems = members[m].get(scen) or []
            if not mems:
                continue
            ds = ws.inputs(m, scen, mems[0], (years[0] - SPINUP_YEARS, years[1]),
                           ["tasmax", "hursmin", "hurs", "sfcWind", "pr", "tas"])
            land = land_mask(ds.lon, ds.lat)
            w = np.cos(np.deg2rad(ds.lat))
            row = {"model": m, "run": run, "member": mems[0]}
            flags = []
            for v in variables:
                if v not in ds:
                    continue
                x = ds[v].isel(time=slice(0, ndays)).mean("time")
                if v == "pr":
                    x = x * 86400.0
                val = float(x.where(land).weighted(w).mean().compute())
                row[f"{v} [{ranges[v][2]}]"] = round(val, 2)
                lo, hi, _ = ranges[v]
                if not (lo <= val <= hi):
                    flags.append(v)
            row["check"] = "ok" if not flags else "IMPLAUSIBLE: " + ", ".join(flags)
            rows.append(row)
    return pd.DataFrame(rows)
