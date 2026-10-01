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


def log(msg):
    print(msg, flush=True)


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
MON_TOKENS = {"Amon", "AMON", "Mon", "mon", "h0", "monthly", "Emon", "Lmon"}
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
                y0=y0, y1=y1)


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
    keep = set(var_candidates) | {"time_bnds", "time_bounds", "time_bnd"}

    def pre(ds):
        dv = [v for v in ds.data_vars if v in keep]
        ds = ds[dv]
        if "t" in ds.dims and "time" not in ds.dims:
            ds = ds.rename(t="time")
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


def _sample(da, n=5):
    return da.isel(time=slice(0, n)).values


def to_target_units(da, canon):
    u = str(da.attrs.get("units", "")).strip()
    ul = u.lower().replace(" ", "")
    s = _sample(da)
    med = float(np.nanmedian(s))
    if canon in ("tas", "tasmax"):
        if ul in ("c", "degc", "celsius", "°c", "deg_c") or (ul == "" and med < 150):
            da = da + 273.15
    elif canon in ("hurs", "hursmin"):
        if ul in ("1", "fraction", "0-1") or float(np.nanmax(s)) <= 1.5:
            da = da * 100.0
        da = da.clip(0, 100)
    elif canon == "pr":
        if ul in ("m/s", "ms-1", "m.s-1", "m/sec", "ms**-1"):
            da = da * 1000.0
        elif ul in ("mm/day", "mmd-1", "mm/d", "mmday-1", "kgm-2d-1"):
            da = da / 86400.0
        elif ul in ("kgm-2s-1", "kg/m2/s", "kgm**-2s**-1", "kg/m^2/s", "mm/s", "mms-1"):
            pass
        else:
            warnings.warn(f"pr units '{u}' not recognised; assuming kg m-2 s-1")
        da = da.clip(min=0)
        m = float(np.nanmean(_sample(da)))
        if m > 1e-2:
            warnings.warn(f"pr mean {m:.3g} kg m-2 s-1 looks too large - check units ('{u}')")
    elif canon in ("sfcWind", "uas", "vas"):
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


def load_inputs(catalog, model, scenario, member, variables, years, bbox=None,
                optional=("hursmin", "tasmax", "tas")):
    """Return Dataset of canonical daily variables for one model/scenario/member."""
    entries = catalog[model][scenario][member]
    label = f"{model}/{scenario}/{member}: "
    out = {}
    for v in variables:
        if v in entries:
            out[v] = load_variable(entries[v], v, years, bbox, label)
        elif v == "sfcWind" and "uas" in entries and "vas" in entries:
            u = load_variable(entries["uas"], "uas", years, bbox, label)
            w = load_variable(entries["vas"], "vas", years, bbox, label)
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
    ds = xr.merge(list(out.values()), join="inner", compat="override")
    ds.attrs = {}
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
def compute_fwi(ds, temp="tasmax", rh="hursmin", rh_fallback="hurs", spinup_years=SPINUP_YEARS,
                keep=("FWI", "ISI", "BUI", "DC", "DMC", "FFMC"), space_chunk=48, season_method=None):
    """Canadian FWI system (xclim) from daily model output.

    Inputs used ("noon-equivalent" proxies, standard for daily GCM output):
      temperature: daily max (tasmax), humidity: daily min RH if available, else daily mean,
      wind: daily mean 10 m speed, precipitation: daily total.
    The first `spinup_years` are computed (moisture codes spin up from default start values)
    and dropped from the output.
    """
    import xclim
    from xclim.indices import cffwis_indices

    rhv = rh if rh in ds else rh_fallback
    tv = temp if temp in ds else "tas"
    if tv != temp:
        warnings.warn(f"{temp} not available; FWI uses {tv}")
    ds = ds.chunk({"time": -1, "lat": space_chunk, "lon": space_chunk})
    # explicit mm/day avoids flux->rate conversion differences between xclim versions
    pr_mmd = (ds["pr"] * 86400.0).assign_attrs(units="mm/d", standard_name="precipitation_amount")
    with xclim.set_options(data_validation="log", cf_compliance="log"):
        out = cffwis_indices(tas=ds[tv], pr=pr_mmd, sfcWind=ds["sfcWind"], hurs=ds[rhv],
                             lat=ds["lat"], season_method=season_method)
    res = {}
    for k in keep:
        v = getattr(out, k).astype("float32").transpose("time", "lat", "lon")
        v.attrs = {"units": "1", "long_name": f"Canadian FWI system: {k}"}
        res[k] = v
    res = xr.Dataset(res)
    y_first = int(ds.time.dt.year.values[0])
    res = res.isel(time=(res.time.dt.year >= y_first + spinup_years).values)
    res.attrs.update(
        fwi_temperature=tv, fwi_humidity=rhv, fwi_wind="sfcWind (daily mean)",
        fwi_precip="pr (daily total)", xclim_version=xclim.__version__,
        spinup_years_dropped=spinup_years, season_method=str(season_method),
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
    """Bilinear interpolation to a global res x res grid (cell centres), periodic in lon."""
    lat = np.arange(-90 + res / 2, 90, res)
    lon = np.arange(-180 + res / 2, 180, res)
    dt = da.dtype
    if dt == bool:
        da = da.astype("float32")
    left = da.isel(lon=slice(-2, None)).assign_coords(lon=lambda d: d.lon - 360)
    right = da.isel(lon=slice(0, 2)).assign_coords(lon=lambda d: d.lon + 360)
    ext = xr.concat([left, da, right], "lon")
    out = ext.interp(lat=lat, lon=lon, method="linear", kwargs={"fill_value": None})
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
        return obj.assign_coords(lat=lat, lon=lon)

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

    def inputs(self, model, scen, member, years, variables):
        p = self._path("inputs", model, scen, member, years)
        if self.stage:
            if not store_exists(p, self.so):
                log(f"  staging inputs -> {p}")
                ds = load_inputs(self.catalog, model, scen, member, variables, years, self.bbox)
                write_zarr(ds, p, self.sc, storage_options=self.so)
            ds = self.snap(model, open_store(p, self.sc, self.so))
            entries = self.catalog[model][scen][member]
            missing = [v for v in variables if v not in ds and _has(entries, v)]
            if missing:
                raise KeyError(f"staged store {p} lacks {missing}; delete it to re-stage")
            return ds
        return self.snap(model, load_inputs(self.catalog, model, scen, member, variables, years, self.bbox))

    def fwi(self, model, scen, member, years, input_vars, spinup_years=SPINUP_YEARS,
            keep=("FWI",), **fwi_kw):
        """Daily FWI for `years` (inputs start `spinup_years` earlier)."""
        p = self._path(f"fwi_{fwi_kw.get('temp', 'tasmax')}", model, scen, member, years)
        if not store_exists(p, self.so):
            yrs_in = (years[0] - spinup_years, years[1])
            ds = self.inputs(model, scen, member, yrs_in, input_vars)
            log(f"  computing FWI {model}/{scen}/{member} {years[0]}-{years[1]}")
            f = compute_fwi(ds, spinup_years=spinup_years, keep=keep, space_chunk=self.sc, **fwi_kw)
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
    (1, ("Cooling + wetting: consistent mitigation", "#2166ac")),
    (2, ("Risk offset: cooling outweighs drying (VPD↓, P↓, FWI↓)", "#2a9d8f")),
    (3, ("Fire-weather paradox: drying outweighs cooling (VPD↓, P↓, FWI↑)", "#e66101")),
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


def choose_fwi_temp(ws, model, rh="hurs", override=None):
    """'tasmax' if every scenario has it (>=1 member with all FWI inputs), else 'tas'.
    Using the same variable in every scenario keeps the comparison consistent."""
    if override:
        return override
    if all(ws.members(model, s, ["tasmax", rh, "sfcWind", "pr"]) for s in SCENARIOS):
        return "tasmax"
    return "tas"


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
        WIND_UV_RATIO[m] = r
        log(f"[wind] {m}: uas/vas-derived wind will be scaled by sfcWind/|uv| "
            f"(median {float(r.median()):.2f}; donor {r.attrs.get('donor')})")
