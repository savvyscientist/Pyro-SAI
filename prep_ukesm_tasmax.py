"""Add UKESM1-1 daily tasmax from A. Duffey's daily_Tmaxmin files to the Pyro SAI catalog.

    python prep_ukesm_tasmax.py            # inspect, match members, copy tasmax, write catalog extra
    python prep_ukesm_tasmax.py --check    # inspect and match only (nothing written)

Source (Slack, A. Duffey 2026-10-02):
    s3://reflective-persistent-prod/alistairduffey/UKESM1-1/<G6-1.5K-SAI|SSP245>/daily_Tmaxmin/T_00N.nc
Each file holds three unnamed temperatures (temp, temp_1, temp_2; UM output converted with CONVSH)
for one ensemble member, numbered 001-003 rather than by CMIP member id. This script:
  1. identifies the daily maximum (highest mean of the three, with a clear gap to the others),
  2. finds which catalog member the file belongs to by comparing the file's mean temperature
     with each member's daily tas (same days; the right member matches weather day by day),
  3. copies tasmax to Zarr in your persistent bucket (only for members that lack tasmax;
     for members that already have one it just reports the agreement as a validation),
  4. writes pyrosai_catalog_extra.json, which pyrosai.load_catalog merges in automatically.
Each file (~6 GB) is downloaded to local disk (/tmp, or $PYROSAI_TMP) and memory-mapped, one at a
time; peak memory is ~3 GB (the tasmax copy).
"""
import contextlib
import json
import os
import shutil
import sys
import tempfile

import numpy as np
import s3fs
import xarray as xr

import pyrosai as P

CHECK_ONLY = "--check" in sys.argv
MODEL = "UKESM1-1"
SRC = "reflective-persistent-prod/alistairduffey/UKESM1-1"
SCEN_DIRS = {"G6-1.5K-SAI": "G6-1.5K-SAI", "ssp245": "SSP245"}
USER = os.environ.get("JUPYTERHUB_USER", "savvyscientist")
DEST = os.environ.get("PYROSAI_UKESM_DEST", f"s3://reflective-persistent-prod/{USER}/pyrosai_data/{MODEL}")
CATALOG = "pyrosai_catalog.json"
EXTRA = "pyrosai_catalog_extra.json"
NDAYS = 60
TMP = os.environ.get("PYROSAI_TMP")      # local scratch disk for one file at a time (default /tmp)

fs = s3fs.S3FileSystem()
cat = P.load_catalog(CATALOG)
extra = json.load(open(EXTRA)) if os.path.exists(EXTRA) else {}


def pick_tmp(size):
    for d in [TMP, "/tmp", os.path.expanduser("~")]:
        if d and os.path.isdir(d) and shutil.disk_usage(d).free > 1.15 * size:
            return d
    raise OSError(f"no local folder with {1.15 * size / 1e9:.1f} GB free; set PYROSAI_TMP")


@contextlib.contextmanager
def opened(f):
    """Download one file to local disk and open it memory-mapped, so only the slices used are
    read into memory (reading the ~6 GB files straight from S3 loads them whole and can
    exhaust the Hub server's memory)."""
    size = fs.info(f)["size"]
    d = tempfile.mkdtemp(dir=pick_tmp(size), prefix="ukesm_tasmax_")
    p = os.path.join(d, os.path.basename(f))
    try:
        print(f"    downloading {size / 1e9:.1f} GB to {d} ...")
        fs.get(f, p)
        ds = xr.open_dataset(p, engine="scipy")       # scipy backend memory-maps local files
        try:
            yield ds
        finally:
            ds.close()
    finally:
        shutil.rmtree(d, ignore_errors=True)


def field(da):
    """Small (time, lat, lon) slice on the pipeline's grid orientation."""
    ds = P.harmonize_coords(da.to_dataset(name="x"))
    return ds["x"].load()


def rms(a, b):
    return float(np.sqrt(((a.values - b.values) ** 2).mean()))


summary = []
for scen, sdir in SCEN_DIRS.items():
    files = sorted(fs.find(f"{SRC}/{sdir}/daily_Tmaxmin"))
    print(f"\n=== {scen}: {len(files)} file(s) in {SRC}/{sdir}/daily_Tmaxmin")
    for f in files:
        print(f"\n--- {f}")
        with opened(f) as ds:
            tvars = [v for v in ds.data_vars if ds[v].ndim >= 3]
            t = ds["t"]
            y0, y1 = int(t.dt.year[0]), int(t.dt.year[-1])
            print(f"    time {str(t.values[0])[:10]} -> {str(t.values[-1])[:10]} (n={t.size}, "
                  f"calendar {getattr(t.values[0], 'calendar', '?')})")
            for v in tvars:
                a = {k: ds[v].attrs[k] for k in ds[v].attrs if k in ("long_name", "units", "stash_item",
                                                                       "stash_section", "source", "name")}
                print(f"    {v}: {a}")
            # compare on the first NDAYS of a year both the file and the catalog cover
            yr = max(y0 + 1, 2020) if scen == "ssp245" else y0 + 1
            sel = t.dt.year == yr
            i0 = int(np.argmax(sel.values))     # first day of that year; slices only (the scipy
            sl = {v: field(ds[v].isel(t=slice(i0, i0 + NDAYS)).load()) for v in tvars}   # backend has no fancy indexing)
            means = {v: float(sl[v].mean()) for v in tvars}
            order = sorted(tvars, key=means.get)          # low -> high
            vmin, vmid, vmax = order[0], order[len(order) // 2], order[-1]
            print("    mean over first %d days of %d: " % (NDAYS, yr)
                  + ", ".join(f"{v}={means[v]:.2f}" for v in tvars))
            ok_order = len(order) == 3 and means[vmax] - means[vmid] > 0.5 and means[vmid] - means[vmin] > 0.5
            print(f"    -> max={vmax}, mean={vmid}, min={vmin}" + ("" if ok_order else "   [UNCLEAR ORDER]"))
            mid = sl[vmid]
            half = (sl[vmax] + sl[vmin]) / 2
            scores = {}
            for mem, e in sorted(cat.get(MODEL, {}).get(scen, {}).items()):
                if "tas" not in e:
                    continue
                try:
                    tas = P.load_variable(e["tas"], "tas", (yr, yr)).isel(time=slice(0, NDAYS)).load()
                    tas = tas.assign_coords(lat=mid.lat, lon=mid.lon)
                    scores[mem] = min(rms(mid, tas), rms(half, tas))
                except Exception as ex:
                    print(f"    {mem}: tas comparison failed ({type(ex).__name__}: {ex})")
            if not scores:
                summary.append((scen, f, "no catalog tas to compare", "-")); continue
            print("    RMS vs catalog tas (K): " + ", ".join(f"{m}={s:.2f}" for m, s in scores.items()))
            ranked = sorted(scores, key=scores.get)
            best = ranked[0]
            clear = len(ranked) == 1 or scores[best] < 0.6 * scores[ranked[1]]
            if not (clear and ok_order):
                print("    -> NOT USED (member match or variable order unclear)")
                summary.append((scen, f, "unclear", "-")); continue
            print(f"    -> member {best}")
            if "tasmax" in cat[MODEL][scen][best] and cat[MODEL][scen][best]["tasmax"].get("kind") != "zarr":
                try:
                    tm = P.load_variable(cat[MODEL][scen][best]["tasmax"], "tasmax", (yr, yr)).isel(
                        time=slice(0, NDAYS)).load().assign_coords(lat=mid.lat, lon=mid.lon)
                    print(f"    validation: RMS(file max, catalog tasmax) = {rms(sl[vmax], tm):.3f} K")
                except Exception as ex:
                    print(f"    validation failed ({type(ex).__name__})")
                summary.append((scen, f, best, "already has tasmax (validated, not copied)")); continue
            dest = f"{DEST}/{scen}/{best}_tasmax_{y0}-{y1}.zarr"
            if CHECK_ONLY:
                summary.append((scen, f, best, f"would copy {vmax} -> {dest}")); continue
            if not P.store_exists(dest):
                da = ds[vmax].isel(ht=0, drop=True) if "ht" in ds[vmax].dims else ds[vmax]
                da = da.rename({"t": "time", "latitude": "lat", "longitude": "lon"}).load().astype("float32")
                da.attrs = {"units": "K", "standard_name": "air_temperature",
                            "long_name": "daily maximum near-surface air temperature",
                            "source": f"s3://{f} variable {vmax}", "member_match": f"RMS vs tas {scores[best]:.2f} K"}
                print(f"    copying {vmax} -> {dest} ...")
                P.write_zarr(da.to_dataset(name="tasmax"), dest)
            extra.setdefault(MODEL, {}).setdefault(scen, {}).setdefault(best, {})["tasmax"] = {
                "kind": "zarr", "store": dest, "var": "tasmax", "source": f"s3://{f}:{vmax}"}
            summary.append((scen, f, best, f"copied -> {dest}"))

print("\n=== summary")
for r in summary:
    print("  ", " | ".join(str(x) for x in r))

if not CHECK_ONLY:
    with open(EXTRA, "w") as fh:
        json.dump(extra, fh, indent=1)
    print(f"\nwrote {EXTRA}")
    # earlier UKESM metric caches were computed with tas: record that, so Task 1 refreshes them
    for d in sorted(os.listdir(".")):
        if d.startswith("pyrosai_output"):
            for md in sorted(os.listdir(d)):
                p = os.path.join(d, md, MODEL)
                if md.startswith("metrics") and os.path.isdir(p) and not os.path.exists(os.path.join(p, "fwi_temp.txt")):
                    open(os.path.join(p, "fwi_temp.txt"), "w").write("tas")
                    print(f"marked {p} as computed with tas")
