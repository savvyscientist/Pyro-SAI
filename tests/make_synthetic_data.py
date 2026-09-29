"""
Generate small synthetic daily datasets that mimic the layout, naming, units, calendars
and time-stamping conventions of the G6-1.5K archive, to test the Pyro SAI pipeline
end-to-end without cloud access. NOT real model output.

Model A ("CESM2-WACCM"-like): CAM names (TREFHTMX, TREFHT, RHREFHT, U10, PRECT[m/s]),
    noleap calendar, daily means stamped at END of day + time_bnds, lon 0..360,
    5-year files, members r1-r3.
Model B ("UKESM1-1"-like): CMOR names (tasmax, tas, hurs, sfcWind, pr[kg m-2 s-1]),
    360_day calendar, 'latitude'/'longitude' dims, descending latitude, no bounds,
    stamps at 12:00, members r1-r2.
"""
import json
import os
import sys

import cftime
import numpy as np
import xarray as xr

ROOT = sys.argv[1] if len(sys.argv) > 1 else "./synthetic_data"
rng = np.random.default_rng(42)

LAT = np.arange(-85, 90, 10.0)     # 18
LON = np.arange(0, 360, 15.0)      # 24


def blocks(y0, y1, n=5):
    out, y = [], y0
    while y <= y1:
        out.append((y, min(y + n - 1, y1)))
        y += n
    return out


def fields(times, scen, member, lat, lon):
    """Return dict of daily fields (time, lat, lon) in SI-ish units."""
    nt = len(times)
    yr = np.array([t.year for t in times], float)
    doy = np.array([t.dayofyr for t in times], float)
    ndays = 360.0 if times[0].calendar == "360_day" else 365.0
    la, lo = np.meshgrid(lat, lon, indexing="ij")
    season = np.cos(2 * np.pi * (doy[:, None, None] - 200) / ndays) * np.sign(la + 1e-6)[None]
    base_t = 300 - 0.55 * np.abs(la)[None] ** 1.15 / 1.5
    warm = np.clip(yr - 2020, 0, None)[:, None, None] * 0.03          # SSP2-4.5 warming
    if scen != "ssp245":
        warm = warm - np.clip(yr - 2035, 0, None)[:, None, None] * 0.03 * (1 + 0.2 * np.cos(np.deg2rad(la)))[None]
    r = np.random.default_rng(abs(hash((scen, member, int(yr[0])))) % 2**32)
    noise = np.zeros((nt,) + la.shape)
    e = r.normal(0, 2.0, noise.shape)
    for i in range(1, nt):
        noise[i] = 0.7 * noise[i - 1] + e[i]
    tas = base_t + 12 * season * (np.abs(la)[None] / 90) + warm + noise
    tasmax = tas + 6 + 2 * (np.abs(la)[None] < 45)
    # precipitation: drier subtropics, wet tropics; SAI dries tropics, HiLLA dries high-lat
    wet = 3.0 + 6 * np.exp(-(la / 12) ** 2) - 2.0 * np.exp(-((np.abs(la) - 28) / 8) ** 2)
    fac = np.ones_like(la)
    if scen == "G6-1.5K-SAI":
        fac = 1 - 0.18 * np.exp(-(la / 15) ** 2) - 0.05
    elif scen == "G6-1.5K-HiLLA":
        fac = 1 - 0.12 * (np.abs(la) > 50) + 0.04 * np.exp(-(la / 15) ** 2)
    ramp = np.clip((yr - 2035) / 10, 0, 1)[:, None, None] if scen != "ssp245" else 0
    fac_t = 1 + (fac[None] - 1) * ramp
    wetday = r.random(noise.shape) < 0.45
    pr_mm = np.where(wetday, r.gamma(0.8, wet[None] * fac_t / 0.8 / 0.45, noise.shape), 0.02)
    rh = np.clip(70 - 25 * np.exp(-((np.abs(la) - 25) / 10) ** 2)[None] - 1.2 * noise
                 + 8 * wetday - (warm * 1.5) + r.normal(0, 5, noise.shape), 5, 100)
    wind = np.abs(3.5 + 1.5 * (np.abs(la) > 40)[None] + r.normal(0, 1.5, noise.shape))
    return dict(tasmax=tasmax, tas=tas, hurs=rh, sfcWind=wind, pr=pr_mm)


def write_model_a(root):
    names = dict(tasmax="TREFHTMX", tas="TREFHT", hurs="RHREFHT", sfcWind="U10", pr="PRECT")
    units = dict(tasmax="K", tas="K", hurs="percent", sfcWind="m/s", pr="m/s")
    runs = {"ssp245": ("SSP245", "b.e21.BWSSP245cmip6.f09_g17.CMIP6-SSP2-4.5-WACCM", [(2019, 2039), (2062, 2084)]),
            "G6-1.5K-SAI": ("G6-1.5K-SAI", "b.e21.BW.f09_g17.SSP245-G6-1p5K-SAI", [(2062, 2084)]),
            "G6-1.5K-HiLLA": ("G6-1.5k-HiLLA", "b.e21.BW.f09_g17.SSP245-G6-1p5K-HiLLA", [(2062, 2084)])}
    for scen, (folder, case, spans) in runs.items():
        for m in (1, 2, 3):
            d = os.path.join(root, "CESM2-WACCM", folder, f"r{m}", "DAY")
            os.makedirs(d, exist_ok=True)
            for s0, s1 in spans:
                for y0, y1 in blocks(s0, s1):
                    days = xr.date_range(f"{y0}-01-01", f"{y1}-12-31", freq="D", calendar="noleap", use_cftime=True)
                    f = fields(list(days), scen, m, LAT, LON)
                    stamp = np.array([t + np.timedelta64(1, "D").astype("timedelta64[s]").item() for t in days])
                    bnds = np.stack([np.array(list(days)), stamp], axis=1)
                    for v, cam in names.items():
                        data = f[v] / 86400.0 / 1000.0 if v == "pr" else f[v]
                        ds = xr.Dataset({cam: (("time", "lat", "lon"), data.astype("float32"), {"units": units[v]}),
                                         "time_bnds": (("time", "nbnd"), bnds)},
                                        coords={"time": ("time", stamp, {"bounds": "time_bnds"}), "lat": LAT, "lon": LON})
                        enc = {"time": {"units": "days since 2015-01-01", "calendar": "noleap"},
                               "time_bnds": {"units": "days since 2015-01-01", "calendar": "noleap"}}
                        fn = f"{case}.00{m}.cam.h1.{cam}.{y0}0101-{y1}1231.nc"
                        ds.to_netcdf(os.path.join(d, fn), encoding=enc)
                    # also a monthly file that discovery must ignore
            os.makedirs(os.path.join(root, "CESM2-WACCM", folder, f"r{m}", "AMON"), exist_ok=True)


def write_model_b(root):
    units = dict(tasmax="K", tas="K", hurs="%", sfcWind="m s-1", pr="kg m-2 s-1")
    runs = {"ssp245": ("SSP245", [(2019, 2039), (2062, 2084)]),
            "G6-1.5K-SAI": ("G6-1.5K-SAI", [(2062, 2084)]),
            "G6-1.5K-HiLLA": ("G6-1p5K-HiLLA", [(2062, 2084)])}
    lat_desc = LAT[::-1]
    for scen, (folder, spans) in runs.items():
        for m in (1, 2):
            for v in units:
                d = os.path.join(root, "UKESM1-1", folder, f"r{m}i1p1f2", "day", v)
                os.makedirs(d, exist_ok=True)
            for s0, s1 in spans:
                for y0, y1 in blocks(s0, s1, 10):
                    days = xr.date_range(f"{y0}-01-01", f"{y1}-12-30", freq="D", calendar="360_day", use_cftime=True)
                    days12 = [cftime.Datetime360Day(t.year, t.month, t.day, 12) for t in days]
                    f = fields(list(days), scen, m + 10, lat_desc, LON)
                    for v in units:
                        data = f[v] / 86400.0 if v == "pr" else f[v]
                        ds = xr.Dataset({v: (("time", "latitude", "longitude"), data.astype("float32"), {"units": units[v]})},
                                        coords={"time": days12, "latitude": lat_desc, "longitude": LON})
                        fn = f"{v}_day_UKESM1-1_{scen}_r{m}i1p1f2_gn_{y0}0101-{y1}1230.nc"
                        ds.to_netcdf(os.path.join(root, "UKESM1-1", folder, f"r{m}i1p1f2", "day", v, fn),
                                     encoding={"time": {"units": "days since 1850-01-01", "calendar": "360_day"}})


if __name__ == "__main__":
    os.makedirs(ROOT, exist_ok=True)
    write_model_a(ROOT)
    write_model_b(ROOT)
    print("synthetic data written to", ROOT)
