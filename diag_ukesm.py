"""Print the grid of every UKESM input variable (headers only) for one member per scenario."""
import pyrosai as P

cat = P.apply_known_issues(P.load_catalog("pyrosai_catalog.json"))
for scen, mem in [("ssp245", "r12i1p1f1"), ("G6-1.5K-SAI", "r12i1p1f2"), ("G6-1.5K-HiLLA", "r12i1p1f2")]:
    e = cat["UKESM1-1"][scen][mem]
    print(f"=== {scen} {mem}", flush=True)
    for v in ("tasmax", "tas", "hurs", "uas", "vas", "pr"):
        if v not in e:
            continue
        try:
            raw = P.open_entry(e[v], P.ALIASES[v] + [e[v].get("var", v)])
            ds = P.harmonize_coords(raw)
            print(v, "| raw dims:", dict(raw.sizes), "| vars:", list(raw.data_vars),
                  "| raw coords:", list(raw.coords), flush=True)
            for c in ("lat", "lon"):
                if c in ds.coords:
                    x = ds[c].values
                    print("     ", c, ds[c].dims, x.shape, "first", x.ravel()[:2], "last", x.ravel()[-2:], flush=True)
                else:
                    print("     ", c, "MISSING; coords:", list(ds.coords), flush=True)
        except Exception as ex:
            print(v, "ERROR", type(ex).__name__, ex, flush=True)
