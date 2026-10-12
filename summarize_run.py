"""Print all text outputs and errors of an executed notebook (no images), cell by cell."""
import json, re, sys
nb = json.load(open(sys.argv[1]))
ansi = re.compile(r"\x1b\[[0-9;]*m")
for i, c in enumerate(nb["cells"]):
    if c["cell_type"] == "markdown":
        head = "".join(c["source"]).splitlines()
        if head and head[0].startswith("#"):
            print(f"\n======== {head[0]}")
        continue
    for o in c.get("outputs", []):
        t = o.get("output_type")
        if t == "stream":
            txt = "".join(o["text"])
        elif t in ("execute_result", "display_data"):
            d = o.get("data", {})
            txt = "[figure]" if "image/png" in d else "".join(d.get("text/plain", ""))
        elif t == "error":
            txt = "### ERROR " + o["ename"] + ": " + o["evalue"] + "\n" + ansi.sub("", "\n".join(o["traceback"]))[-4000:]
        else:
            continue
        lines = [l for l in txt.splitlines() if "warnings.warn(" not in l]
        if lines:
            print(f"--- cell {i}"); print("\n".join(lines)[:6000])
