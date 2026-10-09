#!/usr/bin/env bash
# Run notebooks in the background and collect everything Claude needs in runs/<stamp>_<notebook>/:
#   progress.txt (live), log.txt, <notebook>_executed.ipynb, summary.txt (all text outputs + errors), figures/
# Usage (from a Hub terminal, in this folder):
#   nohup bash run_headless.sh Task1_FWI_P95_Extremes Task2_VPD_Evaporative_Demand > runs.log 2>&1 &
# Then:  git add runs/ && git commit -m "run outputs" && git push
set -u
# only one run at a time: two runs side by side exhaust the server's 30 GB and it is stopped
if command -v flock >/dev/null 2>&1; then
  exec 9>.run_headless.lock
  if ! flock -n 9; then echo "[$(date +%H:%M)] another run_headless.sh is already running - not starting a second one"; exit 1; fi
fi
stamp=$(date +%Y%m%d_%H%M)
for nb in "$@"; do
  nb=${nb%.ipynb}
  d="runs/${stamp}_${nb}"; mkdir -p "$d"
  echo "[$(date +%H:%M)] running $nb -> $d"
  export PYROSAI_PROGRESS="$PWD/$d/progress.txt"      # live progress:  tail runs/*/progress.txt
  jupyter nbconvert --to notebook --execute --allow-errors --ExecutePreprocessor.timeout=-1 \
      "${nb}.ipynb" --output-dir "$d" --output "${nb}_executed.ipynb" > "$d/log.txt" 2>&1
  if [ -f "$d/${nb}_executed.ipynb" ]; then
    python summarize_run.py "$d/${nb}_executed.ipynb" > "$d/summary.txt" 2>&1
  else
    echo "### ERROR: no executed notebook - the run was interrupted (server stopped or process killed)" > "$d/summary.txt"
  fi
  outdir=$(python -c "import re,sys;s=open('local_config.py').read() if __import__('os').path.exists('local_config.py') else '';m=re.search(r'^OUT_DIR\s*=\s*[\"\\'](.+?)[\"\\']',s,re.M);print(m.group(1) if m else './pyrosai_output')")
  if [ -d "$outdir/figures" ]; then mkdir -p "$d/figures"; cp "$outdir"/figures/*.png "$outdir"/figures/*.pdf "$d/figures/" 2>/dev/null; fi
  for f in "$outdir"/*.csv; do [ -f "$f" ] && cp "$f" "$d/"; done
  # durable backup of the results folder (Hub home files can be lost): BACKUP_ROOT in local_config.py
  python - "$outdir" <<'PYEOF'
import os, re, sys
cfg = open("local_config.py").read() if os.path.exists("local_config.py") else ""
m = re.search(r'^BACKUP_ROOT\s*=\s*["\'](.+?)["\']', cfg, re.M)
if m and os.path.isdir(sys.argv[1]):
    import fsspec
    dest = m.group(1).rstrip("/") + "/" + os.path.basename(os.path.normpath(sys.argv[1]))
    fs, path = fsspec.core.url_to_fs(dest)
    fs.put(sys.argv[1].rstrip("/") + "/", path, recursive=True)
    print(f"backed up {sys.argv[1]} -> {dest}")
PYEOF
  echo "[$(date +%H:%M)] done $nb: $(grep -c '^### ERROR' "$d/summary.txt") error(s)"
done
