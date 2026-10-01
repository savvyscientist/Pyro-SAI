#!/usr/bin/env bash
# Run notebooks in the background and collect everything Claude needs in runs/<stamp>_<notebook>/:
#   log.txt, <notebook>_executed.ipynb, summary.txt (all text outputs + errors), figures/
# Usage (from a Hub terminal, in this folder):
#   nohup bash run_headless.sh Task1_FWI_P95_Extremes Task2_VPD_Evaporative_Demand > runs.log 2>&1 &
# Then:  git add runs/ && git commit -m "run outputs" && git push
set -u
stamp=$(date +%Y%m%d_%H%M)
for nb in "$@"; do
  nb=${nb%.ipynb}
  d="runs/${stamp}_${nb}"; mkdir -p "$d"
  echo "[$(date +%H:%M)] running $nb -> $d"
  jupyter nbconvert --to notebook --execute --allow-errors --ExecutePreprocessor.timeout=-1 \
      "${nb}.ipynb" --output-dir "$d" --output "${nb}_executed.ipynb" > "$d/log.txt" 2>&1
  python summarize_run.py "$d/${nb}_executed.ipynb" > "$d/summary.txt" 2>&1
  outdir=$(python -c "import re,sys;s=open('local_config.py').read() if __import__('os').path.exists('local_config.py') else '';m=re.search(r'^OUT_DIR\s*=\s*[\"\\'](.+?)[\"\\']',s,re.M);print(m.group(1) if m else './pyrosai_output')")
  if [ -d "$outdir/figures" ]; then mkdir -p "$d/figures"; cp "$outdir"/figures/*.png "$d/figures/" 2>/dev/null; fi
  for f in "$outdir"/*.csv; do [ -f "$f" ] && cp "$f" "$d/"; done
  echo "[$(date +%H:%M)] done $nb: $(grep -c '^### ERROR' "$d/summary.txt") error(s)"
done
