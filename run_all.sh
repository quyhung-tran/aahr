#!/usr/bin/env bash
# Reproduces every experiment in Section VII of the paper.
#   usage: bash run_all.sh [path/to/ML-EdgeIIoT-dataset.csv] [output_dir]
#   defaults: data/ML-EdgeIIoT-dataset.csv  results
# Requires the Edge-IIoTset CSV (see README.md, "Dataset") and the packages in
# requirements.txt. Every run is seeded; compare against reference_results/
# with verify_results.py afterwards.
set -euo pipefail
cd "$(dirname "$0")"

CSV="${1:-data/ML-EdgeIIoT-dataset.csv}"
OUT="${2:-results}"

if [ ! -f "$CSV" ]; then
  echo "Edge-IIoTset file not found at $CSV (see README.md, 'Dataset')." >&2
  exit 1
fi
mkdir -p "$OUT"

echo "== 1/3 Edge-IIoTset and synthetic experiments (Section VII-A, Table III) =="
python aahr_verification.py --csv "$CSV" --seeds 30 --out "$OUT/aahr_results.json" \
  2>&1 | tee "$OUT/aahr_run.log"

echo "== 2/3 Network simulation (Section VII-B, Table IV) =="
python aahr_netsim.py --seeds 30 --out "$OUT/netsim_results.json" \
  2>&1 | tee "$OUT/netsim_run.log"

echo "== 3/3 Paper tables and quoted numbers =="
python make_paper_tables.py "$OUT" 2>&1 | tee "$OUT/paper_numbers.txt"

echo
echo "Done. Compare with the reference results:  python verify_results.py $OUT reference_results"
