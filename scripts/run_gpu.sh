#!/usr/bin/env bash
# Runs the full GPU job; prints results as base64 tar between markers so they can be read from container logs.
set -uo pipefail
cd "$(dirname "$0")/.."
[ -d kb ] || bash scripts/fetch_solutions.sh kb
mkdir -p out
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv > out/nvidia_smi.txt
python gpu_run.py --sol kb/solutions --out out 2>&1 | tee out/run_log.txt
tar czf out.tgz out && echo "===RESULTS_B64_BEGIN===" && base64 -w 300 out.tgz && echo "===RESULTS_B64_END==="
