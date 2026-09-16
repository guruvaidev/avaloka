#!/bin/bash
# Run every main suite in a safe order. Sources ./env.sh for endpoints/credentials
# (see README). The main suites contain NO resource-bomb items — those live in the
# separate `heavy` suite (run it only against a memory-limited k8s deploy). Ctrl+C-safe;
# re-running resumes from cache (state.json).
set -o pipefail
cd "$(dirname "$0")"
[ -f ./env.sh ] && source ./env.sh
TS=$(date +%m%d_%H%M); LOG=results/all_$TS.console.log
mkdir -p results

# v4 — functionality coverage (ingestion first primes datasets the rest reuse)
for b in F1 F2 F3 F4 F9 F5 F6 F7 F8 F10 F11 F12 F13; do
  python3 run_suite.py --suite v4 --batch $b 2>&1 | tee -a "$LOG"
done
# v2 — evidence-based per connection
for b in ieee-fraud nyc-taxi instacart-mba walmart tier-c-confirmations; do
  python3 run_suite.py --suite v2 --batch $b 2>&1 | tee -a "$LOG"
done
# v1 — legacy Kaggle arcs (dead-file items already pruned)
for b in ieee-fraud walmart-sales nyc-taxi-duration instacart; do
  python3 run_suite.py --suite v1 --batch $b 2>&1 | tee -a "$LOG"
done
echo "ALL SUITES DONE — console log: $LOG"
echo "NOTE: the 'heavy' suite (entire-dataset / wide-file items) is NOT run here —"
echo "run it only on a memory-limited/supervised deploy: python3 run_suite.py --suite heavy --list"
