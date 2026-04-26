#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────
# Ablation runner — runs all 5 variants and prints a summary table.
#
# Usage:
#   ./run_ablations.sh           # runs on default round (3)
#   ./run_ablations.sh 3 0 1 2   # custom rounds/days if supported
#
# Assumes prosperity3bt is on PATH. If it's not, swap the command
# on the `BT_CMD` line below for your backtester invocation.
# ─────────────────────────────────────────────────────────────────

set -u
cd "$(dirname "$0")"

# EDIT THIS if your backtester invocation is different.
BT_CMD="prosperity3bt"
ROUND_ARG="${1:-3}"

FILES=(
  trader_abl_1_hp_only.py
  trader_abl_2_hp_vf.py
  trader_abl_3_hp_vev5400.py
  trader_abl_4_hp_vev5300gamma.py
  trader_abl_5_hp_otm.py
)

LABELS=(
  "HP only (baseline)"
  "HP + VF asymmetric"
  "HP + VEV_5400 long"
  "HP + VEV_5300 gamma"
  "HP + OTM tickets"
)

mkdir -p logs
rm -f logs/summary.txt

echo "Running 5 ablations on round $ROUND_ARG..."
echo

for i in "${!FILES[@]}"; do
  f="${FILES[$i]}"
  label="${LABELS[$i]}"
  log="logs/${f%.py}.log"
  echo "─── [$((i+1))/5] $label  ($f)"
  $BT_CMD "$f" "$ROUND_ARG" > "$log" 2>&1
  # Grab the last "Total profit" line (prosperity3bt prints this at the end).
  total=$(grep -iE "total.*profit|total profit" "$log" | tail -1)
  echo "    $total"
  echo "${label}|${f}|${total}" >> logs/summary.txt
done

echo
echo "═══ SUMMARY ═══"
column -t -s '|' logs/summary.txt
echo
echo "Full logs in ./logs/"
echo
echo "Compute deltas vs ablation #1 (HP only) to get each component's true contribution."
