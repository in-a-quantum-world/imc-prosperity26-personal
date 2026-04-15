"""
Parameter sweep for cfd_technique.py on Kevin Fu's backtester.
Tests all combinations of RE_THRESHOLD, DRIFT_EXTRAP, LAM_MAKE_WIDTH, LAM_DRIFT_LEAN.
"""
import subprocess, re, itertools, sys, os, tempfile, shutil
from pathlib import Path

BASE_FILE  = Path("C:/quant-hacks/imc-prosperity26/training-round/cfd_technique.py")
BT_DIR     = Path("C:/quant-hacks/imc-prosperity26/backtester/imc-prosperity-4-backtester")
BASE_SRC   = BASE_FILE.read_text(encoding="utf-8")

GRID = {
    "RE_THRESHOLD":  [0.3, 0.5, 0.7, 1.0],
    "DRIFT_EXTRAP":  [3, 5, 8],
    "LAM_MAKE_WIDTH": [2, 3, 4],
    "LAM_DRIFT_LEAN": [0.3, 0.5, 0.8],
}

def patch(src, params):
    for k, v in params.items():
        # Replace class-level assignments like:  RE_THRESHOLD = 0.5
        src = re.sub(
            rf"(\s{k}\s*=\s*)[\d.]+",
            lambda m: m.group(1) + str(v),
            src,
        )
    return src

def run_bt(tmp_file):
    result = subprocess.run(
        [sys.executable, "-m", "prosperity4bt", str(tmp_file), "0",
         "--match-trades", "worse"],
        cwd=str(BT_DIR),
        capture_output=True, text=True, timeout=60,
    )
    out = result.stdout + result.stderr
    # Parse total profit — take the LAST occurrence (combined summary, not per-day)
    matches = re.findall(r"Total profit:\s*([\d,]+)", out)
    if not matches:
        return None, out
    profit = int(matches[-1].replace(",", ""))
    # Parse per-day
    days = re.findall(r"Round 0 day (-\d+):\s*([\d,]+)", out)
    return profit, {f"day{d}": int(p.replace(",", "")) for d, p in days}

results = []
combos = list(itertools.product(*GRID.values()))
keys   = list(GRID.keys())
total  = len(combos)

print(f"Running {total} combinations...\n")

for i, vals in enumerate(combos, 1):
    params = dict(zip(keys, vals))
    patched = patch(BASE_SRC, params)

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", prefix="cfd_sweep_",
        dir=BASE_FILE.parent,   # same dir as original so datamodel.py is findable
        delete=False, encoding="utf-8"
    ) as f:
        f.write(patched)
        tmp_path = Path(f.name)

    try:
        profit, details = run_bt(tmp_path)
    finally:
        tmp_path.unlink(missing_ok=True)

    if profit is None:
        print(f"  [{i:3d}/{total}] ERROR  {params}")
        continue

    results.append((profit, params, details))
    if i % 10 == 0 or i == total:
        best_so_far = max(results, key=lambda x: x[0])[0]
        print(f"  [{i:3d}/{total}]  profit={profit:,}  best_so_far={best_so_far:,}  {params}")

# ── Sort and print top 20 ────────────────────────────────────────────────────
results.sort(key=lambda x: x[0], reverse=True)

print("\n" + "="*80)
print(f"{'Rank':>4}  {'Total':>8}  {'RE':>5}  {'Extrap':>6}  {'Width':>5}  {'Lean':>5}  Days")
print("="*80)
for rank, (profit, params, details) in enumerate(results[:20], 1):
    day_str = "  ".join(f"d{k}={v:,}" for k, v in details.items())
    print(
        f"{rank:4d}  {profit:8,}  "
        f"{params['RE_THRESHOLD']:5.1f}  "
        f"{params['DRIFT_EXTRAP']:6d}  "
        f"{params['LAM_MAKE_WIDTH']:5d}  "
        f"{params['LAM_DRIFT_LEAN']:5.1f}  "
        f"{day_str}"
    )

print("\nBaseline (original cfd_technique): 20,375")
print("Baseline (trader_mm_v2):           29,536")

# Save full results to CSV
import csv
out_csv = Path("C:/quant-hacks/imc-prosperity26/backtester/sweep_cfd_results.csv")
with open(out_csv, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["rank", "total_profit", "RE_THRESHOLD", "DRIFT_EXTRAP",
                "LAM_MAKE_WIDTH", "LAM_DRIFT_LEAN"])
    for rank, (profit, params, _) in enumerate(results, 1):
        w.writerow([rank, profit,
                    params["RE_THRESHOLD"], params["DRIFT_EXTRAP"],
                    params["LAM_MAKE_WIDTH"], params["LAM_DRIFT_LEAN"]])
print(f"\nFull results saved to: {out_csv}")
