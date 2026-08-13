#!/usr/bin/env python
"""Extract the training loss curve from a StarVLA run log.

The trainer logs through rich, which soft-wraps each metrics dict across several
terminal lines, so a naive line-wise grep misses most points. This flattens the
log first, then pulls (step, loss) pairs.

Usage:
    python extract_loss_curve.py <run.log> [--csv out.csv]
"""
import argparse
import re
import statistics as st
from pathlib import Path


def extract(log_path: Path):
    flat = re.sub(r"\s+", " ", log_path.read_text(errors="ignore"))
    pairs = re.findall(r"Step (\d+), Loss.*?action_dit_loss': ([0-9.]+)", flat)
    return [(int(s), float(v)) for s, v in pairs]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log", type=Path)
    ap.add_argument("--csv", type=Path, default=None)
    ap.add_argument("--buckets", type=int, default=20)
    args = ap.parse_args()

    pts = extract(args.log)
    if not pts:
        raise SystemExit(f"no loss points found in {args.log}")

    steps = [s for s, _ in pts]
    losses = [v for _, v in pts]
    print(f"points={len(pts)}  steps {steps[0]}..{steps[-1]}")
    print(f"min={min(losses):.4f}  max={max(losses):.4f}")

    # bucketed means — a readable stand-in for a plot
    n = max(1, len(pts) // args.buckets)
    print(f"\n{'step range':>18}  {'mean loss':>9}")
    for i in range(0, len(pts), n):
        chunk = pts[i : i + n]
        lo, hi = chunk[0][0], chunk[-1][0]
        print(f"{lo:>8}-{hi:<8}  {st.mean(v for _, v in chunk):>9.4f}")

    if args.csv:
        args.csv.write_text("step,action_l1_loss\n" + "\n".join(f"{s},{v}" for s, v in pts))
        print(f"\nwrote {args.csv}")


if __name__ == "__main__":
    main()
