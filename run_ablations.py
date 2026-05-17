"""
Ablation runner: invokes train_runner.py --eval-only in sequence for every
filter/σ/TTA combination, parses the summary printed at the end of each run,
and writes a single CSV + Markdown table comparing all configurations.

Usage:
    python run_ablations.py --ckpt path/to/best.pth
    python run_ablations.py --ckpt path/to/best.pth --skip-completed
    python run_ablations.py --ckpt path/to/best.pth --only eskf complementary

Per-experiment log files land in ablation_logs/<run_name>.txt.
Final table lands in ablation_logs/ablation_results.{csv,md}.

Each global-eval pass takes ~50–60 min (1200 scans). With TTA enabled that
roughly doubles. Plan for ~8–10 hours wall-clock for the full sweep.
"""

import argparse
import csv
import os
import re
import subprocess
import sys
from datetime import datetime


# Each entry: (display_label, run_name_tag, [extra args appended to train_runner.py invocation])
EXPERIMENTS = [
    # --- ESKF σ sweep ---
    ("ESKF σ=0.30 (baseline)",   "ablation_eskf_sigma030",      ["--filter", "eskf", "--sigma", "0.30"]),
    ("ESKF σ=0.25",              "ablation_eskf_sigma025",      ["--filter", "eskf", "--sigma", "0.25"]),
    ("ESKF σ=0.15",              "ablation_eskf_sigma015",      ["--filter", "eskf", "--sigma", "0.15"]),
    ("ESKF σ=0.10",              "ablation_eskf_sigma010",      ["--filter", "eskf", "--sigma", "0.10"]),
    # --- ESKF + TTA, using the σ that worked best in prior experimentation; rerun the σ sweep first if needed ---
    ("ESKF σ=0.15 + TTA",        "ablation_eskf_sigma015_tta",  ["--filter", "eskf", "--sigma", "0.15", "--tta"]),
    # --- Complementary filter sweep ---
    ("Complementary α=0.1",      "ablation_comp_a010",          ["--filter", "complementary", "--alpha", "0.1"]),
    ("Complementary α=0.2",      "ablation_comp_a020",          ["--filter", "complementary", "--alpha", "0.2"]),
    ("Complementary α=0.3",      "ablation_comp_a030",          ["--filter", "complementary", "--alpha", "0.3"]),
    # --- Visual only (no filter) ---
    ("Visual only",              "ablation_visual_only",        ["--filter", "none"]),
    ("Visual only + TTA",        "ablation_visual_only_tta",    ["--filter", "none", "--tta"]),
    # --- Extended ablation: filter + TTA combinations and smoothing windows ---
    ("Complementary α=0.3 + TTA",         "ablation_comp_a030_tta",      ["--filter", "complementary", "--alpha", "0.3", "--tta"]),
    ("ESKF σ=0.05 + TTA",                 "ablation_eskf_sigma005_tta",  ["--filter", "eskf", "--sigma", "0.05", "--tta"]),
    ("Visual + TTA + smooth=11",          "ablation_visual_tta_sm11",    ["--filter", "none", "--tta", "--smooth", "11"]),
    ("Complementary α=0.3 + TTA + sm=7",  "ablation_comp_a030_tta_sm7",  ["--filter", "complementary", "--alpha", "0.3", "--tta", "--smooth", "7"]),
]


def parse_summary(log_path):
    """Pull metrics out of the final summary block in a run log."""
    if not os.path.exists(log_path):
        return None
    with open(log_path, "r", encoding="utf-8", errors="replace") as f:
        text = f.read()

    def grab(pattern, cast=str):
        m = re.search(pattern, text)
        if not m:
            return None
        try:
            return cast(m.group(1))
        except (ValueError, TypeError):
            return m.group(1)

    return {
        "val_mae":      grab(r"Validation MAE:\s+([\d.]+) mm", float),
        "val_drift":    grab(r"Validation Drift:\s+([\d.]+) mm", float),
        "val_fdr":      grab(r"Validation FDR:\s+([\d.]+) %", float),
        "global_mae":   grab(r"Average MAE:\s+([\d.]+) mm", float),
        "global_drift": grab(r"Average Final Drift:\s+([\d.]+) mm", float),
        "global_fdr":   grab(r"Average Drift Rate:\s+([\d.]+) %", float),
        "valid_scans":  grab(r"Total valid scans:\s+(\d+)", int),
        "filter_label": grab(r"IMU Verifier:\s+(.+)"),
    }


def fmt(value, suffix=""):
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.2f}{suffix}"
    return f"{value}{suffix}"


def run_one(label, run_name, extra_args, ckpt, logs_dir, skip_completed=False, dry_run=False):
    log_path = os.path.join(logs_dir, f"{run_name}.txt")
    if skip_completed and os.path.exists(log_path) and parse_summary(log_path):
        print(f"  SKIP (already has summary): {log_path}")
        return True

    cmd = [
        sys.executable, "-u", "train_runner.py",
        "--eval-only", ckpt,
        "--name", run_name,
        *extra_args,
    ]
    print(f"  cmd: {' '.join(cmd)}")
    print(f"  log: {log_path}")

    if dry_run:
        return True

    start = datetime.now()
    with open(log_path, "w", encoding="utf-8") as f:
        ret = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
    elapsed_min = (datetime.now() - start).total_seconds() / 60.0

    if ret.returncode == 0:
        print(f"  ✓ done in {elapsed_min:.1f} min")
        return True
    print(f"  ✗ FAILED (exit {ret.returncode}, {elapsed_min:.1f} min) — see {log_path}")
    return False


def write_tables(results, logs_dir):
    """Emit ablation_results.csv and ablation_results.md in logs_dir."""
    csv_path = os.path.join(logs_dir, "ablation_results.csv")
    md_path  = os.path.join(logs_dir, "ablation_results.md")

    cols = ["label", "run_name",
            "val_mae", "val_drift", "val_fdr",
            "global_mae", "global_drift", "global_fdr",
            "valid_scans", "filter_label"]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in results:
            w.writerow({c: r.get(c) for c in cols})
    print(f"\nWrote {csv_path}")

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Ablation results\n\n")
        f.write("| Config | Val MAE (mm) | Val Drift (mm) | Val FDR (%) | "
                "Global MAE (mm) | Global Drift (mm) | Global FDR (%) | Valid scans |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|---:|\n")
        for r in results:
            f.write(f"| {r['label']} | {fmt(r.get('val_mae'))} | {fmt(r.get('val_drift'))} | "
                    f"{fmt(r.get('val_fdr'))} | {fmt(r.get('global_mae'))} | "
                    f"{fmt(r.get('global_drift'))} | {fmt(r.get('global_fdr'))} | "
                    f"{fmt(r.get('valid_scans'))} |\n")
    print(f"Wrote {md_path}")


def main():
    parser = argparse.ArgumentParser(description="Run the full ablation table.")
    parser.add_argument("--ckpt", required=True,
                        help="Path to the best.pth checkpoint to evaluate across configurations.")
    parser.add_argument("--logs-dir", default="ablation_logs",
                        help="Directory where per-run logs and the final tables land.")
    parser.add_argument("--skip-completed", action="store_true",
                        help="Skip runs whose log already exists and contains a parseable summary.")
    parser.add_argument("--only", nargs="+", default=None,
                        help="Filter experiments by substring match against label or run_name. "
                             "Example: --only ESKF complementary")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print commands but don't actually invoke them.")
    args = parser.parse_args()

    if not os.path.exists(args.ckpt):
        print(f"ERROR: checkpoint not found: {args.ckpt}")
        sys.exit(1)

    os.makedirs(args.logs_dir, exist_ok=True)

    selected = EXPERIMENTS
    if args.only:
        needles = [s.lower() for s in args.only]
        selected = [e for e in EXPERIMENTS
                    if any(n in e[0].lower() or n in e[1].lower() for n in needles)]
        if not selected:
            print(f"ERROR: --only filter matched 0 experiments. Available labels:")
            for e in EXPERIMENTS:
                print(f"  {e[0]}")
            sys.exit(1)

    print(f"[{datetime.now()}] Ablation runner")
    print(f"  checkpoint: {args.ckpt}")
    print(f"  logs dir:   {args.logs_dir}")
    print(f"  total runs: {len(selected)}")
    print()

    overall_start = datetime.now()
    for i, (label, run_name, extra_args) in enumerate(selected, 1):
        print(f"[{i}/{len(selected)}] {label}")
        run_one(label, run_name, extra_args, args.ckpt, args.logs_dir,
                skip_completed=args.skip_completed, dry_run=args.dry_run)
        print()

    total_min = (datetime.now() - overall_start).total_seconds() / 60.0
    print(f"[{datetime.now()}] All runs done in {total_min:.1f} min.")

    if args.dry_run:
        print("Dry-run — skipping table generation.")
        return

    # Parse summaries and build results
    results = []
    for label, run_name, _ in selected:
        log_path = os.path.join(args.logs_dir, f"{run_name}.txt")
        summary = parse_summary(log_path) or {}
        results.append({"label": label, "run_name": run_name, **summary})

    write_tables(results, args.logs_dir)

    # Pretty-print summary to stdout
    print()
    print("=" * 110)
    print(f"{'Config':<32} {'Val MAE':>9} {'Val FDR':>9} {'Global MAE':>11} {'Global FDR':>11} {'Valid':>7}")
    print("-" * 110)
    for r in results:
        print(f"{r['label']:<32} "
              f"{fmt(r.get('val_mae')):>9} "
              f"{fmt(r.get('val_fdr'), '%'):>9} "
              f"{fmt(r.get('global_mae')):>11} "
              f"{fmt(r.get('global_fdr'), '%'):>11} "
              f"{fmt(r.get('valid_scans')):>7}")
    print("=" * 110)


if __name__ == "__main__":
    main()
