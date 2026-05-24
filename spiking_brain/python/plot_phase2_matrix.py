"""4-cell matrix comparison: training (fixed pattern / white-noise) x fork (baseline / A / B).

Loads up to 6 CSV files:
    phase2_snapshots.csv              fixed-pattern baseline (100k)
    phase2_a_snapshots.csv            fixed-pattern Fork A (100k)
    phase2_b_snapshots.csv            fixed-pattern Fork B (100k)
    phase2_whitenoise_snapshots.csv   white-noise baseline (1M)
    phase2_a_whitenoise_snapshots.csv white-noise Fork A (100k)
    phase2_b_whitenoise_snapshots.csv white-noise Fork B (100k)

Outputs: phase2_matrix.png (selectivity), prints stats matrix.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


CONFIGS = [
    # (csv_path, training, fork, color, linestyle)
    ("phase2_snapshots.csv",              "fixed",   "baseline", "C0", "-"),
    ("phase2_a_snapshots.csv",            "fixed",   "Fork A",   "C3", "-"),
    ("phase2_b_snapshots.csv",            "fixed",   "Fork B",   "C2", "-"),
    ("phase2_whitenoise_snapshots.csv",   "white",   "baseline", "C0", "--"),
    ("phase2_a_whitenoise_snapshots.csv", "white",   "Fork A",   "C3", "--"),
    ("phase2_b_whitenoise_snapshots.csv", "white",   "Fork B",   "C2", "--"),
    ("phase2_c_whitenoise_snapshots.csv", "white",   "Fork C",   "C4", "--"),
    ("phase2_d_whitenoise_snapshots.csv", "white",   "Fork D",   "C1", "--"),
]


def main() -> None:
    dfs: list[tuple[str, str, str, str, str, pd.DataFrame]] = []
    for path, training, fork, color, ls in CONFIGS:
        p = Path(path)
        if not p.exists():
            print(f"WARN: {p} not found, skipping")
            continue
        df = pd.read_csv(p)
        dfs.append((path, training, fork, color, ls, df))
        print(f"Loaded {training:5s} / {fork:9s}: {len(df)} snapshots, max trial {df['trial'].max():,}")

    if not dfs:
        print("ERROR: no CSV files found", file=sys.stderr)
        sys.exit(1)

    # Plot selectivity over time
    fig, axes = plt.subplots(2, 1, figsize=(13, 9))
    fig.suptitle("Phase 2 Matrix: Training (fixed/white-noise) x Fork (baseline/A/B)",
                 fontsize=14, fontweight="bold")

    # Top: fixed-pattern training
    ax = axes[0]
    for _, training, fork, color, ls, df in dfs:
        if training != "fixed": continue
        ax.plot(df["trial"], df["selectivity"], label=fork, color=color, linewidth=1.0, alpha=0.85)
    ax.axhline(y=0.55, color="green", linestyle=":", alpha=0.4)
    ax.axhline(y=0.40, color="orange", linestyle=":", alpha=0.4)
    ax.set_xlabel("trial"); ax.set_ylabel("selectivity")
    ax.set_title("(top) Fixed pattern training (A-E repeat)")
    ax.legend(loc="upper right", fontsize=9); ax.grid(alpha=0.3)

    # Bottom: white-noise training
    ax = axes[1]
    for _, training, fork, color, ls, df in dfs:
        if training != "white": continue
        ax.plot(df["trial"], df["selectivity"], label=fork, color=color, linewidth=1.0, alpha=0.85)
    ax.axhline(y=0.55, color="green", linestyle=":", alpha=0.4)
    ax.axhline(y=0.40, color="orange", linestyle=":", alpha=0.4)
    ax.set_xlabel("trial"); ax.set_ylabel("selectivity")
    ax.set_title("(bottom) White-noise training")
    ax.legend(loc="upper right", fontsize=9); ax.grid(alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    out = Path("phase2_matrix.png")
    plt.savefig(out, dpi=130, bbox_inches="tight")
    print(f"\nSaved: {out.resolve()}")

    # Stats matrix
    print("\n=== Selectivity statistics matrix ===")
    print(f"  {'training':<8} {'fork':<10}  {'mean':>7}  {'std':>7}  {'>0.40':>6}  {'>0.55':>6}  {'grown':>7}  {'pruned':>7}")
    for _, training, fork, _, _, df in dfs:
        s = df["selectivity"]
        last = df.iloc[-1]
        print(f"  {training:<8} {fork:<10}  {s.mean():>7.4f}  {s.std():>7.4f}  "
              f"{(s>0.40).mean()*100:>5.1f}%  {(s>0.55).mean()*100:>5.1f}%  "
              f"{int(last['axons_grown']):>7}  {int(last['axons_pruned']):>7}")


if __name__ == "__main__":
    main()
