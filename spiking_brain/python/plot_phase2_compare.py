"""Compare Phase 2 baseline / Fork A / Fork B in a single figure.

Input:  phase2_snapshots.csv (baseline)
        phase2_a_snapshots.csv (Fork A: MIN_CONDUCTANCE 40)
        phase2_b_snapshots.csv (Fork B: maturation period 1000 step)
Output: phase2_fork_compare.png

Usage:
    cd spiking_brain_v0.5/spiking_brain
    python python/plot_phase2_compare.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


VARIANTS = [
    ("phase2_snapshots.csv",   "baseline (MIN=20)",  "C0"),
    ("phase2_a_snapshots.csv", "Fork A (MIN=40)",    "C3"),
    ("phase2_b_snapshots.csv", "Fork B (mat=1000)",  "C2"),
]


def main() -> None:
    dfs: dict[str, pd.DataFrame] = {}
    for path, label, _color in VARIANTS:
        p = Path(path)
        if not p.exists():
            print(f"WARN: {p} not found, skipping {label}")
            continue
        dfs[label] = pd.read_csv(p)
        print(f"Loaded {label}: {len(dfs[label])} snapshots")

    if not dfs:
        print("ERROR: no CSV files found", file=sys.stderr)
        sys.exit(1)

    fig, axes = plt.subplots(3, 2, figsize=(15, 11))
    fig.suptitle("Phase 2 Fork Comparison: baseline vs A (MIN=40) vs B (maturation=1000)",
                 fontsize=14, fontweight="bold")

    colors = {label: color for _, label, color in VARIANTS}

    # (1) selectivity
    ax = axes[0, 0]
    for label, df in dfs.items():
        ax.plot(df["trial"], df["selectivity"], label=label, color=colors[label],
                linewidth=1.1, alpha=0.85)
    ax.axhline(y=0.55, color="green", linestyle="--", alpha=0.3)
    ax.axhline(y=0.40, color="orange", linestyle="--", alpha=0.3)
    ax.set_xlabel("trial")
    ax.set_ylabel("selectivity")
    ax.set_title("(1) Selectivity (within - between)")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.3)

    # (2) within-pattern similarity
    ax = axes[0, 1]
    for label, df in dfs.items():
        ax.plot(df["trial"], df["within"], label=label, color=colors[label],
                linewidth=1.1, alpha=0.85)
    ax.set_xlabel("trial")
    ax.set_ylabel("within-pattern cosine")
    ax.set_title("(2) Within-pattern similarity")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)

    # (3) active outputs
    ax = axes[1, 0]
    for label, df in dfs.items():
        ax.plot(df["trial"], df["active"], label=label, color=colors[label],
                linewidth=1.2, alpha=0.85)
    ax.set_xlabel("trial")
    ax.set_ylabel("active outputs (/40)")
    ax.set_title("(3) Active output neurons")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)

    # (4) conductance mean (the key metric for the deadlock)
    ax = axes[1, 1]
    for label, df in dfs.items():
        ax.plot(df["trial"], df["conductance_mean"], label=label + " (mean)",
                color=colors[label], linewidth=1.2, alpha=0.85)
    ax.set_xlabel("trial")
    ax.set_ylabel("conductance mean")
    ax.set_title("(4) Conductance mean (deadlock indicator)")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.3)

    # (5) axons grown cumulative
    ax = axes[2, 0]
    for label, df in dfs.items():
        ax.plot(df["trial"], df["axons_grown"], label=label,
                color=colors[label], linewidth=1.2, alpha=0.85)
    ax.set_xlabel("trial")
    ax.set_ylabel("axons grown (cumulative)")
    ax.set_title("(5) Axon growth (cumulative)")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.3)

    # (6) open synapses (axon survival)
    ax = axes[2, 1]
    for label, df in dfs.items():
        ax.plot(df["trial"], df["open_syn"], label=label,
                color=colors[label], linewidth=1.2, alpha=0.85)
    ax.set_xlabel("trial")
    ax.set_ylabel("open synapses")
    ax.set_title("(6) Open synapses (axon survival)")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    out = Path("phase2_fork_compare.png")
    plt.savefig(out, dpi=130, bbox_inches="tight")
    print(f"\nSaved: {out.resolve()}")

    # Statistics
    print("\n=== Selectivity statistics by variant ===")
    print(f"  {'variant':<22}  {'mean':>7}  {'std':>7}  {'min':>7}  {'max':>7}  {'>0.40':>7}  {'>0.55':>7}")
    for label, df in dfs.items():
        s = df["selectivity"]
        print(f"  {label:<22}  {s.mean():>7.4f}  {s.std():>7.4f}  {s.min():>7.4f}  {s.max():>7.4f}  "
              f"{(s>0.40).mean()*100:>6.1f}%  {(s>0.55).mean()*100:>6.1f}%")

    print("\n=== Final state by variant ===")
    print(f"  {'variant':<22}  {'sel':>7}  {'active':>7}  {'cond_μ':>7}  {'cond_max':>9}  {'grown':>7}  {'open':>6}")
    for label, df in dfs.items():
        last = df.iloc[-1]
        print(f"  {label:<22}  {last['selectivity']:>7.4f}  {int(last['active']):>7}  "
              f"{last['conductance_mean']:>7.2f}  {int(last['conductance_max']):>9}  "
              f"{int(last['axons_grown']):>7}  {int(last['open_syn']):>6}")


if __name__ == "__main__":
    main()
