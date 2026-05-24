"""Plot Phase 2 thermodynamic SNN snapshot evolution.

Input: phase2_snapshots.csv (output of thermo_m1_evaluation)
Output: phase2_evolution.png (6 subplots showing metric time-series)

Usage:
    cd spiking_brain_v0.5/spiking_brain
    python python/plot_phase2_snapshots.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    csv_path = Path("phase2_snapshots.csv")
    if not csv_path.exists():
        print(f"ERROR: {csv_path} not found.", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} snapshots from {csv_path}")

    fig, axes = plt.subplots(3, 2, figsize=(15, 11))
    fig.suptitle(
        f"Phase 2 Thermodynamic SNN -- {df['trial'].max():,} trials (snapshot every 500)",
        fontsize=14, fontweight="bold",
    )

    trials = df["trial"]

    # (1) Selectivity / Within / Between
    ax = axes[0, 0]
    between = df["within"] - df["selectivity"]
    ax.plot(trials, df["selectivity"], label="selectivity", color="C0", linewidth=1.4)
    ax.plot(trials, df["within"], label="within-pattern", color="C1", alpha=0.55, linewidth=0.7)
    ax.plot(trials, between, label="between-pattern", color="C2", alpha=0.55, linewidth=0.7)
    ax.axhline(y=0.55, color="green", linestyle="--", alpha=0.4, label="0.55 (great success band)")
    ax.axhline(y=0.40, color="orange", linestyle="--", alpha=0.4, label="0.40 (right direction)")
    ax.set_xlabel("trial")
    ax.set_ylabel("cosine similarity")
    ax.set_title("(1) Discrimination (selectivity, within, between)")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.3)

    # (2) Active outputs / silent ratio
    ax = axes[0, 1]
    ax.plot(trials, df["active"], label="active outputs (/40)", color="C3", linewidth=1.2)
    ax2 = ax.twinx()
    ax2.plot(trials, df["silent_ratio"], label="silent ratio", color="C4", alpha=0.7, linewidth=1.0)
    ax.set_xlabel("trial")
    ax.set_ylabel("active outputs (out of 40)", color="C3")
    ax2.set_ylabel("silent ratio", color="C4")
    ax.set_title("(2) Active / Silent neurons")
    ax.legend(loc="upper left", fontsize=8)
    ax2.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.3)

    # (3) Entropy mean / max  (habituation indicator)
    ax = axes[1, 0]
    ax.plot(trials, df["entropy_mean"], label="entropy mean", color="C5", linewidth=1.2)
    ax.set_xlabel("trial")
    ax.set_ylabel("entropy mean", color="C5")
    ax2 = ax.twinx()
    ax2.plot(trials, df["entropy_max"], label="entropy max", color="C6", alpha=0.6, linewidth=0.8)
    ax2.set_ylabel("entropy max", color="C6")
    ax.set_title("(3) Local entropy (physical habituation)")
    ax.legend(loc="lower right", fontsize=8)
    ax2.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.3)

    # (4) Enthalpy mean
    ax = axes[1, 1]
    ax.plot(trials, df["enthalpy_mean"], label="enthalpy mean", color="C7", linewidth=1.2)
    ax.set_xlabel("trial")
    ax.set_ylabel("enthalpy mean")
    ax.set_title("(4) Available enthalpy (firing-capable energy)")
    ax.set_ylim(0, 11)
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(alpha=0.3)

    # (5) Conductance (axon growth health)
    ax = axes[2, 0]
    ax.plot(trials, df["conductance_mean"], label="conductance mean", color="C8", linewidth=1.2)
    ax2 = ax.twinx()
    ax2.plot(trials, df["conductance_max"], label="conductance max", color="C9", alpha=0.6, linewidth=0.8)
    ax.set_xlabel("trial")
    ax.set_ylabel("conductance mean", color="C8")
    ax2.set_ylabel("conductance max", color="C9")
    ax.set_title("(5) Conductance (axon growth maturation)")
    ax.legend(loc="upper left", fontsize=8)
    ax2.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.3)

    # (6) Structural change cumulative counts
    ax = axes[2, 1]
    ax.plot(trials, df["axons_grown"], label="axons grown (cumulative)", color="C0", linewidth=1.4)
    ax.plot(trials, df["axons_pruned"], label="axons pruned (cumulative)", color="C1", linewidth=1.0)
    ax.plot(trials, df["open_syn"], label="open synapses", color="C2", alpha=0.5, linewidth=0.8)
    ax.set_xlabel("trial")
    ax.set_ylabel("count")
    ax.set_title("(6) Structural change cumulative (growth / pruning)")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.97])

    out_path = Path("phase2_evolution.png")
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    print(f"Graph saved: {out_path.resolve()}")

    # Statistics
    sel = df["selectivity"]
    print("\n=== selectivity statistics ===")
    print(f"  mean         : {sel.mean():.4f}")
    print(f"  std          : {sel.std():.4f}")
    print(f"  min / max    : {sel.min():.4f} / {sel.max():.4f}")
    print(f"  median       : {sel.median():.4f}")
    print(f"  > 0.40 ratio : {(sel > 0.40).sum()}/{len(sel)} = {(sel > 0.40).mean()*100:.1f}%")
    print(f"  > 0.55 ratio : {(sel > 0.55).sum()}/{len(sel)} = {(sel > 0.55).mean()*100:.1f}%")

    # Compare quarters to check if dynamic equilibrium is reached
    q = len(df) // 4
    print("\n=== selectivity by quarter (dynamic equilibrium check) ===")
    for i in range(4):
        a, b = i*q, (i+1)*q if i < 3 else len(df)
        sub = sel.iloc[a:b]
        start, end = df["trial"].iloc[a], df["trial"].iloc[b-1]
        print(f"  Q{i+1} (trial {start}-{end}): mean {sub.mean():.4f}  std {sub.std():.4f}")

    print(f"\n=== final state ===")
    print(f"  trial            : {df['trial'].iloc[-1]:,}")
    print(f"  selectivity      : {df['selectivity'].iloc[-1]:.4f}")
    print(f"  active outputs   : {df['active'].iloc[-1]}/40")
    print(f"  entropy mean/max : {df['entropy_mean'].iloc[-1]:.1f} / {df['entropy_max'].iloc[-1]}")
    print(f"  axons grown      : {df['axons_grown'].iloc[-1]:,}")
    print(f"  axons pruned     : {df['axons_pruned'].iloc[-1]:,}")


if __name__ == "__main__":
    main()
