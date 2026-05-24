"""Plot Phase 2 thermodynamic SNN with white-noise training (1M trials).

Compare with the previous fixed-pattern training (100k trials) if available.

Input:
    phase2_whitenoise_snapshots.csv  (white-noise training, 1M trials)
    phase2_snapshots.csv             (baseline fixed-pattern, 100k trials, optional)
Output:
    phase2_whitenoise_evolution.png  (white-noise only, 6 subplots)
    phase2_whitenoise_vs_fixed.png   (overlay, if both CSV files exist)

Usage:
    cd spiking_brain_v0.5/spiking_brain
    python python/plot_phase2_whitenoise.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_evolution(df: pd.DataFrame, out_path: Path, title: str) -> None:
    fig, axes = plt.subplots(3, 2, figsize=(15, 11))
    fig.suptitle(title, fontsize=14, fontweight="bold")
    trials = df["trial"]

    ax = axes[0, 0]
    between = df["within"] - df["selectivity"]
    ax.plot(trials, df["selectivity"], label="selectivity", color="C0", linewidth=1.0)
    ax.plot(trials, df["within"], label="within", color="C1", alpha=0.5, linewidth=0.6)
    ax.plot(trials, between, label="between", color="C2", alpha=0.5, linewidth=0.6)
    ax.axhline(y=0.55, color="green", linestyle="--", alpha=0.3)
    ax.axhline(y=0.40, color="orange", linestyle="--", alpha=0.3)
    ax.set_xlabel("trial"); ax.set_ylabel("cosine"); ax.set_title("(1) Selectivity / Within / Between")
    ax.legend(loc="upper right", fontsize=8); ax.grid(alpha=0.3)

    ax = axes[0, 1]
    ax.plot(trials, df["active"], label="active (/40)", color="C3", linewidth=1.0)
    ax2 = ax.twinx()
    ax2.plot(trials, df["silent_ratio"], label="silent ratio", color="C4", alpha=0.6, linewidth=0.8)
    ax.set_xlabel("trial"); ax.set_ylabel("active", color="C3"); ax2.set_ylabel("silent ratio", color="C4")
    ax.set_title("(2) Active / Silent")
    ax.legend(loc="upper left", fontsize=8); ax2.legend(loc="upper right", fontsize=8); ax.grid(alpha=0.3)

    ax = axes[1, 0]
    ax.plot(trials, df["entropy_mean"], label="entropy mean", color="C5", linewidth=1.0)
    ax2 = ax.twinx()
    ax2.plot(trials, df["entropy_max"], label="entropy max", color="C6", alpha=0.6, linewidth=0.8)
    ax.set_xlabel("trial"); ax.set_ylabel("entropy mean", color="C5"); ax2.set_ylabel("entropy max", color="C6")
    ax.set_title("(3) Local Entropy")
    ax.legend(loc="lower right", fontsize=8); ax2.legend(loc="upper right", fontsize=8); ax.grid(alpha=0.3)

    ax = axes[1, 1]
    ax.plot(trials, df["enthalpy_mean"], label="enthalpy mean", color="C7", linewidth=1.0)
    ax.set_xlabel("trial"); ax.set_ylabel("enthalpy mean")
    ax.set_title("(4) Available Enthalpy")
    ax.set_ylim(0, 11); ax.legend(loc="lower right", fontsize=8); ax.grid(alpha=0.3)

    ax = axes[2, 0]
    ax.plot(trials, df["conductance_mean"], label="cond mean", color="C8", linewidth=1.0)
    ax2 = ax.twinx()
    ax2.plot(trials, df["conductance_max"], label="cond max", color="C9", alpha=0.6, linewidth=0.8)
    ax.set_xlabel("trial"); ax.set_ylabel("cond mean", color="C8"); ax2.set_ylabel("cond max", color="C9")
    ax.set_title("(5) Conductance (axon growth maturation)")
    ax.legend(loc="upper left", fontsize=8); ax2.legend(loc="upper right", fontsize=8); ax.grid(alpha=0.3)

    ax = axes[2, 1]
    ax.plot(trials, df["axons_grown"], label="grown (cumulative)", color="C0", linewidth=1.0)
    ax.plot(trials, df["axons_pruned"], label="pruned (cumulative)", color="C1", linewidth=0.8)
    ax.plot(trials, df["open_syn"], label="open synapses", color="C2", alpha=0.5, linewidth=0.6)
    ax.set_xlabel("trial"); ax.set_ylabel("count")
    ax.set_title("(6) Structural change (cumulative)")
    ax.legend(loc="upper left", fontsize=8); ax.grid(alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    print(f"Saved: {out_path.resolve()}")


def print_stats(df: pd.DataFrame, label: str) -> None:
    sel = df["selectivity"]
    print(f"\n=== {label} : selectivity ({len(df)} snapshots, max trial {df['trial'].max():,}) ===")
    print(f"  mean / std    : {sel.mean():.4f} / {sel.std():.4f}")
    print(f"  min / max     : {sel.min():.4f} / {sel.max():.4f}")
    print(f"  median        : {sel.median():.4f}")
    print(f"  > 0.40 ratio  : {(sel>0.40).mean()*100:.1f}%")
    print(f"  > 0.55 ratio  : {(sel>0.55).mean()*100:.1f}%")
    n = len(df)
    for i in range(4):
        a, b = i*n//4, (i+1)*n//4 if i < 3 else n
        sub = sel.iloc[a:b]
        ta, tb = df["trial"].iloc[a], df["trial"].iloc[b-1]
        print(f"  Q{i+1} ({ta:,}-{tb:,}): mean {sub.mean():.4f}  std {sub.std():.4f}")


def main() -> None:
    wn_path = Path("phase2_whitenoise_snapshots.csv")
    if not wn_path.exists():
        print(f"ERROR: {wn_path} not found.", file=sys.stderr)
        sys.exit(1)

    wn = pd.read_csv(wn_path)
    print(f"Loaded {len(wn)} snapshots from {wn_path}")
    plot_evolution(wn, Path("phase2_whitenoise_evolution.png"),
                   f"Phase 2 White-Noise Training -- {wn['trial'].max():,} trials")
    print_stats(wn, "white-noise training (1M)")

    # Optional: overlay with fixed-pattern baseline if available
    base_path = Path("phase2_snapshots.csv")
    if base_path.exists():
        base = pd.read_csv(base_path)
        print(f"\nLoaded {len(base)} baseline snapshots from {base_path}")
        print_stats(base, "fixed-pattern baseline (100k)")

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(wn["trial"], wn["selectivity"], label=f"white-noise (1M)", color="C3", linewidth=0.7, alpha=0.85)
        ax.plot(base["trial"], base["selectivity"], label=f"fixed-pattern (100k)", color="C0", linewidth=0.9, alpha=0.85)
        ax.axhline(y=0.55, color="green", linestyle="--", alpha=0.3)
        ax.axhline(y=0.40, color="orange", linestyle="--", alpha=0.3)
        ax.set_xlabel("trial")
        ax.set_ylabel("selectivity")
        ax.set_title("Selectivity: white-noise training (1M) vs fixed-pattern baseline (100k)")
        ax.legend(loc="upper right")
        ax.grid(alpha=0.3)
        ax.set_xscale("log")
        plt.tight_layout()
        out = Path("phase2_whitenoise_vs_fixed.png")
        plt.savefig(out, dpi=130, bbox_inches="tight")
        print(f"Saved: {out.resolve()}")


if __name__ == "__main__":
    main()
