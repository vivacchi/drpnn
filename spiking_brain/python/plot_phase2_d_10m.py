"""Plot Phase 2 Fork D 10M trials with white-noise training.

Input:  phase2_d_whitenoise_10m_snapshots.csv (1000-step snapshots, 10000 points)
Output: phase2_d_10m_evolution.png

Usage:
    cd spiking_brain_v0.5/spiking_brain
    python python/plot_phase2_d_10m.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main() -> None:
    csv_path = Path("phase2_d_whitenoise_10m_snapshots.csv")
    if not csv_path.exists():
        print(f"ERROR: {csv_path} not found.", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} snapshots from {csv_path}")

    # Moving average for noisy series
    def smooth(s: pd.Series, window: int = 50) -> pd.Series:
        return s.rolling(window=window, min_periods=1, center=True).mean()

    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    fig.suptitle(
        f"Phase 2 Fork D (uniform conductance=50) -- {df['trial'].max():,} trials (snapshot every 1000)",
        fontsize=14, fontweight="bold",
    )

    trials = df["trial"]

    # (1) Selectivity (raw + smoothed)
    ax = axes[0, 0]
    ax.plot(trials, df["selectivity"], color="C0", linewidth=0.4, alpha=0.4, label="selectivity (raw)")
    ax.plot(trials, smooth(df["selectivity"]), color="C0", linewidth=1.3, label="selectivity (smoothed)")
    ax.axhline(y=0.55, color="green", linestyle="--", alpha=0.4, label="0.55 (great)")
    ax.axhline(y=0.40, color="orange", linestyle="--", alpha=0.4, label="0.40 (right dir)")
    ax.set_xlabel("trial"); ax.set_ylabel("selectivity")
    ax.set_title("(1) Selectivity (raw and 50-pt moving avg)")
    ax.legend(loc="upper right", fontsize=8); ax.grid(alpha=0.3)

    # (2) Within / Between
    ax = axes[0, 1]
    between = df["within"] - df["selectivity"]
    ax.plot(trials, smooth(df["within"]), color="C1", linewidth=1.0, label="within (smoothed)")
    ax.plot(trials, smooth(between), color="C2", linewidth=1.0, label="between (smoothed)")
    ax.set_xlabel("trial"); ax.set_ylabel("cosine similarity")
    ax.set_title("(2) Within / Between (smoothed)")
    ax.legend(loc="upper right", fontsize=8); ax.grid(alpha=0.3)

    # (3) Active outputs / silent
    ax = axes[1, 0]
    ax.plot(trials, df["active"], color="C3", linewidth=0.5, alpha=0.5)
    ax.plot(trials, smooth(df["active"]), color="C3", linewidth=1.4, label="active (smoothed)")
    ax.set_xlabel("trial"); ax.set_ylabel("active outputs (/40)")
    ax.set_title("(3) Active output neurons")
    ax.legend(loc="upper right", fontsize=8); ax.grid(alpha=0.3)

    # (4) Local entropy
    ax = axes[1, 1]
    ax.plot(trials, smooth(df["entropy_mean"]), color="C5", linewidth=1.2, label="entropy mean (smoothed)")
    ax2 = ax.twinx()
    ax2.plot(trials, smooth(df["entropy_max"]), color="C6", linewidth=1.0, alpha=0.6, label="entropy max (smoothed)")
    ax.set_xlabel("trial"); ax.set_ylabel("entropy mean", color="C5"); ax2.set_ylabel("entropy max", color="C6")
    ax.set_title("(4) Local entropy")
    ax.legend(loc="lower right", fontsize=8); ax2.legend(loc="upper right", fontsize=8); ax.grid(alpha=0.3)

    # (5) Conductance
    ax = axes[2, 0]
    ax.plot(trials, df["conductance_mean"], color="C8", linewidth=0.8, label="cond mean")
    ax2 = ax.twinx()
    ax2.plot(trials, df["conductance_max"], color="C9", linewidth=0.6, alpha=0.6, label="cond max")
    ax.set_xlabel("trial"); ax.set_ylabel("cond mean", color="C8"); ax2.set_ylabel("cond max", color="C9")
    ax.set_title("(5) Conductance (plastic synapses)")
    ax.legend(loc="upper left", fontsize=8); ax2.legend(loc="upper right", fontsize=8); ax.grid(alpha=0.3)

    # (6) Structural changes cumulative
    ax = axes[2, 1]
    ax.plot(trials, df["axons_grown"], color="C0", linewidth=1.0, label="grown (cumulative)")
    ax.plot(trials, df["axons_pruned"], color="C1", linewidth=1.0, label="pruned (cumulative)")
    ax.plot(trials, df["open_syn"], color="C2", linewidth=0.8, alpha=0.5, label="open synapses")
    ax.set_xlabel("trial"); ax.set_ylabel("count")
    ax.set_title("(6) Structural change (cumulative)")
    ax.legend(loc="upper left", fontsize=8); ax.grid(alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    out = Path("phase2_d_10m_evolution.png")
    plt.savefig(out, dpi=130, bbox_inches="tight")
    print(f"Saved: {out.resolve()}")

    # Statistics
    sel = df["selectivity"]
    print("\n=== selectivity statistics ===")
    print(f"  mean / std    : {sel.mean():.4f} / {sel.std():.4f}")
    print(f"  min / max     : {sel.min():.4f} / {sel.max():.4f}")
    print(f"  median        : {sel.median():.4f}")
    print(f"  > 0.40 ratio  : {(sel>0.40).mean()*100:.1f}%")
    print(f"  > 0.55 ratio  : {(sel>0.55).mean()*100:.1f}%")

    # Divide into 10 segments for trend analysis
    print("\n=== selectivity by decile (long-term trend) ===")
    n = len(df)
    for i in range(10):
        a, b = i*n//10, (i+1)*n//10 if i < 9 else n
        sub = sel.iloc[a:b]
        ta, tb = df["trial"].iloc[a], df["trial"].iloc[b-1]
        print(f"  D{i+1} ({ta:>9,}-{tb:>10,}): mean {sub.mean():.4f}  std {sub.std():.4f}")

    print(f"\n=== final state ===")
    last = df.iloc[-1]
    print(f"  trial            : {int(last['trial']):,}")
    print(f"  selectivity      : {last['selectivity']:.4f}")
    print(f"  active outputs   : {int(last['active'])}/40")
    print(f"  entropy μ/max    : {last['entropy_mean']:.1f} / {int(last['entropy_max'])}")
    print(f"  conductance μ/max: {last['conductance_mean']:.2f} / {int(last['conductance_max'])}")
    print(f"  axons grown      : {int(last['axons_grown']):,}")
    print(f"  axons pruned     : {int(last['axons_pruned']):,}")
    print(f"  open synapses    : {int(last['open_syn']):,}")


if __name__ == "__main__":
    main()
