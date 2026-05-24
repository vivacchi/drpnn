"""Plot Phase 2 Fork F-G1-R1 evolution (vitality + G-1 enthalpy + LTD revival).

Shows the emergence of developmental period (overcompensation → pruning)
followed by stable dynamic equilibrium.

Input:  phase2_f_whitenoise_snapshots.csv
Output: phase2_f_g1_r1_evolution.png, phase2_f_g1_r1_comparison.png
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def smooth(s: pd.Series, window: int = 10) -> pd.Series:
    return s.rolling(window=window, min_periods=1, center=True).mean()


def main() -> None:
    csv_path = Path("phase2_f_whitenoise_snapshots.csv")
    if not csv_path.exists():
        print(f"ERROR: {csv_path} not found.", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} snapshots from {csv_path}")
    print(f"Max trial: {df['trial'].max():,}")

    # ===== Plot 1: Single-fork evolution =====
    fig, axes = plt.subplots(3, 2, figsize=(15, 11))
    fig.suptitle(
        f"Phase 2 Fork F-G1-R1: vitality + LTD revival + enthalpy-based refractory "
        f"-- {df['trial'].max():,} trials",
        fontsize=13, fontweight="bold"
    )
    trials = df["trial"]

    # (1) Selectivity (raw + smoothed)
    ax = axes[0, 0]
    between = df["within"] - df["selectivity"]
    ax.plot(trials, df["selectivity"], color="C0", linewidth=0.4, alpha=0.4, label="selectivity (raw)")
    ax.plot(trials, smooth(df["selectivity"], 20), color="C0", linewidth=1.4, label="selectivity (smoothed)")
    ax.plot(trials, smooth(df["within"], 20), color="C1", alpha=0.6, linewidth=0.8, label="within (smoothed)")
    ax.plot(trials, smooth(between, 20), color="C2", alpha=0.6, linewidth=0.8, label="between (smoothed)")
    ax.axhline(y=0.55, color="green", linestyle="--", alpha=0.3)
    ax.axhline(y=0.40, color="orange", linestyle="--", alpha=0.3)
    ax.set_xlabel("trial")
    ax.set_ylabel("cosine")
    ax.set_title("(1) Selectivity / Within / Between (POST=0.179)")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.3)

    # (2) Active outputs / silent
    ax = axes[0, 1]
    ax.plot(trials, df["active"], color="C3", linewidth=0.5, alpha=0.5)
    ax.plot(trials, smooth(df["active"], 20), color="C3", linewidth=1.4, label="active (smoothed)")
    ax.set_xlabel("trial")
    ax.set_ylabel("active outputs (/40)")
    ax.set_title("(2) Active output neurons")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.3)

    # (3) Open synapses (the key pruning indicator!)
    ax = axes[1, 0]
    ax.plot(trials, df["open_syn"], color="C0", linewidth=1.4, label="open synapses (alive)")
    ax.axhline(y=2800, color="gray", linestyle=":", alpha=0.5, label="baseline 2800 (固定結線数)")
    ax.set_xlabel("trial")
    ax.set_ylabel("open synapses (count)")
    ax.set_title("(3) Pruning dynamics: 過剰接続 → 刈り込み → 動的平衡")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.3)

    # (4) Conductance
    ax = axes[1, 1]
    ax.plot(trials, df["conductance_mean"], color="C8", linewidth=0.5, alpha=0.5)
    ax.plot(trials, smooth(df["conductance_mean"], 20), color="C8", linewidth=1.4, label="cond mean (smoothed)")
    ax2 = ax.twinx()
    ax2.plot(trials, df["conductance_max"], color="C9", linewidth=0.5, alpha=0.5)
    ax2.plot(trials, smooth(df["conductance_max"], 20), color="C9", linewidth=1.0, label="cond max (smoothed)")
    ax.set_xlabel("trial")
    ax.set_ylabel("conductance mean", color="C8")
    ax2.set_ylabel("conductance max", color="C9")
    ax.set_title("(4) Conductance: LTD で平均は低く、強い結線は max=100")
    ax.legend(loc="upper left", fontsize=8)
    ax2.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.3)

    # (5) Entropy (local heat)
    ax = axes[2, 0]
    ax.plot(trials, smooth(df["entropy_mean"], 20), color="C5", linewidth=1.2, label="entropy mean (smoothed)")
    ax2 = ax.twinx()
    ax2.plot(trials, smooth(df["entropy_max"], 20), color="C6", linewidth=0.8, alpha=0.6, label="entropy max (smoothed)")
    ax.set_xlabel("trial")
    ax.set_ylabel("entropy mean", color="C5")
    ax2.set_ylabel("entropy max", color="C6")
    ax.set_title("(5) Local entropy (慣化指標)")
    ax.legend(loc="lower right", fontsize=8)
    ax2.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.3)

    # (6) Cumulative structural change
    ax = axes[2, 1]
    ax.plot(trials, df["axons_grown"], color="C0", linewidth=1.2, label="grown (cumulative)")
    ax.plot(trials, df["axons_pruned"], color="C1", linewidth=1.2, label="pruned (cumulative)")
    ax.set_xlabel("trial")
    ax.set_ylabel("count")
    ax.set_title("(6) Structural change cumulative")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    out = Path("phase2_f_g1_r1_evolution.png")
    plt.savefig(out, dpi=130, bbox_inches="tight")
    print(f"Saved: {out.resolve()}")

    # ===== Plot 2: Comparison with prior forks =====
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    ax.set_title("Selectivity comparison: Fork F-G1-R1 vs prior baselines",
                 fontsize=13, fontweight="bold")

    # Try to load other CSVs for comparison
    comparisons = [
        ("phase2_snapshots.csv", "baseline (fixed pattern 100k)", "C0", 0.41),
        ("phase2_whitenoise_snapshots.csv", "baseline (white-noise 1M)", "C2", 0.06),
    ]
    for path, label, color, _final_post in comparisons:
        p = Path(path)
        if p.exists():
            other = pd.read_csv(p)
            ax.plot(other["trial"], smooth(other["selectivity"], 20),
                    color=color, linewidth=1.0, alpha=0.7, label=f"{label}")

    # Main: Fork F-G1-R1
    ax.plot(trials, smooth(df["selectivity"], 20), color="C3", linewidth=1.8,
            label="Fork F-G1-R1 (white-noise 100k, POST=0.179)")

    ax.axhline(y=0.55, color="green", linestyle="--", alpha=0.3, label="0.55 (great)")
    ax.axhline(y=0.40, color="orange", linestyle="--", alpha=0.3, label="0.40 (good direction)")
    ax.axhline(y=0.179, color="red", linestyle=":", alpha=0.5, label="Fork F-G1-R1 POST = 0.179")
    ax.set_xlabel("trial")
    ax.set_ylabel("selectivity (50-pt smoothed)")
    ax.set_xscale("log")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    out2 = Path("phase2_f_g1_r1_comparison.png")
    plt.savefig(out2, dpi=130, bbox_inches="tight")
    print(f"Saved: {out2.resolve()}")

    # ===== Statistics =====
    print("\n=== Fork F-G1-R1 statistics ===")
    sel = df["selectivity"]
    print(f"  mean / std    : {sel.mean():.4f} / {sel.std():.4f}")
    print(f"  min / max     : {sel.min():.4f} / {sel.max():.4f}")
    print(f"  POST (final)  : 0.179 (from output log)")
    print(f"  > 0.05 ratio  : {(sel > 0.05).mean()*100:.1f}%")
    print(f"  > 0.10 ratio  : {(sel > 0.10).mean()*100:.1f}%")

    print("\n=== Pruning timeline ===")
    print(f"  trial      0: open = 16,200 (initial overcompensation)")
    for t_target in [1000, 2000, 3000, 5000, 10000, 100000]:
        row = df[df["trial"] >= t_target].head(1)
        if not row.empty:
            r = row.iloc[0]
            print(f"  trial {int(r['trial']):>6}: open = {int(r['open_syn']):>6}, pruned = {int(r['axons_pruned']):>6}")

    print(f"\n  最終刈り込み比率: {df.iloc[-1]['axons_pruned']/16205*100:.1f}%")
    print(f"  最終 open: {df.iloc[-1]['open_syn']:.0f} / 16205")


if __name__ == "__main__":
    main()
