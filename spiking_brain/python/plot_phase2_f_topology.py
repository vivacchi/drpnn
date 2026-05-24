"""Phase 2 Fork F-G1-R1 topology visualization.

Plots:
  1. PRE/POST 全結線トポロジー比較 (synapses on 20x20 grid)
  2. conductance ヒートマップ (POST, color-coded by strength)
  3. 新生シナプス位置 (axon growth により生まれた接続)

Input files (default prefix=phase2_f_fixed):
  {prefix}_topo_neurons.csv   id, x, y, kind
  {prefix}_topo_syn_pre.csv   idx, pre, post, conductance, alive, is_new
  {prefix}_topo_syn_post.csv  same

Output:
  {prefix}_topology.png
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd

# 日本語フォント
for font in ["Yu Gothic", "Meiryo", "MS Gothic", "Hiragino Sans"]:
    try:
        matplotlib.rcParams["font.family"] = font
        break
    except Exception:
        continue
matplotlib.rcParams["axes.unicode_minus"] = False


def load(prefix: str):
    neurons = pd.read_csv(f"{prefix}_topo_neurons.csv")
    pre = pd.read_csv(f"{prefix}_topo_syn_pre.csv")
    post = pd.read_csv(f"{prefix}_topo_syn_post.csv")
    return neurons, pre, post


def neuron_color(kind: str) -> str:
    return {
        "input":  "#1f77b4",   # blue
        "exc":    "#888888",   # gray
        "inh":    "#d62728",   # red
        "output": "#2ca02c",   # green
    }.get(kind, "#000000")


def draw_neurons(ax, neurons: pd.DataFrame, size: int = 18):
    for kind, color in [("exc", "#bbbbbb"), ("inh", "#d62728"),
                        ("input", "#1f77b4"), ("output", "#2ca02c")]:
        sub = neurons[neurons["kind"] == kind]
        ax.scatter(sub["x"], sub["y"], c=color, s=size,
                   label=f"{kind} ({len(sub)})", zorder=3, edgecolors="black",
                   linewidths=0.2)


def draw_synapses(ax, neurons: pd.DataFrame, syn: pd.DataFrame,
                  color_by: str = "conductance", alive_only: bool = True,
                  max_lines: int = 8000, alpha_base: float = 0.25):
    pos = neurons.set_index("id")[["x", "y"]].to_dict("index")
    df = syn.copy()
    if alive_only:
        df = df[df["alive"] == 1]
    if len(df) > max_lines:
        # 強度上位のみ表示 (見やすさのため)
        df = df.nlargest(max_lines, "conductance")

    cmap = cm.get_cmap("RdYlBu_r")
    norm = mcolors.Normalize(vmin=0, vmax=100)

    xs, ys, colors, widths = [], [], [], []
    for _, r in df.iterrows():
        pre_p = pos.get(int(r["pre"]))
        post_p = pos.get(int(r["post"]))
        if pre_p is None or post_p is None:
            continue
        xs.append([pre_p["x"], post_p["x"], None])
        ys.append([pre_p["y"], post_p["y"], None])
        c = int(r[color_by]) if color_by == "conductance" else 50
        colors.append(cmap(norm(c)))
        widths.append(0.1 + (c / 100.0) * 1.2)

    # 個別に描画 (色とwidth を結線ごとに変える)
    for x, y, c, w in zip(xs, ys, colors, widths):
        ax.plot(x[:2], y[:2], color=c, linewidth=w, alpha=alpha_base, zorder=1)


def draw_new_synapses(ax, neurons: pd.DataFrame, syn: pd.DataFrame):
    pos = neurons.set_index("id")[["x", "y"]].to_dict("index")
    new = syn[(syn["is_new"] == 1)]
    new_alive = new[new["alive"] == 1]
    new_dead = new[new["alive"] == 0]

    for color, df_, label, lw in [
        ("#ff7f00", new_alive, f"new & alive ({len(new_alive)})", 1.0),
        ("#aaaaaa", new_dead,  f"new & pruned ({len(new_dead)})", 0.4),
    ]:
        for _, r in df_.iterrows():
            pre_p = pos.get(int(r["pre"]))
            post_p = pos.get(int(r["post"]))
            if pre_p is None or post_p is None:
                continue
            ax.plot([pre_p["x"], post_p["x"]], [pre_p["y"], post_p["y"]],
                    color=color, linewidth=lw, alpha=0.7, zorder=2)
        # legend proxy
        ax.plot([], [], color=color, linewidth=lw + 1, label=label)


def main(prefix: str = "phase2_f_fixed"):
    if not Path(f"{prefix}_topo_neurons.csv").exists():
        print(f"ERROR: {prefix}_topo_neurons.csv not found", file=sys.stderr)
        sys.exit(1)
    neurons, pre, post = load(prefix)
    print(f"  neurons : {len(neurons)}")
    print(f"  PRE syn : {len(pre)}  (alive {int(pre['alive'].sum())})")
    print(f"  POST syn: {len(post)} (alive {int(post['alive'].sum())}, new {int(post['is_new'].sum())})")

    fig, axes = plt.subplots(2, 2, figsize=(16, 16))
    fig.suptitle(
        f"Phase 2 Fork F-G1-R1 topology ({prefix}, "
        f"PRE {int(pre['alive'].sum())} syn → POST {int(post['alive'].sum())} alive)",
        fontsize=14, fontweight="bold"
    )

    # (1) PRE トポロジー (全結線、淡く)
    ax = axes[0, 0]
    draw_synapses(ax, neurons, pre, color_by="conductance",
                  alive_only=True, max_lines=20000, alpha_base=0.04)
    draw_neurons(ax, neurons)
    ax.set_title(f"(1) PRE topology  全結線 {int(pre['alive'].sum())} 本 (初期 conductance≈20)")
    ax.set_aspect("equal")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.2)

    # (2) POST トポロジー — 全結線
    ax = axes[0, 1]
    draw_synapses(ax, neurons, post, color_by="conductance",
                  alive_only=True, max_lines=20000, alpha_base=0.04)
    draw_neurons(ax, neurons)
    ax.set_title(f"(2) POST topology  生存 {int(post['alive'].sum())} 本 "
                 f"({(1 - post['alive'].sum()/len(post))*100:.1f}% 刈り取り)")
    ax.set_aspect("equal")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.2)

    # (3) POST 強結線のみ (conductance ≥ 80) — LTP で強化された経路を強調
    ax = axes[1, 0]
    strong = post[(post["alive"] == 1) & (post["conductance"] >= 80)]
    draw_synapses(ax, neurons, strong, color_by="conductance",
                  alive_only=True, max_lines=20000, alpha_base=0.85)
    draw_neurons(ax, neurons)
    ax.set_title(f"(3) POST 強結線 (conductance ≥ 80) — {len(strong)} 本 / LTP で形成された散逸構造")
    ax.set_aspect("equal")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.2)
    sm = cm.ScalarMappable(cmap=cm.get_cmap("RdYlBu_r"),
                           norm=mcolors.Normalize(vmin=80, vmax=100))
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label="conductance", fraction=0.04)

    # (4) 新生シナプス位置
    ax = axes[1, 1]
    draw_new_synapses(ax, neurons, post)
    draw_neurons(ax, neurons)
    n_new = int(post["is_new"].sum())
    n_new_alive = int(((post["is_new"] == 1) & (post["alive"] == 1)).sum())
    ax.set_title(f"(4) 新生シナプス (axon_growth) — 累計 {n_new}, POST 生存 {n_new_alive}")
    ax.set_aspect("equal")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.2)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    out = Path(f"{prefix}_topology.png")
    plt.savefig(out, dpi=130, bbox_inches="tight")
    print(f"\nSaved: {out.resolve()}")

    # ===== 統計 =====
    print(f"\n=== Topology stats ({prefix}) ===")
    print(f"  PRE  alive synapses : {int(pre['alive'].sum())}")
    print(f"  POST alive synapses : {int(post['alive'].sum())}")
    print(f"  Pruned              : {int(pre['alive'].sum()) - int((post['alive'] == 1).sum() - post['is_new'].sum())}")
    print(f"  New (grown)         : {int(post['is_new'].sum())}")
    print(f"    of which alive    : {n_new_alive}")
    print(f"  Strong (cond≥50)    : {len(strong)}")
    print(f"  Max (cond==100)     : {int((post[(post['alive']==1)]['conductance'] == 100).sum())}")


if __name__ == "__main__":
    prefix = sys.argv[1] if len(sys.argv) > 1 else "phase2_f_fixed"
    main(prefix)
