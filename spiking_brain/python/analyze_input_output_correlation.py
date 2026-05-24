"""Phase 2 Fork F-G1-R1 入出力時系列の逆相関解析.

実験仮説:
  ホワイトノイズ訓練後でも、出力ニューロンは「自己組織的に特定の入力時間構造に
  反応する」ようになっているはず。これを以下で検証:

  1. 訓練後期の出力時系列を短時間 window で区切る
  2. 各 window の出力 fingerprint を計算
  3. 出力 fingerprint 同士の cosine 類似度を計算
  4. 高類似度のペアを抽出
  5. それらのペアの **同タイミング入力** で類似度を比較
  6. 「出力類似 → 入力も類似」を相関で確認

入力:
  phase2_f_whitenoise_input_spikes.csv   step, input_idx
  phase2_f_whitenoise_output_spikes.csv  step, output_idx

出力:
  phase2_f_whitenoise_correlation.png    可視化
  コンソールに統計
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

for font in ["Yu Gothic", "Meiryo", "MS Gothic"]:
    try:
        matplotlib.rcParams["font.family"] = font
        break
    except Exception:
        continue
matplotlib.rcParams["axes.unicode_minus"] = False

# 設定
N_INPUT = 20
N_OUTPUT = 40
WINDOW_STEPS = 200          # 100 ms (DT=0.5ms/step)
BIN_STEPS = 20              # 10 ms (出力 fingerprint 用)
TOP_K_PAIRS = 200           # 出力類似度上位ペア数
TRAINING_FRACTION = 0.7     # 後半 30% のスパイクのみ使用 (訓練後期)


def fingerprint_window(events: np.ndarray, t0: int, t1: int, n_unit: int,
                       bin_steps: int) -> np.ndarray:
    """events: shape (M, 2) of [step, unit_idx]. window [t0, t1) 内のスパイクを
    (n_unit × n_bins) のヒストグラムに集計、flatten して返す."""
    n_bins = (t1 - t0) // bin_steps
    fp = np.zeros((n_unit, n_bins), dtype=np.float32)
    mask = (events[:, 0] >= t0) & (events[:, 0] < t1)
    sub = events[mask]
    for step, idx in sub:
        b = int((step - t0) // bin_steps)
        if 0 <= b < n_bins:
            fp[int(idx), b] += 1.0
    return fp.flatten()


def cosine_sim_matrix(X: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n = np.where(n == 0, 1, n)
    Xn = X / n
    return Xn @ Xn.T


def main() -> None:
    prefix = "phase2_f_whitenoise"
    in_path = Path(f"{prefix}_input_spikes.csv")
    out_path = Path(f"{prefix}_output_spikes.csv")
    if not in_path.exists() or not out_path.exists():
        print(f"ERROR: missing CSV. need {in_path} and {out_path}", file=sys.stderr)
        sys.exit(1)

    in_df = pd.read_csv(in_path)
    out_df = pd.read_csv(out_path)
    in_ev = in_df.to_numpy()  # (N, 2) [step, input_idx]
    out_ev = out_df.to_numpy()  # (M, 2)
    print(f"Loaded: input {len(in_ev):,} events, output {len(out_ev):,} events")

    # 訓練後期のみ
    max_step = max(int(in_ev[:, 0].max()), int(out_ev[:, 0].max()))
    start_step = int(max_step * TRAINING_FRACTION)
    in_ev = in_ev[in_ev[:, 0] >= start_step]
    out_ev = out_ev[out_ev[:, 0] >= start_step]
    print(f"Late phase: step [{start_step:,} .. {max_step:,}], "
          f"input {len(in_ev):,} / output {len(out_ev):,}")

    # window 分割
    n_windows = (max_step - start_step) // WINDOW_STEPS
    n_windows = min(n_windows, 5000)  # 計算量上限
    print(f"Windows: {n_windows} × {WINDOW_STEPS} steps each "
          f"(= {WINDOW_STEPS * 0.5} ms / window)")

    out_fps = np.zeros((n_windows, N_OUTPUT * (WINDOW_STEPS // BIN_STEPS)), dtype=np.float32)
    in_fps = np.zeros((n_windows, N_INPUT * (WINDOW_STEPS // BIN_STEPS)), dtype=np.float32)

    for w in range(n_windows):
        t0 = start_step + w * WINDOW_STEPS
        t1 = t0 + WINDOW_STEPS
        out_fps[w] = fingerprint_window(out_ev, t0, t1, N_OUTPUT, BIN_STEPS)
        in_fps[w]  = fingerprint_window(in_ev,  t0, t1, N_INPUT,  BIN_STEPS)

    # 出力が空 (silent) な window を除外
    nonempty = (out_fps.sum(axis=1) > 0)
    out_fps = out_fps[nonempty]
    in_fps = in_fps[nonempty]
    n_active = len(out_fps)
    print(f"Active output windows: {n_active} / {n_windows} ({n_active/n_windows*100:.1f}%)")

    if n_active < 10:
        print("ERROR: too few active windows", file=sys.stderr)
        sys.exit(1)

    # 出力類似度行列
    out_sim = cosine_sim_matrix(out_fps)
    in_sim = cosine_sim_matrix(in_fps)

    # 上三角の (i, j) ペアと類似度
    iu = np.triu_indices(n_active, k=1)
    out_sims_flat = out_sim[iu]
    in_sims_flat = in_sim[iu]

    # 全体の相関 (出力類似 vs 入力類似)
    valid = ~(np.isnan(out_sims_flat) | np.isnan(in_sims_flat))
    out_v = out_sims_flat[valid]
    in_v = in_sims_flat[valid]
    pearson = np.corrcoef(out_v, in_v)[0, 1]
    print(f"\n=== 全ペア相関 (出力類似 vs 入力類似) ===")
    print(f"  Pearson correlation: {pearson:.4f}")
    print(f"  N pairs            : {len(out_v):,}")

    # 出力類似度上位 K ペア
    top_k = min(TOP_K_PAIRS, len(out_v))
    top_idx = np.argsort(out_v)[-top_k:]
    top_out = out_v[top_idx]
    top_in = in_v[top_idx]
    print(f"\n=== 出力類似度上位 {top_k} ペアの入力類似度 ===")
    print(f"  出力類似度 mean ± std : {top_out.mean():.4f} ± {top_out.std():.4f}")
    print(f"  入力類似度 mean ± std : {top_in.mean():.4f} ± {top_in.std():.4f}")
    print(f"  入力類似度 中央値      : {np.median(top_in):.4f}")

    # ランダム K ペアと比較 (帰無分布)
    rand_idx = np.random.choice(len(out_v), top_k, replace=False)
    rand_in = in_v[rand_idx]
    print(f"\n=== 比較: ランダム {top_k} ペア (帰無) ===")
    print(f"  入力類似度 mean ± std : {rand_in.mean():.4f} ± {rand_in.std():.4f}")
    print(f"  入力類似度 中央値      : {np.median(rand_in):.4f}")

    # 統計検定 (上位入力類似度がランダム分布より有意に高いか)
    from scipy import stats
    t_stat, p_val = stats.ttest_ind(top_in, rand_in, equal_var=False)
    print(f"\n  Welch t-test: t = {t_stat:.3f}, p = {p_val:.4e}")
    if p_val < 0.001:
        print(f"  → 有意 (p < 0.001): 出力類似ペアは入力も有意に類似 ✓")
        print(f"  → 自己組織的に特定入力パターンに反応する装置として動作")
    elif p_val < 0.05:
        print(f"  → やや有意 (p < 0.05)")
    else:
        print(f"  → 有意差なし: 入力との対応関係は弱い")

    # ─── 可視化 ───
    fig, axes = plt.subplots(2, 2, figsize=(13, 11))
    fig.suptitle(
        f"Phase 2 Fork F-G1-R1: ホワイトノイズ訓練後の入出力相関 "
        f"(後期 30%, {n_active} active windows)",
        fontsize=13, fontweight="bold"
    )

    # (1) 出力類似 vs 入力類似 散布図
    ax = axes[0, 0]
    sample = np.random.choice(len(out_v), min(20000, len(out_v)), replace=False)
    ax.scatter(out_v[sample], in_v[sample], s=2, alpha=0.2, c="C0")
    ax.scatter(top_out, top_in, s=10, alpha=0.7, c="C3", label=f"top {top_k} (out類似)")
    ax.set_xlabel("output fingerprint 類似度 (cosine)")
    ax.set_ylabel("input fingerprint 類似度 (cosine)")
    ax.set_title(f"(1) 出力類似 vs 入力類似 (Pearson r = {pearson:.3f})")
    ax.legend()
    ax.grid(alpha=0.3)
    lim = (-0.05, 1.05)
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.plot(lim, lim, ":", color="gray", alpha=0.5)

    # (2) ヒストグラム比較
    ax = axes[0, 1]
    ax.hist(top_in, bins=30, alpha=0.6, color="C3",
            label=f"top {top_k} 出力類似ペアの入力類似度", density=True)
    ax.hist(rand_in, bins=30, alpha=0.5, color="C7",
            label="ランダムペア (帰無)", density=True)
    ax.axvline(top_in.mean(), color="C3", linestyle="--",
               label=f"top mean = {top_in.mean():.3f}")
    ax.axvline(rand_in.mean(), color="C7", linestyle="--",
               label=f"random mean = {rand_in.mean():.3f}")
    ax.set_xlabel("input fingerprint 類似度 (cosine)")
    ax.set_ylabel("density")
    ax.set_title(f"(2) 入力類似度分布: top vs random (t={t_stat:.2f}, p={p_val:.2e})")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # (3) 出力類似度の分布
    ax = axes[1, 0]
    ax.hist(out_v, bins=50, color="C0", alpha=0.7)
    ax.set_xlabel("output fingerprint 類似度 (cosine)")
    ax.set_ylabel("count")
    ax.set_title(f"(3) 全 {len(out_v):,} ペアの出力類似度分布")
    ax.axvline(np.median(out_v), color="C3", linestyle="--",
               label=f"median = {np.median(out_v):.3f}")
    ax.legend()
    ax.grid(alpha=0.3)

    # (4) 入力類似度の分布
    ax = axes[1, 1]
    ax.hist(in_v, bins=50, color="C2", alpha=0.7)
    ax.set_xlabel("input fingerprint 類似度 (cosine)")
    ax.set_ylabel("count")
    ax.set_title(f"(4) 全 {len(in_v):,} ペアの入力類似度分布")
    ax.axvline(np.median(in_v), color="C3", linestyle="--",
               label=f"median = {np.median(in_v):.3f}")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    out_fig = Path(f"{prefix}_correlation.png")
    plt.savefig(out_fig, dpi=130, bbox_inches="tight")
    print(f"\nSaved: {out_fig.resolve()}")


if __name__ == "__main__":
    main()
