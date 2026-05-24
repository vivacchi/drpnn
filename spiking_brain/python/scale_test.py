"""
scale_test.py — ネットワーク規模を振って性能・メモリを測る

実行例:
  python scale_test.py
  python scale_test.py --max 1000000   # 100万ニューロンまで試す
  python scale_test.py --device cpu    # CPU と比較

各規模で:
  - 構築時間
  - VRAM 使用量
  - 1試行の所要時間 (= 600 タイムステップ)
  - リアルタイム比 (1秒シミュレートに何秒かかるか)
を出力。RTX 3060 Laptop 6GB だと 30万ニューロン規模までは余裕で乗るはず。
"""

import argparse
import gc
import time

import torch

from brain_torch import BinaryBrain, Config, make_pattern, present_pattern


def run_scale_point(n_cortex: int, device: str, n_warmup: int = 2, n_trials: int = 5):
    """1つの規模で計測。実行時間と VRAM を返す。"""
    # 入力・出力は規模に応じて適度にスケール
    n_input = max(20, n_cortex // 200)
    n_output = max(40, n_cortex // 100)
    fanout = max(40, min(100, n_cortex // 10))  # 10% 結合密度を上限に

    cfg = Config(
        n_cortex=n_cortex,
        n_input=n_input,
        n_output=n_output,
        cortex_fanout=fanout,
    )

    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    gc.collect()

    t0 = time.time()
    try:
        net = BinaryBrain(cfg, device=device)
    except torch.cuda.OutOfMemoryError:
        return {
            "n_cortex": n_cortex,
            "n_synapses": None,
            "build_sec": None,
            "vram_mb": None,
            "trial_ms": None,
            "realtime_factor": None,
            "status": "OOM",
        }
    build_sec = time.time() - t0

    vram_mb = (
        torch.cuda.max_memory_allocated() / 1024**2 if device == "cuda" else 0.0
    )

    pat = make_pattern(cfg.n_input, 42)

    # ウォームアップ (CUDA kernel コンパイル等を除外)
    for k in range(n_warmup):
        present_pattern(net, pat, trial_seed=k)

    if device == "cuda":
        torch.cuda.synchronize()
    t1 = time.time()
    for k in range(n_trials):
        present_pattern(net, pat, trial_seed=1000 + k)
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.time() - t1
    trial_ms = (elapsed / n_trials) * 1000

    # 1試行 = 300ms シミュレーション
    realtime_factor = 300.0 / trial_ms  # >1 ならリアルタイムより速い

    result = {
        "n_cortex": n_cortex,
        "n_synapses": net.n_syn,
        "build_sec": build_sec,
        "vram_mb": vram_mb,
        "trial_ms": trial_ms,
        "realtime_factor": realtime_factor,
        "status": "OK",
    }
    del net
    if device == "cuda":
        torch.cuda.empty_cache()
    gc.collect()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--max", type=int, default=300_000,
                        help="max cortex neuron count to try")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = "cpu"

    if args.device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        props = torch.cuda.get_device_properties(0)
        print(f"  total VRAM: {props.total_memory/1024**3:.2f} GB")
        print(f"  compute capability: {props.major}.{props.minor}\n")

    # 試す規模: 対数スケール
    sizes = [400, 1_000, 3_000, 10_000, 30_000, 100_000, 300_000, 1_000_000]
    sizes = [s for s in sizes if s <= args.max]

    print(f"{'neurons':>10} {'synapses':>12} {'build':>8} {'VRAM':>10} "
          f"{'trial':>10} {'realtime':>10} {'status':>8}")
    print("-" * 76)

    for n in sizes:
        r = run_scale_point(n, args.device)
        if r["status"] == "OOM":
            print(f"{r['n_cortex']:>10}  (out of memory — stop)")
            break
        print(
            f"{r['n_cortex']:>10}"
            f" {r['n_synapses']:>12}"
            f" {r['build_sec']:>7.2f}s"
            f" {r['vram_mb']:>9.1f}M"
            f" {r['trial_ms']:>9.1f}ms"
            f" {r['realtime_factor']:>9.2f}x"
            f"   {r['status']:>6}"
        )

    print("\n注:")
    print("  - realtime_factor > 1 は「実時間より速くシミュレートできる」")
    print("  - 同じハードで CPU 版と比較すると GPU の効きが分かる")
    print("  - VRAM が 4GB を超えたらディスプレイ用も含めて 6GB のヘッドルームを意識")


if __name__ == "__main__":
    main()
