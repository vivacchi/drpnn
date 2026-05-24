"""
brain_torch.py — PyTorch + CUDA 版 1ビット Spiking Neural Network

Rust 版 (binary_brain) と同じアルゴリズムを GPU 上で並列実行する。
ターゲット: RTX 3060 Laptop 6GB (CUDA 12.6)

設計方針:
  - シナプスは Structure-of-Arrays (1D テンソル群) として保持
  - スパイク伝送・STDP・eligibility 更新を全シナプス一括ベクトル演算
  - 配送リングバッファは [max_delay, n_neurons] の 2D テンソル
  - scatter_add で並列イベント配送
  - per-step 一律 eligibility decay (lazy 不要、GPU 上で安い)

Rust 版との対応:
  Network (Rust)               <-> BinaryBrain (Python)
  Synapse fields               <-> syn_pre/post/delay/exists/...
  step(ext_pulses)             <-> step(ext_pulses)
  apply_reward(r)              <-> apply_reward(r)

実行:
  python brain_torch.py              # CUDA 自動検出
  python brain_torch.py --device cpu # 強制 CPU
  python brain_torch.py --neurons 100000  # スケール変更
"""

import argparse
import math
import random
import time
from dataclasses import dataclass, field
from typing import List, Optional, Set, Tuple

import numpy as np
import torch


# =============================================================================
# Config
# =============================================================================
@dataclass
class Config:
    # ネットワーク構成
    n_cortex: int = 400
    n_input: int = 20
    n_output: int = 40
    input_fanout: int = 80
    cortex_fanout: int = 40
    dt_ms: float = 0.5
    seed: int = 7

    # ニューロン (1ビット LIF カウンタモデル)
    exc_threshold: int = 80
    inh_threshold: int = 40
    exc_leak: int = 2
    inh_leak: int = 3
    exc_refr_steps: int = 4
    inh_refr_steps: int = 2
    spike_gain: int = 10  # 興奮性スパイク1本がカウンタに与える寄与
    inh_gain: int = 15    # 抑制性スパイク1本

    # 視床ノイズ
    noise_prob: float = 0.005
    noise_gain: int = 5

    # R-STDP
    tau_stdp_ms: float = 20.0
    tau_e_ms: float = 1000.0
    r_a_plus: float = 0.1
    open_threshold: float = 1.5    # 報酬>0 で結線を「開く」eligibility 閾値
    close_threshold: float = 1.5
    causal_window_ms: float = 80.0


# =============================================================================
# Network
# =============================================================================
class BinaryBrain:
    """1ビット SNN, GPU 上の全テンソル実装"""

    def __init__(self, cfg: Config, device: str = "cuda"):
        self.cfg = cfg
        if device == "cuda" and not torch.cuda.is_available():
            print("[warn] CUDA not available, falling back to CPU")
            device = "cpu"
        self.device = torch.device(device)

        # 再現性のため初期化は CPU 上で
        gen = torch.Generator(device="cpu")
        gen.manual_seed(cfg.seed)

        n_exc = int(cfg.n_cortex * 0.8)
        n_inh = cfg.n_cortex - n_exc
        n_total = cfg.n_input + cfg.n_cortex
        self.n_total = n_total

        # ニューロン種別
        is_inh = torch.zeros(n_total, dtype=torch.bool)
        is_inh[cfg.n_input + n_exc:] = True
        is_input = torch.zeros(n_total, dtype=torch.bool)
        is_input[:cfg.n_input] = True

        # ニューロン状態 (Structure of Arrays)
        threshold = torch.full((n_total,), cfg.exc_threshold, dtype=torch.int32)
        threshold[is_inh] = cfg.inh_threshold
        leak = torch.full((n_total,), cfg.exc_leak, dtype=torch.int32)
        leak[is_inh] = cfg.inh_leak
        refr_period = torch.full((n_total,), cfg.exc_refr_steps, dtype=torch.int32)
        refr_period[is_inh] = cfg.inh_refr_steps

        self.is_inh = is_inh.to(self.device)
        self.is_input = is_input.to(self.device)
        self.threshold = threshold.to(self.device)
        self.leak = leak.to(self.device)
        self.refr_period = refr_period.to(self.device)
        self.counter = torch.zeros(n_total, dtype=torch.int32, device=self.device)
        self.refr_remain = torch.zeros(n_total, dtype=torch.int32, device=self.device)
        self.last_spike = torch.full(
            (n_total,), -1e9, dtype=torch.float32, device=self.device
        )

        # シナプス構築 (ベクトル化、大規模で重要)
        # 設計上の妥協: 「重複なし」サンプリングではなく with-replacement で生成
        # (典型 fanout=40 in pool=400+ で重複率 ~1%、許容範囲)
        exc_start = cfg.n_input
        exc_end = cfg.n_input + n_exc
        inh_start = exc_end
        inh_end = n_total
        cortex_range = inh_end - exc_start  # 興奮性 + 抑制性合計

        # (1) 入力 → 皮質: 固定、強駆動、可塑性なし
        n_inp_syn = cfg.n_input * cfg.input_fanout
        pre_inp = torch.arange(0, cfg.n_input).repeat_interleave(cfg.input_fanout)
        post_inp = torch.randint(0, cortex_range, (n_inp_syn,), generator=gen) + exc_start
        delay_inp = torch.randint(2, 21, (n_inp_syn,), generator=gen)
        exists_inp = torch.ones(n_inp_syn, dtype=torch.bool)
        is_inh_inp = torch.zeros(n_inp_syn, dtype=torch.bool)
        plastic_inp = torch.zeros(n_inp_syn, dtype=torch.bool)

        # (2) 興奮性皮質 → 全皮質: 可塑、初期は半分くらい結線
        n_exc_syn = n_exc * cfg.cortex_fanout
        pre_exc = torch.arange(exc_start, exc_end).repeat_interleave(cfg.cortex_fanout)
        post_exc = torch.randint(0, cortex_range, (n_exc_syn,), generator=gen) + exc_start
        # セルフループ除去 (再サンプル、稀)
        self_loop = post_exc == pre_exc
        max_retry = 5
        while self_loop.any() and max_retry > 0:
            n_redo = int(self_loop.sum().item())
            new_post = torch.randint(0, cortex_range, (n_redo,), generator=gen) + exc_start
            post_exc[self_loop] = new_post
            self_loop = post_exc == pre_exc
            max_retry -= 1
        delay_exc = torch.randint(2, 41, (n_exc_syn,), generator=gen)
        exists_exc = torch.rand(n_exc_syn, generator=gen) < 0.5
        is_inh_exc = torch.zeros(n_exc_syn, dtype=torch.bool)
        plastic_exc = torch.ones(n_exc_syn, dtype=torch.bool)

        # (3) 抑制性 → 興奮性: 固定、短遅延
        n_inh_syn = n_inh * cfg.cortex_fanout
        pre_inh = torch.arange(inh_start, inh_end).repeat_interleave(cfg.cortex_fanout)
        post_inh = torch.randint(0, n_exc, (n_inh_syn,), generator=gen) + exc_start
        delay_inh = torch.full((n_inh_syn,), 2, dtype=torch.int64)
        exists_inh = torch.ones(n_inh_syn, dtype=torch.bool)
        is_inh_inh = torch.ones(n_inh_syn, dtype=torch.bool)
        plastic_inh = torch.zeros(n_inh_syn, dtype=torch.bool)

        # 結合
        syn_pre = torch.cat([pre_inp, pre_exc, pre_inh]).long()
        syn_post = torch.cat([post_inp, post_exc, post_inh]).long()
        syn_delay = torch.cat([delay_inp, delay_exc, delay_inh]).long()
        syn_exists = torch.cat([exists_inp, exists_exc, exists_inh])
        syn_is_inh = torch.cat([is_inh_inp, is_inh_exc, is_inh_inh])
        syn_plastic = torch.cat([plastic_inp, plastic_exc, plastic_inh])

        self.n_syn = syn_pre.numel()
        self.syn_pre = syn_pre.to(self.device)
        self.syn_post = syn_post.to(self.device)
        self.syn_delay = syn_delay.to(self.device)
        self.syn_exists = syn_exists.to(self.device)
        self.syn_is_inh = syn_is_inh.to(self.device)
        self.syn_plastic = syn_plastic.to(self.device)
        self.syn_eligibility = torch.zeros(self.n_syn, dtype=torch.float32, device=self.device)

        self.max_delay = int(syn_delay.max().item()) + 1
        self.delivery_exc = torch.zeros(
            (self.max_delay, n_total), dtype=torch.int32, device=self.device
        )
        self.delivery_inh = torch.zeros(
            (self.max_delay, n_total), dtype=torch.int32, device=self.device
        )
        self.delivery_head = 0

        self.input_neurons = torch.arange(0, cfg.n_input, dtype=torch.int64, device=self.device)
        self.output_neurons = torch.arange(
            exc_start, exc_start + cfg.n_output, dtype=torch.int64, device=self.device
        )

        self.t = 0.0
        self.reconfig_count = 0

        # ノイズ生成器
        self.noise_gen = torch.Generator(device=self.device)
        self.noise_gen.manual_seed(99)

        # 定数 (per-step decay)
        self.decay_per_step = math.exp(-cfg.dt_ms / cfg.tau_e_ms)

    # -------- 状態リセット (重みは保持、膜状態のみクリア) --------
    def reset_state(self, noise_seed: Optional[int] = None) -> None:
        self.counter.zero_()
        self.refr_remain.zero_()
        self.last_spike.fill_(-1e9)
        self.delivery_exc.zero_()
        self.delivery_inh.zero_()
        self.delivery_head = 0
        self.t = 0.0
        if noise_seed is not None:
            self.noise_gen.manual_seed(noise_seed)

    # -------- 1 ステップ進める --------
    @torch.no_grad()
    def step(self, ext_pulses: Optional[torch.Tensor]) -> torch.Tensor:
        """`ext_pulses`: [n_input] int32 テンソル (None なら入力なし)
        戻り値: [n_total] bool 発火マスク"""
        cfg = self.cfg

        # 1. 配送スロットから入力を読み、ゼロクリア
        exc_in = self.delivery_exc[self.delivery_head].clone()
        inh_in = self.delivery_inh[self.delivery_head].clone()
        self.delivery_exc[self.delivery_head].zero_()
        self.delivery_inh[self.delivery_head].zero_()

        # 2. 外部入力 (入力ニューロンに直接加算)
        if ext_pulses is not None:
            exc_in.index_add_(0, self.input_neurons, ext_pulses)

        # 3. 視床ノイズ
        noise = (
            torch.rand(self.n_total, generator=self.noise_gen, device=self.device)
            < cfg.noise_prob
        )

        # 4. カウンタ更新 (不応期にあるニューロンは無視)
        not_refr = self.refr_remain == 0
        # 興奮性入力
        self.counter += (exc_in * cfg.spike_gain) * not_refr.to(torch.int32)
        # 抑制性入力
        self.counter -= (inh_in * cfg.inh_gain) * not_refr.to(torch.int32)
        # ノイズ
        self.counter += (noise.to(torch.int32) * cfg.noise_gain) * not_refr.to(torch.int32)
        # リーク
        self.counter -= self.leak * not_refr.to(torch.int32)
        # 負値はゼロにクランプ
        self.counter.clamp_(min=0)

        # 不応期カウントダウン (>0 のものだけ -1)
        refr_active = self.refr_remain > 0
        self.refr_remain[refr_active] -= 1

        # 5. 発火判定
        fired = (self.counter >= self.threshold) & not_refr
        # 発火後リセット (in-place)
        self.counter.masked_fill_(fired, 0)
        # 不応期セット
        self.refr_remain[fired] = self.refr_period[fired]
        # last_spike 更新
        self.last_spike.masked_fill_(fired, self.t)

        # 6. スパイク配送 (scatter_add)
        pre_fired = fired[self.syn_pre]
        active = pre_fired & self.syn_exists
        if active.any():
            sel = active.nonzero(as_tuple=True)[0]
            slot = (self.delivery_head + self.syn_delay[sel]) % self.max_delay
            post = self.syn_post[sel]
            inh_act = self.syn_is_inh[sel]
            flat = slot * self.n_total + post

            # 興奮性
            if (~inh_act).any():
                exc_flat = flat[~inh_act]
                ones = torch.ones(
                    exc_flat.shape, dtype=torch.int32, device=self.device
                )
                self.delivery_exc.view(-1).scatter_add_(0, exc_flat, ones)
            # 抑制性
            if inh_act.any():
                inh_flat = flat[inh_act]
                ones = torch.ones(
                    inh_flat.shape, dtype=torch.int32, device=self.device
                )
                self.delivery_inh.view(-1).scatter_add_(0, inh_flat, ones)

        # 7. R-STDP eligibility 更新 (全シナプス一括)
        # まず全体に per-step decay
        self.syn_eligibility *= self.decay_per_step

        last_pre = self.last_spike[self.syn_pre]
        last_post = self.last_spike[self.syn_post]
        dt_ltp = self.t - last_pre  # post が今発火 ← pre が dt 前に発火 (LTP)
        dt_ltd = self.t - last_post # pre が今発火 → post が dt 前に発火 (LTD)
        post_fired = fired[self.syn_post]
        pre_fired_now = fired[self.syn_pre]

        window = cfg.causal_window_ms
        tau = cfg.tau_stdp_ms

        ltp_mask = post_fired & (dt_ltp > 0.0) & (dt_ltp < window) & self.syn_plastic
        ltd_mask = pre_fired_now & (dt_ltd > 0.0) & (dt_ltd < window) & self.syn_plastic

        if ltp_mask.any():
            dt_c = torch.clamp(dt_ltp, 0.0, window)
            k = torch.exp(-dt_c / tau) * cfg.r_a_plus
            self.syn_eligibility[ltp_mask] += k[ltp_mask]

        if ltd_mask.any():
            dt_c = torch.clamp(dt_ltd, 0.0, window)
            k = torch.exp(-dt_c / tau) * cfg.r_a_plus
            self.syn_eligibility[ltd_mask] -= k[ltd_mask]

        # 8. 時刻進行
        self.delivery_head = (self.delivery_head + 1) % self.max_delay
        self.t += cfg.dt_ms
        return fired

    # -------- 報酬を適用して構造的可塑性 (= DRP 再構成) を起こす --------
    @torch.no_grad()
    def apply_reward(self, r: float) -> Tuple[int, int]:
        signed_e = r * self.syn_eligibility
        open_mask = (
            (signed_e > self.cfg.open_threshold)
            & (~self.syn_exists)
            & self.syn_plastic
        )
        close_mask = (
            (signed_e < -self.cfg.close_threshold)
            & self.syn_exists
            & self.syn_plastic
        )
        # in-place 更新
        self.syn_exists |= open_mask
        self.syn_exists &= ~close_mask
        n_open = int(open_mask.sum().item())
        n_close = int(close_mask.sum().item())
        self.reconfig_count += n_open + n_close
        self.syn_eligibility.zero_()
        return n_open, n_close

    def n_open(self) -> int:
        return int(self.syn_exists.sum().item())

    def memory_bytes(self) -> int:
        return (
            self.counter.numel() * 4 * 6  # 6 int32 配列分くらい
            + self.syn_pre.numel() * 8 * 3  # 3 int64
            + self.syn_eligibility.numel() * 4  # f32
            + self.syn_exists.numel() * 3  # 3 bool
            + self.delivery_exc.numel() * 4 * 2
        )


# =============================================================================
# パターン生成 / 提示
# =============================================================================
def make_pattern(n_input: int, seed: int) -> torch.Tensor:
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    return torch.rand(n_input, generator=g) * 20.0  # 0-20 ms (CPU テンソル)


@torch.no_grad()
def present_pattern(
    net: BinaryBrain,
    pattern_times: torch.Tensor,
    duration_ms: float = 300.0,
    pulses_per_event: int = 4,
    pulse_width_ms: float = 4.0,
    trial_seed: int = 0,
) -> List[Tuple[int, float]]:
    net.reset_state(noise_seed=trial_seed + 999)
    dt = net.cfg.dt_ms
    n_steps = int(duration_ms / dt)
    pulse_steps = max(1, int(pulse_width_ms / dt))

    fire_steps = (pattern_times / dt).long().to(net.device)
    out_log: List[Tuple[int, float]] = []

    for step in range(n_steps):
        # この時間ステップで点火すべき入力ニューロン
        active = (fire_steps <= step) & (step < fire_steps + pulse_steps)
        ext = (active.to(torch.int32)) * pulses_per_event

        fired = net.step(ext)

        # 出力層のスパイクを抽出 (GPU 上で indexing → CPU に出すのは output だけ)
        out_fired = fired[net.output_neurons]
        if out_fired.any():
            out_idx = out_fired.nonzero(as_tuple=True)[0]
            t_now = net.t
            for oi in out_idx.tolist():
                out_log.append((oi, t_now))

    return out_log


def fingerprint(
    out_log: List[Tuple[int, float]], n_outputs: int, t_end: float, tau: float = 50.0
) -> np.ndarray:
    """出力スパイクログから減衰フィンガープリント (純 NumPy で十分高速)"""
    if not out_log:
        return np.zeros(n_outputs)
    indices = np.fromiter((oi for oi, _ in out_log), dtype=np.int64, count=len(out_log))
    times = np.fromiter((t for _, t in out_log), dtype=np.float32, count=len(out_log))
    weights = np.exp(-(t_end - times) / tau)
    fp = np.zeros(n_outputs, dtype=np.float32)
    np.add.at(fp, indices, weights)
    return fp


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na * nb < 1e-9:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def evaluate_reward(
    out_log: List[Tuple[int, float]],
    target: Set[int],
    window: Tuple[float, float] = (10.0, 250.0),
) -> float:
    hits = sum(1 for oi, t in out_log if oi in target and window[0] <= t <= window[1])
    return 1.0 if hits >= 1 else -0.5


# =============================================================================
# Experiment
# =============================================================================
def run_experiment(cfg: Config, device: str, n_train: int = 200, verbose: bool = True):
    if verbose:
        if torch.cuda.is_available() and device == "cuda":
            print(f"Device: {torch.cuda.get_device_name(0)}")
            props = torch.cuda.get_device_properties(0)
            print(f"  VRAM: {props.total_memory / 1024**3:.1f} GB")
            print(f"  Compute capability: {props.major}.{props.minor}")
        else:
            print(f"Device: CPU")

    t_build = time.time()
    net = BinaryBrain(cfg, device=device)
    if verbose:
        print(f"Network built in {time.time()-t_build:.2f}s")
        print(f"  neurons : {net.n_total}")
        print(f"  synapses: {net.n_syn} total, "
              f"{int(net.syn_plastic.sum().item())} plastic, "
              f"{net.n_open()} initially open")
        if device == "cuda":
            print(f"  VRAM used: {torch.cuda.memory_allocated()/1024**2:.1f} MB")

    pat_a = make_pattern(cfg.n_input, 101)
    pat_b = make_pattern(cfg.n_input, 202)
    target_a: Set[int] = set(range(0, 5))
    target_b: Set[int] = set(range(20, 25))

    n_out = cfg.n_output

    # ===== Phase 1: 訓練前 =====
    if verbose:
        print("\n== Phase 1: pre-training ==")
    fps_a_pre, fps_b_pre = [], []
    hit_a_pre = hit_b_pre = 0
    for trial in range(10):
        log_a = present_pattern(net, pat_a, trial_seed=1000 + trial)
        log_b = present_pattern(net, pat_b, trial_seed=2000 + trial)
        if any(oi in target_a for oi, _ in log_a):
            hit_a_pre += 1
        if any(oi in target_b for oi, _ in log_b):
            hit_b_pre += 1
        fps_a_pre.append(fingerprint(log_a, n_out, 300.0))
        fps_b_pre.append(fingerprint(log_b, n_out, 300.0))
    if verbose:
        print(f"  target hits: A={hit_a_pre}/10  B={hit_b_pre}/10")

    initial_open = net.n_open()

    # ===== Phase 2: 報酬学習 =====
    if verbose:
        print("\n== Phase 2: reward-modulated training ==")
    rng = random.Random(42)
    recent_r: List[float] = []
    t_train = time.time()
    for trial in range(n_train):
        use_a = rng.random() < 0.5
        pat, tgt = (pat_a, target_a) if use_a else (pat_b, target_b)
        log = present_pattern(net, pat, trial_seed=10000 + trial)
        r = evaluate_reward(log, tgt)
        net.apply_reward(r)
        recent_r.append(r)
        if len(recent_r) > 20:
            recent_r.pop(0)
        if verbose and (trial + 1) % 20 == 0:
            avg = sum(recent_r) / len(recent_r)
            print(
                f"  trial {trial+1:>3}/{n_train}  "
                f"recent_avg_r={avg:+.3f}  "
                f"open_syn={net.n_open()}  "
                f"reconfigs={net.reconfig_count}"
            )
    train_time = time.time() - t_train

    # ===== Phase 3: 訓練後 =====
    if verbose:
        print("\n== Phase 3: post-training ==")
    fps_a_post, fps_b_post = [], []
    hit_a_post = hit_b_post = 0
    for trial in range(10):
        log_a = present_pattern(net, pat_a, trial_seed=30000 + trial)
        log_b = present_pattern(net, pat_b, trial_seed=40000 + trial)
        if any(oi in target_a for oi, _ in log_a):
            hit_a_post += 1
        if any(oi in target_b for oi, _ in log_b):
            hit_b_post += 1
        fps_a_post.append(fingerprint(log_a, n_out, 300.0))
        fps_b_post.append(fingerprint(log_b, n_out, 300.0))

    # ===== 結果集計 =====
    def within(fps):
        if len(fps) < 2:
            return 0.0
        s = [cosine(fps[i], fps[j]) for i in range(len(fps)) for j in range(i + 1, len(fps))]
        return float(np.mean(s))

    def between(a, b):
        if not a or not b:
            return 0.0
        s = [cosine(x, y) for x in a for y in b]
        return float(np.mean(s))

    pre_sel = (within(fps_a_pre) + within(fps_b_pre)) / 2 - between(fps_a_pre, fps_b_pre)
    post_sel = (within(fps_a_post) + within(fps_b_post)) / 2 - between(fps_a_post, fps_b_post)

    result = {
        "hit_a_pre": hit_a_pre, "hit_b_pre": hit_b_pre,
        "hit_a_post": hit_a_post, "hit_b_post": hit_b_post,
        "pre_selectivity": pre_sel,
        "post_selectivity": post_sel,
        "initial_open": initial_open,
        "final_open": net.n_open(),
        "total_reconfigs": net.reconfig_count,
        "train_time_sec": train_time,
        "n_neurons": net.n_total,
        "n_synapses": net.n_syn,
    }

    if verbose:
        print(f"\n=== Results ===")
        print(f"  target hit rate:")
        print(f"    A: {hit_a_pre}/10 → {hit_a_post}/10")
        print(f"    B: {hit_b_pre}/10 → {hit_b_post}/10")
        print(f"  selectivity index (within - between):")
        print(f"    pre : {pre_sel:+.3f}")
        print(f"    post: {post_sel:+.3f}")
        print(f"  structural change:")
        print(f"    open synapses: {initial_open} → {net.n_open()}")
        print(f"    total reconfig events: {net.reconfig_count}")
        print(f"  training time: {train_time:.2f}s ({1000*train_time/n_train:.1f} ms/trial)")

    return result, net


# =============================================================================
# Entry point
# =============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--neurons", type=int, default=400,
                        help="cortex neuron count (excludes input layer)")
    parser.add_argument("--input", type=int, default=20)
    parser.add_argument("--output", type=int, default=40)
    parser.add_argument("--fanout", type=int, default=40,
                        help="cortex recurrent fanout")
    parser.add_argument("--train", type=int, default=200, help="training trials")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    cfg = Config(
        n_cortex=args.neurons,
        n_input=args.input,
        n_output=args.output,
        cortex_fanout=args.fanout,
        seed=args.seed,
    )
    run_experiment(cfg, device=args.device, n_train=args.train)


if __name__ == "__main__":
    main()
