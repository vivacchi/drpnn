//! M1 3D 単体 ホワイトノイズ識別 評価 (M0 バイパス)
//!
//! 目的: 「3D 版で within=1.000 飽和」 が
//!   解釈 X (アーキテクチャ問題、 入力非依存) か
//!   解釈 Y (音素特異、 M0 蝸牛応答が似すぎ) かを切り分ける診断テスト
//!
//! 設計:
//!   - M0 蝸牛を バイパス: 直接 input neuron に整数電流を注入
//!   - 5 種の固定 seed で 600-step × 20-channel の独立白色雑音パターンを事前生成
//!     → 互いに無相関、 周波数構造なし、 完全に統計的に異なる入力
//!   - 評価手順は音素版 (m0_m1_3d_pipeline) と完全同一 (within/between/active outputs)
//!
//! 判定:
//!   - within<1.0 + between<within → M1 は入力に応答している (解釈 Y、 音素特異)
//!   - within=1.0 全域飽和 → M1 が入力非依存 (解釈 X、 アーキテクチャ問題)
//!
//! CLI: cargo run --release --bin m1_3d_whitenoise -- [n_train] [fanout] [mode] [grid_d]
//!   例: cargo run --release --bin m1_3d_whitenoise -- 5000 80 random 22

use spiking_brain::phase3_3d::thermo_network_3d::{
    ThermoNetwork3d, ThermoNetwork3dConfig, OverconnectMode,
};
use spiking_brain::trace::{cosine_similarity, OutputTrace};
use rand::prelude::*;
use std::fs::File;
use std::io::Write as IoWrite;

const TRIAL_DURATION_MS: f64 = 300.0;
const DT_MS: f64 = 0.5;
const TRIAL_STEPS: usize = (TRIAL_DURATION_MS / DT_MS) as usize;  // 600 step
const FINGERPRINT_BIN_WIDTH_MS: f64 = 10.0;
const N_INPUT_CH: usize = 20;
const N_PATTERNS: usize = 5;

/// 白色雑音パターン: 600 step × 20 ch の整数電流系列
type NoisePattern = Vec<[i32; N_INPUT_CH]>;

/// 1 seed から 1 noise pattern を生成 (決定論的)
/// 値域: 0..=3 (蝸牛出力と同程度のスケール、 平均 1.5)
fn generate_pattern(seed: u64) -> NoisePattern {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..TRIAL_STEPS)
        .map(|_| {
            let mut row = [0i32; N_INPUT_CH];
            for ch in 0..N_INPUT_CH {
                row[ch] = rng.gen_range(0..=3);
            }
            row
        })
        .collect()
}

/// 1 試行: 事前生成パターンを M1 input neurons に直接注入
fn present_pattern(
    net: &mut ThermoNetwork3d,
    pattern: &NoisePattern,
) -> Vec<(usize, f64)> {
    net.reset_trial_state();
    let trial_start_t = net.current_time;
    let mut out_log: Vec<(usize, f64)> = Vec::new();

    for step in 0..TRIAL_STEPS {
        let ext: Vec<i32> = pattern[step].to_vec();
        let fired = net.step(&ext);
        for nid in fired {
            if let Some(oi) = net.output_index_of(nid) {
                let t_rel = (net.current_time - trial_start_t) as f64 * DT_MS;
                out_log.push((oi, t_rel));
            }
        }
    }
    out_log
}

fn fingerprint(log: &[(usize, f64)], n_out: usize) -> Vec<f64> {
    let mut tr = OutputTrace::new(n_out, 50.0);
    for &(oi, t) in log {
        tr.record_spike(oi, t);
    }
    tr.time_binned_fingerprint(TRIAL_DURATION_MS, FINGERPRINT_BIN_WIDTH_MS)
}

fn mean_pairwise(fps: &[Vec<f64>]) -> f64 {
    let mut sum = 0.0;
    let mut n = 0;
    for i in 0..fps.len() {
        for j in (i + 1)..fps.len() {
            sum += cosine_similarity(&fps[i], &fps[j]);
            n += 1;
        }
    }
    if n == 0 { 0.0 } else { sum / n as f64 }
}

fn mean_between(a: &[Vec<f64>], b: &[Vec<f64>]) -> f64 {
    let mut sum = 0.0;
    let mut n = 0;
    for fa in a {
        for fb in b {
            sum += cosine_similarity(fa, fb);
            n += 1;
        }
    }
    if n == 0 { 0.0 } else { sum / n as f64 }
}

fn compute_selectivity(per_pat_fps: &[Vec<Vec<f64>>]) -> (f64, f64, f64) {
    let within: f64 = per_pat_fps.iter()
        .filter(|fps| fps.len() >= 2)
        .map(|fps| mean_pairwise(fps))
        .sum::<f64>() / per_pat_fps.iter().filter(|fps| fps.len() >= 2).count().max(1) as f64;
    let mut bs = 0.0;
    let mut bn = 0;
    for i in 0..per_pat_fps.len() {
        for j in (i+1)..per_pat_fps.len() {
            bs += mean_between(&per_pat_fps[i], &per_pat_fps[j]);
            bn += 1;
        }
    }
    let between = if bn == 0 { 0.0 } else { bs / bn as f64 };
    (within - between, within, between)
}

fn evaluate(
    net: &mut ThermoNetwork3d,
    patterns: &[NoisePattern],
    n_sample: usize,
    label: &str,
) {
    let n_out = net.output_neurons.len();
    let mut per_pat_fps: Vec<Vec<Vec<f64>>> =
        vec![Vec::with_capacity(n_sample); patterns.len()];
    let mut per_pat_hits: Vec<Vec<u32>> =
        vec![vec![0u32; n_out]; patterns.len()];
    let mut per_pat_spikes: Vec<u64> = vec![0u64; patterns.len()];
    let mut all_spike_times_ms: Vec<f64> = Vec::new();

    for _ in 0..n_sample {
        for (pi, pat) in patterns.iter().enumerate() {
            let log = present_pattern(net, pat);
            per_pat_spikes[pi] += log.len() as u64;
            let mut fired_any = vec![false; n_out];
            for &(oi, t) in &log {
                fired_any[oi] = true;
                all_spike_times_ms.push(t);
            }
            for ni in 0..n_out {
                if fired_any[ni] { per_pat_hits[pi][ni] += 1; }
            }
            per_pat_fps[pi].push(fingerprint(&log, n_out));
        }
    }

    let (selectivity, within, between) = compute_selectivity(&per_pat_fps);
    let active: usize = (0..n_out).filter(|&ni|
        (0..patterns.len()).any(|pi| per_pat_hits[pi][ni] > 0)
    ).count();

    println!("\n  -- {label} --");
    println!("    selectivity   : {:.3}  (within {:.3} - between {:.3})",
        selectivity, within, between);
    println!("    active outputs: {} / {}", active, n_out);
    print!("    hit /pattern  : ");
    for (pi, _pat) in patterns.iter().enumerate() {
        let h = per_pat_hits[pi].iter().filter(|&&c| c > 0).count();
        print!("noise{}:{}/{}  ", pi, h, n_out);
    }
    println!("(across {n_sample} samples)");
    print!("    total spikes  : ");
    for (pi, _pat) in patterns.iter().enumerate() {
        print!("noise{}:{}  ", pi, per_pat_spikes[pi]);
    }
    println!();

    if !all_spike_times_ms.is_empty() {
        let mut times = all_spike_times_ms.clone();
        times.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let n = times.len();
        let p = |q: f64| times[((q * (n as f64 - 1.0)).round() as usize).min(n - 1)];
        let mean: f64 = times.iter().sum::<f64>() / n as f64;
        println!("    出力スパイク時刻分布 (ms):");
        println!("      p10={:.1}  p25={:.1}  p50={:.1}  p75={:.1}  p90={:.1}  mean={:.1}  (n={})",
            p(0.10), p(0.25), p(0.50), p(0.75), p(0.90), mean, n);
    }
}

fn evaluate_internal(
    net: &mut ThermoNetwork3d,
    patterns: &[NoisePattern],
    n_sample: usize,
) {
    let input_set: std::collections::HashSet<usize> =
        net.input_neurons.iter().copied().collect();
    let internal_ids: Vec<usize> = (0..net.n_neurons())
        .filter(|i| !input_set.contains(i)).collect();
    let internal_idx_map: std::collections::HashMap<usize, usize> = internal_ids.iter()
        .enumerate().map(|(idx, &nid)| (nid, idx)).collect();
    let n_internal = internal_ids.len();

    println!("  内部ニューロン数 (出力含む): {}", n_internal);

    let mut per_pat_fps: Vec<Vec<Vec<f64>>> =
        vec![Vec::with_capacity(n_sample); patterns.len()];

    for _ in 0..n_sample {
        for (pi, pat) in patterns.iter().enumerate() {
            net.reset_trial_state();
            let trial_start = net.current_time;
            let mut log: Vec<(usize, f64)> = Vec::new();
            for step in 0..TRIAL_STEPS {
                let ext: Vec<i32> = pat[step].to_vec();
                let fired = net.step(&ext);
                for nid in fired {
                    if let Some(&internal_idx) = internal_idx_map.get(&nid) {
                        let t_rel = (net.current_time - trial_start) as f64 * DT_MS;
                        log.push((internal_idx, t_rel));
                    }
                }
            }
            let mut tr = OutputTrace::new(n_internal, 50.0);
            for &(internal_idx, t) in &log {
                tr.record_spike(internal_idx, t);
            }
            let fp = tr.time_binned_fingerprint(TRIAL_DURATION_MS, FINGERPRINT_BIN_WIDTH_MS);
            per_pat_fps[pi].push(fp);
            let _ = pi;
        }
    }

    let (selectivity, within, between) = compute_selectivity(&per_pat_fps);
    println!("  -- POST (internal {n_internal}) --");
    println!("    selectivity   : {:.3}  (within {:.3} - between {:.3})",
        selectivity, within, between);
    println!("    fingerprint dim: {} × 30 bin = {}", n_internal, n_internal * 30);
}

fn main() {
    let n_train: usize = std::env::args().nth(1)
        .and_then(|s| s.parse().ok()).unwrap_or(100);
    let input_fanout_override: Option<usize> = std::env::args().nth(2)
        .and_then(|s| s.parse().ok());
    let overconnect_arg: String = std::env::args().nth(3)
        .unwrap_or_else(|| "random".to_string());
    let (overconnect_mode, overconnect_fanout): (OverconnectMode, usize) =
        match overconnect_arg.as_str() {
            "hcp" => (OverconnectMode::HcpLocal, 12),
            "hcp2hop" => (OverconnectMode::HcpLocalPlus2hop, 40),
            "full" => (OverconnectMode::FullPair, 9999),
            _ => (OverconnectMode::Random, 40),
        };
    let grid_d_override: Option<i32> = std::env::args().nth(4)
        .and_then(|s| s.parse().ok());

    let n_sample: usize = 20;
    let snap_interval: usize = if n_train >= 100_000 {
        n_train / 200
    } else if n_train >= 500 {
        500
    } else {
        (n_train / 10).max(10)
    };

    println!("== M1 3D 単体 ホワイトノイズ識別 (M0 バイパス) ==");
    println!("  目的: within=1.000 飽和の診断 (アーキテクチャ問題か入力特異か)");
    println!("  入力: 5 種の独立白色雑音 (固定 seed、 0..=3 整数、 600 step × 20 ch)");

    let mut cfg = match grid_d_override {
        Some(gd) => {
            println!("  ★ grid_d オーバーライド: {} (デフォルト 22)", gd);
            ThermoNetwork3dConfig::for_grid(5, 6, gd)
        }
        None => ThermoNetwork3dConfig::default(),
    };
    if let Some(fanout) = input_fanout_override {
        cfg.input_fanout = fanout;
        println!("  ★ input_fanout オーバーライド: {} (デフォルト 80)", fanout);
    }
    cfg.overconnect_mode = overconnect_mode;
    cfg.overconnect_fanout = overconnect_fanout;
    println!("  ★ overconnect_mode: {:?} (fanout={})",
        overconnect_mode, overconnect_fanout);

    let mut net = ThermoNetwork3d::new(cfg);

    // 5 ノイズパターン生成 (seed 1000..=1004)
    println!("\n  5 ノイズパターン生成中 (seed 1000-1004)...");
    let patterns: Vec<NoisePattern> = (0..N_PATTERNS)
        .map(|i| generate_pattern(1000 + i as u64))
        .collect();
    for (i, pat) in patterns.iter().enumerate() {
        let sum: i64 = pat.iter()
            .flat_map(|row| row.iter().copied())
            .map(|v| v as i64).sum();
        let total_ch_steps = pat.len() * N_INPUT_CH;
        println!("    noise{}: {} step × {} ch、 平均電流 = {:.2}",
            i, pat.len(), N_INPUT_CH, sum as f64 / total_ch_steps as f64);
    }
    // パターン間の相関確認 (cosine similarity)
    println!("\n  パターン間相関 (cosine):");
    for i in 0..N_PATTERNS {
        let flat_i: Vec<f64> = patterns[i].iter()
            .flat_map(|row| row.iter().copied().map(|v| v as f64))
            .collect();
        for j in (i+1)..N_PATTERNS {
            let flat_j: Vec<f64> = patterns[j].iter()
                .flat_map(|row| row.iter().copied().map(|v| v as f64))
                .collect();
            let sim = cosine_similarity(&flat_i, &flat_j);
            println!("    noise{} vs noise{}: {:.4}", i, j, sim);
        }
    }

    println!("\n  ネットワーク:");
    println!("    neurons   : {}", net.n_neurons());
    println!("    synapses  : {} (open={})", net.n_synapses(), net.n_open_synapses());

    let mut csv = File::create("phase3_3d_whitenoise_snapshots.csv").expect("csv");
    writeln!(csv, "trial,within,selectivity,active,silent_ratio,entropy_mean,enthalpy_mean,conductance_mean,open_syn,total_syn,axons_grown,axons_pruned,sparsity").unwrap();

    println!("\n== Phase 1: 訓練前評価 ==");
    evaluate(&mut net, &patterns, n_sample, "PRE");

    println!("\n== Phase 2: 訓練 {n_train} 試行 (パターンランダム選択) ==");
    println!("  snap_interval = {snap_interval}");
    let mut rng = StdRng::seed_from_u64(42);
    for trial in 1..=n_train {
        let pi = rng.gen_range(0..patterns.len());
        let _ = present_pattern(&mut net, &patterns[pi]);

        if trial % snap_interval == 0 || trial == n_train {
            let mut per_pat_fps: Vec<Vec<Vec<f64>>> =
                vec![Vec::with_capacity(5); patterns.len()];
            let n_out = net.output_neurons.len();
            let mut per_pat_hits: Vec<Vec<u32>> = vec![vec![0u32; n_out]; patterns.len()];
            for _ in 0..5 {
                for pi in 0..patterns.len() {
                    let log = present_pattern(&mut net, &patterns[pi]);
                    let mut fired_any = vec![false; n_out];
                    for &(oi, _) in &log { fired_any[oi] = true; }
                    for ni in 0..n_out {
                        if fired_any[ni] { per_pat_hits[pi][ni] += 1; }
                    }
                    per_pat_fps[pi].push(fingerprint(&log, n_out));
                }
            }
            let (sel, within, _between) = compute_selectivity(&per_pat_fps);
            let active: usize = (0..n_out).filter(|&ni|
                (0..patterns.len()).any(|pi| per_pat_hits[pi][ni] > 0)
            ).count();
            let silent_ratio = (n_out - active) as f64 / n_out as f64;
            let obs = net.macro_observables();

            println!("  {:>5}  within={:.3}  sel={:.3}  act={}/{}  ent_μ={:.1}  cond_μ={:.1}  open={}  grown/pruned={}/{}",
                trial, within, sel, active, n_out,
                obs.entropy_mean, obs.conductance_mean,
                net.n_open_synapses(), net.axons_grown, net.axons_pruned);

            writeln!(csv, "{},{:.4},{:.4},{},{:.4},{:.2},{:.2},{:.2},{},{},{},{},{:.4}",
                trial, within, sel, active, silent_ratio,
                obs.entropy_mean, obs.enthalpy_mean, obs.conductance_mean,
                net.n_open_synapses(), net.n_synapses(),
                net.axons_grown, net.axons_pruned, obs.sparsity).unwrap();
        }
    }

    println!("\n== Phase 3: 訓練後評価 (出力 layer) ==");
    evaluate(&mut net, &patterns, n_sample, "POST (output)");

    println!("\n══════════════════════════════════════════════════════════");
    println!("== Phase 3b: 内部 layer 全体での再評価 ==");
    println!("══════════════════════════════════════════════════════════");
    evaluate_internal(&mut net, &patterns, n_sample);

    println!("\n══════════════════════════════════════════════════════════");
    println!("  M1 3D 白色雑音識別サマリ");
    println!("  軸索成長 累積={} 刈り取り累積={}", net.axons_grown, net.axons_pruned);
    println!("  open シナプス: {}/{}", net.n_open_synapses(), net.n_synapses());
    println!("  CSV: phase3_3d_whitenoise_snapshots.csv");
}
