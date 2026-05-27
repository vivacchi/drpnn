//! M0 蝸牛 → M1 3D (A1 HCP 配置版) パイプライン評価
//!
//! 設計: M1_3D_DESIGN.md
//!
//! 流れ (2D 版 m0_m1_pipeline.rs と完全同じ、 違いは M1 ネットワークが 3D HCP 配置):
//!   音素 (波形 i16, 16kHz)
//!     → M0 蝸牛 (20 帯域フィルタ + 包絡線 + 閾値発火)
//!     → M1 3D input neurons (底面 z=0 中央 20 セル)
//!     → ThermoNetwork3d.step() — HCP 12 近傍配置 + 物理プロセスは 2D と同一
//!     → 出力 30 ニューロン (上面 z=21 完全充填) の発火パターン
//!     → 時間 bin 化 fingerprint (30 出力 × 30 bin) で識別性評価
//!
//! 2D 版 (POST 0.795 / 1200 dim) との直接比較が目的。
//!
//! CLI:
//!   cargo run --release --bin m0_m1_3d_pipeline [n_train]

use spiking_brain::phase3_3d::thermo_network_3d::{
    ThermoNetwork3d, ThermoNetwork3dConfig, OverconnectMode,
};
use spiking_brain::phase2_f::cochlea::{Cochlea, SAMPLES_PER_STEP};
use spiking_brain::phase2_f::phoneme_synth::{
    standard_syllables, synth_syllable, LfsrNoise, Syllable,
};
use spiking_brain::trace::{cosine_similarity, OutputTrace};
use rand::prelude::*;
use std::fs::File;
use std::io::Write as IoWrite;

const TRIAL_DURATION_MS: f64 = 300.0;
const DT_MS: f64 = 0.5;
const TRIAL_STEPS: usize = (TRIAL_DURATION_MS / DT_MS) as usize;  // 600 step
const FINGERPRINT_BIN_WIDTH_MS: f64 = 10.0;

fn present_syllable(
    net: &mut ThermoNetwork3d,
    cochlea: &mut Cochlea,
    waveform: &[i32],
) -> Vec<(usize, f64)> {
    net.reset_trial_state();
    cochlea.reset();

    let trial_start_t = net.current_time;
    let mut out_log: Vec<(usize, f64)> = Vec::new();

    for step in 0..TRIAL_STEPS {
        let s0 = step * SAMPLES_PER_STEP;
        let mut samples = [0i32; SAMPLES_PER_STEP];
        for i in 0..SAMPLES_PER_STEP {
            let idx = s0 + i;
            if idx < waveform.len() {
                samples[i] = waveform[idx];
            }
        }
        let ext = cochlea.process_step(&samples);
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

fn compute_selectivity(per_syl_fps: &[Vec<Vec<f64>>]) -> (f64, f64, f64) {
    let within: f64 = per_syl_fps.iter()
        .filter(|fps| fps.len() >= 2)
        .map(|fps| mean_pairwise(fps))
        .sum::<f64>() / per_syl_fps.iter().filter(|fps| fps.len() >= 2).count().max(1) as f64;
    let mut bs = 0.0;
    let mut bn = 0;
    for i in 0..per_syl_fps.len() {
        for j in (i+1)..per_syl_fps.len() {
            bs += mean_between(&per_syl_fps[i], &per_syl_fps[j]);
            bn += 1;
        }
    }
    let between = if bn == 0 { 0.0 } else { bs / bn as f64 };
    (within - between, within, between)
}

fn evaluate(
    net: &mut ThermoNetwork3d,
    cochlea: &mut Cochlea,
    syllables: &[Syllable],
    waveforms: &[Vec<i32>],
    n_sample: usize,
    label: &str,
) {
    let n_out = net.output_neurons.len();
    let mut per_syl_fps: Vec<Vec<Vec<f64>>> =
        vec![Vec::with_capacity(n_sample); syllables.len()];
    let mut per_syl_hits: Vec<Vec<u32>> =
        vec![vec![0u32; n_out]; syllables.len()];
    let mut per_syl_spikes: Vec<u64> = vec![0u64; syllables.len()];
    let mut all_spike_times_ms: Vec<f64> = Vec::new();  // 出力到達時刻分布

    for _ in 0..n_sample {
        for (si, _syl) in syllables.iter().enumerate() {
            let log = present_syllable(net, cochlea, &waveforms[si]);
            per_syl_spikes[si] += log.len() as u64;
            let mut fired_any = vec![false; n_out];
            for &(oi, t) in &log {
                fired_any[oi] = true;
                all_spike_times_ms.push(t);
            }
            for ni in 0..n_out {
                if fired_any[ni] { per_syl_hits[si][ni] += 1; }
            }
            per_syl_fps[si].push(fingerprint(&log, n_out));
        }
    }

    let (selectivity, within, between) = compute_selectivity(&per_syl_fps);
    let active: usize = (0..n_out).filter(|&ni|
        (0..syllables.len()).any(|si| per_syl_hits[si][ni] > 0)
    ).count();

    println!("\n  -- {label} --");
    println!("    selectivity   : {:.3}  (within {:.3} - between {:.3})",
        selectivity, within, between);
    println!("    active outputs: {} / {}", active, n_out);
    print!("    hit /syllable : ");
    for (si, syl) in syllables.iter().enumerate() {
        let h = per_syl_hits[si].iter().filter(|&&c| c > 0).count();
        print!("{}:{}/{}  ", syl.label, h, n_out);
    }
    println!("(across {n_sample} samples)");
    print!("    total spikes  : ");
    for (si, syl) in syllables.iter().enumerate() {
        print!("{}:{}  ", syl.label, per_syl_spikes[si]);
    }
    println!();

    // 出力到達時刻分布 (3D 経路長伸長による遅延偏り検証)
    if !all_spike_times_ms.is_empty() {
        let mut times = all_spike_times_ms.clone();
        times.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let n = times.len();
        let p = |q: f64| times[((q * (n as f64 - 1.0)).round() as usize).min(n - 1)];
        let mean: f64 = times.iter().sum::<f64>() / n as f64;
        println!("    出力スパイク時刻分布 (ms, 試行 0-{TRIAL_DURATION_MS:.0}ms 内):");
        println!("      p10={:.1}  p25={:.1}  p50={:.1}  p75={:.1}  p90={:.1}  mean={:.1}  (n={})",
            p(0.10), p(0.25), p(0.50), p(0.75), p(0.90), mean, n);
    }
}

fn present_syllable_internal(
    net: &mut ThermoNetwork3d,
    cochlea: &mut Cochlea,
    waveform: &[i32],
    internal_idx_map: &std::collections::HashMap<usize, usize>,
) -> Vec<(usize, f64)> {
    net.reset_trial_state();
    cochlea.reset();
    let trial_start = net.current_time;
    let mut log: Vec<(usize, f64)> = Vec::new();

    for step in 0..TRIAL_STEPS {
        let s0 = step * SAMPLES_PER_STEP;
        let mut samples = [0i32; SAMPLES_PER_STEP];
        for i in 0..SAMPLES_PER_STEP {
            let idx = s0 + i;
            if idx < waveform.len() {
                samples[i] = waveform[idx];
            }
        }
        let ext = cochlea.process_step(&samples);
        let fired = net.step(&ext);
        for nid in fired {
            if let Some(&internal_idx) = internal_idx_map.get(&nid) {
                let t_rel = (net.current_time - trial_start) as f64 * DT_MS;
                log.push((internal_idx, t_rel));
            }
        }
    }
    log
}

fn evaluate_internal(
    net: &mut ThermoNetwork3d,
    cochlea: &mut Cochlea,
    syllables: &[Syllable],
    waveforms: &[Vec<i32>],
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

    let mut per_syl_fps: Vec<Vec<Vec<f64>>> =
        vec![Vec::with_capacity(n_sample); syllables.len()];

    for _ in 0..n_sample {
        for (si, _syl) in syllables.iter().enumerate() {
            let log = present_syllable_internal(
                net, cochlea, &waveforms[si], &internal_idx_map);
            let mut tr = OutputTrace::new(n_internal, 50.0);
            for &(internal_idx, t) in &log {
                tr.record_spike(internal_idx, t);
            }
            let fp = tr.time_binned_fingerprint(TRIAL_DURATION_MS, FINGERPRINT_BIN_WIDTH_MS);
            per_syl_fps[si].push(fp);
        }
    }

    let (selectivity, within, between) = compute_selectivity(&per_syl_fps);
    println!("  -- POST (internal {n_internal}) --");
    println!("    selectivity   : {:.3}  (within {:.3} - between {:.3})",
        selectivity, within, between);
    println!("    fingerprint dim: {} × 30 bin = {}", n_internal, n_internal * 30);
}

fn main() {
    let n_train: usize = std::env::args().nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(100);
    // CLI 第 2 引数: input_fanout のオーバーライド (仮説 A 検証用)
    // 省略時はデフォルト 80。 2D 密度 (3.81) 整合は 120、 強化版は 160。
    let input_fanout_override: Option<usize> = std::env::args().nth(2)
        .and_then(|s| s.parse().ok());
    // CLI 第 3 引数: 過剰接続モード ("random"|"hcp"|"hcp2hop")
    // 省略時 "random" (2D 互換、 3D デフォルト)
    let overconnect_arg: String = std::env::args().nth(3)
        .unwrap_or_else(|| "random".to_string());
    let (overconnect_mode, overconnect_fanout): (OverconnectMode, usize) =
        match overconnect_arg.as_str() {
            "hcp" => (OverconnectMode::HcpLocal, 12),
            "hcp2hop" => (OverconnectMode::HcpLocalPlus2hop, 40),
            "full" => (OverconnectMode::FullPair, 9999),  // 9999 = 候補プール上限まで使う
            _ => (OverconnectMode::Random, 40),
        };
    // CLI 第 4 引数: grid_d (柱の高さ、 E-2 用)
    // 省略時はデフォルト 22 (660 セル)。 11 で柱半分 (330 セル、 経路長半減)。
    let grid_d_override: Option<i32> = std::env::args().nth(4)
        .and_then(|s| s.parse().ok());
    // CLI 第 5 引数: placement ("block" 既定 | "cone")
    let placement: String = std::env::args().nth(5)
        .unwrap_or_else(|| "block".to_string());

    let n_sample: usize = 20;
    let snap_interval: usize = if n_train >= 100_000 {
        n_train / 200
    } else if n_train >= 500 {
        500
    } else {
        (n_train / 10).max(10)
    };

    println!("== M0+M1 3D パイプライン: 音素 5 種識別 (HCP 12 近傍) ==");
    println!("  音素: pa, ki, tu, se, mo");
    println!("  M0: 20 帯域蝸牛 (ERB スケール 50Hz-4kHz)");
    println!("  M1 3D: grid 5×6×22 = 660 セル (柱状 HCP)");
    println!("    底面 z=0 (30): 入力 20 中央 + 内部 10");
    println!("    内部 z=1..20 (600): 興奮 500 + 抑制 110 (18%)");
    println!("    上面 z=21 (30): 出力 30 完全充填");
    println!("  評価: 時間 bin 化 fingerprint (30 出力 × 30 bin = 900 dim)");

    // placement モード判定
    // cone: 5×6×22 cone shape (入力 20 → 出力 30、 561 cells)
    // spindle: 5×8×22 紡錘形 (入力 20 → 中間ピーク 40 → 出力 30、 775 cells)
    // block (default): 5×6×grid_d full block
    let mut cfg = match placement.as_str() {
        "cone" => {
            println!("  ★ placement: cone (5×6×22、 561 cells、 入力 20 → 出力 30 fan-out)");
            ThermoNetwork3dConfig::cone_default()
        }
        "spindle" => {
            println!("  ★ placement: spindle (5×8×22、 775 cells、 入力 20 → 中間ピーク 40 → 出力 30)");
            ThermoNetwork3dConfig::spindle_default()
        }
        _ => match grid_d_override {
            Some(gd) => {
                println!("  ★ grid_d オーバーライド: {} (デフォルト 22)", gd);
                ThermoNetwork3dConfig::for_grid(5, 6, gd)
            }
            None => ThermoNetwork3dConfig::default(),
        },
    };
    if let Some(fanout) = input_fanout_override {
        cfg.input_fanout = fanout;
        println!("  ★ input_fanout オーバーライド: {} (デフォルト 80)", fanout);
    }
    cfg.overconnect_mode = overconnect_mode;
    cfg.overconnect_fanout = overconnect_fanout;
    println!("  ★ overconnect_mode: {:?} (fanout={})",
        overconnect_mode, overconnect_fanout);
    println!("  UP/DOWN 状態: 無効 (dense 入力では過剰刺激、§5.12.7-A)");
    let cfg_for_print = cfg.clone();
    let mut net = ThermoNetwork3d::new(cfg);
    let mut cochlea = Cochlea::new();
    let syllables = standard_syllables();
    let mut noise = LfsrNoise::new(0xACE1);

    println!("\n  音節波形を生成中...");
    let mut waveforms: Vec<Vec<i32>> = Vec::with_capacity(syllables.len());
    for syl in &syllables {
        let wave = synth_syllable(syl, &mut noise);
        let rms = ((wave.iter().map(|&x| (x as i64) * (x as i64)).sum::<i64>()
            / wave.len() as i64) as f64).sqrt();
        println!("    {}: {} samples, RMS={:.0}", syl.label, wave.len(), rms);
        waveforms.push(wave);
    }

    println!("\n  ネットワーク:");
    println!("    neurons   : {}", net.n_neurons());
    println!("    synapses  : {} (open={})", net.n_synapses(), net.n_open_synapses());
    println!("    fanout    : input={}, grid={}×{}×{}",
        cfg_for_print.input_fanout,
        cfg_for_print.grid_w, cfg_for_print.grid_h, cfg_for_print.grid_d);

    // z 層ごとの神経分布 (抑制ランダム配置の検証用)
    println!("\n  z 層別ニューロン分布 (input/exc/inh):");
    println!("    z   | input | exc | inh | inh%");
    println!("    ----|-------|-----|-----|-----");
    let dist = net.layer_distribution();
    let mut total_inh_internal = 0usize;
    let mut total_internal = 0usize;
    for (z, ninp, nexc, ninh) in &dist {
        let layer_total = nexc + ninh;
        let inh_pct = if layer_total > 0 {
            100.0 * (*ninh as f64) / (layer_total as f64)
        } else { 0.0 };
        println!("    {:>3} | {:>5} | {:>3} | {:>3} | {:>4.1}%",
            z, ninp, nexc, ninh, inh_pct);
        total_inh_internal += ninh;
        total_internal += layer_total;
    }
    let global_inh_pct = 100.0 * (total_inh_internal as f64) / (total_internal as f64);
    println!("    合計内部 (出力含む): {} 興奮 + {} 抑制 = {} ({}抑制比 {:.1}%)",
        total_internal - total_inh_internal, total_inh_internal, total_internal,
        if (global_inh_pct - 18.0).abs() < 2.0 { "✓ " } else { "" },
        global_inh_pct);

    let mut csv = File::create("phase3_3d_phoneme_snapshots.csv").expect("csv");
    writeln!(csv, "trial,within,selectivity,active,silent_ratio,entropy_mean,entropy_max,enthalpy_mean,conductance_mean,conductance_max,open_syn,total_syn,axons_grown,axons_pruned,sparsity").unwrap();

    println!("\n== Phase 1: 訓練前評価 ==");
    evaluate(&mut net, &mut cochlea, &syllables, &waveforms, n_sample, "PRE");

    println!("\n== Phase 2: 訓練 {n_train} 試行 (音節ランダム選択) ==");
    println!("  snap_interval = {snap_interval}");

    let mut rng = StdRng::seed_from_u64(42);
    for trial in 1..=n_train {
        let si = rng.gen_range(0..syllables.len());
        let _ = present_syllable(&mut net, &mut cochlea, &waveforms[si]);

        if trial % snap_interval == 0 || trial == n_train {
            let mut per_syl_fps: Vec<Vec<Vec<f64>>> =
                vec![Vec::with_capacity(5); syllables.len()];
            let n_out = net.output_neurons.len();
            let mut per_syl_hits: Vec<Vec<u32>> = vec![vec![0u32; n_out]; syllables.len()];
            for _ in 0..5 {
                for si in 0..syllables.len() {
                    let log = present_syllable(&mut net, &mut cochlea, &waveforms[si]);
                    let mut fired_any = vec![false; n_out];
                    for &(oi, _) in &log { fired_any[oi] = true; }
                    for ni in 0..n_out {
                        if fired_any[ni] { per_syl_hits[si][ni] += 1; }
                    }
                    per_syl_fps[si].push(fingerprint(&log, n_out));
                }
            }
            let (sel, within, _between) = compute_selectivity(&per_syl_fps);
            let active: usize = (0..n_out).filter(|&ni|
                (0..syllables.len()).any(|si| per_syl_hits[si][ni] > 0)
            ).count();
            let silent_ratio = (n_out - active) as f64 / n_out as f64;
            let obs = net.macro_observables();

            println!("  {:>5}  within={:.3}  sel={:.3}  act={}/{}  ent_μ={:.1}  cond_μ={:.1}  open={}  grown/pruned={}/{}",
                trial, within, sel, active, n_out,
                obs.entropy_mean, obs.conductance_mean,
                net.n_open_synapses(), net.axons_grown, net.axons_pruned);

            writeln!(csv, "{},{:.4},{:.4},{},{:.4},{:.2},{},{:.2},{:.2},{},{},{},{},{},{:.4}",
                trial, within, sel, active, silent_ratio,
                obs.entropy_mean, obs.entropy_max, obs.enthalpy_mean,
                obs.conductance_mean, obs.conductance_max,
                net.n_open_synapses(), net.n_synapses(),
                net.axons_grown, net.axons_pruned, obs.sparsity).unwrap();
        }
    }

    println!("\n== Phase 3: 訓練後評価 (出力 layer 30) ==");
    evaluate(&mut net, &mut cochlea, &syllables, &waveforms, n_sample, "POST (output)");

    println!("\n══════════════════════════════════════════════════════════");
    println!("== Phase 3b: 内部 640 ニューロン全体での再評価 ==");
    println!("══════════════════════════════════════════════════════════");
    evaluate_internal(&mut net, &mut cochlea, &syllables, &waveforms, n_sample);

    println!("\n══════════════════════════════════════════════════════════");
    println!("  M0+M1 3D 音素識別サマリ");
    println!("  軸索成長 累積={} 刈り取り累積={}", net.axons_grown, net.axons_pruned);
    println!("  open シナプス: {}/{}", net.n_open_synapses(), net.n_synapses());
    println!("  CSV: phase3_3d_phoneme_snapshots.csv");
}
