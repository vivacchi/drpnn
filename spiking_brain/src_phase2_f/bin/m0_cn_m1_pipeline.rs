//! M0 蝸牛 → M0.5 蝸牛神経核 → M1 (A1) パイプライン
//!
//! 設計: M0_5_COCHLEAR_NUCLEUS_DESIGN.md
//!
//! 診断で判明: M0→M1 直結だと M1 が入力非依存のデフォルトアトラクタに落ちる
//! (per-pair between ≈ 1.0)。 原因は蝸牛出力 (密な包絡 rate) に時間構造が無いこと。
//!
//! 本実装: 間に蝸牛神経核 (3 細胞型) を挟み、 音を時間/周波数/オンセットに分解:
//!   M0 蝸牛 (20ch) → M0.5 (Octopus 4 + Bushy 20 + Stellate 20 = 44ch) → M1 (44入力)
//!
//! 検証: per-pair M1 出力 between が 1.0 から下がるか (アトラクタ脱出)
//!
//! CLI: cargo run --release --bin m0_cn_m1_pipeline [n_train]

use spiking_brain::phase2_f::thermo_network::{ThermoNetwork, ThermoNetworkConfig};
use spiking_brain::phase2_f::cochlea::{Cochlea, SAMPLES_PER_STEP};
use spiking_brain::phase2_f::cochlear_nucleus::CochlearNucleus;
use spiking_brain::phase2_f::phoneme_synth::{
    standard_syllables, synth_syllable_scaled, LfsrNoise, Syllable,
};
use spiking_brain::trace::{cosine_similarity, OutputTrace};
use rand::prelude::*;

const TRIAL_DURATION_MS: f64 = 300.0;
const DT_MS: f64 = 0.5;
const TRIAL_STEPS: usize = (TRIAL_DURATION_MS / DT_MS) as usize;
const FINGERPRINT_BIN_WIDTH_MS: f64 = 10.0;

/// 1 trial: M0 → M0.5 → M1
fn present_syllable(
    net: &mut ThermoNetwork,
    cochlea: &mut Cochlea,
    cn: &mut CochlearNucleus,
    waveform: &[i32],
) -> Vec<(usize, f64)> {
    net.reset_trial_state();
    cochlea.reset();
    cn.reset();

    let trial_start_t = net.current_time;
    let mut out_log: Vec<(usize, f64)> = Vec::new();

    for step in 0..TRIAL_STEPS {
        let s0 = step * SAMPLES_PER_STEP;
        let mut samples = [0i32; SAMPLES_PER_STEP];
        for i in 0..SAMPLES_PER_STEP {
            let idx = s0 + i;
            if idx < waveform.len() { samples[i] = waveform[idx]; }
        }
        let coch = cochlea.process_step(&samples);     // 20ch
        let cn_out = cn.process_step(&coch);            // 44ch
        let fired = net.step(&cn_out);                  // M1 (44入力)
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
    for &(oi, t) in log { tr.record_spike(oi, t); }
    tr.time_binned_fingerprint(TRIAL_DURATION_MS, FINGERPRINT_BIN_WIDTH_MS)
}

fn mean_pairwise(fps: &[Vec<f64>]) -> f64 {
    let mut sum = 0.0; let mut n = 0;
    for i in 0..fps.len() {
        for j in (i + 1)..fps.len() {
            sum += cosine_similarity(&fps[i], &fps[j]); n += 1;
        }
    }
    if n == 0 { 0.0 } else { sum / n as f64 }
}

fn mean_between(a: &[Vec<f64>], b: &[Vec<f64>]) -> f64 {
    let mut sum = 0.0; let mut n = 0;
    for fa in a { for fb in b { sum += cosine_similarity(fa, fb); n += 1; } }
    if n == 0 { 0.0 } else { sum / n as f64 }
}

fn compute_selectivity(per_syl_fps: &[Vec<Vec<f64>>]) -> (f64, f64, f64) {
    let within: f64 = per_syl_fps.iter()
        .filter(|fps| fps.len() >= 2)
        .map(|fps| mean_pairwise(fps))
        .sum::<f64>() / per_syl_fps.iter().filter(|fps| fps.len() >= 2).count().max(1) as f64;
    let mut bs = 0.0; let mut bn = 0;
    for i in 0..per_syl_fps.len() {
        for j in (i+1)..per_syl_fps.len() {
            bs += mean_between(&per_syl_fps[i], &per_syl_fps[j]); bn += 1;
        }
    }
    let between = if bn == 0 { 0.0 } else { bs / bn as f64 };
    (within - between, within, between)
}

fn evaluate(
    net: &mut ThermoNetwork, cochlea: &mut Cochlea, cn: &mut CochlearNucleus,
    syllables: &[Syllable], waveforms: &[Vec<i32>], n_sample: usize, label: &str,
) {
    let n_out = net.output_neurons.len();
    let mut per_syl_fps: Vec<Vec<Vec<f64>>> = vec![Vec::with_capacity(n_sample); syllables.len()];
    let mut per_syl_hits: Vec<Vec<u32>> = vec![vec![0u32; n_out]; syllables.len()];
    for _ in 0..n_sample {
        for si in 0..syllables.len() {
            let log = present_syllable(net, cochlea, cn, &waveforms[si]);
            let mut fired_any = vec![false; n_out];
            for &(oi, _) in &log { fired_any[oi] = true; }
            for ni in 0..n_out { if fired_any[ni] { per_syl_hits[si][ni] += 1; } }
            per_syl_fps[si].push(fingerprint(&log, n_out));
        }
    }
    let (sel, within, between) = compute_selectivity(&per_syl_fps);
    let active: usize = (0..n_out).filter(|&ni|
        (0..syllables.len()).any(|si| per_syl_hits[si][ni] > 0)).count();
    println!("\n  -- {label} --");
    println!("    selectivity   : {:.3}  (within {:.3} - between {:.3})", sel, within, between);
    println!("    active outputs: {} / {}", active, n_out);
    print!("    hit /syllable : ");
    for (si, syl) in syllables.iter().enumerate() {
        let h = per_syl_hits[si].iter().filter(|&&c| c > 0).count();
        print!("{}:{}/{}  ", syl.label, h, n_out);
    }
    println!();
}

/// per-pair 混同分析 (M1 平均出力の音素間 between)
fn perpair(
    net: &mut ThermoNetwork, cochlea: &mut Cochlea, cn: &mut CochlearNucleus,
    syllables: &[Syllable], waveforms: &[Vec<i32>], n_sample: usize,
) {
    let n_out = net.output_neurons.len();
    let n_syl = syllables.len();
    let mut mean_fp: Vec<Vec<f64>> = vec![Vec::new(); n_syl];
    for si in 0..n_syl {
        let mut acc: Vec<f64> = Vec::new();
        for _ in 0..n_sample {
            let log = present_syllable(net, cochlea, cn, &waveforms[si]);
            let fp = fingerprint(&log, n_out);
            if acc.is_empty() { acc = vec![0.0; fp.len()]; }
            for (a, v) in acc.iter_mut().zip(fp.iter()) { *a += v; }
        }
        for a in acc.iter_mut() { *a /= n_sample as f64; }
        mean_fp[si] = acc;
    }
    println!("\n  ── per-pair M1 出力 between (平均 fingerprint) ──");
    let mut sum = 0.0; let mut n = 0;
    for i in 0..n_syl {
        for j in (i+1)..n_syl {
            let zi = mean_fp[i].iter().all(|&v| v == 0.0);
            let zj = mean_fp[j].iter().all(|&v| v == 0.0);
            let b = if zi || zj { 0.0 } else { cosine_similarity(&mean_fp[i], &mean_fp[j]) };
            let note = if zi || zj { " (一方無応答)" } else { "" };
            println!("    {}-{}: {:.3}{}", syllables[i].label, syllables[j].label, b, note);
            sum += b; n += 1;
        }
    }
    println!("    平均 = {:.3}  (M0→M1 直結 baseline は 0.956〜1.0)", sum / n as f64);
}

fn main() {
    let n_train: usize = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(100);
    // CLI 第 2 引数: 音素の時間圧縮速度 (1.0 標準 200ms、 3.0 で 67ms = STDP 窓内)
    let speed: f64 = std::env::args().nth(2).and_then(|s| s.parse().ok()).unwrap_or(1.0);
    // CLI 第 3 引数: 減衰遅延倍率 (1 標準、 10 で conductance/vitality 減衰を 10x 遅く)
    // 時間スケール整合のもう半分: 痕跡を提示間隔より長持ちさせ音素固有構造を累積
    let decay_slow: i32 = std::env::args().nth(3).and_then(|s| s.parse().ok()).unwrap_or(1);
    let n_sample = 20;
    let snap_interval = if n_train >= 500 { 500 } else { (n_train / 10).max(10) };

    println!("== M0 蝸牛 → M0.5 蝸牛神経核 → M1 パイプライン ==");
    println!("  M0:   40 帯域蝸牛 (旧 20→40 拡張、 ki/se 分化改善)");
    println!("  M0.5: Octopus 4 + Bushy 40 + Stellate 40 = 84ch (3 細胞型 信号分解)");
    println!("  M1:   84 入力 → 40 出力 (grid 20×26)");
    println!("  音素: pa, ki, tu, se, mo");
    println!("  ★ 時間圧縮速度: {}x (音素長 {:.0}ms、 STDP 窓 80ms)", speed, 200.0 / speed);

    let mut cfg = ThermoNetworkConfig::for_m1_cn_40();
    if decay_slow > 1 {
        cfg.conductance_decay_interval *= decay_slow;
        cfg.vitality_decay_interval *= decay_slow;
        println!("  ★ 減衰 {}x 遅延: conductance {}step, vitality {}step",
            decay_slow, cfg.conductance_decay_interval, cfg.vitality_decay_interval);
    }
    let mut net = ThermoNetwork::new(cfg);
    let mut cochlea = Cochlea::new();
    let mut cn = CochlearNucleus::new();
    let syllables = standard_syllables();
    let mut noise = LfsrNoise::new(0xACE1);
    // 圧縮音素を生成し、 trial (300ms=4800sample) を埋めるよう反復 tile
    // speed=1: 200ms 音素 ×1 (+100ms 無音、 従来)、 speed=3: 67ms 音素を ~4 回反復
    let trial_samples = (TRIAL_DURATION_MS * 16000.0 / 1000.0) as usize;  // 4800
    let waveforms: Vec<Vec<i32>> = syllables.iter()
        .map(|s| {
            let base = synth_syllable_scaled(s, &mut noise, speed);
            if speed <= 1.0 {
                base  // 従来通り (tile しない)
            } else {
                // 圧縮音素を trial 長まで反復 tile (反復曝露)
                let mut tiled = Vec::with_capacity(trial_samples);
                while tiled.len() < trial_samples {
                    tiled.extend_from_slice(&base);
                }
                tiled.truncate(trial_samples);
                tiled
            }
        }).collect();

    println!("\n  M1: neurons={}, synapses={} (in={}, out={})",
        net.n_neurons(), net.n_synapses(), net.input_neurons.len(), net.output_neurons.len());

    println!("\n== Phase 1: 訓練前評価 ==");
    evaluate(&mut net, &mut cochlea, &mut cn, &syllables, &waveforms, n_sample, "PRE");

    println!("\n== Phase 2: 訓練 {n_train} 試行 ==");
    let mut rng = StdRng::seed_from_u64(42);
    for trial in 1..=n_train {
        let si = rng.gen_range(0..syllables.len());
        let _ = present_syllable(&mut net, &mut cochlea, &mut cn, &waveforms[si]);
        if trial % snap_interval == 0 || trial == n_train {
            let mut fps: Vec<Vec<Vec<f64>>> = vec![Vec::with_capacity(5); syllables.len()];
            let n_out = net.output_neurons.len();
            let mut hits: Vec<Vec<u32>> = vec![vec![0u32; n_out]; syllables.len()];
            for _ in 0..5 {
                for si in 0..syllables.len() {
                    let log = present_syllable(&mut net, &mut cochlea, &mut cn, &waveforms[si]);
                    let mut fa = vec![false; n_out];
                    for &(oi, _) in &log { fa[oi] = true; }
                    for ni in 0..n_out { if fa[ni] { hits[si][ni] += 1; } }
                    fps[si].push(fingerprint(&log, n_out));
                }
            }
            let (sel, within, _b) = compute_selectivity(&fps);
            let active: usize = (0..n_out).filter(|&ni|
                (0..syllables.len()).any(|si| hits[si][ni] > 0)).count();
            println!("  trial {:>5}: sel={:.3} within={:.3} active={}/{} open={}",
                trial, sel, within, active, n_out, net.n_open_synapses());
        }
    }

    println!("\n== Phase 3: 訓練後評価 ==");
    evaluate(&mut net, &mut cochlea, &mut cn, &syllables, &waveforms, n_sample, "POST");
    perpair(&mut net, &mut cochlea, &mut cn, &syllables, &waveforms, n_sample);

    println!("\n  軸索成長 {} 刈り取り {} open {}/{}",
        net.axons_grown, net.axons_pruned, net.n_open_synapses(), net.n_synapses());
}
