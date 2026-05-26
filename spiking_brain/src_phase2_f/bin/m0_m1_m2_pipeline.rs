//! M0 蝸牛 → M1 (A1) → M2 (A2) パイプライン (3 階層統合).
//!
//! 設計: M2_A2_DESIGN.md
//!
//! 階層構造:
//!   音声波形 (16 kHz)
//!     → M0 蝸牛 (20 帯域)         [既存]
//!     → M1 A1 (20→40)             [既存、Fork F-G1-R1 v2]
//!     → M2 A2 (20→40)             [本実装]
//!     → 出力 fingerprint (時間 bin 化)
//!
//! M1 出力 40 → M2 入力 20 の集約:
//!   M2 入力 i は M1 出力 {2i, 2i+1} の発火イベントを電流として受け取る。
//!   これにより「情報の絞り込み」が物理プロセスとして実装される (Bizley & Cohen 2013)。
//!
//! M2 は M1 と同じ ThermoNetwork (Fork F-G1-R1 v2) を使う:
//!   - 同じ 6 原理、 同じ STDP + vitality + 軸索成長 + UP/DOWN
//!   - 異なるインスタンス、 異なる seed、 同じ config テンプレート
//!   - 「階層内の M0 → M1 → M2 は同じ熱力学原理の繰り返し適用」(設計の orthogonality)

use spiking_brain::phase2_f::thermo_network::{ThermoNetwork, ThermoNetworkConfig};
use spiking_brain::phase2_f::cochlea::{Cochlea, SAMPLES_PER_STEP};
use spiking_brain::phase2_f::phoneme_synth::{
    standard_syllables, synth_syllable, LfsrNoise, Syllable,
};
use spiking_brain::trace::{cosine_similarity, OutputTrace};
use rand::prelude::*;
use std::fs::File;
use std::io::Write as IoWrite;

// ─── 定数 ───
const TRIAL_DURATION_MS: f64 = 300.0;
const DT_MS: f64 = 0.5;
const TRIAL_STEPS: usize = (TRIAL_DURATION_MS / DT_MS) as usize;  // 600
const FINGERPRINT_BIN_WIDTH_MS: f64 = 10.0;
const INPUT_CURRENT_M2: i32 = 60;

// ─── M1→M2 集約配線: 40 出力 → 20 入力 ───
// M2 入力 i = M1 出力 {2i, 2i+1} の発火イベントを集約
const M2_INPUT_GROUPING: usize = 2;  // M1 出力 2 個を M2 入力 1 個に集約

// ─── 1 trial 実行 ───

/// 1 trial を実行 (M0 → M1 → M2 階層パイプライン).
/// 戻り値:
///   m1_out_log: (出力 idx, t_rel_ms) for M1 出力
///   m2_out_log: (出力 idx, t_rel_ms) for M2 出力
fn present_syllable_3layer(
    m1: &mut ThermoNetwork,
    m2: &mut ThermoNetwork,
    cochlea: &mut Cochlea,
    waveform: &[i32],
) -> (Vec<(usize, f64)>, Vec<(usize, f64)>) {
    m1.reset_trial_state();
    m2.reset_trial_state();
    cochlea.reset();

    let m1_trial_start = m1.current_time;
    let n_m2_input = m2.input_neurons.len();
    let mut m1_out_log: Vec<(usize, f64)> = Vec::new();
    let mut m2_out_log: Vec<(usize, f64)> = Vec::new();

    for step in 0..TRIAL_STEPS {
        // (1) M0 蝸牛: 8 sample → 20 帯域電流
        let s0 = step * SAMPLES_PER_STEP;
        let mut samples = [0i32; SAMPLES_PER_STEP];
        for i in 0..SAMPLES_PER_STEP {
            let idx = s0 + i;
            if idx < waveform.len() { samples[i] = waveform[idx]; }
        }
        let m0_output = cochlea.process_step(&samples);

        // (2) M1: M0 出力を入力電流として step
        let m1_fired = m1.step(&m0_output);
        for nid in &m1_fired {
            if let Some(oi) = m1.output_index_of(*nid) {
                let t_rel = (m1.current_time - m1_trial_start) as f64 * DT_MS;
                m1_out_log.push((oi, t_rel));
            }
        }

        // (3) M1 → M2 集約: M1 の発火出力 40 → M2 の電流 20
        let mut m2_input = vec![0i32; n_m2_input];
        for &nid in &m1_fired {
            if let Some(oi) = m1.output_index_of(nid) {
                let m2_idx = oi / M2_INPUT_GROUPING;
                if m2_idx < n_m2_input {
                    m2_input[m2_idx] = m2_input[m2_idx].saturating_add(INPUT_CURRENT_M2);
                }
            }
        }

        // (4) M2: 集約電流を入力として step
        let m2_fired = m2.step(&m2_input);
        for nid in m2_fired {
            if let Some(oi) = m2.output_index_of(nid) {
                let t_rel = (m1.current_time - m1_trial_start) as f64 * DT_MS;
                m2_out_log.push((oi, t_rel));
            }
        }
    }
    (m1_out_log, m2_out_log)
}

// ─── fingerprint と評価 ───

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

fn evaluate_3layer(
    m1: &mut ThermoNetwork,
    m2: &mut ThermoNetwork,
    cochlea: &mut Cochlea,
    syllables: &[Syllable],
    waveforms: &[Vec<i32>],
    n_sample: usize,
    label: &str,
) {
    let n_m1_out = m1.output_neurons.len();
    let n_m2_out = m2.output_neurons.len();
    let mut m1_fps: Vec<Vec<Vec<f64>>> =
        vec![Vec::with_capacity(n_sample); syllables.len()];
    let mut m2_fps: Vec<Vec<Vec<f64>>> =
        vec![Vec::with_capacity(n_sample); syllables.len()];

    for _ in 0..n_sample {
        for (si, _syl) in syllables.iter().enumerate() {
            let (m1_log, m2_log) = present_syllable_3layer(m1, m2, cochlea, &waveforms[si]);
            m1_fps[si].push(fingerprint(&m1_log, n_m1_out));
            m2_fps[si].push(fingerprint(&m2_log, n_m2_out));
        }
    }

    let (m1_sel, m1_w, m1_b) = compute_selectivity(&m1_fps);
    let (m2_sel, m2_w, m2_b) = compute_selectivity(&m2_fps);

    println!("\n  -- {label} --");
    println!("    M1 出力 ({} ニューロン): selectivity={:.3} (within {:.3} - between {:.3})",
        n_m1_out, m1_sel, m1_w, m1_b);
    println!("    M2 出力 ({} ニューロン): selectivity={:.3} (within {:.3} - between {:.3})",
        n_m2_out, m2_sel, m2_w, m2_b);
}

// ─── main ───

fn main() {
    let n_train: usize = std::env::args().nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(100);

    println!("== M0+M1+M2 統合パイプライン (3 階層) ==");
    println!("  M0 蝸牛: 20 帯域 ERB");
    println!("  M1 A1  : 20→40 (Fork F-G1-R1 v2)");
    println!("  M2 A2  : 20→40 (M1 出力 40 を集約して 20 入力に)");
    println!("  集約則: M2 入力 i = M1 出力 {{2i, 2i+1}} の発火イベント");
    println!("  音素   : pa, ki, tu, se, mo");

    // M1 (M0 の出力 20 channel を入力に受ける)
    let mut cfg_m1 = ThermoNetworkConfig::default();
    cfg_m1.enable_up_down = false;  // dense 入力なので OFF (§5.12.7-A)
    cfg_m1.seed = 300;
    let mut m1 = ThermoNetwork::new(cfg_m1);

    // M2 (M1 出力 40 を 20 に集約した電流を入力に受ける)
    let mut cfg_m2 = ThermoNetworkConfig::default();
    cfg_m2.enable_up_down = false;  // 同じく dense 入力
    cfg_m2.seed = 301;              // M1 と異なる seed (独立構造)
    let mut m2 = ThermoNetwork::new(cfg_m2);

    let mut cochlea = Cochlea::new();
    let syllables = standard_syllables();
    let mut noise = LfsrNoise::new(0xACE1);

    // 音節波形を事前生成
    println!("\n  音節波形を生成中...");
    let mut waveforms: Vec<Vec<i32>> = Vec::with_capacity(syllables.len());
    for syl in &syllables {
        let wave = synth_syllable(syl, &mut noise);
        waveforms.push(wave);
    }

    println!("\n  M1: neurons={}, synapses={}", m1.n_neurons(), m1.n_synapses());
    println!("  M2: neurons={}, synapses={}", m2.n_neurons(), m2.n_synapses());

    // Phase 1: PRE 評価
    println!("\n== Phase 1: 訓練前評価 ==");
    evaluate_3layer(&mut m1, &mut m2, &mut cochlea, &syllables, &waveforms, 10, "PRE");

    // Phase 2: 訓練
    let snap_interval = (n_train / 10).max(10);
    println!("\n== Phase 2: 訓練 {n_train} 試行 ==");
    let mut rng = StdRng::seed_from_u64(42);
    for trial in 1..=n_train {
        let si = rng.gen_range(0..syllables.len());
        let _ = present_syllable_3layer(&mut m1, &mut m2, &mut cochlea, &waveforms[si]);

        if trial % snap_interval == 0 || trial == n_train {
            let obs_m1 = m1.macro_observables();
            let obs_m2 = m2.macro_observables();
            println!("  trial {:>5}: M1 open={} cond μ={:.1} | M2 open={} cond μ={:.1}",
                trial, m1.n_open_synapses(), obs_m1.conductance_mean,
                m2.n_open_synapses(), obs_m2.conductance_mean);
        }
    }

    // Phase 3: POST 評価
    println!("\n== Phase 3: 訓練後評価 ==");
    evaluate_3layer(&mut m1, &mut m2, &mut cochlea, &syllables, &waveforms, 20, "POST");

    println!("\n══════════════════════════════════════════════════════════");
    println!("  M0+M1+M2 統合サマリ");
    println!("  M1 軸索成長 {} 刈り取り {} open {}/{}",
        m1.axons_grown, m1.axons_pruned, m1.n_open_synapses(), m1.n_synapses());
    println!("  M2 軸索成長 {} 刈り取り {} open {}/{}",
        m2.axons_grown, m2.axons_pruned, m2.n_open_synapses(), m2.n_synapses());
}
