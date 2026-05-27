//! M0 蝸牛 → M1 (A1) → M2 (A2) パイプライン B 案 (3 階層、 M2 内部収束)
//!
//! 設計: M2_A2_DESIGN.md
//!
//! A 案 (m0_m1_m2_pipeline.rs) との違い:
//!   - A 案: M1 出力 40 → 外部 2:1 集約 → M2 入力 20 (集約判断機構が外部)
//!   - B 案: M1 出力 40 → M2 入力 40 (1:1 直結) → M2 内部で 40→20 収束
//!
//! M2 構成 (ThermoNetworkConfig::for_m2()):
//!   - grid 20×22 = 440 (M1 と同じ規模)
//!   - 入力 40 (y=0,1 の 2 行)
//!   - 出力 20 (y=21 の 1 行)
//!   - 内部 380 (興奮 304 + 抑制 76、 18% 抑制比)
//!
//! 期待:
//!   B 案では「40→20 集約」を M2 内部の散逸構造として自然に形成、
//!   A 案で起きた「外部集約による情報損失」 を回避できるはず。

use spiking_brain::phase2_f::thermo_network::{ThermoNetwork, ThermoNetworkConfig};
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
const TRIAL_STEPS: usize = (TRIAL_DURATION_MS / DT_MS) as usize;  // 600
/// M1 用 fingerprint bin 幅 (10ms、 一次聴覚野の時間精度に合致)
const FINGERPRINT_BIN_WIDTH_MS_M1: f64 = 10.0;
/// M2 用 fingerprint bin 幅 (50ms、 二次聴覚野の積分窓 10-100ms と合致)
/// M2_A2_DESIGN.md §1.1: A2 は「時間積分窓が長い (10-100 ms)、 抽象度上昇」
const FINGERPRINT_BIN_WIDTH_MS_M2: f64 = 50.0;
/// M1 出力 → M2 入力 1:1 直結時の電流値
const INPUT_CURRENT_M2: i32 = 60;

/// 1 trial 実行 (M0 → M1 → M2、 M1→M2 は 1:1 直結)
fn present_syllable_v2(
    m1: &mut ThermoNetwork,
    m2: &mut ThermoNetwork,
    cochlea: &mut Cochlea,
    waveform: &[i32],
) -> (Vec<(usize, f64)>, Vec<(usize, f64)>) {
    m1.reset_trial_state();
    m2.reset_trial_state();
    cochlea.reset();

    let m1_trial_start = m1.current_time;
    let n_m2_input = m2.input_neurons.len();  // 40
    let mut m1_out_log: Vec<(usize, f64)> = Vec::new();
    let mut m2_out_log: Vec<(usize, f64)> = Vec::new();

    for step in 0..TRIAL_STEPS {
        let s0 = step * SAMPLES_PER_STEP;
        let mut samples = [0i32; SAMPLES_PER_STEP];
        for i in 0..SAMPLES_PER_STEP {
            let idx = s0 + i;
            if idx < waveform.len() { samples[i] = waveform[idx]; }
        }
        let m0_output = cochlea.process_step(&samples);

        let m1_fired = m1.step(&m0_output);
        for &nid in &m1_fired {
            if let Some(oi) = m1.output_index_of(nid) {
                let t_rel = (m1.current_time - m1_trial_start) as f64 * DT_MS;
                m1_out_log.push((oi, t_rel));
            }
        }

        // M1 → M2 1:1 直結: M1 出力 i → M2 入力 i (集約なし)
        let mut m2_input = vec![0i32; n_m2_input];
        for &nid in &m1_fired {
            if let Some(oi) = m1.output_index_of(nid) {
                if oi < n_m2_input {
                    m2_input[oi] = m2_input[oi].saturating_add(INPUT_CURRENT_M2);
                }
            }
        }

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

fn fingerprint(log: &[(usize, f64)], n_out: usize, bin_width_ms: f64) -> Vec<f64> {
    let mut tr = OutputTrace::new(n_out, 50.0);
    for &(oi, t) in log {
        tr.record_spike(oi, t);
    }
    tr.time_binned_fingerprint(TRIAL_DURATION_MS, bin_width_ms)
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

fn evaluate_v2(
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
    let mut m1_active = vec![false; n_m1_out];
    let mut m2_active = vec![false; n_m2_out];

    for _ in 0..n_sample {
        for (si, _syl) in syllables.iter().enumerate() {
            let (m1_log, m2_log) = present_syllable_v2(m1, m2, cochlea, &waveforms[si]);
            for &(oi, _) in &m1_log { m1_active[oi] = true; }
            for &(oi, _) in &m2_log { m2_active[oi] = true; }
            m1_fps[si].push(fingerprint(&m1_log, n_m1_out, FINGERPRINT_BIN_WIDTH_MS_M1));
            m2_fps[si].push(fingerprint(&m2_log, n_m2_out, FINGERPRINT_BIN_WIDTH_MS_M2));
        }
    }

    let (m1_sel, m1_w, m1_b) = compute_selectivity(&m1_fps);
    let (m2_sel, m2_w, m2_b) = compute_selectivity(&m2_fps);
    let m1_active_count = m1_active.iter().filter(|&&v| v).count();
    let m2_active_count = m2_active.iter().filter(|&&v| v).count();

    println!("\n  -- {label} --");
    println!("    M1 出力 ({} ニューロン): sel={:.3} (within {:.3} - between {:.3}) active={}/{}",
        n_m1_out, m1_sel, m1_w, m1_b, m1_active_count, n_m1_out);
    println!("    M2 出力 ({} ニューロン): sel={:.3} (within {:.3} - between {:.3}) active={}/{}",
        n_m2_out, m2_sel, m2_w, m2_b, m2_active_count, n_m2_out);
}

fn main() {
    let n_train: usize = std::env::args().nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(100);

    println!("== M0+M1+M2 統合パイプライン B 案 (M2 内部 40→20 収束) ==");
    println!("  M0 蝸牛: 20 帯域 ERB");
    println!("  M1 A1  : 20→40 (Fork F-G1-R1 v2)");
    println!("  M2 A2  : 40→20 (M1 出力 40 を 1:1 直結、 内部で収束)");
    println!("  音素   : pa, ki, tu, se, mo");

    // M1 (デフォルト構成 20→40)
    let mut cfg_m1 = ThermoNetworkConfig::default();
    cfg_m1.enable_up_down = false;
    cfg_m1.seed = 300;
    let mut m1 = ThermoNetwork::new(cfg_m1);

    // M2 (for_m2: 40→20、 grid 20×22)
    let cfg_m2 = ThermoNetworkConfig::for_m2();
    let mut m2 = ThermoNetwork::new(cfg_m2);

    let mut cochlea = Cochlea::new();
    let syllables = standard_syllables();
    let mut noise = LfsrNoise::new(0xACE1);

    println!("\n  音節波形を生成中...");
    let mut waveforms: Vec<Vec<i32>> = Vec::with_capacity(syllables.len());
    for syl in &syllables {
        let wave = synth_syllable(syl, &mut noise);
        waveforms.push(wave);
    }

    println!("\n  M1: neurons={}, synapses={} (in={}, out={})",
        m1.n_neurons(), m1.n_synapses(),
        m1.input_neurons.len(), m1.output_neurons.len());
    println!("  M2: neurons={}, synapses={} (in={}, out={})",
        m2.n_neurons(), m2.n_synapses(),
        m2.input_neurons.len(), m2.output_neurons.len());

    let mut csv = File::create("phase2_f_m0m1m2v2_snapshots.csv").expect("csv");
    writeln!(csv, "trial,m1_within,m1_sel,m1_active,m2_within,m2_sel,m2_active,m1_open,m2_open").unwrap();

    println!("\n== Phase 1: 訓練前評価 ==");
    evaluate_v2(&mut m1, &mut m2, &mut cochlea, &syllables, &waveforms, 10, "PRE");

    let snap_interval = if n_train >= 500 { 500 } else { (n_train / 10).max(10) };
    println!("\n== Phase 2: 訓練 {n_train} 試行 ==");
    let mut rng = StdRng::seed_from_u64(42);
    for trial in 1..=n_train {
        let si = rng.gen_range(0..syllables.len());
        let _ = present_syllable_v2(&mut m1, &mut m2, &mut cochlea, &waveforms[si]);

        if trial % snap_interval == 0 || trial == n_train {
            // mini eval (5 samples)
            let mut m1_fps: Vec<Vec<Vec<f64>>> =
                vec![Vec::with_capacity(5); syllables.len()];
            let mut m2_fps: Vec<Vec<Vec<f64>>> =
                vec![Vec::with_capacity(5); syllables.len()];
            let mut m1_active = vec![false; m1.output_neurons.len()];
            let mut m2_active = vec![false; m2.output_neurons.len()];
            for _ in 0..5 {
                for si in 0..syllables.len() {
                    let (m1_log, m2_log) = present_syllable_v2(&mut m1, &mut m2, &mut cochlea, &waveforms[si]);
                    for &(oi, _) in &m1_log { m1_active[oi] = true; }
                    for &(oi, _) in &m2_log { m2_active[oi] = true; }
                    m1_fps[si].push(fingerprint(&m1_log, m1.output_neurons.len(), FINGERPRINT_BIN_WIDTH_MS_M1));
                    m2_fps[si].push(fingerprint(&m2_log, m2.output_neurons.len(), FINGERPRINT_BIN_WIDTH_MS_M2));
                }
            }
            let (m1_sel, m1_w, _) = compute_selectivity(&m1_fps);
            let (m2_sel, m2_w, _) = compute_selectivity(&m2_fps);
            let m1_act = m1_active.iter().filter(|&&v| v).count();
            let m2_act = m2_active.iter().filter(|&&v| v).count();

            println!("  trial {:>5}: M1 sel={:.3} (w={:.3}) act={}/{} open={} | M2 sel={:.3} (w={:.3}) act={}/{} open={}",
                trial, m1_sel, m1_w, m1_act, m1.output_neurons.len(), m1.n_open_synapses(),
                m2_sel, m2_w, m2_act, m2.output_neurons.len(), m2.n_open_synapses());

            writeln!(csv, "{},{:.4},{:.4},{},{:.4},{:.4},{},{},{}",
                trial, m1_w, m1_sel, m1_act,
                m2_w, m2_sel, m2_act,
                m1.n_open_synapses(), m2.n_open_synapses()).unwrap();
        }
    }

    println!("\n== Phase 3: 訓練後評価 ==");
    evaluate_v2(&mut m1, &mut m2, &mut cochlea, &syllables, &waveforms, 20, "POST");

    println!("\n══════════════════════════════════════════════════════════");
    println!("  M0+M1+M2 V2 サマリ");
    println!("  M1 軸索成長 {} 刈り取り {} open {}/{}",
        m1.axons_grown, m1.axons_pruned, m1.n_open_synapses(), m1.n_synapses());
    println!("  M2 軸索成長 {} 刈り取り {} open {}/{}",
        m2.axons_grown, m2.axons_pruned, m2.n_open_synapses(), m2.n_synapses());
    println!("  CSV: phase2_f_m0m1m2v2_snapshots.csv");
}
