//! M0 蝸牛 → M0.5 蝸牛神経核 → M1 (A1) → M2 (A2) 4 段パイプライン
//!
//! 設計: M0_5_COCHLEAR_NUCLEUS_DESIGN.md + M2_A2_DESIGN.md
//!
//! 背景: §5.14 で M1 が音素を分化するようになった (sel 0.516, active 19/40)。
//! 以前 M2 が効かなかったのは M1 が collapse していたため (garbage in)。
//! M1 が分化した今、 M2 が有用な抽象化を獲得できるか再挑戦。
//!
//! 構成:
//!   cochlea(20) → cochlear_nucleus(44) → M1(44→40) → M2(40→20)
//!   - M1: for_m1_cn、 speed 圧縮 + decay_slow (§5.14 最適)
//!   - M2: for_m2 (40 入力 1:1 直結、 内部 40→20 収束)、 causal_window=320 (音節スケール)
//!   - M1, M2 とも decay_slow 適用 (経路保存)
//!
//! CLI: m0_cn_m1_m2_pipeline [n_train] [speed] [decay_slow]

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
const FP_BIN_M1: f64 = 10.0;   // M1: 一次聴覚野の時間精度
const FP_BIN_M2: f64 = 10.0;   // M2: 修正1検証で 50ms は逆効果だったため 10ms 維持
const INPUT_CURRENT_M2: i32 = 60;

/// 1 trial: M0 → M0.5 → M1 → M2
fn present(
    m1: &mut ThermoNetwork, m2: &mut ThermoNetwork,
    cochlea: &mut Cochlea, cn: &mut CochlearNucleus,
    waveform: &[i32],
) -> (Vec<(usize, f64)>, Vec<(usize, f64)>) {
    m1.reset_trial_state();
    m2.reset_trial_state();
    cochlea.reset();
    cn.reset();
    let t0 = m1.current_time;
    let n_m2_in = m2.input_neurons.len();
    let mut m1_log = Vec::new();
    let mut m2_log = Vec::new();

    for step in 0..TRIAL_STEPS {
        let s0 = step * SAMPLES_PER_STEP;
        let mut samples = [0i32; SAMPLES_PER_STEP];
        for i in 0..SAMPLES_PER_STEP {
            let idx = s0 + i;
            if idx < waveform.len() { samples[i] = waveform[idx]; }
        }
        let coch = cochlea.process_step(&samples);
        let cn_out = cn.process_step(&coch);
        let m1_fired = m1.step(&cn_out);
        for &nid in &m1_fired {
            if let Some(oi) = m1.output_index_of(nid) {
                let tr = (m1.current_time - t0) as f64 * DT_MS;
                m1_log.push((oi, tr));
            }
        }
        // M1 出力 40 → M2 入力 40 (1:1 直結)
        let mut m2_in = vec![0i32; n_m2_in];
        for &nid in &m1_fired {
            if let Some(oi) = m1.output_index_of(nid) {
                if oi < n_m2_in { m2_in[oi] = m2_in[oi].saturating_add(INPUT_CURRENT_M2); }
            }
        }
        let m2_fired = m2.step(&m2_in);
        for nid in m2_fired {
            if let Some(oi) = m2.output_index_of(nid) {
                let tr = (m1.current_time - t0) as f64 * DT_MS;
                m2_log.push((oi, tr));
            }
        }
    }
    (m1_log, m2_log)
}

fn fingerprint(log: &[(usize, f64)], n_out: usize, bin: f64) -> Vec<f64> {
    let mut tr = OutputTrace::new(n_out, 50.0);
    for &(oi, t) in log { tr.record_spike(oi, t); }
    tr.time_binned_fingerprint(TRIAL_DURATION_MS, bin)
}

fn mean_pairwise(fps: &[Vec<f64>]) -> f64 {
    let mut s = 0.0; let mut n = 0;
    for i in 0..fps.len() { for j in (i+1)..fps.len() { s += cosine_similarity(&fps[i], &fps[j]); n += 1; } }
    if n == 0 { 0.0 } else { s / n as f64 }
}
fn mean_between(a: &[Vec<f64>], b: &[Vec<f64>]) -> f64 {
    let mut s = 0.0; let mut n = 0;
    for x in a { for y in b { s += cosine_similarity(x, y); n += 1; } }
    if n == 0 { 0.0 } else { s / n as f64 }
}
fn selectivity(per: &[Vec<Vec<f64>>]) -> (f64, f64, f64) {
    let within: f64 = per.iter().filter(|f| f.len() >= 2).map(|f| mean_pairwise(f)).sum::<f64>()
        / per.iter().filter(|f| f.len() >= 2).count().max(1) as f64;
    let mut bs = 0.0; let mut bn = 0;
    for i in 0..per.len() { for j in (i+1)..per.len() { bs += mean_between(&per[i], &per[j]); bn += 1; } }
    let between = if bn == 0 { 0.0 } else { bs / bn as f64 };
    (within - between, within, between)
}

fn evaluate(
    m1: &mut ThermoNetwork, m2: &mut ThermoNetwork,
    cochlea: &mut Cochlea, cn: &mut CochlearNucleus,
    syllables: &[Syllable], waveforms: &[Vec<i32>], n_sample: usize, label: &str,
) {
    let n1 = m1.output_neurons.len();
    let n2 = m2.output_neurons.len();
    let mut f1: Vec<Vec<Vec<f64>>> = vec![Vec::new(); syllables.len()];
    let mut f2: Vec<Vec<Vec<f64>>> = vec![Vec::new(); syllables.len()];
    let mut a1 = vec![false; n1];
    let mut a2 = vec![false; n2];
    for _ in 0..n_sample {
        for si in 0..syllables.len() {
            let (l1, l2) = present(m1, m2, cochlea, cn, &waveforms[si]);
            for &(oi, _) in &l1 { a1[oi] = true; }
            for &(oi, _) in &l2 { a2[oi] = true; }
            f1[si].push(fingerprint(&l1, n1, FP_BIN_M1));
            f2[si].push(fingerprint(&l2, n2, FP_BIN_M2));
        }
    }
    let (s1, w1, b1) = selectivity(&f1);
    let (s2, w2, b2) = selectivity(&f2);
    println!("\n  -- {label} --");
    println!("    M1 ({}出力): sel={:.3} (within {:.3} - between {:.3}) active={}/{}",
        n1, s1, w1, b1, a1.iter().filter(|&&v| v).count(), n1);
    println!("    M2 ({}出力): sel={:.3} (within {:.3} - between {:.3}) active={}/{}",
        n2, s2, w2, b2, a2.iter().filter(|&&v| v).count(), n2);
}

fn main() {
    let n_train: usize = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(100);
    let speed: f64 = std::env::args().nth(2).and_then(|s| s.parse().ok()).unwrap_or(3.0);
    let decay_slow: i32 = std::env::args().nth(3).and_then(|s| s.parse().ok()).unwrap_or(30);
    let n_sample = 20;
    let snap = if n_train >= 500 { 500 } else { (n_train / 10).max(10) };

    println!("== M0 → M0.5 蝸牛神経核 → M1 → M2 4 段パイプライン ==");
    println!("  cochlea(20) → CN(44) → M1(44→40) → M2(40→20)");
    println!("  speed={}, decay_slow={} (§5.14 最適を M1/M2 両方に)", speed, decay_slow);

    // M1: for_m1_cn + decay_slow
    let mut cfg1 = ThermoNetworkConfig::for_m1_cn();
    cfg1.conductance_decay_interval *= decay_slow;
    cfg1.vitality_decay_interval *= decay_slow;
    let mut m1 = ThermoNetwork::new(cfg1);

    // M2: for_m2 + decay_slow (causal_window=320 は for_m2 既定)
    let mut cfg2 = ThermoNetworkConfig::for_m2();
    cfg2.conductance_decay_interval *= decay_slow;
    cfg2.vitality_decay_interval *= decay_slow;
    let mut m2 = ThermoNetwork::new(cfg2);

    let mut cochlea = Cochlea::new();
    let mut cn = CochlearNucleus::new();
    let syllables = standard_syllables();
    let mut noise = LfsrNoise::new(0xACE1);
    let trial_samples = (TRIAL_DURATION_MS * 16000.0 / 1000.0) as usize;
    let waveforms: Vec<Vec<i32>> = syllables.iter().map(|s| {
        let base = synth_syllable_scaled(s, &mut noise, speed);
        if speed <= 1.0 { base } else {
            let mut t = Vec::with_capacity(trial_samples);
            while t.len() < trial_samples { t.extend_from_slice(&base); }
            t.truncate(trial_samples); t
        }
    }).collect();

    println!("\n  M1: {} neurons, {} syn | M2: {} neurons, {} syn",
        m1.n_neurons(), m1.n_synapses(), m2.n_neurons(), m2.n_synapses());

    println!("\n== Phase 1: 訓練前評価 ==");
    evaluate(&mut m1, &mut m2, &mut cochlea, &mut cn, &syllables, &waveforms, n_sample, "PRE");

    println!("\n== Phase 2: 訓練 {n_train} 試行 ==");
    let mut rng = StdRng::seed_from_u64(42);
    for trial in 1..=n_train {
        let si = rng.gen_range(0..syllables.len());
        let _ = present(&mut m1, &mut m2, &mut cochlea, &mut cn, &waveforms[si]);
        if trial % snap == 0 || trial == n_train {
            let obs1 = m1.macro_observables();
            let obs2 = m2.macro_observables();
            println!("  trial {:>5}: M1 open={} cond_μ={:.1} | M2 open={} cond_μ={:.1}",
                trial, m1.n_open_synapses(), obs1.conductance_mean,
                m2.n_open_synapses(), obs2.conductance_mean);
        }
    }

    println!("\n== Phase 3: 訓練後評価 ==");
    evaluate(&mut m1, &mut m2, &mut cochlea, &mut cn, &syllables, &waveforms, n_sample, "POST");

    println!("\n  M1 軸索成長 {} 刈り取り {} | M2 軸索成長 {} 刈り取り {}",
        m1.axons_grown, m1.axons_pruned, m2.axons_grown, m2.axons_pruned);
}
