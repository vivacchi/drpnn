//! M0 → M0.5 蝸牛神経核 → M1 → M1.5 皮質中継 → M2 (5 段)
//!
//! 設計: M1_5_CORTICAL_RELAY_DESIGN.md
//!
//! M1→M2 境界に M1.5 リレー核を挿入し、M2 collapse が解消するか検証する。
//! mode で M1.5 の機構を切り替え:
//!   none  : M1 出力を素通し (baseline = 現状の M1→M2 直結、collapse するはず)
//!   coinc : 案 B ランダム部分集合 同時性検出器バンク (非線形派生特徴)
//!
//! M1 は relay/M2 から影響を受けない (前向き) ため、mode 間で M1 の進化は同一。
//! よって M2 の差だけを比較できる。
//!
//! CLI: m0_cn_m1_relay_m2_pipeline [mode=coinc] [n_train=2000] [speed=3] [decay_slow=30]

use spiking_brain::phase2_f::thermo_network::{ThermoNetwork, ThermoNetworkConfig};
use spiking_brain::phase2_f::cochlea::{Cochlea, SAMPLES_PER_STEP};
use spiking_brain::phase2_f::cochlear_nucleus::CochlearNucleus;
use spiking_brain::phase2_f::cortical_relay::CoincidenceRelay;
use spiking_brain::phase2_f::phoneme_synth::{standard_syllables, synth_syllable_scaled, LfsrNoise, Syllable};
use spiking_brain::trace::{cosine_similarity, OutputTrace};
use rand::prelude::*;

const TRIAL_DURATION_MS: f64 = 300.0;
const DT_MS: f64 = 0.5;
const TRIAL_STEPS: usize = (TRIAL_DURATION_MS / DT_MS) as usize;
const FP_BIN: f64 = 10.0;
const INPUT_CURRENT_M2: i32 = 60;

#[derive(Clone, Copy, PartialEq)]
enum Mode { None, Coinc }

/// 1 trial: M0 → CN → M1 → (M1.5) → M2。M1.5 出力ラスタも返す。
fn present(
    m1: &mut ThermoNetwork, m2: &mut ThermoNetwork,
    cochlea: &mut Cochlea, cn: &mut CochlearNucleus,
    relay: &mut Option<CoincidenceRelay>, mode: Mode,
    waveform: &[i32],
) -> (Vec<(usize, f64)>, Vec<(usize, f64)>, Vec<(usize, f64)>) {
    m1.reset_trial_state();
    m2.reset_trial_state();
    cochlea.reset();
    cn.reset();
    if let Some(r) = relay { r.reset(); }
    let t0 = m1.current_time;
    let n_m2_in = m2.input_neurons.len();
    let n_m1_out = m1.output_neurons.len();
    let mut m1_log = Vec::new();
    let mut m15_log = Vec::new();
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

        // M1 出力を発火電流ベクトルに
        let mut m1_vec = vec![0i32; n_m1_out];
        let tr = (m1.current_time - t0) as f64 * DT_MS;
        for &nid in &m1_fired {
            if let Some(oi) = m1.output_index_of(nid) {
                m1_vec[oi] = INPUT_CURRENT_M2;
                m1_log.push((oi, tr));
            }
        }

        // M1.5 変換
        let m15_vec: Vec<i32> = match mode {
            Mode::None => m1_vec.clone(),
            Mode::Coinc => relay.as_mut().unwrap().process_step(&m1_vec),
        };
        for (ch, &v) in m15_vec.iter().enumerate() {
            if v > 0 { m15_log.push((ch, tr)); }
        }

        // M2 入力へ (1:1)
        let mut m2_in = vec![0i32; n_m2_in];
        for (ch, &v) in m15_vec.iter().enumerate() {
            if v > 0 && ch < n_m2_in { m2_in[ch] = m2_in[ch].saturating_add(INPUT_CURRENT_M2); }
        }
        let m2_fired = m2.step(&m2_in);
        for nid in m2_fired {
            if let Some(oi) = m2.output_index_of(nid) { m2_log.push((oi, tr)); }
        }
    }
    (m1_log, m15_log, m2_log)
}

fn fingerprint(log: &[(usize, f64)], n_out: usize) -> Vec<f64> {
    let mut tr = OutputTrace::new(n_out, 50.0);
    for &(oi, t) in log { tr.record_spike(oi, t); }
    tr.time_binned_fingerprint(TRIAL_DURATION_MS, FP_BIN)
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
/// per-pair 平均 (音素間 平均 fingerprint cosine) — 真の分化指標
fn perpair(per: &[Vec<Vec<f64>>]) -> f64 {
    let n = per.len();
    let means: Vec<Vec<f64>> = per.iter().map(|fps| {
        if fps.is_empty() { return Vec::new(); }
        let mut acc = vec![0.0; fps[0].len()];
        for f in fps { for (a, v) in acc.iter_mut().zip(f) { *a += v; } }
        for a in acc.iter_mut() { *a /= fps.len() as f64; }
        acc
    }).collect();
    let mut s = 0.0; let mut c = 0;
    for i in 0..n { for j in (i+1)..n {
        let zi = means[i].iter().all(|&v| v == 0.0);
        let zj = means[j].iter().all(|&v| v == 0.0);
        s += if zi || zj { 0.0 } else { cosine_similarity(&means[i], &means[j]) };
        c += 1;
    }}
    if c == 0 { 0.0 } else { s / c as f64 }
}

fn evaluate(
    m1: &mut ThermoNetwork, m2: &mut ThermoNetwork, cochlea: &mut Cochlea, cn: &mut CochlearNucleus,
    relay: &mut Option<CoincidenceRelay>, mode: Mode,
    syllables: &[Syllable], waveforms: &[Vec<i32>], n_sample: usize, n_det: usize, label: &str,
) {
    let n1 = m1.output_neurons.len();
    let n2 = m2.output_neurons.len();
    let mut f1: Vec<Vec<Vec<f64>>> = vec![Vec::new(); syllables.len()];
    let mut f15: Vec<Vec<Vec<f64>>> = vec![Vec::new(); syllables.len()];
    let mut f2: Vec<Vec<Vec<f64>>> = vec![Vec::new(); syllables.len()];
    let mut a2 = vec![false; n2];
    for _ in 0..n_sample {
        for si in 0..syllables.len() {
            let (l1, l15, l2) = present(m1, m2, cochlea, cn, relay, mode, &waveforms[si]);
            for &(oi, _) in &l2 { a2[oi] = true; }
            f1[si].push(fingerprint(&l1, n1));
            f15[si].push(fingerprint(&l15, n_det));
            f2[si].push(fingerprint(&l2, n2));
        }
    }
    let (_s1, _w1, _b1) = selectivity(&f1);
    let (s2, w2, b2) = selectivity(&f2);
    println!("\n  -- {label} --");
    println!("    M1   per-pair = {:.3}", perpair(&f1));
    println!("    M1.5 per-pair = {:.3}  (検出器 {} ch)", perpair(&f15), n_det);
    println!("    M2   sel={:.3} within={:.3} between={:.3} per-pair={:.3} active={}/{}",
        s2, w2, b2, perpair(&f2), a2.iter().filter(|&&v| v).count(), n2);
}

fn main() {
    let mode_s = std::env::args().nth(1).unwrap_or_else(|| "coinc".into());
    let mode = if mode_s == "none" { Mode::None } else { Mode::Coinc };
    let n_train: usize = std::env::args().nth(2).and_then(|s| s.parse().ok()).unwrap_or(2000);
    let speed: f64 = std::env::args().nth(3).and_then(|s| s.parse().ok()).unwrap_or(3.0);
    let decay_slow: i32 = std::env::args().nth(4).and_then(|s| s.parse().ok()).unwrap_or(30);
    let n_sample = 20;
    let snap = if n_train >= 500 { 500 } else { (n_train / 10).max(10) };

    println!("== M0 → CN → M1 → M1.5({}) → M2 (5 段) ==", mode_s);
    println!("  speed={} decay_slow={}", speed, decay_slow);

    let mut cfg1 = ThermoNetworkConfig::for_m1_cn_40();
    cfg1.conductance_decay_interval *= decay_slow;
    cfg1.vitality_decay_interval *= decay_slow;
    let mut m1 = ThermoNetwork::new(cfg1);
    let n_m1_out = m1.output_neurons.len();

    let mut cfg2 = ThermoNetworkConfig::for_m2();
    cfg2.conductance_decay_interval *= decay_slow;
    cfg2.vitality_decay_interval *= decay_slow;
    let mut m2 = ThermoNetwork::new(cfg2);
    let n_det = m2.input_neurons.len();  // M1.5 検出器数 = M2 入力数 (40)

    let mut relay = match mode {
        Mode::None => None,
        Mode::Coinc => Some(CoincidenceRelay::new(n_m1_out, n_det, 0x1F5)),
    };

    let mut cochlea = Cochlea::new();
    let mut cn = CochlearNucleus::new();
    let syllables = standard_syllables();
    let mut noise = LfsrNoise::new(0xACE1);
    let trial_samples = (TRIAL_DURATION_MS * 16.0) as usize;
    let waveforms: Vec<Vec<i32>> = syllables.iter().map(|s| {
        let base = synth_syllable_scaled(s, &mut noise, speed);
        if speed <= 1.0 { base } else {
            let mut t = Vec::with_capacity(trial_samples);
            while t.len() < trial_samples { t.extend_from_slice(&base); }
            t.truncate(trial_samples); t
        }
    }).collect();

    println!("  M1: {}→{} | M1.5 検出器: {} | M2: {}→{}",
        m1.input_neurons.len(), n_m1_out, n_det, m2.input_neurons.len(), m2.output_neurons.len());

    println!("\n== 訓練前 ==");
    evaluate(&mut m1, &mut m2, &mut cochlea, &mut cn, &mut relay, mode, &syllables, &waveforms, n_sample, n_det, "PRE");

    println!("\n== 訓練 {n_train} 試行 ==");
    let mut rng = StdRng::seed_from_u64(42);
    for trial in 1..=n_train {
        let si = rng.gen_range(0..syllables.len());
        let _ = present(&mut m1, &mut m2, &mut cochlea, &mut cn, &mut relay, mode, &waveforms[si]);
        if trial % snap == 0 || trial == n_train {
            println!("  trial {:>5}: M1 open={} | M2 open={} within(参考)",
                trial, m1.n_open_synapses(), m2.n_open_synapses());
        }
    }

    println!("\n== 訓練後 ==");
    evaluate(&mut m1, &mut m2, &mut cochlea, &mut cn, &mut relay, mode, &syllables, &waveforms, n_sample, n_det, "POST");
    println!("\n  M1 成長{} 刈{} | M2 成長{} 刈{}", m1.axons_grown, m1.axons_pruned, m2.axons_grown, m2.axons_pruned);
}
