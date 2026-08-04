//! M1 出力の時間構造 probe (M1.5 皮質中継 設計の土台)
//!
//! 目的: M1.5 (リレー核 #2) を設計する前に、 M1 出力が時間的にどう見えるかを実測する。
//!   - M1 出力は密か疎か (spike 総数、 active neuron 数)
//!   - オンセット構造があるか (population 発火が trial 内でいつ立つか)
//!   - 音素間で「時間プロファイル」が違うか (identity だけでなく timing で分化するか)
//!
//! これにより「蝸牛神経核を M1 出力に naive copy」の罠を避け、 M1 出力の実際の統計に
//! 合わせて M1.5 の Octopus/Bushy/Stellate 相当を設計する。
//!
//! CLI: cargo run --release --bin m1_output_probe [n_train=3000] [speed=3] [decay_slow=30]

use spiking_brain::phase2_f::thermo_network::{ThermoNetwork, ThermoNetworkConfig};
use spiking_brain::phase2_f::cochlea::{Cochlea, SAMPLES_PER_STEP};
use spiking_brain::phase2_f::cochlear_nucleus::CochlearNucleus;
use spiking_brain::phase2_f::phoneme_synth::{
    standard_syllables, synth_syllable_scaled, LfsrNoise, Syllable,
};
use spiking_brain::trace::cosine_similarity;
use rand::prelude::*;

const TRIAL_DURATION_MS: f64 = 300.0;
const DT_MS: f64 = 0.5;
const TRIAL_STEPS: usize = (TRIAL_DURATION_MS / DT_MS) as usize;  // 600
const N_BINS: usize = 30;       // 10ms bin
const BIN_MS: f64 = TRIAL_DURATION_MS / N_BINS as f64;

/// 1 trial: M0 → M0.5 → M1、 M1 出力ラスタ (oi, t_rel_ms) を返す
fn present(
    net: &mut ThermoNetwork, cochlea: &mut Cochlea, cn: &mut CochlearNucleus, waveform: &[i32],
) -> Vec<(usize, f64)> {
    net.reset_trial_state();
    cochlea.reset();
    cn.reset();
    let t0 = net.current_time;
    let mut out_log = Vec::new();
    for step in 0..TRIAL_STEPS {
        let s0 = step * SAMPLES_PER_STEP;
        let mut samples = [0i32; SAMPLES_PER_STEP];
        for i in 0..SAMPLES_PER_STEP {
            let idx = s0 + i;
            if idx < waveform.len() { samples[i] = waveform[idx]; }
        }
        let coch = cochlea.process_step(&samples);
        let cn_out = cn.process_step(&coch);
        for nid in net.step(&cn_out) {
            if let Some(oi) = net.output_index_of(nid) {
                out_log.push((oi, (net.current_time - t0) as f64 * DT_MS));
            }
        }
    }
    out_log
}

fn main() {
    let n_train: usize = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(3000);
    let speed: f64 = std::env::args().nth(2).and_then(|s| s.parse().ok()).unwrap_or(3.0);
    let decay_slow: i32 = std::env::args().nth(3).and_then(|s| s.parse().ok()).unwrap_or(30);
    let n_sample = 20;

    println!("== M1 出力 時間構造 probe (M1.5 設計用) ==");
    println!("  構成: cochlea 40ch → CN 84ch → M1 (for_m1_cn_40)、 speed={} decay_slow={}", speed, decay_slow);

    let mut cfg = ThermoNetworkConfig::for_m1_cn_40();
    if decay_slow > 1 {
        cfg.conductance_decay_interval *= decay_slow;
        cfg.vitality_decay_interval *= decay_slow;
    }
    let mut net = ThermoNetwork::new(cfg);
    let mut cochlea = Cochlea::new();
    let mut cn = CochlearNucleus::new();
    let syllables = standard_syllables();
    let mut noise = LfsrNoise::new(0xACE1);
    let trial_samples = (TRIAL_DURATION_MS * 16.0) as usize;  // 4800
    let waveforms: Vec<Vec<i32>> = syllables.iter().map(|s| {
        let base = synth_syllable_scaled(s, &mut noise, speed);
        if speed <= 1.0 { base } else {
            let mut t = Vec::with_capacity(trial_samples);
            while t.len() < trial_samples { t.extend_from_slice(&base); }
            t.truncate(trial_samples); t
        }
    }).collect();
    let n_out = net.output_neurons.len();

    // 訓練
    println!("  訓練 {} 試行 ...", n_train);
    let mut rng = StdRng::seed_from_u64(42);
    for _ in 0..n_train {
        let si = rng.gen_range(0..syllables.len());
        let _ = present(&mut net, &mut cochlea, &mut cn, &waveforms[si]);
    }

    // 各音素の M1 出力を測定
    println!("\n  ── 音素別 M1 出力 統計 ({} sample 平均) ──", n_sample);
    println!("  音素 | spike総数 | active | 初発火ms | ピークbin(ms) | 時間集中度");
    // pop_profile[si][bin] = その bin で発火した「延べ spike 数」平均 (population 時間プロファイル)
    let mut pop_profile: Vec<Vec<f64>> = vec![vec![0.0; N_BINS]; syllables.len()];
    // coincidence_profile[si][bin] = その bin で発火した「異なる出力ニューロン数」平均 (Octopus が見る量)
    let mut coinc_profile: Vec<Vec<f64>> = vec![vec![0.0; N_BINS]; syllables.len()];

    for (si, syl) in syllables.iter().enumerate() {
        let mut tot_spikes = 0.0;
        let mut active_sum = 0.0;
        let mut onset_sum = 0.0f64;
        let mut onset_cnt = 0.0f64;
        for _ in 0..n_sample {
            let log = present(&mut net, &mut cochlea, &mut cn, &waveforms[si]);
            tot_spikes += log.len() as f64;
            let mut fired = vec![false; n_out];
            let mut first_ms = f64::INFINITY;
            // bin ごとに distinct neuron を数える
            let mut bin_neurons: Vec<Vec<bool>> = vec![vec![false; n_out]; N_BINS];
            for &(oi, t) in &log {
                fired[oi] = true;
                if t < first_ms { first_ms = t; }
                let b = ((t / BIN_MS) as usize).min(N_BINS - 1);
                pop_profile[si][b] += 1.0;
                bin_neurons[b][oi] = true;
            }
            for b in 0..N_BINS {
                coinc_profile[si][b] += bin_neurons[b].iter().filter(|&&x| x).count() as f64;
            }
            active_sum += fired.iter().filter(|&&x| x).count() as f64;
            if first_ms.is_finite() { onset_sum += first_ms; onset_cnt += 1.0; }
        }
        for b in 0..N_BINS { pop_profile[si][b] /= n_sample as f64; coinc_profile[si][b] /= n_sample as f64; }
        // ピーク bin
        let (peak_bin, peak_val) = pop_profile[si].iter().enumerate()
            .fold((0usize, 0.0f64), |(bi, bv), (i, &v)| if v > bv { (i, v) } else { (bi, bv) });
        // 時間集中度 = ピーク bin の spike 数 / 全 spike 数 (1.0 に近いほど 1 bin に集中 = 鋭いオンセット)
        let total: f64 = pop_profile[si].iter().sum();
        let concentration = if total > 0.0 { peak_val / total } else { 0.0 };
        let _ = peak_val;
        println!("  {:>4} | {:>8.1} | {:>3.0}/{} | {:>7.1} | bin{:>2}({:>3.0}ms) | {:.2}",
            syl.label, tot_spikes / n_sample as f64, active_sum / n_sample as f64, n_out,
            onset_sum / onset_cnt.max(1.0), peak_bin, peak_bin as f64 * BIN_MS, concentration);
    }

    // population 時間プロファイル (identity 無視、 timing のみ) の音素間 cosine
    // → M1 出力が「いつ発火するか」だけで音素を分けられるか
    println!("\n  ── population 時間プロファイル (timing のみ、 identity 無視) 音素間 cosine ──");
    let mut sum = 0.0; let mut cnt = 0;
    for i in 0..syllables.len() {
        for j in (i+1)..syllables.len() {
            let c = cosine_similarity(&pop_profile[i], &pop_profile[j]);
            println!("    {}-{}: {:.3}", syllables[i].label, syllables[j].label, c);
            sum += c; cnt += 1;
        }
    }
    println!("    平均 = {:.3}  (1.0 に近い = timing だけでは分化しない → identity が主)", sum / cnt as f64);

    // population 発火時間プロファイルの平均形状 (全音素平均) — オンセット型か持続型か
    println!("\n  ── 全音素平均 population プロファイル (10ms bin、 発火延べ数) ──");
    print!("    ");
    for b in 0..N_BINS {
        let avg: f64 = (0..syllables.len()).map(|si| pop_profile[si][b]).sum::<f64>() / syllables.len() as f64;
        print!("{:>4.0}", avg);
        if b % 10 == 9 { print!(" |"); }
    }
    println!("\n    (各値 = その 10ms における延べ発火数。 冒頭に集中 = オンセット型、 平坦 = 持続型)");

    // Octopus 相当が見る量: bin ごとの「同時発火した distinct 出力ニューロン数」ピーク
    println!("\n  ── coincidence (bin ごと distinct 発火ニューロン数) ピーク ──");
    for (si, syl) in syllables.iter().enumerate() {
        let peak = coinc_profile[si].iter().cloned().fold(0.0f64, f64::max);
        let peak_bin = coinc_profile[si].iter().enumerate()
            .fold((0usize, 0.0f64), |(bi, bv), (i, &v)| if v > bv { (i, v) } else { (bi, bv) }).0;
        println!("    {:>4}: ピーク {:.1} 個同時 @ bin{}({:.0}ms)  [Octopus 閾値設計の参考]",
            syl.label, peak, peak_bin, peak_bin as f64 * BIN_MS);
    }
}

fn _unused(_: &Syllable) {}
