//! M1.5 皮質中継 効果 probe (M2 を挟む前の安価な検証)
//!
//! 案 A (遅延多様リレー) の核心仮説を M2 なしで検証する:
//!   M1 出力は timing が音素非依存 (cosine 0.954)。M1.5 を通すと identity が timing に
//!   展開され、timing-only cosine が下がる (= timing が音素を運ぶようになる) か?
//!
//! 反証: M1.5 出力の timing-only cosine が下がらなければ、遅延展開だけでは分離可能な
//!   時間構造にできない (案 A 棄却 → 案 B へ)。
//!
//! CLI: cargo run --release --bin relay_probe [n_train=3000] [speed=3] [decay_slow=30] [seed=43981]

use spiking_brain::phase2_f::thermo_network::{ThermoNetwork, ThermoNetworkConfig};
use spiking_brain::phase2_f::cochlea::{Cochlea, SAMPLES_PER_STEP, FIRE_CURRENT};
use spiking_brain::phase2_f::cochlear_nucleus::CochlearNucleus;
use spiking_brain::phase2_f::cortical_relay::CorticalRelay;
use spiking_brain::phase2_f::phoneme_synth::{standard_syllables, synth_syllable_scaled, LfsrNoise};
use spiking_brain::trace::cosine_similarity;
use rand::prelude::*;

const TRIAL_DURATION_MS: f64 = 300.0;
const DT_MS: f64 = 0.5;
const TRIAL_STEPS: usize = (TRIAL_DURATION_MS / DT_MS) as usize;  // 600
// timing 解像度: 遅延幅 (1-30ms) を捉えるため細かい 2ms bin
const FINE_BIN_MS: f64 = 2.0;
const N_FINE: usize = (TRIAL_DURATION_MS / FINE_BIN_MS) as usize;  // 150

/// M1 出力 (発火電流ベクトル列) を 1 trial 分そのまま記録し、M1.5 通過後も記録する。
/// 戻り値: (m1_pop[N_FINE], relay_pop[N_FINE])
///   pop = 各 fine-bin での延べ発火チャネル数 (identity 無視、timing のみ)
fn present_and_relay(
    net: &mut ThermoNetwork, cochlea: &mut Cochlea, cn: &mut CochlearNucleus,
    relay: &mut CorticalRelay, waveform: &[i32], n_out: usize,
) -> (Vec<f64>, Vec<f64>) {
    net.reset_trial_state();
    cochlea.reset();
    cn.reset();
    relay.reset();
    let t0 = net.current_time;
    let mut m1_pop = vec![0.0f64; N_FINE];
    let mut relay_pop = vec![0.0f64; N_FINE];

    for step in 0..TRIAL_STEPS {
        let s0 = step * SAMPLES_PER_STEP;
        let mut samples = [0i32; SAMPLES_PER_STEP];
        for i in 0..SAMPLES_PER_STEP {
            let idx = s0 + i;
            if idx < waveform.len() { samples[i] = waveform[idx]; }
        }
        let coch = cochlea.process_step(&samples);
        let cn_out = cn.process_step(&coch);
        let fired = net.step(&cn_out);

        // M1 出力を発火電流ベクトルに変換
        let mut m1_vec = vec![0i32; n_out];
        let t_rel = (net.current_time - t0) as f64 * DT_MS;
        let fbin = ((t_rel / FINE_BIN_MS) as usize).min(N_FINE - 1);
        for nid in fired {
            if let Some(oi) = net.output_index_of(nid) {
                m1_vec[oi] = FIRE_CURRENT;
                m1_pop[fbin] += 1.0;
            }
        }
        // M1.5 通過
        let relay_out = relay.process_step(&m1_vec);
        for &v in &relay_out {
            if v > 0 { relay_pop[fbin] += 1.0; }
        }
    }
    (m1_pop, relay_pop)
}

fn mean_pair_cosine(profiles: &[Vec<f64>], labels: &[&str], title: &str) -> f64 {
    println!("\n  ── {} ──", title);
    let mut sum = 0.0; let mut cnt = 0;
    for i in 0..profiles.len() {
        for j in (i+1)..profiles.len() {
            let c = cosine_similarity(&profiles[i], &profiles[j]);
            println!("    {}-{}: {:.3}", labels[i], labels[j], c);
            sum += c; cnt += 1;
        }
    }
    let avg = sum / cnt as f64;
    println!("    平均 = {:.3}", avg);
    avg
}

fn main() {
    let n_train: usize = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(3000);
    let speed: f64 = std::env::args().nth(2).and_then(|s| s.parse().ok()).unwrap_or(3.0);
    let decay_slow: i32 = std::env::args().nth(3).and_then(|s| s.parse().ok()).unwrap_or(30);
    let seed: u16 = std::env::args().nth(4).and_then(|s| s.parse().ok()).unwrap_or(43981);
    let n_sample = 20;

    println!("== M1.5 皮質中継 効果 probe (案 A: 遅延多様リレー) ==");
    println!("  構成: cochlea 40 → CN 84 → M1 → M1.5 (遅延 seed={})、speed={} decay_slow={}", seed, speed, decay_slow);

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
    let trial_samples = (TRIAL_DURATION_MS * 16.0) as usize;
    let waveforms: Vec<Vec<i32>> = syllables.iter().map(|s| {
        let base = synth_syllable_scaled(s, &mut noise, speed);
        if speed <= 1.0 { base } else {
            let mut t = Vec::with_capacity(trial_samples);
            while t.len() < trial_samples { t.extend_from_slice(&base); }
            t.truncate(trial_samples); t
        }
    }).collect();
    let n_out = net.output_neurons.len();
    let mut relay = CorticalRelay::new(n_out, seed);

    // 訓練 (M1 のみ。M1.5 は固定なので訓練不要)
    println!("  訓練 {} 試行 ...", n_train);
    let mut rng = StdRng::seed_from_u64(42);
    for _ in 0..n_train {
        let si = rng.gen_range(0..syllables.len());
        net.reset_trial_state(); cochlea.reset(); cn.reset();
        for step in 0..TRIAL_STEPS {
            let s0 = step * SAMPLES_PER_STEP;
            let mut samples = [0i32; SAMPLES_PER_STEP];
            for i in 0..SAMPLES_PER_STEP {
                let idx = s0 + i;
                if idx < waveforms[si].len() { samples[i] = waveforms[si][idx]; }
            }
            let coch = cochlea.process_step(&samples);
            let cn_out = cn.process_step(&coch);
            let _ = net.step(&cn_out);
        }
    }

    // 音素別に M1 / M1.5 の timing-only プロファイルを平均
    let mut m1_prof: Vec<Vec<f64>> = vec![vec![0.0; N_FINE]; syllables.len()];
    let mut relay_prof: Vec<Vec<f64>> = vec![vec![0.0; N_FINE]; syllables.len()];
    for si in 0..syllables.len() {
        for _ in 0..n_sample {
            let (m1p, rp) = present_and_relay(&mut net, &mut cochlea, &mut cn, &mut relay, &waveforms[si], n_out);
            for b in 0..N_FINE { m1_prof[si][b] += m1p[b]; relay_prof[si][b] += rp[b]; }
        }
        for b in 0..N_FINE { m1_prof[si][b] /= n_sample as f64; relay_prof[si][b] /= n_sample as f64; }
    }

    let labels: Vec<&str> = syllables.iter().map(|s| s.label).collect();
    let m1_c = mean_pair_cosine(&m1_prof, &labels, "M1 出力 timing-only 音素間 cosine (2ms bin)");
    let relay_c = mean_pair_cosine(&relay_prof, &labels, "M1.5 出力 timing-only 音素間 cosine (2ms bin)");

    println!("\n  ── 判定 ──");
    println!("    M1   timing-only cosine = {:.3}", m1_c);
    println!("    M1.5 timing-only cosine = {:.3}  (Δ {:+.3})", relay_c, relay_c - m1_c);
    if relay_c < 0.85 {
        println!("    → 案 A 成立の兆し: timing が音素を運ぶようになった (< 0.85)。M2 パイプラインへ進む価値あり。");
    } else if relay_c < m1_c - 0.05 {
        println!("    → 部分的効果: 低下はするが 0.85 未満に届かず。遅延幅/seed 調整 or 案 B 検討。");
    } else {
        println!("    → 案 A 反証の兆し: timing-only が下がらない。遅延展開だけでは不十分 → 案 B へ。");
    }

    // 遅延割り当ての分散も参考表示
    let d = relay.delays();
    let dmin = d.iter().min().unwrap(); let dmax = d.iter().max().unwrap();
    let dmean: f64 = d.iter().map(|&x| x as f64).sum::<f64>() / d.len() as f64;
    println!("    遅延分布: min={} max={} mean={:.1} step (×0.5ms)", dmin, dmax, dmean);
}
