//! M0 蝸牛 → M1 DRAM 物理モデル版 パイプライン評価
//!
//! Phase 2 fork F (m0_m1_pipeline.rs) の DRAM 物理モデル対応版。
//! M0 蝸牛と評価方法は完全同一、 M1 のみ DramNetwork に置換。
//!
//! 目的: STAGE2GAMMA V2 設計の Rust シミュレーションで
//!       Phase 2 fork F (POST=0.795) に対する性能比較を行う。
//!
//! CLI:
//!   cargo run --release --bin m0_m1_dram_pipeline [n_train] [input_scale]
//!
//! デフォルト: n_train=100, input_scale=50 (cochlea 出力 × 50 = DRAM charge)

use spiking_brain::phase2_dram::dram_network::{DramNetwork, DramNetworkConfig};
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
    net: &mut DramNetwork,
    cochlea: &mut Cochlea,
    waveform: &[i32],
    input_scale: i32,
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
        // M0 蝸牛 → 20 ch 整数電流
        let cochlea_out = cochlea.process_step(&samples);
        // DRAM charge スケール (Phase 2 の current 0-10 → DRAM charge 0-500 程度)
        let ext: Vec<i32> = cochlea_out.iter().map(|&c| c * input_scale).collect();

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
    net: &mut DramNetwork,
    cochlea: &mut Cochlea,
    syllables: &[Syllable],
    waveforms: &[Vec<i32>],
    n_sample: usize,
    input_scale: i32,
    label: &str,
) {
    let n_out = net.output_neurons.len();
    let mut per_syl_fps: Vec<Vec<Vec<f64>>> =
        vec![Vec::with_capacity(n_sample); syllables.len()];
    let mut per_syl_hits: Vec<Vec<u32>> =
        vec![vec![0u32; n_out]; syllables.len()];
    let mut per_syl_spikes: Vec<u64> = vec![0u64; syllables.len()];
    let mut all_spike_times_ms: Vec<f64> = Vec::new();

    for _ in 0..n_sample {
        for (si, _syl) in syllables.iter().enumerate() {
            let log = present_syllable(net, cochlea, &waveforms[si], input_scale);
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

    if !all_spike_times_ms.is_empty() {
        let mut times = all_spike_times_ms.clone();
        times.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let n = times.len();
        let p = |q: f64| times[((q * (n as f64 - 1.0)).round() as usize).min(n - 1)];
        let mean: f64 = times.iter().sum::<f64>() / n as f64;
        println!("    出力スパイク時刻分布 (ms, 0-{TRIAL_DURATION_MS:.0}ms 内):");
        println!("      p10={:.1}  p25={:.1}  p50={:.1}  p75={:.1}  p90={:.1}  mean={:.1}  (n={})",
            p(0.10), p(0.25), p(0.50), p(0.75), p(0.90), mean, n);
    }
}

fn main() {
    let n_train: usize = std::env::args().nth(1)
        .and_then(|s| s.parse().ok()).unwrap_or(100);
    let input_scale: i32 = std::env::args().nth(2)
        .and_then(|s| s.parse().ok()).unwrap_or(50);

    let n_sample: usize = 20;
    let snap_interval: usize = if n_train >= 100_000 {
        n_train / 200
    } else if n_train >= 500 {
        500
    } else {
        (n_train / 10).max(10)
    };

    println!("== M0+M1 DRAM 物理モデル パイプライン: 音素 5 種識別 ==");
    println!("  音素: pa, ki, tu, se, mo");
    println!("  M0: 20 帯域蝸牛 (Phase 2 と共通)");
    println!("  M1: DRAM 物理モデル (Vth ばらつき + 個別セル書き込み + PWM 重み + ringbuffer)");
    println!("  ★ input_scale: {} (cochlea 出力 × {} = DRAM charge)", input_scale, input_scale);

    let cfg = DramNetworkConfig::default();
    let cfg_for_print = cfg.clone();
    let mut net = DramNetwork::new(cfg);
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
    println!("    neurons        : {}", net.n_neurons());
    println!("    synapses       : {} (alive={})", net.n_synapses(), net.n_alive_synapses());
    println!("    fanout         : input={}, overconnect={}, inhibitory={}",
        cfg_for_print.input_fanout, cfg_for_print.overconnect_fanout,
        cfg_for_print.inhibitory_fanout);
    println!("    ring_size      : {} (max_delay={})",
        cfg_for_print.ring_size, cfg_for_print.delay_range.1);

    let obs0 = net.macro_observables();
    println!("\n  初期 macro 観察量:");
    println!("    Vth 分布: mean={:.1}, std={:.1}", obs0.vth_mean, obs0.vth_std);
    println!("    pulse_width: mean={:.1}, max={}", obs0.pulse_width_mean, obs0.pulse_width_max);

    let mut csv = File::create("phase2_dram_phoneme_snapshots.csv").expect("csv");
    writeln!(csv, "trial,within,selectivity,active,charge_mean,pulse_width_mean,alive_syn,total_syn,sparsity").unwrap();

    println!("\n== Phase 1: 訓練前評価 ==");
    evaluate(&mut net, &mut cochlea, &syllables, &waveforms, n_sample, input_scale, "PRE");

    println!("\n== Phase 2: 訓練 {n_train} 試行 (音節ランダム選択) ==");
    println!("  snap_interval = {snap_interval}");
    let mut rng = StdRng::seed_from_u64(42);
    for trial in 1..=n_train {
        let si = rng.gen_range(0..syllables.len());
        let _ = present_syllable(&mut net, &mut cochlea, &waveforms[si], input_scale);

        if trial % snap_interval == 0 || trial == n_train {
            let mut per_syl_fps: Vec<Vec<Vec<f64>>> =
                vec![Vec::with_capacity(5); syllables.len()];
            let n_out = net.output_neurons.len();
            let mut per_syl_hits: Vec<Vec<u32>> = vec![vec![0u32; n_out]; syllables.len()];
            for _ in 0..5 {
                for si in 0..syllables.len() {
                    let log = present_syllable(&mut net, &mut cochlea, &waveforms[si], input_scale);
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
            let obs = net.macro_observables();

            println!("  {:>5}  within={:.3}  sel={:.3}  act={}/{}  charge_μ={:.1}  pw_μ={:.1}  alive={}",
                trial, within, sel, active, n_out,
                obs.charge_mean, obs.pulse_width_mean, net.n_alive_synapses());

            writeln!(csv, "{},{:.4},{:.4},{},{:.2},{:.2},{},{},{:.4}",
                trial, within, sel, active,
                obs.charge_mean, obs.pulse_width_mean,
                net.n_alive_synapses(), net.n_synapses(), obs.sparsity).unwrap();
        }
    }

    println!("\n== Phase 3: 訓練後評価 (出力 layer) ==");
    evaluate(&mut net, &mut cochlea, &syllables, &waveforms, n_sample, input_scale, "POST (output)");

    println!("\n══════════════════════════════════════════════════════════");
    println!("  M0+M1 DRAM 音素識別サマリ");
    println!("  総発火数 (累積): {}", net.spikes_total);
    println!("  刈り取りシナプス: {}", net.pruned_total);
    println!("  alive シナプス: {}/{}", net.n_alive_synapses(), net.n_synapses());
    println!("  CSV: phase2_dram_phoneme_snapshots.csv");
}
