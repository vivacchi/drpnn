//! M0 蝸牛 → M1 (A1) パイプライン評価.
//!
//! 設計: M0_COCHLEA_DESIGN.md §5
//!
//! 流れ:
//!   音素 (波形 i16, 16kHz)
//!     → M0 蝸牛 (20 帯域フィルタ + 包絡線 + 閾値発火)
//!     → M1 input neurons への external current (20 次元)
//!     → M1 ThermoNetwork.step()
//!     → 出力 40 ニューロンの発火パターン
//!     → 時間 bin 化 fingerprint で識別性評価
//!
//! CLI:
//!   cargo run --release --bin m0_m1_pipeline [n_train]
//!
//! デフォルト n_train=100 (動作確認モード)、10000 で本評価.

use spiking_brain::phase2_f::thermo_network::{ThermoNetwork, ThermoNetworkConfig};
use spiking_brain::phase2_f::cochlea::{Cochlea, SAMPLES_PER_STEP};
use spiking_brain::phase2_f::phoneme_synth::{
    standard_syllables, synth_syllable, LfsrNoise, Syllable,
};
use spiking_brain::trace::{cosine_similarity, OutputTrace};
use rand::prelude::*;
use std::fs::File;
use std::io::Write as IoWrite;

// ──────────────────────────────────────────────────────────────
// 定数
// ──────────────────────────────────────────────────────────────

const TRIAL_DURATION_MS: f64 = 300.0;
const DT_MS: f64 = 0.5;
const TRIAL_STEPS: usize = (TRIAL_DURATION_MS / DT_MS) as usize;  // 600 step
const SYLLABLE_DURATION_MS: f64 = 200.0;
const SYLLABLE_SAMPLES: usize = ((SYLLABLE_DURATION_MS / 1000.0) * 16000.0) as usize;  // 3200
const FINGERPRINT_BIN_WIDTH_MS: f64 = 10.0;

// ──────────────────────────────────────────────────────────────
// 1 試行: 音素を M0+M1 に流す
// ──────────────────────────────────────────────────────────────

/// 1 trial = 300ms. 音節 (200ms) を先頭に提示、その後は無音.
/// 戻り値: 出力ニューロンの (idx, t_rel_ms) ログ.
fn present_syllable(
    net: &mut ThermoNetwork,
    cochlea: &mut Cochlea,
    waveform: &[i32],
) -> Vec<(usize, f64)> {
    net.reset_trial_state();
    cochlea.reset();

    let trial_start_t = net.current_time;
    let mut out_log: Vec<(usize, f64)> = Vec::new();

    for step in 0..TRIAL_STEPS {
        // 音波サンプルを取得 (足りない範囲は 0 = 無音)
        let s0 = step * SAMPLES_PER_STEP;
        let mut samples = [0i32; SAMPLES_PER_STEP];
        for i in 0..SAMPLES_PER_STEP {
            let idx = s0 + i;
            if idx < waveform.len() {
                samples[i] = waveform[idx];
            }
        }
        // M0: 8 サンプル → 20 次元電流ベクトル
        let ext = cochlea.process_step(&samples);

        // M1: step 1 回
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

// ──────────────────────────────────────────────────────────────
// 時間 bin 化 fingerprint (thermo_m1_evaluation と同じ)
// ──────────────────────────────────────────────────────────────

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

// ──────────────────────────────────────────────────────────────
// 評価
// ──────────────────────────────────────────────────────────────

fn evaluate(
    net: &mut ThermoNetwork,
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

    for _ in 0..n_sample {
        for (si, _syl) in syllables.iter().enumerate() {
            let log = present_syllable(net, cochlea, &waveforms[si]);
            per_syl_spikes[si] += log.len() as u64;
            let mut fired_any = vec![false; n_out];
            for &(oi, _) in &log { fired_any[oi] = true; }
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
}

// ──────────────────────────────────────────────────────────────
// main
// ──────────────────────────────────────────────────────────────

fn main() {
    let n_train: usize = std::env::args().nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(100);

    let n_sample: usize = 20;
    let snap_interval: usize = if n_train >= 100_000 {
        n_train / 200
    } else if n_train >= 500 {
        500
    } else {
        (n_train / 10).max(10)
    };

    println!("== M0+M1 パイプライン: 音素 5 種識別 ==");
    println!("  音素: pa, ki, tu, se, mo");
    println!("  M0: 20 帯域蝸牛 (ERB スケール 50Hz-4kHz)");
    println!("  M1: Fork F-G1-R1 v2 (8 近傍, 1セル1ニューロン)");
    println!("  評価: 時間 bin 化 fingerprint (40 出力 × 30 bin)");

    let cfg = ThermoNetworkConfig::default();
    let cfg_for_print = cfg.clone();
    let mut net = ThermoNetwork::new(cfg);
    let mut cochlea = Cochlea::new();
    let syllables = standard_syllables();
    let mut noise = LfsrNoise::new(0xACE1);

    // 5 音節の波形を事前生成 (決定論的)
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
    println!("    fanout    : input={}, grid={}x{}",
        cfg_for_print.input_fanout, cfg_for_print.grid_width, cfg_for_print.grid_height);

    // CSV 出力
    let mut csv = File::create("phase2_f_phoneme_snapshots.csv").expect("csv");
    writeln!(csv, "trial,within,selectivity,active,silent_ratio,entropy_mean,entropy_max,enthalpy_mean,conductance_mean,conductance_max,open_syn,plastic_syn,axons_grown,axons_pruned,sparsity").unwrap();

    // ─── Phase 1: PRE 評価 ───
    println!("\n== Phase 1: 訓練前評価 ==");
    evaluate(&mut net, &mut cochlea, &syllables, &waveforms, n_sample, "PRE");

    // ─── Phase 2: 訓練 ───
    println!("\n== Phase 2: 訓練 {n_train} 試行 (音節ランダム選択) ==");
    println!("  snap_interval = {snap_interval}");

    let mut rng = StdRng::seed_from_u64(42);
    for trial in 1..=n_train {
        let si = rng.gen_range(0..syllables.len());
        let _ = present_syllable(&mut net, &mut cochlea, &waveforms[si]);

        if trial % snap_interval == 0 || trial == n_train {
            // mini 評価 (5 sample)
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
                net.n_open_synapses(), net.n_plastic_synapses(),
                net.axons_grown, net.axons_pruned, obs.sparsity).unwrap();
        }
    }

    // ─── Phase 3: POST 評価 ───
    println!("\n== Phase 3: 訓練後評価 ==");
    evaluate(&mut net, &mut cochlea, &syllables, &waveforms, n_sample, "POST");

    println!("\n══════════════════════════════════════════════════════════");
    println!("  M0+M1 音素識別サマリ");
    println!("  軸索成長 累積={} 刈り取り累積={}", net.axons_grown, net.axons_pruned);
    println!("  open シナプス: {}/{}", net.n_open_synapses(), net.n_synapses());
    println!("  CSV: phase2_f_phoneme_snapshots.csv");
}
