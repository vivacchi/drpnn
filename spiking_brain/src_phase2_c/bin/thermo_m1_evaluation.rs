//! Phase 2 評価ランナー (熱力学版 SNN)
//!
//! PHASE2_INSTRUCTION.md §5 に従う:
//!   - 3000 試行、5 パターン (新 A-E) ランダム選択
//!   - 報酬なし、target なし、外部評価なし
//!   - 500 試行ごとに snapshot (entropy 分布、enthalpy、conductance、シナプス数、軸索成長累積)
//!
//! CLI:
//!   cargo run --release --bin thermo_m1_evaluation -- [n_train]
//!
//! デフォルト n_train = 100 (動作確認モード)、3000 で正式評価。

use spiking_brain::phase2_c::thermo_network::{ThermoNetwork, ThermoNetworkConfig};
use spiking_brain::trace::{cosine_similarity, OutputTrace};
use rand::prelude::*;
use rand::seq::SliceRandom;
use std::fs::File;
use std::io::Write as IoWrite;

// ─────────────────────────────────────────────────────────
// パターン生成 (Phase 1 と同じ 新 A-E)
// ─────────────────────────────────────────────────────────

/// 全パターン 0-95ms (5ms step × 20 入力)、時間順序のみ異なる
fn make_eval_pattern_set(n_input: usize) -> [(char, Vec<f64>); 5] {
    let step = 5.0;
    let a: Vec<f64> = (0..n_input).map(|i| i as f64 * step).collect();
    let b: Vec<f64> = (0..n_input).map(|i| (n_input - 1 - i) as f64 * step).collect();

    let mut c = vec![0.0; n_input];
    let mut order_c: Vec<usize> = Vec::with_capacity(n_input);
    for i in (0..n_input).step_by(2) { order_c.push(i); }
    for i in (1..n_input).step_by(2) { order_c.push(i); }
    for (pos, &input_idx) in order_c.iter().enumerate() {
        c[input_idx] = pos as f64 * step;
    }

    let mut d = vec![0.0; n_input];
    for i in 0..n_input {
        let input_idx = (i + 10) % n_input;
        d[input_idx] = i as f64 * step;
    }

    let mut rng = StdRng::seed_from_u64(0xE_E_E_E);
    let mut order_e: Vec<usize> = (0..n_input).collect();
    order_e.shuffle(&mut rng);
    let mut e = vec![0.0; n_input];
    for (pos, &input_idx) in order_e.iter().enumerate() {
        e[input_idx] = pos as f64 * step;
    }

    [('A', a), ('B', b), ('C', c), ('D', d), ('E', e)]
}

// ─────────────────────────────────────────────────────────
// パターン提示
// ─────────────────────────────────────────────────────────

const TRIAL_DURATION_MS: f64 = 300.0;
const DT_MS: f64 = 0.5;
const PULSE_WIDTH_MS: f64 = 4.0;
const INPUT_CURRENT: i32 = 60; // 入力ニューロンを発火させる十分な電流

/// 1 試行を提示し、出力ニューロンの発火ログを返す。
/// ログの時刻は **試行開始からの相対 ms** (fingerprint 計算で t_end=TRIAL_DURATION と整合)。
/// ネットワーク内部の current_time は連続 (STDP の last_spike_time との整合性のため)。
fn present_pattern(
    net: &mut ThermoNetwork,
    pattern_times: &[f64],
) -> Vec<(usize, f64)> {
    net.reset_trial_state();

    let trial_start_t_ms = net.current_time as f64 * DT_MS;

    let n_steps = (TRIAL_DURATION_MS / DT_MS) as usize;
    let pulse_steps = (PULSE_WIDTH_MS / DT_MS).max(1.0) as i64;
    let fire_steps: Vec<i64> = pattern_times.iter().map(|&t| (t / DT_MS) as i64).collect();

    let mut external_input = vec![0i32; pattern_times.len()];
    let mut out_log: Vec<(usize, f64)> = Vec::new();

    for step in 0..n_steps {
        for k in 0..pattern_times.len() {
            let fs = fire_steps[k];
            external_input[k] = if (step as i64) >= fs && (step as i64) < fs + pulse_steps {
                INPUT_CURRENT
            } else {
                0
            };
        }
        let fired = net.step(&external_input);
        for nid in fired {
            if let Some(oi) = net.output_index_of(nid) {
                let t_abs_ms = net.current_time as f64 * DT_MS;
                let t_rel_ms = t_abs_ms - trial_start_t_ms;
                out_log.push((oi, t_rel_ms));
            }
        }
    }
    out_log
}

/// 訓練用: 確率的ホワイトノイズ入力。
/// 各 step で各 input neuron が独立に確率 p_per_step で発火 (パルス幅 8 step)。
/// 内部機構 (NN) は決定論的だが、外部刺激 (環境) はランダム — 生物の感覚入力と整合。
fn present_white_noise(
    net: &mut ThermoNetwork,
    n_input: usize,
    p_per_step: f64,
    rng: &mut StdRng,
) -> Vec<(usize, f64)> {
    net.reset_trial_state();

    let trial_start_t_ms = net.current_time as f64 * DT_MS;
    let n_steps = (TRIAL_DURATION_MS / DT_MS) as usize;
    let pulse_steps = (PULSE_WIDTH_MS / DT_MS).max(1.0) as i32;

    // 各 input neuron のパルス残り step (>0 ならパルス継続中)
    let mut pulse_remaining = vec![0i32; n_input];
    let mut external_input = vec![0i32; n_input];
    let mut out_log: Vec<(usize, f64)> = Vec::new();

    for _step in 0..n_steps {
        for k in 0..n_input {
            if pulse_remaining[k] > 0 {
                pulse_remaining[k] -= 1;
                external_input[k] = INPUT_CURRENT;
            } else if rng.gen::<f64>() < p_per_step {
                pulse_remaining[k] = pulse_steps - 1; // この step + 残り (pulse_steps-1) step
                external_input[k] = INPUT_CURRENT;
            } else {
                external_input[k] = 0;
            }
        }
        let fired = net.step(&external_input);
        for nid in fired {
            if let Some(oi) = net.output_index_of(nid) {
                let t_abs_ms = net.current_time as f64 * DT_MS;
                out_log.push((oi, t_abs_ms - trial_start_t_ms));
            }
        }
    }
    out_log
}

fn fingerprint_from_log(log: &[(usize, f64)], n_out: usize, t_end: f64, tau: f64) -> Vec<f64> {
    let mut tr = OutputTrace::new(n_out, tau);
    for &(oi, t) in log { tr.record_spike(oi, t); }
    tr.fingerprint(t_end)
}

// ─────────────────────────────────────────────────────────
// 評価 (5 性質)
// ─────────────────────────────────────────────────────────

struct EvalResult {
    per_pattern_fps: Vec<Vec<Vec<f64>>>,
    per_pattern_hit: Vec<Vec<u32>>,
    per_pattern_total_spikes: Vec<u64>,
    selectivity: f64,
    within: f64,
    between: f64,
    active: usize,
}

fn evaluate_patterns(
    net: &mut ThermoNetwork,
    patterns: &[(char, Vec<f64>)],
    n_sample: usize,
) -> EvalResult {
    let n_out = net.output_neurons.len();
    let n_pat = patterns.len();
    let mut per_pattern_fps: Vec<Vec<Vec<f64>>> = vec![Vec::with_capacity(n_sample); n_pat];
    let mut per_pattern_hit: Vec<Vec<u32>> = vec![vec![0u32; n_out]; n_pat];
    let mut per_pattern_total_spikes: Vec<u64> = vec![0u64; n_pat];

    for _s in 0..n_sample {
        for (pi, (_label, pat)) in patterns.iter().enumerate() {
            let log = present_pattern(net, pat);
            per_pattern_total_spikes[pi] += log.len() as u64;
            let mut fired_any = vec![false; n_out];
            for &(oi, _) in &log { fired_any[oi] = true; }
            for ni in 0..n_out {
                if fired_any[ni] { per_pattern_hit[pi][ni] += 1; }
            }
            per_pattern_fps[pi].push(fingerprint_from_log(&log, n_out, TRIAL_DURATION_MS, 50.0));
        }
    }

    let (selectivity, within, between) = compute_selectivity(&per_pattern_fps);
    let mut active = 0;
    for ni in 0..n_out {
        for pi in 0..n_pat {
            if per_pattern_hit[pi][ni] > 0 { active += 1; break; }
        }
    }

    EvalResult { per_pattern_fps, per_pattern_hit, per_pattern_total_spikes, selectivity, within, between, active }
}

fn mean_pairwise(fps: &[Vec<f64>]) -> f64 {
    let mut sims = Vec::new();
    for i in 0..fps.len() {
        for j in (i + 1)..fps.len() {
            sims.push(cosine_similarity(&fps[i], &fps[j]));
        }
    }
    if sims.is_empty() { 0.0 } else { sims.iter().sum::<f64>() / sims.len() as f64 }
}

fn mean_between(a: &[Vec<f64>], b: &[Vec<f64>]) -> f64 {
    let mut sims = Vec::new();
    for x in a { for y in b { sims.push(cosine_similarity(x, y)); } }
    if sims.is_empty() { 0.0 } else { sims.iter().sum::<f64>() / sims.len() as f64 }
}

fn compute_selectivity(per_pattern_fps: &[Vec<Vec<f64>>]) -> (f64, f64, f64) {
    let mut within_sum = 0.0;
    let mut within_n = 0;
    for fps in per_pattern_fps {
        if fps.len() >= 2 {
            within_sum += mean_pairwise(fps);
            within_n += 1;
        }
    }
    let mut between_sum = 0.0;
    let mut between_n = 0;
    for i in 0..per_pattern_fps.len() {
        for j in (i + 1)..per_pattern_fps.len() {
            between_sum += mean_between(&per_pattern_fps[i], &per_pattern_fps[j]);
            between_n += 1;
        }
    }
    let within  = if within_n  > 0 { within_sum  / within_n as f64  } else { 0.0 };
    let between = if between_n > 0 { between_sum / between_n as f64 } else { 0.0 };
    (within - between, within, between)
}

fn print_eval(label: &str, r: &EvalResult, n_sample: usize) {
    println!("\n  -- {label} --");
    println!("    selectivity   : {:.3}  (within {:.3} - between {:.3})",
        r.selectivity, r.within, r.between);
    println!("    active outputs: {} / {}", r.active, r.per_pattern_hit[0].len());
    print!("    hit /pattern  : ");
    for pi in 0..r.per_pattern_hit.len() {
        let h = r.per_pattern_hit[pi].iter().filter(|&&c| c > 0).count();
        print!("{}:{}/{}  ", ['A','B','C','D','E'][pi], h, r.per_pattern_hit[0].len());
    }
    println!("(across {n_sample} samples)");
    print!("    total spikes  : ");
    for pi in 0..r.per_pattern_total_spikes.len() {
        print!("{}:{}  ", ['A','B','C','D','E'][pi], r.per_pattern_total_spikes[pi]);
    }
    println!();
}

// ─────────────────────────────────────────────────────────
// main
// ─────────────────────────────────────────────────────────

fn main() {
    let n_train: usize = std::env::args().nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(100); // デフォルトは動作確認モード

    let n_sample: usize = 20;
    // snap_interval: 試行数に応じて約 200 snapshot に収まるよう動的に
    //   1M 試行 → 5000 刻み、10万 → 500 刻み、それ以下は適宜
    let snap_interval: usize = if n_train >= 100_000 {
        n_train / 200
    } else if n_train >= 500 {
        500
    } else {
        (n_train / 10).max(10)
    };

    // ホワイトノイズ訓練の発火確率 (per step、per input neuron)
    // 1 / (TRIAL_DURATION_MS / DT_MS) = 1/600、平均 1 spike/input/trial で固定パターンと同じ
    let p_white_noise: f64 = 1.0 / 600.0;

    println!("== Phase 2 Fork C (MIN_COND=40 + MATURATION=1000) ホワイトノイズ訓練版 ==");
    println!("  訓練入力 = 確率的ホワイトノイズ (p={:.5}/step/input)", p_white_noise);
    println!("  内部機構は決定論的、外部刺激は環境のランダム性そのまま");
    println!("  評価入力 = 固定 A-E (比較可能性のため)");

    let cfg = ThermoNetworkConfig::default();
    println!("\n  グリッド: {}x{}", cfg.grid_width, cfg.grid_height);
    println!("  ニューロン: input={} exc={} inh={}",
        cfg.n_input, cfg.n_excitatory, cfg.n_inhibitory);
    println!("  入力 fanout: {}", cfg.input_fanout);
    println!("  軸索成長周期: {} step", cfg.axon_growth_interval);

    let mut net = ThermoNetwork::new(cfg);
    println!("\n  実構築:");
    println!("    neurons     : {}", net.n_neurons());
    println!("    synapses    : {} (open={}, plastic={})",
        net.n_synapses(), net.n_open_synapses(), net.n_plastic_synapses());
    println!("    memory      : {} KB", net.memory_bytes() / 1024);

    let n_input = net.input_neurons.len();
    let patterns = make_eval_pattern_set(n_input);
    println!("\n  評価パターン (新 A-E, 0-95ms 5ms step):");
    for (label, pat) in &patterns {
        println!("    {label}: {:?}", &pat[..pat.len().min(8)]);
    }

    // CSV 出力 (ホワイトノイズ訓練版は別ファイル名にして既存結果を保護)
    let mut csv_snap = File::create("phase2_c_whitenoise_snapshots.csv").expect("snap csv");
    writeln!(csv_snap, "trial,within,selectivity,active,silent_ratio,entropy_mean,entropy_max,enthalpy_mean,conductance_mean,conductance_max,open_syn,plastic_syn,axons_grown,axons_pruned,sparsity,entropy_std,conductance_std,syn_growth_rate").unwrap();

    // ─────────────────────────────────────────────────
    // Phase 1: 訓練前評価 (PRE)
    // ─────────────────────────────────────────────────
    println!("\n== Phase 1: 訓練前評価 ==");
    let pre = evaluate_patterns(&mut net, &patterns, n_sample);
    print_eval("PRE", &pre, n_sample);

    // ─────────────────────────────────────────────────
    // Phase 2: 訓練 (報酬なし、ただパターンを提示するだけ)
    // ─────────────────────────────────────────────────
    println!("\n== Phase 2: 訓練 {n_train} 試行 (ホワイトノイズ入力、報酬なし、target なし、apply_* なし) ==");
    println!("  snap_interval = {snap_interval} step");
    println!("  trial  within  select  active  silent  ent_μ  ent_max  enth_μ  cond_μ/max  open  grown/pruned");

    let mut rng = StdRng::seed_from_u64(42);
    for trial in 1..=n_train {
        // 訓練入力: 確率的ホワイトノイズ (内部機構は決定論的、外部刺激のみランダム)
        let _ = present_white_noise(&mut net, n_input, p_white_noise, &mut rng);

        if trial % snap_interval == 0 || trial == n_train {
            let mini = evaluate_patterns(&mut net, &patterns, 5);
            let active = mini.active;
            let n_out = mini.per_pattern_hit[0].len();
            let silent_ratio = (n_out - active) as f64 / n_out as f64;
            let obs = net.macro_observables();

            println!("  {:>5}  {:.3}   {:.3}   {:>2}/{:>2}  {:.2}   {:.1}   {}   {:.1}   {:.1}/{}   {}   {}/{}  sp={:.3}",
                trial, mini.within, mini.selectivity, active, n_out,
                silent_ratio, obs.entropy_mean, obs.entropy_max, obs.enthalpy_mean,
                obs.conductance_mean, obs.conductance_max, net.n_open_synapses(),
                net.axons_grown, net.axons_pruned, obs.sparsity);

            writeln!(csv_snap, "{},{:.4},{:.4},{},{:.4},{:.2},{},{:.2},{:.2},{},{},{},{},{},{:.4},{:.3},{:.3},{:.3}",
                trial, mini.within, mini.selectivity, active, silent_ratio,
                obs.entropy_mean, obs.entropy_max, obs.enthalpy_mean,
                obs.conductance_mean, obs.conductance_max,
                net.n_open_synapses(), net.n_plastic_synapses(),
                net.axons_grown, net.axons_pruned,
                obs.sparsity, obs.entropy_std, obs.conductance_std, obs.syn_growth_rate).unwrap();
        }
    }

    // ─────────────────────────────────────────────────
    // Phase 3: 訓練後評価 (POST)
    // ─────────────────────────────────────────────────
    println!("\n== Phase 3: 訓練後評価 ==");
    let post = evaluate_patterns(&mut net, &patterns, n_sample);
    print_eval("POST", &post, n_sample);

    // ─────────────────────────────────────────────────
    // 要約
    // ─────────────────────────────────────────────────
    println!("\n══════════════════════════════════════════════════════════");
    println!("              Phase 2 動作確認サマリ");
    println!("══════════════════════════════════════════════════════════");
    println!("  PRE  selectivity={:.3}  within={:.3}  active={}/{}",
        pre.selectivity, pre.within, pre.active, post.per_pattern_hit[0].len());
    println!("  POST selectivity={:.3}  within={:.3}  active={}/{}",
        post.selectivity, post.within, post.active, post.per_pattern_hit[0].len());
    let (ent_mean, ent_max) = net.entropy_stats();
    let (cond_mean, cond_max) = net.conductance_stats();
    println!("  最終 entropy μ={:.1} max={}", ent_mean, ent_max);
    println!("  最終 conductance μ={:.1} max={}", cond_mean, cond_max);
    println!("  軸索成長 累積={} 刈り取り累積={}", net.axons_grown, net.axons_pruned);
    println!("  open シナプス: {}/{}", net.n_open_synapses(), net.n_synapses());

    println!("\n  CSV: phase2_snapshots.csv");
}
