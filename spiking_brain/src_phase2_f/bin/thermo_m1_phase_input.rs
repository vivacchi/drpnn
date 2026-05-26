//! Phase 2 評価ランナー (4 位相ローテーション訓練版)
//!
//! 訓練入力: 決定論的な 4 位相ローテーション
//!   - 50 step (25ms) ごとにグループが切り替わる
//!   - グループ 1: input 0-4   (25ms 発火)
//!   - グループ 2: input 5-9   (次の 25ms)
//!   - グループ 3: input 10-14
//!   - グループ 4: input 15-19
//!   - 1 試行 300ms = 6 ローテーション × 50ms (12 phase 完結)
//!
//! 性質:
//!   - 固定 5 パターン (時間相関大) と ホワイトノイズ (相関ゼロ) の中間
//!   - 規則的だが特定パターンではない → 「軸索成長が起きるか」の境界条件
//!
//! 評価: 固定 A-E (Phase 1 と同じ、比較のため)
//!
//! CLI:
//!   cargo run --release --bin thermo_m1_phase_input -- [n_train]

use spiking_brain::phase2::thermo_network::{ThermoNetwork, ThermoNetworkConfig};
use spiking_brain::trace::{cosine_similarity, OutputTrace};
use rand::prelude::*;
use rand::seq::SliceRandom;
use std::fs::File;
use std::io::Write as IoWrite;

// ─────────────────────────────────────────────────────────
// 評価パターン (固定 A-E、PHASE 1 と同じ)
// ─────────────────────────────────────────────────────────

fn make_eval_pattern_set(n_input: usize) -> [(char, Vec<f64>); 5] {
    let step = 5.0;
    let a: Vec<f64> = (0..n_input).map(|i| i as f64 * step).collect();
    let b: Vec<f64> = (0..n_input).map(|i| (n_input - 1 - i) as f64 * step).collect();
    let mut c = vec![0.0; n_input];
    let mut order_c: Vec<usize> = Vec::with_capacity(n_input);
    for i in (0..n_input).step_by(2) { order_c.push(i); }
    for i in (1..n_input).step_by(2) { order_c.push(i); }
    for (pos, &input_idx) in order_c.iter().enumerate() { c[input_idx] = pos as f64 * step; }
    let mut d = vec![0.0; n_input];
    for i in 0..n_input {
        let input_idx = (i + 10) % n_input;
        d[input_idx] = i as f64 * step;
    }
    let mut rng = StdRng::seed_from_u64(0xE_E_E_E);
    let mut order_e: Vec<usize> = (0..n_input).collect();
    order_e.shuffle(&mut rng);
    let mut e = vec![0.0; n_input];
    for (pos, &input_idx) in order_e.iter().enumerate() { e[input_idx] = pos as f64 * step; }
    [('A', a), ('B', b), ('C', c), ('D', d), ('E', e)]
}

const TRIAL_DURATION_MS: f64 = 300.0;
const DT_MS: f64 = 0.5;
const PULSE_WIDTH_MS: f64 = 4.0;
const INPUT_CURRENT: i32 = 60;

/// 評価用: パターン提示
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
            } else { 0 };
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

/// 訓練用: 4 位相ローテーション入力
/// 50 step ごとにグループが切り替わる、完全決定論的
fn present_phase_rotation(
    net: &mut ThermoNetwork,
    n_input: usize,
    phase_duration_step: i32, // 例: 50 (= 25ms)
    n_groups: usize,          // 例: 4
) -> Vec<(usize, f64)> {
    net.reset_trial_state();
    let trial_start_t_ms = net.current_time as f64 * DT_MS;
    let n_steps = (TRIAL_DURATION_MS / DT_MS) as usize;
    let group_size = (n_input + n_groups - 1) / n_groups; // ceiling

    let mut external_input = vec![0i32; n_input];
    let mut out_log: Vec<(usize, f64)> = Vec::new();

    for step in 0..n_steps {
        let phase = (step as i32 / phase_duration_step) as usize % n_groups;
        let start = phase * group_size;
        let end = ((phase + 1) * group_size).min(n_input);
        for k in 0..n_input {
            external_input[k] = if k >= start && k < end { INPUT_CURRENT } else { 0 };
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

/// 時間 bin 化 fingerprint (PAPER §5.9 で確立した正しい指標、聴覚モデル向け).
/// 2026-05-25: 旧 `tr.fingerprint(t_end)` (時間平滑化、tau=50ms) から切替。
/// thermo_m1_evaluation.rs / m0_m1_pipeline.rs / internal_state_probe.rs と統一。
/// 過去の C10 (4 位相ローテーション) 結果は旧評価 (時間平滑化) なので、新評価と直接比較不可。
const FINGERPRINT_BIN_WIDTH_MS: f64 = 10.0;
fn fingerprint_from_log(log: &[(usize, f64)], n_out: usize, t_end: f64, _tau: f64) -> Vec<f64> {
    let mut tr = OutputTrace::new(n_out, 50.0);
    for &(oi, t) in log { tr.record_spike(oi, t); }
    tr.time_binned_fingerprint(t_end, FINGERPRINT_BIN_WIDTH_MS)
}

struct EvalResult {
    per_pattern_fps: Vec<Vec<Vec<f64>>>,
    per_pattern_hit: Vec<Vec<u32>>,
    per_pattern_total_spikes: Vec<u64>,
    selectivity: f64,
    within: f64,
    between: f64,
    active: usize,
}

fn evaluate_patterns(net: &mut ThermoNetwork, patterns: &[(char, Vec<f64>)], n_sample: usize) -> EvalResult {
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
    let mut within_sum = 0.0; let mut within_n = 0;
    for fps in per_pattern_fps {
        if fps.len() >= 2 { within_sum += mean_pairwise(fps); within_n += 1; }
    }
    let mut between_sum = 0.0; let mut between_n = 0;
    for i in 0..per_pattern_fps.len() {
        for j in (i + 1)..per_pattern_fps.len() {
            between_sum += mean_between(&per_pattern_fps[i], &per_pattern_fps[j]);
            between_n += 1;
        }
    }
    let within = if within_n > 0 { within_sum / within_n as f64 } else { 0.0 };
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

fn main() {
    let n_train: usize = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(100);
    let n_sample: usize = 20;
    let snap_interval: usize = if n_train >= 100_000 { n_train / 200 }
        else if n_train >= 500 { 500 }
        else { (n_train / 10).max(10) };

    // 4 位相ローテーションのパラメータ
    let phase_duration_step: i32 = 50; // 25ms ごと
    let n_groups: usize = 4;            // 4 phase

    println!("== Phase 2 (4位相ローテーション訓練版) ==");
    println!("  訓練入力 = 4 phase rotation (group 5 inputs each, 25ms/phase, 12cycle/trial)");
    println!("  評価入力 = 固定 A-E (Phase 1 と同じ、比較のため)");

    let cfg = ThermoNetworkConfig::default();
    let mut net = ThermoNetwork::new(cfg);
    println!("\n  neurons={}, synapses={} (open={})", net.n_neurons(), net.n_synapses(), net.n_open_synapses());

    let n_input = net.input_neurons.len();
    let patterns = make_eval_pattern_set(n_input);

    let mut csv_snap = File::create("phase2_phaseinput_snapshots.csv").expect("snap csv");
    writeln!(csv_snap, "trial,within,selectivity,active,silent_ratio,entropy_mean,entropy_max,enthalpy_mean,conductance_mean,conductance_max,open_syn,plastic_syn,axons_grown,axons_pruned,sparsity,entropy_std,conductance_std,syn_growth_rate").unwrap();

    println!("\n== Phase 1: 訓練前評価 ==");
    let pre = evaluate_patterns(&mut net, &patterns, n_sample);
    print_eval("PRE", &pre, n_sample);

    println!("\n== Phase 2: 4位相ローテーション訓練 {n_train} 試行 ==");
    println!("  snap_interval = {snap_interval} step");

    for trial in 1..=n_train {
        let _ = present_phase_rotation(&mut net, n_input, phase_duration_step, n_groups);

        if trial % snap_interval == 0 || trial == n_train {
            let mini = evaluate_patterns(&mut net, &patterns, 5);
            let active = mini.active;
            let n_out = mini.per_pattern_hit[0].len();
            let silent_ratio = (n_out - active) as f64 / n_out as f64;
            let obs = net.macro_observables();

            println!("  {:>5}  within={:.3}  sel={:.3}  active={:>2}/{:>2}  silent={:.2}  ent_μ={:.1}  cond_μ/max={:.1}/{}  sp={:.3}  grown={}",
                trial, mini.within, mini.selectivity, active, n_out,
                silent_ratio, obs.entropy_mean, obs.conductance_mean, obs.conductance_max,
                obs.sparsity, net.axons_grown);

            writeln!(csv_snap, "{},{:.4},{:.4},{},{:.4},{:.2},{},{:.2},{:.2},{},{},{},{},{},{:.4},{:.3},{:.3},{:.3}",
                trial, mini.within, mini.selectivity, active, silent_ratio,
                obs.entropy_mean, obs.entropy_max, obs.enthalpy_mean,
                obs.conductance_mean, obs.conductance_max,
                net.n_open_synapses(), net.n_plastic_synapses(),
                net.axons_grown, net.axons_pruned,
                obs.sparsity, obs.entropy_std, obs.conductance_std, obs.syn_growth_rate).unwrap();
        }
    }

    println!("\n== Phase 3: 訓練後評価 ==");
    let post = evaluate_patterns(&mut net, &patterns, n_sample);
    print_eval("POST", &post, n_sample);

    println!("\n══════════════════════════════════════════════════════════");
    println!("  4位相ローテーション訓練 サマリ ({n_train} trials)");
    println!("══════════════════════════════════════════════════════════");
    println!("  PRE  sel={:.3} within={:.3} active={}", pre.selectivity, pre.within, pre.active);
    println!("  POST sel={:.3} within={:.3} active={}", post.selectivity, post.within, post.active);
    let obs = net.macro_observables();
    println!("  entropy μ/max/std    : {:.1} / {} / {:.2}", obs.entropy_mean, obs.entropy_max, obs.entropy_std);
    println!("  conductance μ/max/std: {:.2} / {} / {:.2}", obs.conductance_mean, obs.conductance_max, obs.conductance_std);
    println!("  sparsity             : {:.4}", obs.sparsity);
    println!("  axons grown/pruned   : {}/{}", net.axons_grown, net.axons_pruned);
    println!("\n  CSV: phase2_phaseinput_snapshots.csv");
}
