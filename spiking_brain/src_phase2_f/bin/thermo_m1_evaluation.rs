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

use spiking_brain::phase2_f::thermo_network::{ThermoNetwork, ThermoNetworkConfig};
use spiking_brain::trace::{cosine_similarity, OutputTrace};
use rand::prelude::*;
use rand::seq::SliceRandom;
use rand_distr::{Normal, Distribution};
use std::fs::File;
use std::io::Write as IoWrite;

// ─────────────────────────────────────────────────────────
// トポロジー可視化用ダンプ
// ─────────────────────────────────────────────────────────
fn dump_neurons_csv(net: &ThermoNetwork, path: &str) {
    let mut f = File::create(path).expect("neurons csv");
    writeln!(f, "id,x,y,kind").unwrap();
    let input_set: std::collections::HashSet<usize> =
        net.input_neurons.iter().copied().collect();
    let output_set: std::collections::HashSet<usize> =
        net.output_neurons.iter().copied().collect();
    for (i, n) in net.neurons.iter().enumerate() {
        let kind = if input_set.contains(&i) {
            "input"
        } else if n.is_inhibitory {
            "inh"
        } else if output_set.contains(&i) {
            "output"
        } else {
            "exc"
        };
        writeln!(f, "{},{},{},{}", i, n.position.0, n.position.1, kind).unwrap();
    }
}

fn dump_synapses_csv(net: &ThermoNetwork, initial_count: usize, path: &str) {
    let mut f = File::create(path).expect("synapse csv");
    writeln!(f, "idx,pre,post,conductance,alive,is_new").unwrap();
    for (i, s) in net.synapses.iter().enumerate() {
        let is_new = if i >= initial_count { 1 } else { 0 };
        writeln!(
            f, "{},{},{},{},{},{}",
            i, s.pre, s.post, s.conductance,
            if s.alive { 1 } else { 0 }, is_new
        ).unwrap();
    }
}

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

/// 訓練用: 時空間相関のあるホワイトノイズ入力 (生物的に妥当)。
///
/// 旧 present_white_noise は各 input が完全独立に発火 → 同時発火パターンがほぼ存在しない
/// → 「情報を持たないノイズ」になっていた問題を修正。
///
/// 新仕様:
///   - 各 step で確率 P_EVENT_PER_STEP で「event」が発生
///   - event 発生時、ランダム中心 c の周辺で k 個の input が同時パルス
///   - k = 平均 MEAN_CLUSTER_SIZE のポアソン的 (1 以上)
///   - 中心からの偏差は標準偏差 SPATIAL_SIGMA のガウシアン (= 空間相関)
///   - 同時発火することで polychronization の素材となる時空間構造が生まれる
///
/// 生物的解釈:
///   - 音: 倍音構造 (複数周波数チャンネル同時活性化)
///   - 視覚: エッジ (隣接ピクセル同時応答)
///   - 体性感覚: 接触面 (面状のニューロン群が同時応答)
const P_EVENT_PER_STEP: f64 = 1.0 / 120.0;       // 1 trial (600 step) で平均 5 event
const MEAN_CLUSTER_SIZE: f64 = 3.0;              // event ごとの平均同時発火数
const SPATIAL_SIGMA: f64 = 2.5;                  // 空間相関 (input idx 軸のガウシアン σ)

fn present_correlated_noise_with_log(
    net: &mut ThermoNetwork,
    n_input: usize,
    rng: &mut StdRng,
    mut in_log_opt: Option<&mut Vec<(usize, i64)>>,
    mut out_log_abs_opt: Option<&mut Vec<(usize, i64)>>,
) -> Vec<(usize, f64)> {
    net.reset_trial_state();

    let trial_start_t_ms = net.current_time as f64 * DT_MS;
    let n_steps = (TRIAL_DURATION_MS / DT_MS) as usize;
    let pulse_steps = (PULSE_WIDTH_MS / DT_MS).max(1.0) as i32;

    let mut pulse_remaining = vec![0i32; n_input];
    let mut external_input = vec![0i32; n_input];
    let mut out_log: Vec<(usize, f64)> = Vec::new();
    let normal = Normal::new(0.0f64, SPATIAL_SIGMA).unwrap();

    for _step in 0..n_steps {
        let step_abs = net.current_time as i64;

        // 既存パルスの継続
        for k in 0..n_input {
            if pulse_remaining[k] > 0 {
                pulse_remaining[k] -= 1;
                external_input[k] = INPUT_CURRENT;
            } else {
                external_input[k] = 0;
            }
        }

        // event 発生判定
        if rng.gen::<f64>() < P_EVENT_PER_STEP {
            // クラスターサイズ k (一様 0..2*mean、平均 mean、1 以上)
            let k_cluster = ((rng.gen::<f64>() * MEAN_CLUSTER_SIZE * 2.0)
                .round() as usize).max(1).min(n_input);
            // クラスター中心 c
            let center = rng.gen_range(0..n_input) as f64;

            let mut selected: std::collections::HashSet<usize> =
                std::collections::HashSet::new();
            let mut attempts = 0;
            while selected.len() < k_cluster && attempts < k_cluster * 10 + 5 {
                let offset = normal.sample(rng);
                let idx = (center + offset).round() as i32;
                if idx >= 0 && (idx as usize) < n_input {
                    selected.insert(idx as usize);
                }
                attempts += 1;
            }

            for &k in &selected {
                pulse_remaining[k] = pulse_steps - 1;
                external_input[k] = INPUT_CURRENT;
                if let Some(ref mut log) = in_log_opt {
                    log.push((k, step_abs));
                }
            }
        }

        let fired = net.step(&external_input);
        for nid in fired {
            if let Some(oi) = net.output_index_of(nid) {
                let t_abs_ms = net.current_time as f64 * DT_MS;
                out_log.push((oi, t_abs_ms - trial_start_t_ms));
                if let Some(ref mut log) = out_log_abs_opt {
                    log.push((oi, net.current_time as i64));
                }
            }
        }
    }
    out_log
}

/// 訓練用: 確率的ホワイトノイズ入力 (旧版・各 input 独立)。
/// 各 step で各 input neuron が独立に確率 p_per_step で発火 (パルス幅 8 step)。
/// 後方互換のため残してあるが、新規実験では present_correlated_noise_with_log を推奨。
fn present_white_noise(
    net: &mut ThermoNetwork,
    n_input: usize,
    p_per_step: f64,
    rng: &mut StdRng,
) -> Vec<(usize, f64)> {
    let mut _dummy_in: Option<&mut Vec<(usize, i64)>> = None;
    let mut _dummy_out: Option<&mut Vec<(usize, i64)>> = None;
    present_white_noise_with_log(net, n_input, p_per_step, rng, _dummy_in, _dummy_out)
}

/// ホワイトノイズ + 入力/出力スパイクの絶対時刻 (step) を任意で記録。
/// in_log / out_log を渡せば、その vec に push される (絶対時刻 step、abs)
///   in_log:  (input_idx, step_abs)  — 各 input neuron が「パルス開始した step」
///   out_log_abs: (output_idx, step_abs)
fn present_white_noise_with_log(
    net: &mut ThermoNetwork,
    n_input: usize,
    p_per_step: f64,
    rng: &mut StdRng,
    mut in_log_opt: Option<&mut Vec<(usize, i64)>>,
    mut out_log_abs_opt: Option<&mut Vec<(usize, i64)>>,
) -> Vec<(usize, f64)> {
    net.reset_trial_state();

    let trial_start_t_ms = net.current_time as f64 * DT_MS;
    let n_steps = (TRIAL_DURATION_MS / DT_MS) as usize;
    let pulse_steps = (PULSE_WIDTH_MS / DT_MS).max(1.0) as i32;

    let mut pulse_remaining = vec![0i32; n_input];
    let mut external_input = vec![0i32; n_input];
    let mut out_log: Vec<(usize, f64)> = Vec::new();

    for _step in 0..n_steps {
        let step_abs = net.current_time as i64;
        for k in 0..n_input {
            if pulse_remaining[k] > 0 {
                pulse_remaining[k] -= 1;
                external_input[k] = INPUT_CURRENT;
            } else if rng.gen::<f64>() < p_per_step {
                pulse_remaining[k] = pulse_steps - 1;
                external_input[k] = INPUT_CURRENT;
                // パルス開始イベントを記録 (pulse_steps 連続発火の代表点)
                if let Some(ref mut log) = in_log_opt {
                    log.push((k, step_abs));
                }
            } else {
                external_input[k] = 0;
            }
        }
        let fired = net.step(&external_input);
        for nid in fired {
            if let Some(oi) = net.output_index_of(nid) {
                let t_abs_ms = net.current_time as f64 * DT_MS;
                out_log.push((oi, t_abs_ms - trial_start_t_ms));
                if let Some(ref mut log) = out_log_abs_opt {
                    log.push((oi, net.current_time as i64));
                }
            }
        }
    }
    out_log
}

// ─────────────────────────────────────────────────────────
// 評価指標 fingerprint (時間 bin 化、聴覚モデル向け)
// ─────────────────────────────────────────────────────────
//
// 旧: t_end 時点での指数減衰加重 (n_output 次元、時間情報は tau=50ms で平滑化)
// 新: bin_width ごとの発火カウント (n_output × n_bins 次元、時間構造を直接保持)
//
// 聴覚モデルとして「いつ・どこで発火したか」を直接比較できる
//   bin_width = 10 ms → 300ms / 10ms = 30 bin
//   n_output = 40 → 40 × 30 = 1200 次元の fingerprint

/// 時間 bin 幅 (ms)。聴覚 phase locking (~5ms 精度) と音節レベル (~50ms) の間で 10ms 採用
const FINGERPRINT_BIN_WIDTH_MS: f64 = 10.0;

fn fingerprint_from_log(log: &[(usize, f64)], n_out: usize, t_end: f64, _tau: f64) -> Vec<f64> {
    // tau は互換性のため引数に残すが、時間 bin 化版では未使用
    let mut tr = OutputTrace::new(n_out, 50.0);
    for &(oi, t) in log { tr.record_spike(oi, t); }
    tr.time_binned_fingerprint(t_end, FINGERPRINT_BIN_WIDTH_MS)
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
    // CLI: thermo_m1_evaluation_f [n_train] [mode] [record]
    //   mode:   "whitenoise" (デフォルト、相関ノイズ) / "whitenoise_old" (旧・独立)
    //           / "fixed" (固定 A-E パターン)
    //   record: "record" を指定すると入出力スパイクログを CSV 出力 (ホワイトノイズ系のみ)
    let n_train: usize = std::env::args().nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(100);
    let train_mode: String = std::env::args().nth(2).unwrap_or_else(|| "whitenoise".to_string());
    let use_fixed_pattern = train_mode == "fixed";
    let use_old_whitenoise = train_mode == "whitenoise_old";
    let record_spikes: bool = std::env::args().nth(3).as_deref() == Some("record");

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

    let mode_label = if use_fixed_pattern { "固定 A-E パターン" }
        else if use_old_whitenoise { "ホワイトノイズ (旧・独立)" }
        else { "ホワイトノイズ (相関・推奨)" };
    println!("== Phase 2 Fork F-G1-R1 (vitality + LTD復活 + enthalpy 統合 + 過剰接続) {mode_label}訓練版 ==");
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
    let csv_filename = if use_fixed_pattern {
        "phase2_f_fixed_snapshots.csv"
    } else {
        "phase2_f_whitenoise_snapshots.csv"
    };
    let mut csv_snap = File::create(csv_filename).expect("snap csv");
    writeln!(csv_snap, "trial,within,selectivity,active,silent_ratio,entropy_mean,entropy_max,enthalpy_mean,conductance_mean,conductance_max,open_syn,plastic_syn,axons_grown,axons_pruned,sparsity,entropy_std,conductance_std,syn_growth_rate").unwrap();

    // ─────────────────────────────────────────────────
    // Phase 1: 訓練前評価 (PRE)
    // ─────────────────────────────────────────────────
    // ネットワーク構築直後の総数を「初期 (非新生)」として記録
    // PRE 評価中も axon_growth が走るので、構築直後を基準にしないと「新生」判定がブレる
    let initial_syn_count = net.synapses.len();
    let topo_prefix = if use_fixed_pattern { "phase2_f_fixed" } else { "phase2_f_whitenoise" };
    dump_neurons_csv(&net, &format!("{topo_prefix}_topo_neurons.csv"));

    println!("\n== Phase 1: 訓練前評価 ==");
    let pre = evaluate_patterns(&mut net, &patterns, n_sample);
    print_eval("PRE", &pre, n_sample);

    // PRE 時点 (= PRE 評価後) のトポロジーをダンプ
    dump_synapses_csv(&net, initial_syn_count, &format!("{topo_prefix}_topo_syn_pre.csv"));
    println!("  topology dump (PRE): {topo_prefix}_topo_syn_pre.csv ({} syn, {} new since build)",
        net.synapses.len(), net.synapses.len() - initial_syn_count);

    // ─────────────────────────────────────────────────
    // Phase 2: 訓練 (報酬なし、ただパターンを提示するだけ)
    // ─────────────────────────────────────────────────
    println!("\n== Phase 2: 訓練 {n_train} 試行 ({mode_label}入力、報酬なし、target なし、apply_* なし) ==");
    println!("  snap_interval = {snap_interval} step");
    println!("  trial  within  select  active  silent  ent_μ  ent_max  enth_μ  cond_μ/max  open  grown/pruned");

    let mut rng = StdRng::seed_from_u64(42);

    // 入出力スパイクログ (record モードのみ使用、ホワイトノイズ専用)
    let do_record = record_spikes && !use_fixed_pattern;
    let mut in_spike_log: Vec<(usize, i64)> = Vec::new();
    let mut out_spike_log_abs: Vec<(usize, i64)> = Vec::new();
    if do_record {
        println!("  [record] 入出力スパイクログを CSV 出力します");
    }

    for trial in 1..=n_train {
        if use_fixed_pattern {
            // 固定 A-E パターンからランダム選択
            let pi = rng.gen_range(0..patterns.len());
            let pat = &patterns[pi].1;
            let _ = present_pattern(&mut net, pat);
        } else if use_old_whitenoise {
            // 旧 (独立) ホワイトノイズ
            if do_record {
                let _ = present_white_noise_with_log(
                    &mut net, n_input, p_white_noise, &mut rng,
                    Some(&mut in_spike_log),
                    Some(&mut out_spike_log_abs),
                );
            } else {
                let _ = present_white_noise(&mut net, n_input, p_white_noise, &mut rng);
            }
        } else {
            // 相関ノイズ (デフォルト)
            if do_record {
                let _ = present_correlated_noise_with_log(
                    &mut net, n_input, &mut rng,
                    Some(&mut in_spike_log),
                    Some(&mut out_spike_log_abs),
                );
            } else {
                let _ = present_correlated_noise_with_log(
                    &mut net, n_input, &mut rng, None, None,
                );
            }
        }

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
    // record モード: 入出力スパイクログを CSV に書き出す
    // ─────────────────────────────────────────────────
    if do_record {
        let in_path = format!("{topo_prefix}_input_spikes.csv");
        let out_path = format!("{topo_prefix}_output_spikes.csv");
        let mut f_in = File::create(&in_path).expect("input spikes csv");
        writeln!(f_in, "step,input_idx").unwrap();
        for &(idx, step) in &in_spike_log {
            writeln!(f_in, "{},{}", step, idx).unwrap();
        }
        let mut f_out = File::create(&out_path).expect("output spikes csv");
        writeln!(f_out, "step,output_idx").unwrap();
        for &(idx, step) in &out_spike_log_abs {
            writeln!(f_out, "{},{}", step, idx).unwrap();
        }
        println!("  [record] spike logs: {} ({} events) / {} ({} events)",
            in_path, in_spike_log.len(), out_path, out_spike_log_abs.len());
    }

    // ─────────────────────────────────────────────────
    // Phase 3: 訓練後評価 (POST)
    // ─────────────────────────────────────────────────
    // POST トポロジーをダンプ (評価前、訓練直後の状態)
    dump_synapses_csv(&net, initial_syn_count, &format!("{topo_prefix}_topo_syn_post.csv"));
    println!("  topology dump (POST): {topo_prefix}_topo_syn_post.csv ({} syn, {} new)",
        net.synapses.len(), net.synapses.len() - initial_syn_count);

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

    println!("\n  CSV: {csv_filename}");
}
