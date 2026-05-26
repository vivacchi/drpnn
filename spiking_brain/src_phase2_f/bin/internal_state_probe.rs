//! 内部状態レパートリ検証 (Kenet 2003 Phase 2 版)
//!
//! 仮説 (PAPER §5.12.3 より):
//!   訓練後の M1 を無刺激で走らせると、出力ニューロンの自発活動が
//!   「訓練刺激への応答パターン」と類似する。これは Kenet 2003 が視覚野で
//!   観察した「内部状態レパートリ」現象の Phase 2 版である。
//!
//! 実験プロトコル:
//!   Phase A: 訓練 (固定 A-E 10k, 仮想 M0 = M1 input spont=2)
//!   Phase B: 訓練後の各刺激への応答 fingerprint を 20 sample ずつ取得
//!     → 「学習応答 fingerprint 集合」5 パターン × 20 = 100 fingerprint
//!   Phase C: 外部刺激 = 0 で N=5000 step 走らせて、200 step 単位の
//!     window 25 個分の出力 fingerprint を取得
//!     → 「自発活動 fingerprint 集合」25 fingerprint
//!   Phase D: 各自発活動 fp と各刺激応答 fp の cos sim を計算
//!     - 自発 vs 刺激応答 (signal): 25 × 100 ペア
//!     - 自発 vs ランダム (帰無): シャッフルしたfp と比較
//!     - 自発の最大類似度がランダムより有意に高ければ仮説支持
//!
//! CLI:
//!   cargo run --release --bin internal_state_probe [n_train]
//!   デフォルト n_train=10000

use spiking_brain::phase2_f::thermo_network::{ThermoNetwork, ThermoNetworkConfig};
use spiking_brain::trace::{cosine_similarity, OutputTrace};
use rand::prelude::*;
use std::fs::File;
use std::io::Write as IoWrite;

const TRIAL_DURATION_MS: f64 = 300.0;
const DT_MS: f64 = 0.5;
const TRIAL_STEPS: usize = (TRIAL_DURATION_MS / DT_MS) as usize;  // 600
const FINGERPRINT_BIN_WIDTH_MS: f64 = 10.0;
const N_SAMPLE_RESPONSE: usize = 20;  // 各刺激への応答 sample 数
const SPONTANEOUS_WINDOW_STEPS: usize = 200;  // 100 ms window
const N_SPONTANEOUS_WINDOWS: usize = 50;
const PULSE_WIDTH_MS: f64 = 4.0;
const INPUT_CURRENT: i32 = 60;

// ──────────────────────────────────────────────────────────────
// 固定 A-E パターン (thermo_m1_evaluation と同じ)
// ──────────────────────────────────────────────────────────────

fn make_eval_pattern_set(n_input: usize) -> Vec<(char, Vec<f64>)> {
    let step = 5.0;
    let a: Vec<f64> = (0..n_input).map(|i| i as f64 * step).collect();
    let b: Vec<f64> = (0..n_input).map(|i| (n_input - 1 - i) as f64 * step).collect();
    let mut c = vec![0.0; n_input];
    for i in 0..n_input {
        c[i] = if i % 2 == 0 {
            (i as f64 / 2.0) * step
        } else {
            ((n_input / 2 + i / 2) as f64) * step
        };
    }
    let d: Vec<f64> = (0..n_input)
        .map(|i| (n_input / 2 + i) as f64 * step % (n_input as f64 * step))
        .collect();
    let e: Vec<f64> = vec![60.0, 10.0, 50.0, 80.0, 15.0, 90.0, 55.0, 0.0,
                            25.0, 70.0, 35.0, 5.0, 45.0, 20.0, 75.0, 30.0,
                            65.0, 40.0, 85.0, 95.0]
        .into_iter().take(n_input).collect();
    vec![('A', a), ('B', b), ('C', c), ('D', d), ('E', e)]
}

// ──────────────────────────────────────────────────────────────
// 1 trial: 刺激あり (固定パターン) / 無し (自発活動)
// ──────────────────────────────────────────────────────────────

fn present_pattern(net: &mut ThermoNetwork, pattern_times: &[f64]) -> Vec<(usize, f64)> {
    net.reset_trial_state();
    let trial_start_t_ms = net.current_time as f64 * DT_MS;
    let n_steps = TRIAL_STEPS;
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
                let t_abs = net.current_time as f64 * DT_MS;
                out_log.push((oi, t_abs - trial_start_t_ms));
            }
        }
    }
    out_log
}

/// 自発活動の N step 連続記録. 戻り値: (output_idx, time_step) の絶対時刻ログ.
/// 出力 layer のみ.
fn record_spontaneous(net: &mut ThermoNetwork, n_steps: usize, n_input: usize) -> Vec<(usize, i32)> {
    let external_input = vec![0i32; n_input];
    let mut log: Vec<(usize, i32)> = Vec::new();
    for _ in 0..n_steps {
        let fired = net.step(&external_input);
        let t_step = net.current_time;
        for nid in fired {
            if let Some(oi) = net.output_index_of(nid) {
                log.push((oi, t_step));
            }
        }
    }
    log
}

/// 内部全ニューロン (入力以外、興奮性 + 抑制性) の自発活動を記録.
/// 戻り値: (internal_idx, time_step). internal_idx は input_neurons を除いた連番.
fn record_spontaneous_internal(
    net: &mut ThermoNetwork,
    n_steps: usize,
    n_input: usize,
) -> (Vec<(usize, i32)>, Vec<usize>) {
    // input 以外のニューロン idx の集合 (input_neurons はソート済み前提)
    let input_set: std::collections::HashSet<usize> = net.input_neurons.iter().copied().collect();
    let internal_ids: Vec<usize> = (0..net.n_neurons()).filter(|i| !input_set.contains(i)).collect();
    let internal_idx_map: std::collections::HashMap<usize, usize> = internal_ids.iter()
        .enumerate().map(|(idx, &nid)| (nid, idx)).collect();

    let external_input = vec![0i32; n_input];
    let mut log: Vec<(usize, i32)> = Vec::new();
    for _ in 0..n_steps {
        let fired = net.step(&external_input);
        let t_step = net.current_time;
        for nid in fired {
            if let Some(&internal_idx) = internal_idx_map.get(&nid) {
                log.push((internal_idx, t_step));
            }
        }
    }
    (log, internal_ids)
}

/// 刺激パターンを提示し、内部全ニューロンの発火を絶対時刻で記録.
/// 戻り値: (internal_idx, time_step_abs). 比較のため trial 開始 step 情報も別途必要.
fn present_pattern_internal(
    net: &mut ThermoNetwork,
    pattern_times: &[f64],
    internal_idx_map: &std::collections::HashMap<usize, usize>,
) -> (Vec<(usize, i32)>, i32) {
    net.reset_trial_state();
    let trial_start = net.current_time;
    let pulse_steps = (PULSE_WIDTH_MS / DT_MS).max(1.0) as i64;
    let fire_steps: Vec<i64> = pattern_times.iter().map(|&t| (t / DT_MS) as i64).collect();
    let mut external_input = vec![0i32; pattern_times.len()];
    let mut log: Vec<(usize, i32)> = Vec::new();

    for step in 0..TRIAL_STEPS {
        for k in 0..pattern_times.len() {
            let fs = fire_steps[k];
            external_input[k] = if (step as i64) >= fs && (step as i64) < fs + pulse_steps {
                INPUT_CURRENT
            } else {
                0
            };
        }
        let fired = net.step(&external_input);
        let t_step = net.current_time;
        for nid in fired {
            if let Some(&internal_idx) = internal_idx_map.get(&nid) {
                log.push((internal_idx, t_step));
            }
        }
    }
    (log, trial_start)
}

// ──────────────────────────────────────────────────────────────
// fingerprint
// ──────────────────────────────────────────────────────────────

fn fingerprint_from_log_rel(log: &[(usize, f64)], n_out: usize) -> Vec<f64> {
    let mut tr = OutputTrace::new(n_out, 50.0);
    for &(oi, t) in log {
        tr.record_spike(oi, t);
    }
    tr.time_binned_fingerprint(TRIAL_DURATION_MS, FINGERPRINT_BIN_WIDTH_MS)
}

/// 自発活動ログから固定 window の fingerprint を抽出.
fn window_fingerprint(
    spike_log: &[(usize, i32)],
    window_start_step: i32,
    window_steps: usize,
    n_out: usize,
) -> Vec<f64> {
    let win_end = window_start_step + window_steps as i32;
    let window_duration_ms = window_steps as f64 * DT_MS;
    let mut tr = OutputTrace::new(n_out, 50.0);
    for &(oi, t_step) in spike_log {
        if t_step >= window_start_step && t_step < win_end {
            let t_rel = (t_step - window_start_step) as f64 * DT_MS;
            tr.record_spike(oi, t_rel);
        }
    }
    tr.time_binned_fingerprint(window_duration_ms, FINGERPRINT_BIN_WIDTH_MS)
}

// ──────────────────────────────────────────────────────────────
// main
// ──────────────────────────────────────────────────────────────

fn main() {
    let n_train: usize = std::env::args().nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(10000);

    println!("== 内部状態レパートリ検証 (Kenet 2003 Phase 2 版) ==");
    println!("  訓練: 固定 A-E パターン × {n_train} 試行 (仮想 M0 = M1 input spont=2)");
    println!("  検証: 訓練後の自発活動が刺激応答パターンと類似するか");

    let mut cfg = ThermoNetworkConfig::default();
    cfg.enable_up_down = true;  // PAPER §5.12.7-A: UP/DOWN 状態を有効化
    let mut net = ThermoNetwork::new(cfg);
    let n_input = net.input_neurons.len();
    println!("  UP/DOWN 状態: 有効 (up_offset=20, period 100-300ms 個体差)");
    let n_out = net.output_neurons.len();
    let patterns = make_eval_pattern_set(n_input);

    println!("\n  ネットワーク: neurons={}, synapses={}", net.n_neurons(), net.n_synapses());
    println!("  patterns: {} (n_input={}), n_output={}",
        patterns.len(), n_input, n_out);

    // ─── Phase A: 訓練 ───
    println!("\n== Phase A: 訓練 {n_train} 試行 ==");
    let mut rng = StdRng::seed_from_u64(42);
    let mut last_log_trial = 0;
    for trial in 1..=n_train {
        let pi = rng.gen_range(0..patterns.len());
        let pat = &patterns[pi].1;
        let _ = present_pattern(&mut net, pat);
        if trial % (n_train / 10).max(1) == 0 {
            println!("    trial {:>6}: open={}, grown={}, pruned={}",
                trial, net.n_open_synapses(), net.axons_grown, net.axons_pruned);
            last_log_trial = trial;
        }
    }
    let _ = last_log_trial;

    // ─── Phase B: 刺激応答 fingerprint 取得 ───
    println!("\n== Phase B: 訓練後の刺激応答 fingerprint 取得 ==");
    let mut response_fps: Vec<Vec<Vec<f64>>> =
        vec![Vec::with_capacity(N_SAMPLE_RESPONSE); patterns.len()];
    for _ in 0..N_SAMPLE_RESPONSE {
        for (pi, (_label, pat)) in patterns.iter().enumerate() {
            let log = present_pattern(&mut net, pat);
            response_fps[pi].push(fingerprint_from_log_rel(&log, n_out));
        }
    }
    let total_resp: usize = response_fps.iter().map(|v| v.len()).sum();
    println!("    取得した刺激応答 fp: {} 個 (5 パターン × {})", total_resp, N_SAMPLE_RESPONSE);

    // 各パターンの平均 fingerprint (代表ベクトル)
    let pattern_centroids: Vec<Vec<f64>> = response_fps.iter().map(|fps| {
        let mut centroid = vec![0.0f64; fps[0].len()];
        for fp in fps {
            for (i, &v) in fp.iter().enumerate() { centroid[i] += v; }
        }
        for v in centroid.iter_mut() { *v /= fps.len() as f64; }
        centroid
    }).collect();

    // ─── Phase C: 自発活動 fingerprint 取得 ───
    println!("\n== Phase C: 自発活動を {N_SPONTANEOUS_WINDOWS} 個の window で記録 ==");
    let spontaneous_steps = SPONTANEOUS_WINDOW_STEPS * N_SPONTANEOUS_WINDOWS;
    let start_step = net.current_time;
    let spike_log = record_spontaneous(&mut net, spontaneous_steps, n_input);
    println!("    自発スパイク総数: {} ({} step 間)", spike_log.len(), spontaneous_steps);

    // 各 window の fingerprint
    let mut spontaneous_fps: Vec<Vec<f64>> = Vec::with_capacity(N_SPONTANEOUS_WINDOWS);
    for w in 0..N_SPONTANEOUS_WINDOWS {
        let win_start = start_step + (w * SPONTANEOUS_WINDOW_STEPS) as i32;
        let fp = window_fingerprint(&spike_log, win_start, SPONTANEOUS_WINDOW_STEPS, n_out);
        spontaneous_fps.push(fp);
    }

    // 非空 (発火がある) window のみ採用
    let active_spontaneous: Vec<&Vec<f64>> = spontaneous_fps.iter()
        .filter(|fp| fp.iter().any(|&v| v > 0.0))
        .collect();
    println!("    非空 (発火あり) window: {}/{}", active_spontaneous.len(), N_SPONTANEOUS_WINDOWS);

    if active_spontaneous.is_empty() {
        println!("\n  ⚠️  自発活動が観測されませんでした。M1 input spontaneous_input が必要かも。");
        return;
    }

    // ─── Phase D: 解析 ───
    println!("\n== Phase D: 自発 vs 刺激応答 類似度解析 ==");

    // 自発 fp と各パターン centroid の cos sim
    // 各自発 fp で「最も近いパターン」と「平均 cos sim」を取る
    let mut max_sims: Vec<f64> = Vec::new();
    let mut all_sims: Vec<f64> = Vec::new();
    let mut best_pattern_counts = vec![0usize; patterns.len()];

    for fp in &active_spontaneous {
        let sims: Vec<f64> = pattern_centroids.iter()
            .map(|c| cosine_similarity(fp, c))
            .collect();
        let (best_idx, &best_sim) = sims.iter().enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap();
        max_sims.push(best_sim);
        best_pattern_counts[best_idx] += 1;
        for s in sims { all_sims.push(s); }
    }

    let max_mean = max_sims.iter().sum::<f64>() / max_sims.len() as f64;
    let all_mean = all_sims.iter().sum::<f64>() / all_sims.len() as f64;

    println!("\n  自発活動 fp の解析:");
    println!("    max cos sim (各 fp の最良パターン) 平均: {:.4}", max_mean);
    println!("    全ペア cos sim (自発 × 5パターン) 平均: {:.4}", all_mean);
    print!("    best パターン分布: ");
    for (pi, (l, _)) in patterns.iter().enumerate() {
        print!("{}:{} ", l, best_pattern_counts[pi]);
    }
    println!();

    // ─── 帰無分布: シャッフルした fp との比較 ───
    println!("\n  帰無対照: シャッフル fingerprint との比較");
    let mut shuffle_max_sims: Vec<f64> = Vec::new();
    let mut shuffle_rng = StdRng::seed_from_u64(123);
    let fp_dim = pattern_centroids[0].len();

    for _ in 0..active_spontaneous.len() {
        // ランダム fingerprint を生成 (自発の総スパイク量分布に近づける)
        let mean_total: f64 = active_spontaneous.iter()
            .map(|fp| fp.iter().sum::<f64>())
            .sum::<f64>() / active_spontaneous.len() as f64;
        let n_events = mean_total as usize;
        let mut shuffled = vec![0.0f64; fp_dim];
        for _ in 0..n_events {
            let idx = shuffle_rng.gen_range(0..fp_dim);
            shuffled[idx] += 1.0;
        }
        let sims: Vec<f64> = pattern_centroids.iter()
            .map(|c| cosine_similarity(&shuffled, c))
            .collect();
        let best_sim = sims.iter().cloned()
            .fold(f64::NEG_INFINITY, f64::max);
        shuffle_max_sims.push(best_sim);
    }
    let shuffle_max_mean = shuffle_max_sims.iter().sum::<f64>() / shuffle_max_sims.len() as f64;
    println!("    シャッフル max cos sim 平均: {:.4}", shuffle_max_mean);

    // Welch t-test (簡易版): self-coded
    fn welch_t(a: &[f64], b: &[f64]) -> (f64, f64) {
        let mean_a = a.iter().sum::<f64>() / a.len() as f64;
        let mean_b = b.iter().sum::<f64>() / b.len() as f64;
        let var_a = a.iter().map(|&x| (x - mean_a).powi(2)).sum::<f64>() / (a.len() - 1) as f64;
        let var_b = b.iter().map(|&x| (x - mean_b).powi(2)).sum::<f64>() / (b.len() - 1) as f64;
        let se = ((var_a / a.len() as f64) + (var_b / b.len() as f64)).sqrt();
        let t = (mean_a - mean_b) / se;
        // 自由度は両群サイズの合計 - 2 として簡易化 (本来 Welch-Satterthwaite)
        let df = (a.len() + b.len() - 2) as f64;
        // ナイーブな p 近似: |t| > 2 で p < 0.05、|t| > 3 で p < 0.001 程度
        // ここでは t のみ報告
        let _ = df;
        (t, 0.0) // p 値は近似困難なので t のみ
    }
    let (t_stat, _) = welch_t(&max_sims, &shuffle_max_sims);

    println!("\n  ─── 検定結果 ───");
    println!("    Welch t-statistic (max sim, 自発 vs シャッフル): t = {:.3}", t_stat);
    println!("    判定基準: |t| > 2 で p < 0.05, > 3 で p < 0.001");
    if t_stat > 3.0 {
        println!("    → 強く有意 (p < 0.001): 自発活動が訓練刺激パターンと有意に類似");
        println!("    → Kenet 2003「内部状態レパートリ」現象を Phase 2 で再現 ✓");
    } else if t_stat > 2.0 {
        println!("    → 有意 (p < 0.05): 自発活動と刺激パターンに関連性あり");
    } else if t_stat > 1.0 {
        println!("    → 弱い傾向: シャッフルよりやや高いが有意ではない");
    } else {
        println!("    → 有意差なし: 自発活動は刺激パターンと独立");
    }

    // ─── CSV 出力 (出力 layer) ───
    let mut csv = File::create("phase2_f_internal_state_probe.csv").expect("csv");
    writeln!(csv, "type,window_or_sample_idx,best_pattern_idx,best_cos_sim").unwrap();
    for (i, (&max_sim, fp)) in max_sims.iter().zip(active_spontaneous.iter()).enumerate() {
        let best_idx = pattern_centroids.iter().enumerate()
            .map(|(pi, c)| (pi, cosine_similarity(fp, c)))
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .unwrap().0;
        writeln!(csv, "spontaneous,{},{},{:.4}", i, best_idx, max_sim).unwrap();
    }
    for (i, &max_sim) in shuffle_max_sims.iter().enumerate() {
        writeln!(csv, "shuffle,{},-1,{:.4}", i, max_sim).unwrap();
    }
    println!("\n  CSV: phase2_f_internal_state_probe.csv");

    // ─────────────────────────────────────────────────
    // Phase B'-D': 内部 380 ニューロン全体での再評価 (仮説 1 検証)
    // ─────────────────────────────────────────────────
    println!("\n══════════════════════════════════════════════════════════");
    println!("== 仮説 1 検証: 内部 380 ニューロン全体での評価 ==");
    println!("══════════════════════════════════════════════════════════");

    // input 以外の全 neuron について internal_idx を割り当て
    let input_set: std::collections::HashSet<usize> = net.input_neurons.iter().copied().collect();
    let internal_ids: Vec<usize> = (0..net.n_neurons()).filter(|i| !input_set.contains(i)).collect();
    let internal_idx_map: std::collections::HashMap<usize, usize> = internal_ids.iter()
        .enumerate().map(|(idx, &nid)| (nid, idx)).collect();
    let n_internal = internal_ids.len();
    println!("  内部ニューロン数: {}", n_internal);

    // Phase B': 刺激応答 fingerprint (内部 380 × 30 bin = 11400 次元)
    println!("\n== Phase B': 内部全体の刺激応答 fingerprint 取得 ==");
    let mut internal_response_fps: Vec<Vec<Vec<f64>>> =
        vec![Vec::with_capacity(N_SAMPLE_RESPONSE); patterns.len()];
    for _ in 0..N_SAMPLE_RESPONSE {
        for (pi, (_label, pat)) in patterns.iter().enumerate() {
            let (log, trial_start) = present_pattern_internal(&mut net, pat, &internal_idx_map);
            // log は (internal_idx, t_step_abs)、trial 開始からの相対時刻に変換して time_binned fp に
            let mut tr = OutputTrace::new(n_internal, 50.0);
            for &(internal_idx, t_step) in &log {
                let t_rel = (t_step - trial_start) as f64 * DT_MS;
                tr.record_spike(internal_idx, t_rel);
            }
            let fp = tr.time_binned_fingerprint(TRIAL_DURATION_MS, FINGERPRINT_BIN_WIDTH_MS);
            internal_response_fps[pi].push(fp);
        }
    }
    println!("    取得した内部刺激応答 fp: 5 × {} (次元 {})",
        N_SAMPLE_RESPONSE, internal_response_fps[0][0].len());

    // centroid
    let internal_centroids: Vec<Vec<f64>> = internal_response_fps.iter().map(|fps| {
        let mut centroid = vec![0.0f64; fps[0].len()];
        for fp in fps {
            for (i, &v) in fp.iter().enumerate() { centroid[i] += v; }
        }
        for v in centroid.iter_mut() { *v /= fps.len() as f64; }
        centroid
    }).collect();

    // Phase C': 内部自発活動を 50 window × 200 step で記録
    println!("\n== Phase C': 内部 380 ニューロン自発活動を記録 ==");
    let spontaneous_steps2 = SPONTANEOUS_WINDOW_STEPS * N_SPONTANEOUS_WINDOWS;
    let start_step2 = net.current_time;
    let (internal_spike_log, _) = record_spontaneous_internal(&mut net, spontaneous_steps2, n_input);
    println!("    自発スパイク総数 (内部): {} ({} step 間)",
        internal_spike_log.len(), spontaneous_steps2);

    // window fingerprint
    let mut internal_spont_fps: Vec<Vec<f64>> = Vec::with_capacity(N_SPONTANEOUS_WINDOWS);
    for w in 0..N_SPONTANEOUS_WINDOWS {
        let win_start = start_step2 + (w * SPONTANEOUS_WINDOW_STEPS) as i32;
        let fp = window_fingerprint_internal(&internal_spike_log, win_start, SPONTANEOUS_WINDOW_STEPS, n_internal);
        internal_spont_fps.push(fp);
    }
    let active_internal: Vec<&Vec<f64>> = internal_spont_fps.iter()
        .filter(|fp| fp.iter().any(|&v| v > 0.0))
        .collect();
    println!("    非空 (発火あり) window: {}/{}", active_internal.len(), N_SPONTANEOUS_WINDOWS);

    if active_internal.is_empty() {
        println!("\n  ⚠️  内部自発活動も観測されず");
        return;
    }

    // Phase D': 解析
    println!("\n== Phase D': 内部自発 vs 内部刺激応答 ==");
    let mut int_max_sims: Vec<f64> = Vec::new();
    let mut int_all_sims: Vec<f64> = Vec::new();
    let mut int_best_pattern_counts = vec![0usize; patterns.len()];

    for fp in &active_internal {
        let sims: Vec<f64> = internal_centroids.iter()
            .map(|c| cosine_similarity(fp, c))
            .collect();
        let (best_idx, &best_sim) = sims.iter().enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap();
        int_max_sims.push(best_sim);
        int_best_pattern_counts[best_idx] += 1;
        for s in sims { int_all_sims.push(s); }
    }
    let int_max_mean = int_max_sims.iter().sum::<f64>() / int_max_sims.len() as f64;
    let int_all_mean = int_all_sims.iter().sum::<f64>() / int_all_sims.len() as f64;

    println!("\n  内部自発活動 fp の解析:");
    println!("    max cos sim 平均: {:.4}", int_max_mean);
    println!("    全ペア cos sim 平均: {:.4}", int_all_mean);
    print!("    best パターン分布: ");
    for (pi, (l, _)) in patterns.iter().enumerate() {
        print!("{}:{} ", l, int_best_pattern_counts[pi]);
    }
    println!();

    // 帰無対照: シャッフル fp
    let mut int_shuffle_max_sims: Vec<f64> = Vec::new();
    let mut shuffle_rng2 = StdRng::seed_from_u64(456);
    let fp_dim2 = internal_centroids[0].len();
    let mean_total: f64 = active_internal.iter()
        .map(|fp| fp.iter().sum::<f64>())
        .sum::<f64>() / active_internal.len() as f64;
    for _ in 0..active_internal.len() {
        let n_events = mean_total as usize;
        let mut shuffled = vec![0.0f64; fp_dim2];
        for _ in 0..n_events {
            let idx = shuffle_rng2.gen_range(0..fp_dim2);
            shuffled[idx] += 1.0;
        }
        let best_sim = internal_centroids.iter()
            .map(|c| cosine_similarity(&shuffled, c))
            .fold(f64::NEG_INFINITY, f64::max);
        int_shuffle_max_sims.push(best_sim);
    }
    let int_shuffle_mean = int_shuffle_max_sims.iter().sum::<f64>() / int_shuffle_max_sims.len() as f64;

    println!("    シャッフル max cos sim 平均: {:.4}", int_shuffle_mean);

    fn welch_t2(a: &[f64], b: &[f64]) -> f64 {
        let mean_a = a.iter().sum::<f64>() / a.len() as f64;
        let mean_b = b.iter().sum::<f64>() / b.len() as f64;
        let var_a = a.iter().map(|&x| (x - mean_a).powi(2)).sum::<f64>() / (a.len() - 1) as f64;
        let var_b = b.iter().map(|&x| (x - mean_b).powi(2)).sum::<f64>() / (b.len() - 1) as f64;
        let se = ((var_a / a.len() as f64) + (var_b / b.len() as f64)).sqrt();
        (mean_a - mean_b) / se
    }
    let t_internal = welch_t2(&int_max_sims, &int_shuffle_max_sims);

    println!("\n  ─── 内部評価の検定結果 ───");
    println!("    Welch t-statistic: t = {:.3}", t_internal);
    if t_internal > 3.0 {
        println!("    → 強く有意 (p<0.001): 内部レパートリ存在を強く支持 ✓");
        println!("    → 出力 layer だけでは見えなかった「内部状態レパートリ」を検出");
    } else if t_internal > 2.0 {
        println!("    → 有意 (p<0.05): 内部レパートリの傾向あり");
    } else if t_internal > 1.0 {
        println!("    → 弱い傾向: 出力 layer よりは強いが、有意ではない");
    } else {
        println!("    → 有意差なし: 内部 layer でも内部レパートリ未検出");
        println!("    → 仮説 1 (出力 layer 限定が原因) は棄却、別の改善が必要");
    }

    // 比較サマリ
    println!("\n══════════════════════════════════════════════════════════");
    println!("  最終サマリ: 出力 layer vs 内部 380 ニューロン");
    println!("══════════════════════════════════════════════════════════");
    println!("  出力 layer  : t = {:>6.3}, max_sim_mean = {:.4} (shuffle {:.4})",
        t_stat, max_mean, shuffle_max_mean);
    println!("  内部 380   : t = {:>6.3}, max_sim_mean = {:.4} (shuffle {:.4})",
        t_internal, int_max_mean, int_shuffle_mean);
}

/// 内部ニューロン版の window fingerprint.
fn window_fingerprint_internal(
    spike_log: &[(usize, i32)],
    window_start_step: i32,
    window_steps: usize,
    n_internal: usize,
) -> Vec<f64> {
    let win_end = window_start_step + window_steps as i32;
    let window_duration_ms = window_steps as f64 * DT_MS;
    let mut tr = OutputTrace::new(n_internal, 50.0);
    for &(internal_idx, t_step) in spike_log {
        if t_step >= window_start_step && t_step < win_end {
            let t_rel = (t_step - window_start_step) as f64 * DT_MS;
            tr.record_spike(internal_idx, t_rel);
        }
    }
    tr.time_binned_fingerprint(window_duration_ms, FINGERPRINT_BIN_WIDTH_MS)
}
