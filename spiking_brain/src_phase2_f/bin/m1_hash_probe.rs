//! M1 に SHA-256 ハッシュを入力する実験
//!
//! SHA-256 出力は avalanche 効果でほぼ一様乱数 (構造なし)。 これを M1 に与え、
//! 「構造のない高エントロピー入力」 に M1 がどう応答するかを観察する。
//!
//! M1: 標準 ThermoNetworkConfig::default() (20 入力, 40 出力, 未訓練)。
//! M0/蝸牛をバイパスし、 ハッシュを M1 入力層へ直接注入。
//!
//! 入力マッピング: hash 32 バイトの先頭 20 バイトを 20 入力に。
//!   input[i] = byte[i] / 8 (0-31 の graded 持続電流、 byte 値が大きいほど強く駆動)
//!
//! 3 手法 (CLI 第1引数):
//!   chain    : h_{n+1} = SHA256(h_n)、 毎 trial 新しいハッシュ (直近ハッシュを繰り返し)
//!   same     : 同一ハッシュを毎 trial (within=1.0 の確認、 "やっぱりね")
//!   feedback : M1 出力 → SHA256 → 次入力 (ネットワークとハッシュの再帰ループ)
//!
//! CLI: m1_hash_probe [method] [n_trials]

use spiking_brain::phase2_f::thermo_network::{ThermoNetwork, ThermoNetworkConfig};
use spiking_brain::phase2_f::sha256::sha256;
use spiking_brain::trace::{cosine_similarity, OutputTrace};

const TRIAL_DURATION_MS: f64 = 300.0;
const DT_MS: f64 = 0.5;
const TRIAL_STEPS: usize = (TRIAL_DURATION_MS / DT_MS) as usize;
const FP_BIN: f64 = 10.0;

/// hash 32 バイト → M1 入力電流ベクトル (先頭 n_input バイト、 byte/8 の graded)
fn hash_to_input(h: &[u8; 32], n_input: usize) -> Vec<i32> {
    (0..n_input).map(|i| (h[i] as i32) / 8).collect()
}

/// 1 trial 実行 (入力を持続注入)。 戻り値: (出力ログ, 出力ニューロン別スパイク数)
fn run_trial(net: &mut ThermoNetwork, input: &[i32]) -> (Vec<(usize, f64)>, Vec<u32>) {
    net.reset_trial_state();
    let t0 = net.current_time;
    let n_out = net.output_neurons.len();
    let mut log = Vec::new();
    let mut counts = vec![0u32; n_out];
    for _step in 0..TRIAL_STEPS {
        let fired = net.step(input);
        for nid in fired {
            if let Some(oi) = net.output_index_of(nid) {
                let tr = (net.current_time - t0) as f64 * DT_MS;
                log.push((oi, tr));
                counts[oi] += 1;
            }
        }
    }
    (log, counts)
}

fn fingerprint(log: &[(usize, f64)], n_out: usize) -> Vec<f64> {
    let mut tr = OutputTrace::new(n_out, 50.0);
    for &(oi, t) in log { tr.record_spike(oi, t); }
    tr.time_binned_fingerprint(TRIAL_DURATION_MS, FP_BIN)
}

/// 出力ニューロン別スパイク数 → バイト列 (feedback でハッシュ入力に再エンコード)
fn output_to_bytes(counts: &[u32]) -> Vec<u8> {
    counts.iter().map(|&c| (c & 0xff) as u8).collect()
}

fn mean_pairwise(fps: &[Vec<f64>]) -> f64 {
    let mut s = 0.0; let mut n = 0;
    for i in 0..fps.len() {
        for j in (i + 1)..fps.len() {
            s += cosine_similarity(&fps[i], &fps[j]); n += 1;
        }
    }
    if n == 0 { 0.0 } else { s / n as f64 }
}

fn main() {
    let method = std::env::args().nth(1).unwrap_or_else(|| "chain".to_string());
    let n_trials: usize = std::env::args().nth(2).and_then(|s| s.parse().ok()).unwrap_or(50);

    println!("== M1 SHA-256 ハッシュ入力実験 ==");
    println!("  M1: 標準 (20 入力, 40 出力, 未訓練)");
    println!("  入力: hash 先頭 20 バイト → input[i]=byte[i]/8 (持続)");
    println!("  手法: {}  試行数: {}", method, n_trials);

    let cfg = ThermoNetworkConfig::default();
    let mut net = ThermoNetwork::new(cfg);
    let n_out = net.output_neurons.len();

    let mut fps: Vec<Vec<f64>> = Vec::new();
    let mut total_spikes: u64 = 0;
    let mut active_union = vec![false; n_out];

    match method.as_str() {
        "chain" => {
            // h_{n+1} = SHA256(h_n)
            let mut h = sha256(b"drpnn-seed");
            println!("\n  trial | active | spikes | (出力ニューロン別 先頭8)");
            for trial in 0..n_trials {
                let input = hash_to_input(&h, net.input_neurons.len());
                let (log, counts) = run_trial(&mut net, &input);
                total_spikes += log.len() as u64;
                for (oi, c) in counts.iter().enumerate() { if *c > 0 { active_union[oi] = true; } }
                fps.push(fingerprint(&log, n_out));
                if trial < 10 || trial % 10 == 0 {
                    let active = counts.iter().filter(|&&c| c > 0).count();
                    let head: Vec<String> = counts.iter().take(8).map(|c| c.to_string()).collect();
                    println!("  {:>5} |  {:>3}   |  {:>4}  | [{}]",
                        trial, active, log.len(), head.join(","));
                }
                h = sha256(&h);  // 次のハッシュ
            }
            let between = mean_pairwise(&fps);
            println!("\n  ── chain サマリ ──");
            println!("  異なるハッシュ {} 種に対する出力 fingerprint の平均 between = {:.3}", n_trials, between);
            println!("  active union: {}/{}、 総スパイク {}", active_union.iter().filter(|&&v| v).count(), n_out, total_spikes);
            println!("  → between 低 = M1 は異なるハッシュに異なる応答 (入力感応)");
            println!("    between 高 = ハッシュに依らず同じ応答 (入力非依存アトラクタ)");
        }
        "same" => {
            // 同一ハッシュを繰り返し
            let h = sha256(b"drpnn-seed");
            let input = hash_to_input(&h, net.input_neurons.len());
            for trial in 0..n_trials {
                let (log, counts) = run_trial(&mut net, &input);
                total_spikes += log.len() as u64;
                for (oi, c) in counts.iter().enumerate() { if *c > 0 { active_union[oi] = true; } }
                fps.push(fingerprint(&log, n_out));
                if trial < 5 {
                    let active = counts.iter().filter(|&&c| c > 0).count();
                    println!("  trial {:>3}: active={} spikes={}", trial, active, log.len());
                }
            }
            let within = mean_pairwise(&fps);
            println!("\n  ── same サマリ ──");
            println!("  同一ハッシュ {} 回提示の within (fingerprint 一致度) = {:.4}", n_trials, within);
            println!("  → within≈1.0 なら決定論的応答の確認 (やっぱりね)");
            println!("    (試行間に entropy/conductance が持ち越されるので完全 1.0 とは限らない)");
        }
        "feedback" => {
            // M1 出力 → SHA256 → 次入力 の再帰ループ。 サイクル検出
            let mut h = sha256(b"drpnn-seed");
            let mut seen: Vec<[u8; 32]> = Vec::new();
            let mut cycle_found: Option<(usize, usize)> = None;
            println!("\n  trial | active | spikes | hash 先頭4バイト");
            for trial in 0..n_trials {
                // サイクル検出: 同じハッシュが既出か
                if let Some(prev) = seen.iter().position(|x| *x == h) {
                    cycle_found = Some((prev, trial));
                    println!("  → サイクル検出! trial {} の状態が trial {} で再出現 (周期 {})",
                        prev, trial, trial - prev);
                    break;
                }
                seen.push(h);
                let input = hash_to_input(&h, net.input_neurons.len());
                let (log, counts) = run_trial(&mut net, &input);
                total_spikes += log.len() as u64;
                for (oi, c) in counts.iter().enumerate() { if *c > 0 { active_union[oi] = true; } }
                let active = counts.iter().filter(|&&c| c > 0).count();
                if trial < 15 {
                    println!("  {:>5} |  {:>3}   |  {:>4}  | {:02x}{:02x}{:02x}{:02x}",
                        trial, active, log.len(), h[0], h[1], h[2], h[3]);
                }
                // 出力をハッシュして次の入力に
                let out_bytes = output_to_bytes(&counts);
                h = sha256(&out_bytes);
            }
            println!("\n  ── feedback サマリ ──");
            match cycle_found {
                Some((start, end)) => println!("  再帰ループは周期 {} のサイクルに収束 (trial {}→{})", end - start, start, end),
                None => println!("  {} 試行ではサイクル未検出 (周期 > {} or 非周期的)", n_trials, n_trials),
            }
            println!("  active union: {}/{}", active_union.iter().filter(|&&v| v).count(), n_out);
            println!("  → 決定論的なので必ず有限周期のサイクルに入る (状態空間有限)");
        }
        _ => {
            println!("  未知の手法: {} (chain/same/feedback のいずれか)", method);
        }
    }
}
