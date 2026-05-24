//! 1ビット版実験ランナー: 同じ報酬学習タスクを binary ネットで実行。
//!
//! 連続版 (continuous_brain) と直接比較できる結果を出すのが目的。
//! 加えて DRP の「再構成イベント数」を計測する (DRP コンフィグレーション
//! 書き換え頻度の見積もり)。

use spiking_brain::binary_network::{
    fingerprint_from_log, make_pulse_pattern, present_pulse_pattern,
    BinaryNetwork, BinaryNetworkConfig, BinaryStdpParams,
};
use spiking_brain::trace::cosine_similarity;
use rand::prelude::*;

fn within(fps: &[Vec<f64>]) -> f64 {
    let mut sims = Vec::new();
    for i in 0..fps.len() {
        for j in (i + 1)..fps.len() {
            sims.push(cosine_similarity(&fps[i], &fps[j]));
        }
    }
    if sims.is_empty() { 0.0 } else { sims.iter().sum::<f64>() / sims.len() as f64 }
}
fn between(a: &[Vec<f64>], b: &[Vec<f64>]) -> f64 {
    let mut sims = Vec::new();
    for x in a { for y in b { sims.push(cosine_similarity(x, y)); } }
    if sims.is_empty() { 0.0 } else { sims.iter().sum::<f64>() / sims.len() as f64 }
}

fn evaluate_reward(log: &[(usize, f64)], target: &[usize], window_ms: (f64, f64)) -> f32 {
    let hits = log.iter().filter(|&&(oi, t)| {
        target.contains(&oi) && t >= window_ms.0 && t <= window_ms.1
    }).count();
    if hits >= 1 { 1.0 } else { -0.5 }
}

fn main() {
    let cfg = BinaryNetworkConfig::default();
    let stdp = BinaryStdpParams::default();
    let mut net = BinaryNetwork::new(cfg, stdp);

    println!("== 1-bit Network ==");
    println!("  neurons     : {}", net.n_neurons());
    println!("  synapses    : {} total, {} plastic, {} open at start",
        net.n_synapses(), net.n_plastic_synapses(), net.n_open_synapses());
    let (dmin, dmax) = net.delay_range_ms();
    println!("  delay range : {:.1} - {:.1} ms", dmin, dmax);
    println!("  memory      : {} KB", net.memory_bytes() / 1024);

    let n_input = net.input_neurons.len();
    let n_out = net.output_neurons.len();
    let pattern_a = make_pulse_pattern(n_input, 101);
    let pattern_b = make_pulse_pattern(n_input, 202);
    let target_a: Vec<usize> = (0..5).collect();
    let target_b: Vec<usize> = (20..25).collect();

    // ---------- Phase 1: pre-training ----------
    println!("\n== Phase 1: pre-training ==");
    let mut fps_a_pre = Vec::new();
    let mut fps_b_pre = Vec::new();
    let mut hit_a_pre = 0; let mut hit_b_pre = 0;
    for trial in 0..10 {
        let log_a = present_pulse_pattern(&mut net, &pattern_a, 300.0, 4, 4.0, 1000 + trial);
        let log_b = present_pulse_pattern(&mut net, &pattern_b, 300.0, 4, 4.0, 2000 + trial);
        if log_a.iter().any(|&(oi, _)| target_a.contains(&oi)) { hit_a_pre += 1; }
        if log_b.iter().any(|&(oi, _)| target_b.contains(&oi)) { hit_b_pre += 1; }
        fps_a_pre.push(fingerprint_from_log(&log_a, n_out, 300.0, 50.0));
        fps_b_pre.push(fingerprint_from_log(&log_b, n_out, 300.0, 50.0));
    }
    println!("  target hits: A={}/10  B={}/10", hit_a_pre, hit_b_pre);
    let initial_open = net.n_open_synapses();

    // ---------- Phase 2: training with reward ----------
    println!("\n== Phase 2: reward-modulated structural plasticity ==");
    let mut rng = StdRng::seed_from_u64(42);
    let n_train = 200;
    let mut recent_reward: Vec<f32> = Vec::new();
    for trial in 0..n_train {
        let use_a = rng.gen::<f64>() < 0.5;
        let (pat, tgt) = if use_a { (&pattern_a, &target_a) } else { (&pattern_b, &target_b) };
        let log = present_pulse_pattern(&mut net, pat, 300.0, 4, 4.0, 10_000 + trial);
        let reward = evaluate_reward(&log, tgt, (10.0, 250.0));
        net.apply_reward(reward);
        recent_reward.push(reward);
        if recent_reward.len() > 20 { recent_reward.remove(0); }
        if (trial + 1) % 20 == 0 {
            let avg = recent_reward.iter().sum::<f32>() / recent_reward.len() as f32;
            println!("  trial {:>3}/{}  recent_avg_reward={:+.3}  open_synapses={}  total_reconfigs={}",
                trial+1, n_train, avg, net.n_open_synapses(), net.reconfig_count);
        }
    }

    // ---------- Phase 3: post-training ----------
    println!("\n== Phase 3: post-training ==");
    let mut fps_a_post = Vec::new();
    let mut fps_b_post = Vec::new();
    let mut hit_a_post = 0; let mut hit_b_post = 0;
    for trial in 0..10 {
        let log_a = present_pulse_pattern(&mut net, &pattern_a, 300.0, 4, 4.0, 30_000 + trial);
        let log_b = present_pulse_pattern(&mut net, &pattern_b, 300.0, 4, 4.0, 40_000 + trial);
        if log_a.iter().any(|&(oi, _)| target_a.contains(&oi)) { hit_a_post += 1; }
        if log_b.iter().any(|&(oi, _)| target_b.contains(&oi)) { hit_b_post += 1; }
        fps_a_post.push(fingerprint_from_log(&log_a, n_out, 300.0, 50.0));
        fps_b_post.push(fingerprint_from_log(&log_b, n_out, 300.0, 50.0));
    }

    println!("\n== Results ==");
    println!("  target hit rate (1-bit version):");
    println!("    A: {}/10 → {}/10", hit_a_pre, hit_a_post);
    println!("    B: {}/10 → {}/10", hit_b_pre, hit_b_post);
    let pre_sel = (within(&fps_a_pre) + within(&fps_b_pre)) / 2.0 - between(&fps_a_pre, &fps_b_pre);
    let post_sel = (within(&fps_a_post) + within(&fps_b_post)) / 2.0 - between(&fps_a_post, &fps_b_post);
    println!("  selectivity index (within - between):");
    println!("    pre : {:.3}", pre_sel);
    println!("    post: {:.3}", post_sel);
    println!("  structural change:");
    println!("    open synapses: {} → {}", initial_open, net.n_open_synapses());
    println!("    total reconfig events: {}", net.reconfig_count);
}
