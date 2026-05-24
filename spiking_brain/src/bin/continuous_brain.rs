//! 連続重み版の実験ランナー。
//!
//! Phase 1: 訓練前のフィンガープリント
//! Phase 2: 自己組織化のみ (即時 STDP, 報酬なし) — 旧来の挙動
//! Phase 3: 報酬変調 R-STDP — 「特定の出力ニューロンが Aで光る」を強化
//! Phase 4: 訓練後フィンガープリント + 報酬学習の効果測定

use spiking_brain::network::{
    fingerprint_from_log, make_pattern, present_pattern, Network, NetworkConfig, StdpParams,
};
use spiking_brain::trace::cosine_similarity;
use rand::prelude::*;
use std::fs::File;
use std::io::{BufWriter, Write};

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

fn write_fps_csv(path: &str, fps: &[Vec<f64>]) -> std::io::Result<()> {
    let f = File::create(path)?;
    let mut w = BufWriter::new(f);
    for fp in fps {
        let line: Vec<String> = fp.iter().map(|x| format!("{:.6}", x)).collect();
        writeln!(w, "{}", line.join(","))?;
    }
    Ok(())
}

/// 試行ログから「目標ニューロン群が光ったか」「いつ光ったか」を判定して報酬を返す。
fn evaluate_reward(log: &[(usize, f64)], target_outputs: &[usize], window_ms: (f64, f64)) -> f64 {
    let mut hits = 0;
    for &(oi, t) in log {
        if target_outputs.contains(&oi) && t >= window_ms.0 && t <= window_ms.1 {
            hits += 1;
        }
    }
    if hits >= 1 { 1.0 } else { -0.5 }
}

fn main() {
    let config = NetworkConfig::default();
    let stdp = StdpParams {
        // pure R-STDP に近づける: 即時STDPは効かせる、ただし weight 動かしすぎないように弱める
        immediate_scale: 0.3,
        ..StdpParams::default()
    };
    let mut net = Network::new(config, stdp);

    println!("== Network ==");
    println!("  neurons : {}", net.n_neurons());
    println!("  synapses: {} (plastic: {})", net.n_synapses(), net.n_plastic_synapses());
    let (dmin, dmax) = net.delay_range_ms();
    println!("  delay   : {:.1} - {:.1} ms", dmin, dmax);

    let n_input = net.input_neurons.len();
    let n_out = net.output_neurons.len();
    let pattern_a = make_pattern(n_input, 101);
    let pattern_b = make_pattern(n_input, 202);

    // 「正解」出力ニューロンを指定:
    //   パターンA → 出力 0..5 のいずれかが光る
    //   パターンB → 出力 20..25 のいずれかが光る
    let target_a: Vec<usize> = (0..5).collect();
    let target_b: Vec<usize> = (20..25).collect();

    // ---------- Phase 1: 訓練前 ----------
    println!("\n== Phase 1: pre-training fingerprints ==");
    let mut fps_a_pre = Vec::new();
    let mut fps_b_pre = Vec::new();
    let mut hit_a_pre = 0;
    let mut hit_b_pre = 0;
    for trial in 0..10 {
        let log_a = present_pattern(&mut net, &pattern_a, 300.0, 40.0, 4.0, 0.5, 1000 + trial);
        let log_b = present_pattern(&mut net, &pattern_b, 300.0, 40.0, 4.0, 0.5, 2000 + trial);
        if log_a.iter().any(|&(oi, _)| target_a.contains(&oi)) { hit_a_pre += 1; }
        if log_b.iter().any(|&(oi, _)| target_b.contains(&oi)) { hit_b_pre += 1; }
        fps_a_pre.push(fingerprint_from_log(&log_a, n_out, 300.0, 50.0));
        fps_b_pre.push(fingerprint_from_log(&log_b, n_out, 300.0, 50.0));
    }
    println!("  target hits: A={}/10  B={}/10", hit_a_pre, hit_b_pre);

    // ---------- Phase 2: 報酬変調学習 ----------
    println!("\n== Phase 2: reward-modulated training ==");
    let mut rng = StdRng::seed_from_u64(42);
    let n_train = 200;
    let mut recent_reward: Vec<f64> = Vec::new();
    for trial in 0..n_train {
        let use_a = rng.gen::<f64>() < 0.5;
        let (pattern, target) = if use_a {
            (&pattern_a, &target_a)
        } else {
            (&pattern_b, &target_b)
        };
        let log = present_pattern(&mut net, pattern, 300.0, 40.0, 4.0, 0.5, 10_000 + trial);
        let reward = evaluate_reward(&log, target, (10.0, 250.0));
        net.apply_reward(reward);
        recent_reward.push(reward);
        if recent_reward.len() > 20 { recent_reward.remove(0); }
        if (trial + 1) % 20 == 0 {
            let avg = recent_reward.iter().sum::<f64>() / recent_reward.len() as f64;
            println!("  trial {:>3}/{}  recent_avg_reward={:+.3}  mean_w={:.3}  total_elig={:.3}",
                trial + 1, n_train, avg, net.mean_plastic_weight(),
                net.total_eligibility_magnitude());
        }
    }

    // ---------- Phase 3: 訓練後 ----------
    println!("\n== Phase 3: post-training fingerprints ==");
    let mut fps_a_post = Vec::new();
    let mut fps_b_post = Vec::new();
    let mut hit_a_post = 0;
    let mut hit_b_post = 0;
    for trial in 0..10 {
        let log_a = present_pattern(&mut net, &pattern_a, 300.0, 40.0, 4.0, 0.5, 30_000 + trial);
        let log_b = present_pattern(&mut net, &pattern_b, 300.0, 40.0, 4.0, 0.5, 40_000 + trial);
        if log_a.iter().any(|&(oi, _)| target_a.contains(&oi)) { hit_a_post += 1; }
        if log_b.iter().any(|&(oi, _)| target_b.contains(&oi)) { hit_b_post += 1; }
        fps_a_post.push(fingerprint_from_log(&log_a, n_out, 300.0, 50.0));
        fps_b_post.push(fingerprint_from_log(&log_b, n_out, 300.0, 50.0));
    }

    println!("\n== Results ==");
    println!("  target hit rate:");
    println!("    A: {}/10 → {}/10", hit_a_pre, hit_a_post);
    println!("    B: {}/10 → {}/10", hit_b_pre, hit_b_post);
    println!("  selectivity (within - between):");
    let pre_sel = (within(&fps_a_pre) + within(&fps_b_pre)) / 2.0 - between(&fps_a_pre, &fps_b_pre);
    let post_sel = (within(&fps_a_post) + within(&fps_b_post)) / 2.0 - between(&fps_a_post, &fps_b_post);
    println!("    pre : {:.3}", pre_sel);
    println!("    post: {:.3}", post_sel);

    std::fs::create_dir_all("out").ok();
    write_fps_csv("out/cont_fps_a_pre.csv",  &fps_a_pre).unwrap();
    write_fps_csv("out/cont_fps_b_pre.csv",  &fps_b_pre).unwrap();
    write_fps_csv("out/cont_fps_a_post.csv", &fps_a_post).unwrap();
    write_fps_csv("out/cont_fps_b_post.csv", &fps_b_post).unwrap();
    println!("\nCSVs written to out/");
}
