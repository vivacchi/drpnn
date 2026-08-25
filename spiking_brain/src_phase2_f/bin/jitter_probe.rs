//! 背景活動ノイズは M1 にどれだけ入っているか (S10 診断・2026-08-25)
//!
//! ## 経緯
//!
//! ユーザーの狙いは「人間の脳内の活動ノイズ／JEPA のノイズ追加と同じ」。
//! `ThermoNeuron::spontaneous_jitter` として実装し、振幅 0-8 を掃引したところ
//! **全振幅で G19 (再現性) が崩壊**した:
//!
//!   jitter  selectivity  within  between  per-pair
//!        0        0.699   0.966    0.267     0.548
//!        1        0.161   0.373    0.211     0.728
//!        8        0.002   0.045    0.043     0.486
//!
//! `within` は「同じ音素を繰り返し与えたときの応答類似度」。0.966→0.373 は
//! **ネットワークが自分自身の応答を再現できなくなった**ことを意味する。
//! (jitter=6/8 の per-pair 改善は「全部壊れたから統計量が良く見える」型の罠。)
//!
//! ## この診断が問うこと
//!
//! なぜ最小振幅 1 でも壊れるのか。**正解は実験の設計側にある量だけを測る**:
//!   - 無音時の自発発火率 (入力を与えていないのは実験者)
//!   - 膜電位が閾値に達するまでの step 数と、その間に積もるノイズの大きさ
//!   - 決定論性 (同じ条件で 2 回走らせて完全一致するか)
//!
//! CLI: jitter_probe

use spiking_brain::phase2_f::thermo_network::{ThermoNetwork, ThermoNetworkConfig};
use spiking_brain::phase2_f::cochlea::N_BANDS;

/// DT_MS = 0.5ms → 2000 step/秒
const STEPS_PER_SEC: f64 = 2000.0;
const SILENCE_STEPS: usize = 20_000; // 10 秒
const AMPLITUDES: [i32; 7] = [0, 1, 2, 3, 4, 6, 8];
/// M0.5 蝸牛神経核の出力チャンネル数 (M1 の入力数)
const N_CN_OUT: usize = 84;

/// 無音 (入力電流ゼロ) で走らせ、出力ニューロンの自発発火率 [Hz] を返す。
fn silent_output_rate(amplitude: i32) -> (f64, usize) {
    let mut net = ThermoNetwork::new(ThermoNetworkConfig::for_m1_cn_40());
    net.set_spontaneous_jitter(amplitude);
    let zero = vec![0i32; N_CN_OUT];
    let n_out = net.output_neurons.len();
    let mut spikes = 0usize;
    let mut fired_any = vec![false; n_out];
    for _ in 0..SILENCE_STEPS {
        for nid in net.step(&zero) {
            if let Some(oi) = net.output_index_of(nid) {
                spikes += 1;
                fired_any[oi] = true;
            }
        }
    }
    let rate = spikes as f64 * STEPS_PER_SEC / (SILENCE_STEPS as f64 * n_out as f64);
    (rate, fired_any.iter().filter(|&&b| b).count())
}

/// 決定論性: 同じ振幅で 2 回走らせて出力スパイク列が完全一致するか。
fn deterministic(amplitude: i32) -> bool {
    let run = || {
        let mut net = ThermoNetwork::new(ThermoNetworkConfig::for_m1_cn_40());
        net.set_spontaneous_jitter(amplitude);
        let zero = vec![0i32; N_CN_OUT];
        let mut log = Vec::new();
        for t in 0..2000 {
            for nid in net.step(&zero) {
                log.push((t, nid));
            }
        }
        log
    };
    run() == run()
}

/// 膜電位のランダムウォーク: 振幅 A のノイズが n step で積もる典型的な大きさ。
/// 決定論的 LFSR を実際に回して測る (理屈でなく実測)。
fn accumulated_noise(amplitude: i32, steps: usize) -> f64 {
    use spiking_brain::phase2_f::thermo_neuron::ThermoNeuron;
    let mut worst = 0i32;
    let mut sum_abs = 0i64;
    let n_trial = 200;
    for k in 0..n_trial {
        let mut n = ThermoNeuron::excitatory((0, 0));
        n.spontaneous_jitter = amplitude;
        n.jitter_state = (0xACE1u16).wrapping_add((k as u16).wrapping_mul(2654));
        let mut acc = 0i32;
        for _ in 0..steps {
            acc += n.next_jitter();
        }
        worst = worst.max(acc.abs());
        sum_abs += acc.abs() as i64;
    }
    let _ = worst;
    sum_abs as f64 / n_trial as f64
}

/// 無音時に**入力ニューロン**が発火するか (除外ガードが生きているかの検査)。
///
/// thermo_network.rs のガードは `spontaneous_input == 0 && leak == 0` で
/// 入力ニューロンを除外するつもりだが、`ThermoNeuron::input()` は
/// `spontaneous_input: 2, leak: 1` なので**条件に掛からず素通りする**。
/// 結果、入力ニューロンにも `idx % 4` の自発入力が配られている疑いがある。
fn silent_input_layer() {
    let mut net = ThermoNetwork::new(ThermoNetworkConfig::for_m1_cn_40());
    let n_in = net.input_neurons.len();
    let ids: std::collections::HashSet<usize> = net.input_neurons.iter().cloned().collect();
    let spont: Vec<i32> = net.input_neurons.iter().map(|&i| net.neurons[i].spontaneous_input).collect();
    let leak: Vec<i32> = net.input_neurons.iter().map(|&i| net.neurons[i].leak).collect();
    let zero = vec![0i32; N_CN_OUT];
    let mut counts = vec![0u32; n_in];
    let idx_of: std::collections::HashMap<usize, usize> =
        net.input_neurons.iter().enumerate().map(|(k, &i)| (i, k)).collect();
    for _ in 0..SILENCE_STEPS {
        for nid in net.step(&zero) {
            if ids.contains(&nid) {
                counts[idx_of[&nid]] += 1;
            }
        }
    }
    let active = counts.iter().filter(|&&c| c > 0).count();
    let rates: Vec<f64> = counts.iter()
        .map(|&c| c as f64 * STEPS_PER_SEC / SILENCE_STEPS as f64).collect();
    let mx = rates.iter().cloned().fold(0.0f64, f64::max);
    println!();
    println!("--- 無音時の入力ニューロン (除外ガードの検査) ---");
    println!("入力ニューロン数 {} / 無音で発火したもの {} 個", n_in, active);
    println!("最大発火率 {:.1} Hz", mx);
    println!("spontaneous_input の分布: {:?}", {
        let mut h = std::collections::BTreeMap::new();
        for v in spont.iter() { *h.entry(*v).or_insert(0) += 1; }
        h
    });
    println!("leak の分布: {:?}", {
        let mut h = std::collections::BTreeMap::new();
        for v in leak.iter() { *h.entry(*v).or_insert(0) += 1; }
        h
    });
    println!("ガード条件 (spontaneous_input==0 && leak==0) に掛かる入力ニューロン: {} 個",
        (0..n_in).filter(|&k| spont[k] == 0 && leak[k] == 0).count());
    if active > 0 {
        println!("**ガードは死んでいる**: 無音でも入力層が発火している。");
        println!("  設計上は「入力ニューロンは受信専用トランスデューサ」のはず。");
    } else {
        println!("ガードは生きている (無音で入力層は沈黙)。");
    }
}

fn main() {
    silent_input_layer();
    println!("=== 背景活動ノイズの診断 (M1: for_m1_cn_40・入力 {}ch) ===", N_CN_OUT);
    println!("N_BANDS={} ・ 無音 {:.0} 秒", N_BANDS, SILENCE_STEPS as f64 / STEPS_PER_SEC);
    println!();
    println!("振幅  無音時の出力自発発火率  発火した出力  30step で積もるノイズ  決定論性");
    for &a in AMPLITUDES.iter() {
        let (rate, active) = silent_output_rate(a);
        let acc = accumulated_noise(a, 30);
        let det = deterministic(a);
        println!(
            "{:>4}  {:>20.1} Hz  {:>10}/40  {:>20.1}  {}",
            a, rate, active, acc,
            if det { "PASS" } else { "**FAIL**" }
        );
    }

    println!();
    println!("--- 読み方 ---");
    println!("出力ニューロンの threshold_base は 30 前後。");
    println!("「30step で積もるノイズ」が閾値 30 に対してどれだけ大きいかが、");
    println!("信号 (シナプス入力) がノイズに埋もれるかどうかを決める。");
    println!("整数の膜電位では **±1/step が最小の刻み**なので、");
    println!("これより穏やかなノイズはこの表現のままでは作れない。");
}
