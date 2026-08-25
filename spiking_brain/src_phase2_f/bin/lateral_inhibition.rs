//! 側方抑制は母音の識別を上げるか (2026-08-26)
//!
//! ## なぜ
//!
//! 倍音つき刺激で母音の識別率が **35% で頭打ち**になり、
//! `(N_BANDS, Q, 閾値)` のどの組み合わせでも動かなかった (`m0_design_v2`)。
//! Q を 60 倍 (0.1-6.0)、帯域数を 3 倍 (40-120) 振っても変わらない。
//!
//! **原因**: 母音はどれも同じ帯域群にエネルギーが広がっており、**違うのは重みだけ**。
//! コサインは大きな共通成分に支配され、母音の差はその上の小さな摂動にすぎない。
//! (中心化しても 25% で改善しない = 共通成分を一律に引くだけでは足りない。)
//!
//! **側方抑制**は共通成分を**局所的に**差し引くので、この形に真正面から効くはず。
//!
//! ## この機構の位置づけ（正直に書く）
//!
//! **設計書には無い新規の機構。** 設計書 (Bushy) の「抑制」は
//! **時間方向の適応**(持続を抑えて立ち上がりを強調) であって、
//! 周波数方向の側方抑制ではない。
//! これまでの 4 件 (自発発火 / phase locking / 対数圧縮 / OHC 選択性) は
//! 「設計されたが未実装」だったが、これは違う。
//!
//! ただし 6 原理には触れない: 隣からの抑制は**局所的**(原理1)、
//! 抑制性シナプスは**物理プロセス**(原理2)、整数・決定論的(原理3/4)。
//! AGC の「全帯域の平均で割る」とは性質が違う。
//!
//! ## ゲート (実測前に固定)
//!
//!   G59 母音の識別率: 5 母音 × 4 F0 = 20 条件の leave-one-out 1-NN。
//!       **M0.5 の出力**で測る (抑制は M0.5 に入るので)。チャンス 15.8%。
//!       正解の出どころ = どれが同じ母音かは実験者が決めた。
//!   G60 壊さない    : 無音で発火しない / 沈黙する母音が出ない
//!
//! **採用規則 (先に宣言)**: G60 を満たすうち G59 最大。
//! **抑制なし (現行) を上回らなければ不採用。**
//!
//! CLI: lateral_inhibition

use spiking_brain::phase2_f::cochlea::{Cochlea, SAMPLES_PER_STEP};
use spiking_brain::phase2_f::cochlear_nucleus::{CochlearNucleus, N_CN_OUTPUT};
use spiking_brain::phase2_f::phoneme_synth::{synth_vowel_f0, vowels};

const VOWEL_MS: f64 = 170.0;
const F0S: [f64; 4] = [100.0, 150.0, 200.0, 250.0];
const INHIBITIONS: [i32; 8] = [0, 10, 20, 30, 40, 50, 70, 90];

/// M0.5 の出力チャネルごとのスパイク数
fn cn_counts(wave: &[i32], inhib: i32) -> Vec<u32> {
    let mut co = Cochlea::new();
    let mut cn = CochlearNucleus::new();
    cn.lateral_inhibition_percent = inhib;
    let mut counts = vec![0u32; N_CN_OUTPUT];
    for chunk in wave.chunks(SAMPLES_PER_STEP) {
        if chunk.len() < SAMPLES_PER_STEP {
            break;
        }
        let out = co.process_step(chunk);
        for (i, &v) in cn.process_step(&out).iter().enumerate() {
            if v != 0 {
                counts[i] += 1;
            }
        }
    }
    counts
}

fn cosine(a: &[u32], b: &[u32]) -> f64 {
    let dot: f64 = a.iter().zip(b.iter()).map(|(&x, &y)| x as f64 * y as f64).sum();
    let na: f64 = a.iter().map(|&x| (x as f64).powi(2)).sum::<f64>().sqrt();
    let nb: f64 = b.iter().map(|&x| (x as f64).powi(2)).sum::<f64>().sqrt();
    if na == 0.0 || nb == 0.0 {
        0.0
    } else {
        dot / (na * nb)
    }
}

fn main() {
    let vs = vowels();
    println!("=== 側方抑制は母音の識別を上げるか ===");
    println!("M0.5 の出力 ({} ch) で測る ・ 5 母音 × 4 F0 = 20 条件", N_CN_OUTPUT);
    println!("主指標 G59 = leave-one-out 1-NN の母音識別率 (チャンス 15.8%)");
    println!("採用規則: G60(壊さない) のうち G59 最大。抑制なしを上回らなければ不採用。");
    println!();
    println!("抑制[%]  G59識別率  同一母音の最小  別母音の最大  沈黙  無音時の発火");

    let mut best: Option<(i32, f64)> = None;
    for &inhib in INHIBITIONS.iter() {
        // 無音での発火 (G60)
        let silence_spikes: u32 = {
            let mut co = Cochlea::new();
            let mut cn = CochlearNucleus::new();
            cn.lateral_inhibition_percent = inhib;
            let mut n = 0u32;
            for _ in 0..4000 {
                let out = co.process_step(&[0i32; SAMPLES_PER_STEP]);
                n += cn.process_step(&out).iter().filter(|&&v| v != 0).count() as u32;
            }
            n
        };

        let mut conds: Vec<(usize, Vec<u32>)> = Vec::new();
        let mut silent = 0usize;
        for (k, v) in vs.iter().enumerate() {
            for &f0 in F0S.iter() {
                let c = cn_counts(&synth_vowel_f0(v, f0, VOWEL_MS), inhib);
                if c.iter().all(|&x| x == 0) {
                    silent += 1;
                }
                conds.push((k, c));
            }
        }
        // leave-one-out 1-NN
        let mut hit = 0usize;
        for i in 0..conds.len() {
            let mut best_j = (-2.0f64, usize::MAX);
            for j in 0..conds.len() {
                if i == j {
                    continue;
                }
                let c = cosine(&conds[i].1, &conds[j].1);
                if c > best_j.0 {
                    best_j = (c, conds[j].0);
                }
            }
            if best_j.1 == conds[i].0 {
                hit += 1;
            }
        }
        let ident = hit as f64 / conds.len() as f64;
        // 同一母音の最小 / 別母音の最大
        let mut min_same = f64::INFINITY;
        let mut max_diff = f64::NEG_INFINITY;
        for i in 0..conds.len() {
            for j in (i + 1)..conds.len() {
                let c = cosine(&conds[i].1, &conds[j].1);
                if conds[i].0 == conds[j].0 {
                    min_same = min_same.min(c);
                } else {
                    max_diff = max_diff.max(c);
                }
            }
        }
        let ok = silent == 0 && silence_spikes == 0;
        println!(
            "{:>6}  {:>9.1}%  {:>14.3}  {:>12.3}  {:>4}  {:>12}  {}",
            inhib,
            ident * 100.0,
            min_same,
            max_diff,
            silent,
            silence_spikes,
            if ok { "" } else { "**G60 FAIL**" }
        );
        if ok {
            match best {
                Some((_, b)) if b >= ident => {}
                _ => best = Some((inhib, ident)),
            }
        }
    }

    println!();
    match best {
        Some((inhib, ident)) => {
            println!("採用規則の結果: **側方抑制 {}%** (識別率 {:.1}%)", inhib, ident * 100.0);
            if inhib == 0 {
                println!("  = **抑制なしが最良。側方抑制は不採用。**");
            } else {
                println!("  抑制なしを上回った → 採用候補");
            }
        }
        None => println!("**G60 を満たす設定なし — 不成立。**"),
    }
}
