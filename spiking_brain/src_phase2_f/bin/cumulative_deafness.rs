//! 累積失聴 — M0.5 は繰り返し提示で聞こえなくなるか (2026-08-25)
//!
//! ## なぜ最優先か
//!
//! 元の目的は「コーパスを流してストリーミングで学ぶ」こと。
//! 刺激列が長くなるほど聞こえなくなるなら、その目的は原理的に達成できない。
//!
//! 独立監査の指摘: 「M0.5 の entropy 適応は有界でない — 60 音節で実効閾値 43 倍、
//! 応答 30→2 発。『適応』ではなく**累積失聴**」。
//!
//! そして `ThermoNeuron::reset_state` は **`local_entropy` を意図的に保持**する
//! (「試行間で持ち越して、慣化と回復の動力学を生かす」)。
//! つまりパイプラインの `cn.reset()` はエントロピーを消さない。
//! **本セッションの M1 測定すべてに掛かる可能性がある。**
//! (n_train=500 で基準が 0.767→0.581 に下がったのを「M1 側の別問題」と書いたが、
//!  これが原因かもしれない。)
//!
//! ## ゲート (実測前に固定)
//!
//! 正解の出どころ = **同じ刺激を繰り返し与えたのは実験者**。
//! §2.2 の「同じものに同じく応じるか」そのもの。
//!
//!   G47a M0 の累積  : 蝸牛の応答が繰り返しで減衰しないか (蝸牛に適応は無いので
//!                     減らないはず = 対照。減っていたら測定系の別の問題)
//!   G47b M0.5 の累積: 蝸牛神経核の応答が繰り返しで減衰しないか
//!   G47c 回復       : 無音を挟めば回復するか。何 ms の無音が要るか
//!
//! CLI: cumulative_deafness

use spiking_brain::phase2_f::cochlea::{Cochlea, SAMPLES_PER_STEP};
use spiking_brain::phase2_f::cochlear_nucleus::{CochlearNucleus, N_CN_OUTPUT, N_OCTOPUS};
use spiking_brain::phase2_f::phoneme_synth::{
    standard_syllables, synth_syllable_scaled, LfsrNoise,
};

const N_PRESENT: usize = 120;
/// 途中経過を出す提示番号
const CHECKPOINTS: [usize; 8] = [1, 2, 5, 10, 20, 40, 80, 120];

/// 1 提示ぶんの (M0 スパイク, M0.5 スパイク, M0.5 発火ch数, Octopus スパイク)
fn present(
    cochlea: &mut Cochlea,
    cn: &mut CochlearNucleus,
    wave: &[i32],
    do_reset: bool,
) -> (u32, u32, usize, u32) {
    if do_reset {
        cochlea.reset();
        cn.reset();
    }
    let (mut m0, mut m05, mut oct) = (0u32, 0u32, 0u32);
    let mut active = vec![false; N_CN_OUTPUT];
    for chunk in wave.chunks(SAMPLES_PER_STEP) {
        if chunk.len() < SAMPLES_PER_STEP {
            break;
        }
        let out = cochlea.process_step(chunk);
        m0 += out.iter().filter(|&&v| v != 0).count() as u32;
        let cn_out = cn.process_step(&out);
        for (i, &v) in cn_out.iter().enumerate() {
            if v != 0 {
                m05 += 1;
                active[i] = true;
                if i < N_OCTOPUS {
                    oct += 1;
                }
            }
        }
    }
    (m0, m05, active.iter().filter(|&&b| b).count(), oct)
}

/// 無音を n_ms だけ流す (回復の測定用)
fn silence(cochlea: &mut Cochlea, cn: &mut CochlearNucleus, n_ms: f64) {
    let steps = (n_ms / 0.5) as usize;
    let zero = [0i32; SAMPLES_PER_STEP];
    for _ in 0..steps {
        let out = cochlea.process_step(&zero);
        let _ = cn.process_step(&out);
    }
}

fn main() {
    let syl = standard_syllables()[0]; // pa
    let mut noise = LfsrNoise::new(0xACE1);
    let wave = synth_syllable_scaled(&syl, &mut noise, 1.0);

    println!("=== 累積失聴 — M0.5 は繰り返し提示で聞こえなくなるか ===");
    println!("音節 {} を {} 回連続提示 ・ 提示ごとに reset() を呼ぶ (パイプラインと同じ)",
             syl.label, N_PRESENT);
    println!("※ ThermoNeuron::reset_state は local_entropy を意図的に保持するので、");
    println!("   reset() を呼んでも適応は持ち越される。");
    println!();

    for &do_reset in [true, false].iter() {
        println!("--- reset() を{}呼ぶ ---", if do_reset { "" } else { "呼ばない (連続ストリーム)" });
        println!("提示回  M0スパイク  M0.5スパイク  M0.5発火ch数  Octopusスパイク");
        let mut cochlea = Cochlea::new();
        let mut cn = CochlearNucleus::new();
        let mut first = (0u32, 0u32, 0usize, 0u32);
        for i in 1..=N_PRESENT {
            let r = present(&mut cochlea, &mut cn, &wave, do_reset);
            if i == 1 {
                first = r;
            }
            if CHECKPOINTS.contains(&i) {
                println!("{:>6}  {:>10}  {:>12}  {:>12}  {:>15}", i, r.0, r.1, r.2, r.3);
            }
            if i == N_PRESENT {
                let g47a = r.0 as f64 / first.0.max(1) as f64;
                let g47b = r.1 as f64 / first.1.max(1) as f64;
                println!();
                println!("  G47a M0 の保存率  : {:.1}%  {}", g47a * 100.0,
                         if g47a >= 0.95 { "PASS (蝸牛に適応は無い)" } else { "**FAIL — 蝸牛が減っている**" });
                println!("  G47b M0.5 の保存率: {:.1}%  {}", g47b * 100.0,
                         if g47b >= 0.95 { "PASS" } else { "**FAIL — 累積失聴**" });

                // G47c 回復
                if g47b < 0.95 {
                    println!();
                    println!("  G47c 回復: 無音を挟んでから同じ音を出す");
                    println!("    無音[ms]  M0.5スパイク  初回比");
                    for &ms in [0.0f64, 100.0, 500.0, 2000.0, 10000.0].iter() {
                        let mut c2 = cochlea.clone();
                        let mut n2 = cn.clone();
                        silence(&mut c2, &mut n2, ms);
                        let rr = present(&mut c2, &mut n2, &wave, do_reset);
                        println!("    {:>8.0}  {:>12}  {:>6.1}%",
                                 ms, rr.1, rr.1 as f64 / first.1.max(1) as f64 * 100.0);
                    }
                }
            }
        }
        println!();
    }
}
