//! 蝸牛の整数 biquad は自己発振しているか (S6・2026-08-25)
//!
//! ## 発見
//!
//! `cochlea_gates_v2` の帯域健全性検査で、**現行 production 設定 (F_MAX=4000) の
//! 40 帯域中 24 本が、インパルス応答を 2000 サンプル走らせても減衰しきらない**ことが判明。
//!
//!   帯域 0 fc=  50.0Hz  前半max|y|=1876  末尾max|y|=2491  (減衰どころか増加)
//!   帯域 1 fc=  70.1Hz  前半max|y|=1349  末尾max|y|=1349  (完全に無減衰)
//!   帯域 2 fc=  91.7Hz  前半max|y|= 770  末尾max|y|= 770
//!   帯域 3 fc= 114.9Hz  前半max|y|= 490  末尾max|y|= 490
//!
//! 母音 F1 が乗る帯域も含まれる (/a/ 820Hz=帯域19、/u/ 331Hz=帯域10)。
//! つまり「発火」の一部は刺激への応答でなく自励振動でありうる。
//! **これは過去の M1 実測値すべてに掛かる交絡である。**
//!
//! ## 仮説と検証
//!
//! `BandpassBiquad::process` は `(acc >> 15) as i32` で Q15 を戻している。
//! 算術右シフトは**切り捨て** (負値では -∞ 方向) なので直流バイアスが乗る。
//! 極が単位円のすぐ内側にあるとき (低周波ほど近い)、このバイアスが
//! リミットサイクルを維持・成長させる — 固定小数点 IIR の古典的病理。
//!
//! 検証は 3 つの丸めモードを同じ係数で比べるだけ。正解はフィルタ理論側にある:
//! **安定な線形フィルタのインパルス応答はゼロに収束する**。
//!   trunc  : 現行 `acc >> 15`
//!   round  : `(acc + (1 << 14)) >> 15`  (最近接への丸め)
//!   magtrunc: ゼロ方向への切り捨て (符号対称)
//!
//! さらに極半径 r = sqrt(a2_q/32768) を係数から直接出し、
//! 「どの帯域が単位円に近すぎるか」を周波数の関数として示す。
//!
//! CLI: biquad_limit_cycle

use spiking_brain::phase2_f::cochlea::{
    erb_q_factor, erb_spaced_freqs, BandpassBiquad, F_MIN_HZ, N_BANDS,
};
use spiking_brain::phase2_f::phoneme_synth::SAMPLE_RATE_HZ;

const F_MAX: f64 = 4000.0; // 現行 production 値
const N_STEPS: usize = 4000;
const TAIL_FROM: usize = 3900;

#[derive(Clone, Copy, PartialEq)]
enum Rounding {
    Trunc,
    Round,
    MagTrunc,
}

impl Rounding {
    fn name(self) -> &'static str {
        match self {
            Rounding::Trunc => "trunc (現行)",
            Rounding::Round => "round (最近接)",
            Rounding::MagTrunc => "magtrunc (ゼロ方向)",
        }
    }
    fn apply(self, acc: i64) -> i32 {
        match self {
            Rounding::Trunc => (acc >> 15) as i32,
            Rounding::Round => ((acc + (1 << 14)) >> 15) as i32,
            Rounding::MagTrunc => (acc / 32768) as i32, // 整数除算はゼロ方向
        }
    }
}

/// 同じ係数で丸めモードだけ差し替えた biquad を走らせる。
fn impulse_tail(fc: f64, mode: Rounding) -> (i32, i32) {
    let proto = BandpassBiquad::new(fc, erb_q_factor(fc), SAMPLE_RATE_HZ);
    let (b0, b1, b2, a1, a2) = (
        proto.b0 as i64,
        proto.b1 as i64,
        proto.b2 as i64,
        proto.a1 as i64,
        proto.a2 as i64,
    );
    let (mut x1, mut x2, mut y1, mut y2) = (0i64, 0i64, 0i64, 0i64);
    let (mut early, mut tail) = (0i32, 0i32);
    for n in 0..N_STEPS {
        let x0: i64 = if n == 0 { 10000 } else { 0 };
        let acc = b0 * x0 + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2;
        let y0 = mode.apply(acc);
        x2 = x1;
        x1 = x0;
        y2 = y1;
        y1 = y0 as i64;
        let a = y0.abs();
        if n < 500 {
            early = early.max(a);
        } else if n >= TAIL_FROM {
            tail = tail.max(a);
        }
    }
    (early, tail)
}

fn main() {
    let freqs = erb_spaced_freqs(F_MIN_HZ, F_MAX, N_BANDS);
    println!("=== 蝸牛 整数 biquad の自己発振検査 (F_MAX={:.0}・{} サンプル) ===", F_MAX, N_STEPS);
    println!("正解の出どころ: 安定な線形フィルタのインパルス応答はゼロに収束する (フィルタ理論)");
    println!();

    // --- 1) 丸めモード比較 ---
    println!("丸めモード              残存帯域数/{}  最悪の末尾max|y|  該当帯域(先頭5本のfc)", N_BANDS);
    let modes = [Rounding::Trunc, Rounding::Round, Rounding::MagTrunc];
    for &m in modes.iter() {
        let mut bad = Vec::new();
        let mut worst = 0i32;
        for &fc in freqs.iter() {
            let (early, tail) = impulse_tail(fc, m);
            if early == 0 || tail != 0 {
                bad.push(fc);
                worst = worst.max(tail);
            }
        }
        let head: Vec<String> = bad.iter().take(5).map(|f| format!("{:.0}", f)).collect();
        println!(
            "{:<22} {:>12}  {:>16}  {}",
            m.name(),
            bad.len(),
            worst,
            if head.is_empty() { "-".to_string() } else { head.join(", ") }
        );
    }

    // --- 2) 極半径を係数から直接出す ---
    println!();
    println!("--- 極半径 r = sqrt(a2/32768) と末尾残存 (現行 trunc) ---");
    println!("帯域   fc(Hz)   a1(Q15)   a2(Q15)     極半径 r   1-r        末尾max|y|");
    for (i, &fc) in freqs.iter().enumerate() {
        let bq = BandpassBiquad::new(fc, erb_q_factor(fc), SAMPLE_RATE_HZ);
        let r = ((bq.a2 as f64) / 32768.0).abs().sqrt();
        let (_e, tail) = impulse_tail(fc, Rounding::Trunc);
        // 全帯域は多いので、末尾残存があるものと最初の数本だけ出す
        if tail != 0 || i < 4 {
            println!(
                "{:>4}  {:>7.1}  {:>8}  {:>8}     {:.6}   {:.6}   {:>10}",
                i, fc, bq.a1, bq.a2, r, 1.0 - r, tail
            );
        }
    }

    // --- 3) 減衰に必要なサンプル数の理論値と実際 ---
    println!();
    println!("--- 理論減衰時間との比較 (現行 trunc) ---");
    println!("極半径 r の理論では |y| は r^n で減る。r^n < 1 になる n = ln(1/y0)/ln(1/r)。");
    for &i in [0usize, 1, 10, 19, 39].iter() {
        let fc = freqs[i];
        let bq = BandpassBiquad::new(fc, erb_q_factor(fc), SAMPLE_RATE_HZ);
        let r = ((bq.a2 as f64) / 32768.0).abs().sqrt();
        let (early, tail) = impulse_tail(fc, Rounding::Trunc);
        let n_theory = if r < 1.0 && early > 0 {
            ((early as f64).ln() / (1.0 / r).ln()) as i64
        } else {
            -1
        };
        println!(
            "  帯域{:>2} fc={:>7.1}Hz  r={:.6}  理論減衰 {:>8} サンプル  実測末尾max|y|={}",
            i, fc, r, n_theory, tail
        );
    }
}
