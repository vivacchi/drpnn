//! モーラ境界と子音-母音境界を検算する (2026-08-27)
//!
//! ## なぜ
//!
//! 入り口監査が 2 つ主張した (**まだ出荷コードで確かめていない**):
//!
//! - **モーラ境界を音響が越えない** — 各モーラの母音が 10ms の release ramp でゼロに落ち、
//!   次のモーラがゼロから始まるので、**「連続発話」は「孤立モーラの連結」と同一**である。
//!   本当なら §14.19/§14.26 の「文脈」軸は**協調調音ではなく適応だけ**を測っていたことになる。
//! - **子音-母音境界に谷ができる** — 子音にも母音にも ramp があるので、
//!   **その谷の位置は `CONSONANT_STEPS = 60` (= 30ms) = 2窓の境界とちょうど同じ。**
//!   本当なら **2窓の利得の一部は「谷を見ているだけ」**かもしれない。
//!
//! ## ゲート (実測前に固定・以後動かさない)
//!
//! - **V6 モーラ境界を音響が越えるか**: 連続合成と個別連結が**バイト同一か**。
//!   *同一なら越えていない。* 母音だけのモーラを使う (雑音を消費しないので条件が揃う)。
//! - **V7 子音-母音境界に谷があるか**: 1ms ごとの包絡を dB で出す。
//!   谷の**位置**と**深さ**を記録する。*深さは判定しない。記述である。*
//! - **V8 谷は 2窓の境界と一致するか**: 谷の最小点が 30ms ± 3ms に入るか。
//!
//! ## 予測 (実測前)
//!
//! - **V6 はバイト同一になるはず** (各モーラが独立に合成され連結されるだけ)。
//! - **V7 谷はあるはず** (母音に 10ms の attack、子音にも 5ms の release)。**深さは分からない。**
//!
//! CLI: boundary_check

use spiking_brain::phase2_f::kana::{moras_from_kana, synth_utterance, MORA_MS};
use spiking_brain::phase2_f::phoneme_synth::{LfsrNoise, SAMPLE_RATE_HZ};

const F0: f64 = 150.0;
const SEED: u16 = 0xACE1;

fn synth(k: &str, noise: &mut LfsrNoise) -> Vec<i32> {
    let (m, sk) = moras_from_kana(k);
    assert_eq!(sk, 0, "未対応: {}", k);
    synth_utterance(&m, F0, noise)
}

fn rms(w: &[i32]) -> f64 {
    (w.iter().map(|&s| (s as f64) * (s as f64)).sum::<f64>() / w.len().max(1) as f64).sqrt()
}

fn main() {
    println!("=== モーラ境界と子音-母音境界を検算する ===");
    println!();
    println!("【なぜ】入り口監査が2つ主張した (**まだ出荷コードで確かめていない**):");
    println!("  ・**モーラ境界を音響が越えない** -> 本当なら §14.19/§14.26 の「文脈」軸は");
    println!("    **協調調音ではなく適応だけ**を測っていたことになる");
    println!("  ・**子音-母音境界に谷** -> その位置は CONSONANT_STEPS=60 (=30ms) = **2窓の境界と同じ**");
    println!("    本当なら **2窓の利得の一部は『谷を見ているだけ』**かもしれない");
    println!();
    println!("【予測・事前】V6 はバイト同一になるはず / V7 谷はあるはず・**深さは分からない**");

    // ---------- V6 モーラ境界 ----------
    println!();
    println!("--- V6 モーラ境界を音響が越えるか ---");
    for pair in ["あい", "おえ", "いあ"].iter() {
        let mut n0 = LfsrNoise::new(SEED);
        let cont = synth(pair, &mut n0);
        let cs: Vec<char> = pair.chars().collect();
        let mut cat: Vec<i32> = Vec::new();
        for c in cs.iter() {
            let mut n = LfsrNoise::new(SEED);
            cat.extend(synth(&c.to_string(), &mut n));
        }
        println!("  {} : 連続合成 {} サンプル / 個別連結 {} サンプル -> {}",
                 pair, cont.len(), cat.len(),
                 if cont == cat { "**バイト同一 = 音響は境界を越えていない**" } else { "違う (越えている)" });
    }

    // ---------- V7 / V8 子音-母音境界 ----------
    println!();
    println!("--- V7/V8 子音-母音境界の包絡 (1ms ごと・最大を 0dB とする) ---");
    println!("  子音区間 = 0-30ms / 母音区間 = 30-120ms。**2窓の境界は 30ms。**");
    println!();
    let step = (SAMPLE_RATE_HZ / 1000.0) as usize;   // 1ms
    for k in ["か", "な", "さ", "あ"].iter() {
        let mut n = LfsrNoise::new(SEED);
        let w = synth(k, &mut n);
        let nb = (MORA_MS as usize).min(w.len() / step);
        let env: Vec<f64> = (0..nb).map(|i| rms(&w[i * step..((i + 1) * step).min(w.len())])).collect();
        let peak = env.iter().cloned().fold(0f64, f64::max).max(1e-9);
        let db: Vec<f64> = env.iter().map(|&e| 20.0 * (e / peak).max(1e-9).log10()).collect();
        // 20-45ms の最小点 (境界の谷)
        let lo = 20usize.min(nb - 1);
        let hi = 45usize.min(nb);
        let (mut vi, mut vmin) = (lo, f64::INFINITY);
        for i in lo..hi { if db[i] < vmin { vmin = db[i]; vi = i; } }
        print!("  {} 20-45ms: ", k);
        for i in lo..hi.min(lo + 25) { print!("{:>5.0}", db[i]); }
        println!();
        println!("      -> **谷は {}ms で {:.1} dB**  (2窓の境界 30ms との差 {}ms)",
                 vi, vmin, (vi as i64 - 30));
    }

    println!();
    println!("  V8 谷は 2窓の境界 (30ms±3ms) と一致するか -> 上の「差」を見よ。");
    println!();
    println!("  【この検算が答えないこと】**谷があることと、2窓の利得が谷由来であることは別。**");
    println!("  それを分けるには**谷を埋めて 2窓を測り直す**必要がある。**まだやっていない。**");
}
