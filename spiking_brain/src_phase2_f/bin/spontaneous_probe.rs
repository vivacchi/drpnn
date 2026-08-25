//! M0 蝸牛の自発発火 (聴神経 spontaneous rate) — S9 検証・2026-08-25
//!
//! ## 経緯
//!
//! `M0_COCHLEA_DESIGN.md` §3.6 は 2026-05-24 に
//! 「**M0 蝸牛が聴神経 spontaneous rate を含むスパイク列を生成する**／
//! 各帯域の自発発火頻度を **50-100 Hz** スケールで設定／
//! M1 input neuron の `spontaneous_input = 0` のまま維持」を**確定した設計**として記載し、
//! `// cochlea.rs (Step 3 で実装)` と書いていたが、**実装されていなかった**。
//!
//! 結果、蝸牛 → M0.5 蝸牛神経核 (3 細胞型とも `spontaneous_input = 0`) → M1 input
//! という入力経路に自発活動がゼロだった (M1 の内部ニューロンだけ `idx % 4` で持つ)。
//!
//! 実装: 帯域ごとに独立な決定論的 LFSR を**包絡線検出器の入力**に加算する。
//! 生物で spontaneous rate が生じるのは内有毛細胞→聴神経シナプスであり、
//! 機械的フィルタリングの**下流**だから。帯域間で独立にするのは、相関ノイズだと
//! 広帯域オンセットに見えて M0.5 の Octopus 細胞が偽発火するため。
//! 原理 3「乱数を使わない」を満たすため乱数ではなく LFSR を使う。
//!
//! ## ゲート (実測前に固定)
//!
//!   G13 全帯域が自発発火する           : 無音時に 40 帯域すべて rate > 0
//!   G14 設計指定のレートに入る         : 帯域レートの中央値が 50-100 Hz
//!        **この 50-100 Hz は設計書 §3.6 が指定した値。こちらで決めた閾値ではない。**
//!   G15 決定論的な帯域間個体差がある   : 全帯域が同一レートでない
//!   G16 決定論性                       : 2 回走らせて完全一致
//!   G17 刺激応答を壊さない             : 母音 5 種の F1 帯域が保たれ、無音母音が出ない
//!
//! 振幅の選定規則 (先に宣言): **中央値レートが設計範囲の中央 75 Hz に最も近い振幅**を採る。
//!
//! CLI: spontaneous_probe

use spiking_brain::phase2_f::cochlea::{
    erb_spaced_freqs, Cochlea, F_MIN_HZ, F_MAX_HZ, N_BANDS, SAMPLES_PER_STEP,
    SPONTANEOUS_INDIVIDUALITY, SPONTANEOUS_RATE_TARGET_HZ,
};
use spiking_brain::phase2_f::phoneme_synth::{synth_vowel, vowels};

/// DT_MS = 0.5ms → 2000 step/秒
const STEPS_PER_SEC: f64 = 2000.0;
const SILENCE_STEPS: usize = 20_000; // 10 秒
const VOWEL_MS: f64 = 170.0;
const AMPLITUDES: [i32; 12] = [2, 4, 6, 8, 10, 12, 16, 20, 24, 32, 48, 64];
/// 選定規則: 設計範囲の中央
const TARGET_MID_HZ: f64 = 75.0;

/// 無音を流して帯域ごとの自発発火レート [Hz] を測る。
fn spontaneous_rates(amplitude: i32) -> Vec<f64> {
    let mut c = Cochlea::new();
    c.spontaneous_amplitude = amplitude;
    let silence = [0i32; SAMPLES_PER_STEP];
    let mut counts = vec![0u32; N_BANDS];
    for _ in 0..SILENCE_STEPS {
        let out = c.process_step(&silence);
        for ch in 0..N_BANDS {
            if out[ch] != 0 {
                counts[ch] += 1;
            }
        }
    }
    counts
        .iter()
        .map(|&n| n as f64 * STEPS_PER_SEC / SILENCE_STEPS as f64)
        .collect()
}

/// 母音を流したときの発火帯域集合 (G17 用)。
fn vowel_active_bands(amplitude: i32) -> Vec<Vec<usize>> {
    vowels()
        .iter()
        .map(|v| {
            let wave = synth_vowel(v, VOWEL_MS);
            let mut c = Cochlea::new();
            c.spontaneous_amplitude = amplitude;
            let mut counts = vec![0u32; N_BANDS];
            for chunk in wave.chunks(SAMPLES_PER_STEP) {
                if chunk.len() < SAMPLES_PER_STEP {
                    break;
                }
                let out = c.process_step(chunk);
                for ch in 0..N_BANDS {
                    if out[ch] != 0 {
                        counts[ch] += 1;
                    }
                }
            }
            (0..N_BANDS).filter(|&i| counts[i] > 0).collect()
        })
        .collect()
}

fn median(v: &[f64]) -> f64 {
    let mut s = v.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = s.len();
    if n % 2 == 1 { s[n / 2] } else { (s[n / 2 - 1] + s[n / 2]) / 2.0 }
}

fn main() {
    let (lo_hz, hi_hz) = SPONTANEOUS_RATE_TARGET_HZ;
    println!("=== M0 自発発火の掃引 (無音 {:.0} 秒・N_BANDS={}) ===",
             SILENCE_STEPS as f64 / STEPS_PER_SEC, N_BANDS);
    println!("設計指定レート: {:.0}-{:.0} Hz (M0_COCHLEA_DESIGN.md §3.6)", lo_hz, hi_hz);
    println!("帯域間個体差: idx % {} (M1 の idiom と同じ)", SPONTANEOUS_INDIVIDUALITY);
    println!();

    println!("振幅  無音帯域  レート中央値  最小    最大    個体差クラス別の中央値");
    let mut rows: Vec<(i32, Vec<f64>)> = Vec::new();
    for &a in AMPLITUDES.iter() {
        let r = spontaneous_rates(a);
        let silent = r.iter().filter(|&&v| v == 0.0).count();
        let mn = r.iter().cloned().fold(f64::INFINITY, f64::min);
        let mx = r.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let per_class: Vec<String> = (0..SPONTANEOUS_INDIVIDUALITY)
            .map(|k| {
                let sub: Vec<f64> = (0..N_BANDS)
                    .filter(|&ch| ch % SPONTANEOUS_INDIVIDUALITY == k)
                    .map(|ch| r[ch])
                    .collect();
                format!("{:.0}", median(&sub))
            })
            .collect();
        println!(
            "{:>4}  {:>8}  {:>12.1}  {:>6.1}  {:>6.1}  [{}]",
            a, silent, median(&r), mn, mx, per_class.join(", ")
        );
        rows.push((a, r));
    }

    // --- 選定規則: 中央値が 75 Hz に最も近い振幅 ---
    let best = rows
        .iter()
        .min_by(|a, b| {
            (median(&a.1) - TARGET_MID_HZ)
                .abs()
                .partial_cmp(&(median(&b.1) - TARGET_MID_HZ).abs())
                .unwrap()
        })
        .unwrap();
    let (amp, rates) = (best.0, &best.1);
    println!();
    println!("選定規則の結果: 振幅 = {} (中央値 {:.1} Hz)", amp, median(rates));

    // --- ゲート判定 ---
    println!();
    println!("--- ゲート判定 (振幅 {}) ---", amp);
    let g13 = rates.iter().all(|&v| v > 0.0);
    let med = median(rates);
    let g14 = med >= lo_hz && med <= hi_hz;
    let g15 = rates.iter().any(|&v| v != rates[0]);
    let g16 = spontaneous_rates(amp) == *rates;

    // G17: 母音応答が保たれるか (自発発火 OFF のときの発火帯域を含んでいること)
    let base = vowel_active_bands(0);
    let with = vowel_active_bands(amp);
    let names = ["a", "i", "u", "e", "o"];
    let mut g17 = true;
    let mut detail = Vec::new();
    for k in 0..base.len() {
        let kept = base[k].iter().all(|b| with[k].contains(b));
        let silent = with[k].is_empty();
        if !kept || silent {
            g17 = false;
        }
        detail.push(format!(
            "{}: F1帯域{}{} ({}本→{}本)",
            names[k],
            if kept { "保持" } else { "**消失**" },
            if silent { "・無音" } else { "" },
            base[k].len(),
            with[k].len()
        ));
    }

    println!("G13 全帯域が自発発火     : {}  (無音帯域 {})",
             if g13 { "PASS" } else { "FAIL" },
             rates.iter().filter(|&&v| v == 0.0).count());
    println!("G14 設計指定 {:.0}-{:.0} Hz  : {}  (中央値 {:.1} Hz)",
             lo_hz, hi_hz, if g14 { "PASS" } else { "FAIL" }, med);
    println!("G15 帯域間個体差         : {}", if g15 { "PASS" } else { "FAIL" });
    println!("G16 決定論性             : {}", if g16 { "PASS" } else { "FAIL" });
    println!("G17 刺激応答を壊さない   : {}", if g17 { "PASS" } else { "FAIL" });
    for d in detail.iter() {
        println!("      {}", d);
    }

    println!();
    println!("総合: {}", if g13 && g14 && g15 && g16 && g17 { "**全ゲート PASS**" } else { "**FAIL あり**" });

    // 参考: 帯域中心と自発レートの対応 (先頭・末尾)
    let freqs = erb_spaced_freqs(F_MIN_HZ, F_MAX_HZ, N_BANDS);
    println!();
    println!("--- 帯域別 自発レート (振幅 {}) ---", amp);
    for ch in 0..N_BANDS {
        if ch < 6 || ch >= N_BANDS - 4 {
            println!("  帯域{:>2} fc={:>7.1}Hz  個体差{}  {:>6.1} Hz",
                     ch, freqs[ch], 1 + ch % SPONTANEOUS_INDIVIDUALITY, rates[ch]);
        } else if ch == 6 {
            println!("  ...");
        }
    }
}
