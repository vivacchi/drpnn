//! 蝸牛の校正は帯域ごとに成り立っているか (S3 検証・2026-08-25)
//!
//! ## 見つけた真因
//!
//! `FIRE_THRESHOLD = 200` は「純音 amp 8000 想定」で置かれている (cochlea.rs)。
//! 母音波形のピークは 7794 で確かにその設計点にある。**しかし閾値は帯域ごとに効く。**
//! 1 本の帯域に届くのは最強フォルマント F1 の振幅 4000 だけ = 設計点の半分。
//!
//! 実測 (F_MAX=4000・圧縮後包絡線のピーク / 閾値 200):
//!   a: F1 204 ○ / F2 167 × / F3 108 ×
//!   i: F1 216 ○ / F2 138 × / F3 131 ×
//!   u: F1 202 ○ / F2 177 × / F3  92 ×
//!   e: F1 203 ○ / F2 157 × / F3 112 ×
//!   o: F1 207 ○ / F2 190 × / F3 111 ×
//!
//! 全母音の符号が 138-216 に密集し、閾値 200 がその真ん中を切っている。
//! 結果、**どの母音も帯域 1 本しか鳴らず F2 は一度も鳴らない**。
//! /e/ と /o/ は F1 が同じ 500Hz なので蝸牛出力が完全に同一 = 構造的に区別不能。
//! これは S1 で発覚した「ピークで校正したがノイズは RMS で効く」と同じ型の取り違え
//! (校正した量と、効いている量が違う)。
//!
//! ## ゲート (実測前に固定)
//!
//!   G4  サ行が聞こえるか : se 総スパイク > 2 かつ 発火帯域数 >= 2
//!   G5' 母音の区別       : (a) 5 母音すべて総スパイク > 0
//!                          (b) 10 ペアすべてで 40 帯域スパイクベクトルが相異なる
//!        正解 = 10/10。実験者が 5 つ別々のフォルマントを与えたから。
//!        基準 (ゲイン1x・F_MAX=4000) は 9/10 で FAIL (e/o 同一) → 棄却域は空でない。
//!        (a) を必須にするのは、無音の母音が「異なる」と数えられて
//!        見せかけの改善を作るのを防ぐため。
//!   G6  数値健全性       : 最上帯域 biquad のインパルス応答 > 0 かつ有界
//!
//! 掃引: 帯域ゲイン {1x(現状), 2x(設計点 8000 復元)} × F_MAX {4000,5000,6000,7000,7500}
//!   ゲイン 2x は恣意的な値ではない。FIRE_THRESHOLD が宣言している設計点
//!   「純音 amp 8000」に、最強フォルマント (4000) を合わせる倍率がちょうど 2。
//!
//! 採用規則 (先に宣言): 3 ゲート全通過のうち **ゲインの小さい方を優先**、次に F_MAX 最大。
//! 無ければ不成立として報告し、定数は変えない。
//!
//! ゲインは**提示側の増幅**として実装する (母音テーブルも定数も書き換えない)。
//!
//! CLI: cochlea_calibration

use spiking_brain::phase2_f::cochlea::{
    erb_q_factor, erb_spaced_freqs, BandpassBiquad, Cochlea, EnvelopeDetector, FireGenerator,
    ENV_LEAK_SHIFT, FIRE_REFRACTORY_STEPS, FIRE_THRESHOLD, F_MIN_HZ, N_BANDS, SAMPLES_PER_STEP,
};
use spiking_brain::phase2_f::phoneme_synth::{F0_DEFAULT_HZ, 
    standard_syllables, synth_consonant_banded, synth_vowel, vowels, LfsrNoise, SAMPLE_RATE_HZ,
};

const CONSONANT_MS: f64 = 30.0;
const VOWEL_MS: f64 = 170.0;
const SEED: u16 = 0xACE1;
const F_MAX_CANDIDATES: [f64; 5] = [4000.0, 5000.0, 6000.0, 7000.0, 7500.0];
const GAIN_CANDIDATES: [i32; 2] = [1, 2];
/// 基準 (現状) の se 総スパイク数 — S1 で実測済み
const BASE_SE_SPIKES: u32 = 2;

fn cochlea_with_fmax(f_max: f64) -> Cochlea {
    let center_freqs = erb_spaced_freqs(F_MIN_HZ, f_max, N_BANDS);
    let bands = center_freqs
        .iter()
        .map(|&fc| BandpassBiquad::new(fc, erb_q_factor(fc), SAMPLE_RATE_HZ))
        .collect();
    let envelopes = (0..N_BANDS).map(|_| EnvelopeDetector::new(ENV_LEAK_SHIFT)).collect();
    let fire_gens = (0..N_BANDS)
        .map(|_| FireGenerator::new(FIRE_THRESHOLD, FIRE_REFRACTORY_STEPS))
        .collect();
    Cochlea { bands, envelopes, fire_gens, center_freqs, ..Cochlea::new() }
}

/// 波形にゲインをかけて蝸牛に通し、帯域ごとのスパイク数を返す。
fn band_spikes(wave: &[i32], f_max: f64, gain: i32) -> [u32; N_BANDS] {
    let mut c = cochlea_with_fmax(f_max);
    let mut counts = [0u32; N_BANDS];
    for chunk in wave.chunks(SAMPLES_PER_STEP) {
        if chunk.len() < SAMPLES_PER_STEP {
            break;
        }
        let amplified: Vec<i32> = chunk.iter().map(|&x| x.saturating_mul(gain)).collect();
        let out = c.process_step(&amplified);
        for ch in 0..N_BANDS {
            if out[ch] != 0 {
                counts[ch] += 1;
            }
        }
    }
    counts
}

fn top_band_impulse(f_max: f64) -> (f64, f64) {
    let freqs = erb_spaced_freqs(F_MIN_HZ, f_max, N_BANDS);
    let fc = *freqs.last().unwrap();
    let mut bp = BandpassBiquad::new(fc, erb_q_factor(fc), SAMPLE_RATE_HZ);
    let (mut energy, mut peak) = (0.0f64, 0.0f64);
    for i in 0..2000 {
        let y = bp.process(if i == 0 { 10000 } else { 0 }) as f64;
        energy += y.abs();
        peak = peak.max(y.abs());
    }
    (energy, peak)
}

struct Cell {
    gain: i32,
    f_max: f64,
    se_spikes: u32,
    se_bands: usize,
    silent_vowels: usize,
    distinct_pairs: usize,
    vowel_active_bands: Vec<usize>,
    impulse_energy: f64,
    impulse_peak: f64,
}

fn measure(gain: i32, f_max: f64) -> Cell {
    let se_c = standard_syllables()[3].consonant;
    let mut n = LfsrNoise::new(SEED);
    let se = band_spikes(&synth_consonant_banded(se_c, CONSONANT_MS, F0_DEFAULT_HZ, &mut n), f_max, gain);

    let profiles: Vec<[u32; N_BANDS]> = vowels()
        .iter()
        .map(|v| band_spikes(&synth_vowel(v, VOWEL_MS), f_max, gain))
        .collect();
    let silent = profiles.iter().filter(|p| p.iter().sum::<u32>() == 0).count();
    let mut distinct = 0usize;
    for i in 0..profiles.len() {
        for j in (i + 1)..profiles.len() {
            if profiles[i] != profiles[j] {
                distinct += 1;
            }
        }
    }
    let (energy, peak) = top_band_impulse(f_max);
    Cell {
        gain,
        f_max,
        se_spikes: se.iter().sum(),
        se_bands: se.iter().filter(|&&v| v > 0).count(),
        silent_vowels: silent,
        distinct_pairs: distinct,
        vowel_active_bands: profiles
            .iter()
            .map(|p| p.iter().filter(|&&v| v > 0).count())
            .collect(),
        impulse_energy: energy,
        impulse_peak: peak,
    }
}

fn main() {
    println!("=== 蝸牛校正の掃引 (N_BANDS = {} 固定・FIRE_THRESHOLD = {}) ===", N_BANDS, FIRE_THRESHOLD);
    println!("ゲートは実測前に固定 (ファイル冒頭の doc コメント参照)");

    let cells: Vec<Cell> = GAIN_CANDIDATES
        .iter()
        .flat_map(|&g| F_MAX_CANDIDATES.iter().map(move |&f| (g, f)))
        .map(|(g, f)| measure(g, f))
        .collect();

    println!();
    println!("ゲイン F_MAX  se発火 se帯域  無音母音  区別ペア/10  母音ごとの発火帯域数   インパルスE");
    for c in cells.iter() {
        println!(
            "{:>4}x {:>5.0}  {:>6} {:>6}  {:>8}  {:>10}   {:<20} {:>10.0}",
            c.gain,
            c.f_max,
            c.se_spikes,
            c.se_bands,
            c.silent_vowels,
            c.distinct_pairs,
            format!("{:?}", c.vowel_active_bands),
            c.impulse_energy
        );
    }

    println!();
    println!("ゲイン F_MAX   G4(サ行)   G5'(母音)                  G6(数値)   総合");
    let mut passing: Vec<&Cell> = Vec::new();
    for c in cells.iter() {
        let g4 = c.se_spikes > BASE_SE_SPIKES && c.se_bands >= 2;
        let g5 = c.silent_vowels == 0 && c.distinct_pairs == 10;
        let g6 = c.impulse_energy > 0.0 && c.impulse_peak.is_finite() && c.impulse_peak < 1e9;
        let all = g4 && g5 && g6;
        println!(
            "{:>4}x {:>5.0}   {:<4}       {:<4} (無音{} 区別{}/10)      {:<4}       {}",
            c.gain,
            c.f_max,
            if g4 { "PASS" } else { "FAIL" },
            if g5 { "PASS" } else { "FAIL" },
            c.silent_vowels,
            c.distinct_pairs,
            if g6 { "PASS" } else { "FAIL" },
            if all { "採用候補" } else { "-" }
        );
        if all {
            passing.push(c);
        }
    }

    println!();
    // 採用規則: ゲイン最小を優先、次に F_MAX 最大
    let chosen = passing
        .iter()
        .min_by(|a, b| {
            a.gain
                .cmp(&b.gain)
                .then(b.f_max.partial_cmp(&a.f_max).unwrap())
        })
        .copied();
    match chosen {
        Some(c) => println!(
            "採用規則の結果: ゲイン {}x ・ F_MAX = {:.0} Hz\n  (se {}発/{}帯域 <- 基準 {}発/1帯域 ・ 母音の区別 {}/10 <- 基準 9/10 ・ 母音ごとの発火帯域数 {:?} <- 基準 [1,1,1,1,1])",
            c.gain, c.f_max, c.se_spikes, c.se_bands, BASE_SE_SPIKES,
            c.distinct_pairs, c.vowel_active_bands
        ),
        None => println!("採用規則の結果: **該当なし**。定数は変えない。"),
    }
}
