//! 蝸牛の上限周波数 F_MAX は上げられるか (S2 検証・2026-08-25)
//!
//! 背景: S1 で /s/ (帯域 3000-8000Hz・中心 4899Hz) が `F_MAX_HZ = 4000` の外にあり、
//! 蝸牛がサ行をほぼ聞き取れないことが判明した (スパイク 2 発 / 1 帯域)。
//! ここでは**定数を書き換えずに** F_MAX を掃引し、上げてよいかを測る。
//!
//! `N_BANDS` は 40 に固定する (M1 の入力数が N_BANDS に縛られているため)。
//! したがって F_MAX を上げると、母音の住む 50-2500Hz の分解能は**下がる**。
//! これがトレードオフの本体であり、G5 はそれが壊れないかを見る。
//!
//! ゲート (実測前に固定):
//!   G4 サ行が聞こえるか : se の総スパイク > 基準(F_MAX=4000)の値 かつ 発火帯域数 >= 2
//!   G5 母音を壊さないか : 5 母音の相互コサイン類似度の最大値 <= 基準の値
//!      棄却域の事前検査 : 基準の最大類似度が 1.000 なら G5 に棄却域が無い → 「測れていない」
//!   G6 数値健全性       : 最上帯域 biquad のインパルス応答エネルギー > 0 かつ有界
//!
//! 採用規則 (先に宣言): G4・G5・G6 をすべて満たす F_MAX のうち最大を採る。
//! 無ければ案 A 不成立として報告し、F_MAX_HZ は変えない。
//!
//! 正解の出どころ: 帯域とフォルマントを指定したのは実験者。
//! 「音素として正しく聞こえるか」は測っていない (正解を持たないので計量できない)。
//!
//! CLI: fmax_sweep

use spiking_brain::phase2_f::cochlea::{
    erb_q_factor, erb_spaced_freqs, BandpassBiquad, Cochlea, EnvelopeDetector, FireGenerator,
    compress_sqrt, ENV_LEAK_SHIFT, FIRE_REFRACTORY_STEPS, FIRE_THRESHOLD, F_MIN_HZ, N_BANDS,
    SAMPLES_PER_STEP,
};
use spiking_brain::phase2_f::phoneme_synth::{
    standard_syllables, synth_consonant_banded, synth_vowel, vowels, LfsrNoise, SAMPLE_RATE_HZ,
};

const CONSONANT_MS: f64 = 30.0;
const VOWEL_MS: f64 = 170.0;
const SEED: u16 = 0xACE1;
const CANDIDATES: [f64; 6] = [4000.0, 5000.0, 6000.0, 7000.0, 7500.0, 8000.0];
/// 母音のフォルマントが収まる帯域 (診断表示用・ゲートではない)
const VOWEL_RANGE_HZ: f64 = 2500.0;

/// F_MAX だけ差し替えた蝸牛を作る (N_BANDS は 40 固定)。
/// `Cochlea` の全フィールドが pub なので、構造体リテラルで組める。
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
    Cochlea { bands, envelopes, fire_gens, center_freqs }
}

fn band_spikes(wave: &[i32], f_max: f64) -> [u32; N_BANDS] {
    let mut c = cochlea_with_fmax(f_max);
    let mut counts = [0u32; N_BANDS];
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
    counts
}

fn cosine(a: &[u32; N_BANDS], b: &[u32; N_BANDS]) -> f64 {
    let dot: f64 = (0..N_BANDS).map(|i| a[i] as f64 * b[i] as f64).sum();
    let na: f64 = (0..N_BANDS).map(|i| (a[i] as f64).powi(2)).sum::<f64>().sqrt();
    let nb: f64 = (0..N_BANDS).map(|i| (b[i] as f64).powi(2)).sum::<f64>().sqrt();
    if na == 0.0 || nb == 0.0 {
        0.0
    } else {
        dot / (na * nb)
    }
}

/// 最上帯域 biquad のインパルス応答: (エネルギー, 最大絶対値)
fn top_band_impulse(f_max: f64) -> (f64, f64) {
    let freqs = erb_spaced_freqs(F_MIN_HZ, f_max, N_BANDS);
    let fc = *freqs.last().unwrap();
    let mut bp = BandpassBiquad::new(fc, erb_q_factor(fc), SAMPLE_RATE_HZ);
    let mut energy = 0.0f64;
    let mut peak = 0.0f64;
    for i in 0..2000 {
        let x = if i == 0 { 10000 } else { 0 };
        let y = bp.process(x) as f64;
        energy += y.abs();
        peak = peak.max(y.abs());
    }
    (energy, peak)
}

struct Row {
    f_max: f64,
    top_fc: f64,
    bands_below_vowel_range: usize,
    se_spikes: u32,
    se_bands: usize,
    vowel_max_sim: f64,
    cons_max_sim: f64,
    impulse_energy: f64,
    impulse_peak: f64,
}

fn measure(f_max: f64) -> Row {
    // --- se (摩擦音単体) ---
    let se_consonant = standard_syllables()[3].consonant;
    let mut noise = LfsrNoise::new(SEED);
    let se_wave = synth_consonant_banded(se_consonant, CONSONANT_MS, &mut noise);
    let se = band_spikes(&se_wave, f_max);

    // --- 5 母音単体 (G5 の対象) ---
    let vowel_profiles: Vec<[u32; N_BANDS]> = vowels()
        .iter()
        .map(|v| band_spikes(&synth_vowel(v, VOWEL_MS), f_max))
        .collect();
    let mut vowel_max = 0.0f64;
    for i in 0..vowel_profiles.len() {
        for j in (i + 1)..vowel_profiles.len() {
            vowel_max = vowel_max.max(cosine(&vowel_profiles[i], &vowel_profiles[j]));
        }
    }

    // --- 5 子音単体 (診断・ゲートではない) ---
    let cons_profiles: Vec<[u32; N_BANDS]> = standard_syllables()
        .iter()
        .map(|s| {
            let mut n = LfsrNoise::new(SEED);
            band_spikes(&synth_consonant_banded(s.consonant, CONSONANT_MS, &mut n), f_max)
        })
        .collect();
    let mut cons_max = 0.0f64;
    for i in 0..cons_profiles.len() {
        for j in (i + 1)..cons_profiles.len() {
            cons_max = cons_max.max(cosine(&cons_profiles[i], &cons_profiles[j]));
        }
    }

    let freqs = erb_spaced_freqs(F_MIN_HZ, f_max, N_BANDS);
    let (energy, peak) = top_band_impulse(f_max);
    Row {
        f_max,
        top_fc: *freqs.last().unwrap(),
        bands_below_vowel_range: freqs.iter().filter(|&&f| f < VOWEL_RANGE_HZ).count(),
        se_spikes: se.iter().sum(),
        se_bands: se.iter().filter(|&&v| v > 0).count(),
        vowel_max_sim: vowel_max,
        cons_max_sim: cons_max,
        impulse_energy: energy,
        impulse_peak: peak,
    }
}

/// 母音が蝸牛でどう写っているかを丸ごと出す (計器の健全性診断)。
fn dump_vowels(f_max: f64) {
    println!();
    println!("--- 母音プロファイル診断 (F_MAX = {:.0}) ---", f_max);
    let vs = vowels();
    let profiles: Vec<[u32; N_BANDS]> = vs
        .iter()
        .map(|v| band_spikes(&synth_vowel(v, VOWEL_MS), f_max))
        .collect();
    let names = ["a", "i", "u", "e", "o"];
    println!("母音  F1     F2     総スパイク 発火帯域  発火した帯域の中心Hz");
    let freqs = erb_spaced_freqs(F_MIN_HZ, f_max, N_BANDS);
    for (k, p) in profiles.iter().enumerate() {
        let total: u32 = p.iter().sum();
        let active: Vec<String> = (0..N_BANDS)
            .filter(|&i| p[i] > 0)
            .map(|i| format!("{:.0}x{}", freqs[i], p[i]))
            .collect();
        println!(
            "{:>4}  {:>5.0}  {:>5.0}  {:>9}  {:>7}  {}",
            names[k], vs[k].formants_hz[0], vs[k].formants_hz[1],
            total, active.len(), active.join(" ")
        );
    }
    println!("コサイン類似度:");
    print!("     ");
    for n in names.iter() {
        print!("{:>8}", n);
    }
    println!();
    for i in 0..profiles.len() {
        print!("{:>4} ", names[i]);
        for j in 0..profiles.len() {
            print!("{:>8.3}", cosine(&profiles[i], &profiles[j]));
        }
        println!();
    }
}

/// 各帯域の「圧縮後包絡線の最大値」を閾値と並べて出す。
/// 発火の有無でなく**閾値からどれだけ離れているか**を見るための計器。
fn dump_band_levels(f_max: f64) {
    println!();
    println!("--- 帯域レベル診断 (F_MAX = {:.0} ・ FIRE_THRESHOLD = {}) ---", f_max, FIRE_THRESHOLD);
    let freqs = erb_spaced_freqs(F_MIN_HZ, f_max, N_BANDS);
    let vs = vowels();
    let names = ["a", "i", "u", "e", "o"];
    println!("母音  フォルマント(Hz×振幅)              最寄り帯域の圧縮包絡線ピーク / 閾値200");
    for (k, v) in vs.iter().enumerate() {
        let wave = synth_vowel(v, VOWEL_MS);
        // 圧縮後包絡線のピークを帯域ごとに追う
        let mut c = cochlea_with_fmax(f_max);
        let mut peaks = vec![0i32; N_BANDS];
        for chunk in wave.chunks(SAMPLES_PER_STEP) {
            if chunk.len() < SAMPLES_PER_STEP {
                break;
            }
            for &x in chunk {
                for ch in 0..N_BANDS {
                    let y = c.bands[ch].process(x);
                    c.envelopes[ch].process(y);
                }
            }
            for ch in 0..N_BANDS {
                let comp = compress_sqrt(c.envelopes[ch].env);
                if comp > peaks[ch] {
                    peaks[ch] = comp;
                }
            }
        }
        // 各フォルマントに最も近い帯域を引く
        let mut cells = Vec::new();
        for f in 0..3 {
            let fhz = v.formants_hz[f];
            let amp = v.amplitudes[f];
            let (bi, _) = freqs
                .iter()
                .enumerate()
                .min_by(|a, b| (a.1 - fhz).abs().partial_cmp(&(b.1 - fhz).abs()).unwrap())
                .unwrap();
            let pk = peaks[bi];
            cells.push(format!(
                "F{}={:.0}Hz×{} -> 帯域{:.0}Hz ピーク{:>4}{}",
                f + 1, fhz, amp, freqs[bi], pk,
                if pk >= FIRE_THRESHOLD { " ○発火" } else { " ×無音" }
            ));
        }
        println!("{:>4}  {}", names[k], cells.join("  |  "));
    }
    let top: Vec<String> = (0..N_BANDS).map(|i| format!("{:.0}", freqs[i])).collect();
    println!("帯域中心(Hz): {}", top.join(" "));
}

fn main() {
    dump_band_levels(4000.0);
    for f in [4000.0, 6000.0, 7000.0] {
        dump_vowels(f);
    }
    println!();
    println!("=== F_MAX 掃引 (N_BANDS = {} 固定) ===", N_BANDS);
    let rows: Vec<Row> = CANDIDATES.iter().map(|&f| measure(f)).collect();
    let base = &rows[0];

    println!();
    println!("F_MAX  最上帯域  母音域帯域数  se発火 se帯域  母音最大類似  子音最大類似  インパルス(E/peak)");
    for r in rows.iter() {
        println!(
            "{:>5.0}  {:>8.1}  {:>12}  {:>6} {:>6}  {:>12.4}  {:>12.4}  {:>9.0} / {:.0}",
            r.f_max,
            r.top_fc,
            r.bands_below_vowel_range,
            r.se_spikes,
            r.se_bands,
            r.vowel_max_sim,
            r.cons_max_sim,
            r.impulse_energy,
            r.impulse_peak
        );
    }

    // --- G5 の棄却域 事前検査 ---
    let g5_measurable = base.vowel_max_sim < 1.0 - 1e-9;
    println!();
    println!(
        "[棄却域の事前検査] 基準(F_MAX=4000)の母音最大類似度 = {:.6} -> G5 は{}",
        base.vowel_max_sim,
        if g5_measurable {
            "測れる (棄却域あり)"
        } else {
            "**測れていない** (棄却域が空)"
        }
    );

    println!();
    println!("F_MAX     G4(サ行)           G5(母音)           G6(数値)          総合");
    let mut accepted: Option<&Row> = None;
    for r in rows.iter().skip(1) {
        let g4 = r.se_spikes > base.se_spikes && r.se_bands >= 2;
        let g5 = g5_measurable && r.vowel_max_sim <= base.vowel_max_sim + 1e-9;
        let g6 = r.impulse_energy > 0.0 && r.impulse_peak.is_finite() && r.impulse_peak < 1e9;
        let all = g4 && g5 && g6;
        println!(
            "{:>5.0}     {:<4} ({:>3}発/{:>2}帯)  {:<4} ({:.4})      {:<4} (E={:>7.0})  {}",
            r.f_max,
            if g4 { "PASS" } else { "FAIL" },
            r.se_spikes,
            r.se_bands,
            if g5 { "PASS" } else { "FAIL" },
            r.vowel_max_sim,
            if g6 { "PASS" } else { "FAIL" },
            r.impulse_energy,
            if all { "採用候補" } else { "-" }
        );
        if all {
            accepted = Some(r); // 候補は昇順なので、最後に残ったものが最大
        }
    }

    println!();
    match accepted {
        Some(r) => println!(
            "採用規則の結果: F_MAX = {:.0} Hz\n  (se {}発/{}帯域 <- 基準 {}発/{}帯域 ・ 母音最大類似 {:.4} <- 基準 {:.4} ・ 母音域の帯域数 {} <- 基準 {})",
            r.f_max,
            r.se_spikes,
            r.se_bands,
            base.se_spikes,
            base.se_bands,
            r.vowel_max_sim,
            base.vowel_max_sim,
            r.bands_below_vowel_range,
            base.bands_below_vowel_range
        ),
        None => println!("採用規則の結果: **該当なし — 案 A 不成立**。F_MAX_HZ は変えない。"),
    }
}
