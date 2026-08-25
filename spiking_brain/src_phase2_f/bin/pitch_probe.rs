//! F0（音程）は表現されるか / 音程不変性は成り立つか (2026-08-25)
//!
//! ## 経緯
//!
//! ユーザーの問い:「低いあ～と高いあ～は同じあだけど音程が違うじゃない？
//! でも同じものとして認識できているでしょ？」
//!
//! 実装を調べたところ **F0 という概念自体が存在しなかった**:
//! `Vowel` に F0 のフィールドが無く、`synth_vowel` は
//! **フォルマント周波数そのものに純音を 3 本立てているだけ**だった。
//! `f0` / `fundamental` / `pitch` / `glottal` / 位相同期 / 周期性 は実装に一件も無い。
//!
//! `synth_vowel_f0` を新設した（声帯パルス列 → 全極フォルマント共鳴器）。
//! 倍音が F0 間隔で並び、共鳴器がその振幅を形づくる = 本物の音声と同じ構造。
//! **音程を変えても包絡は動かない**ので、同じ母音として読めるはず — を測る。
//!
//! ## ゲート（実測前に**完全に**指定する）
//!
//! 前回 G26 を「F0 の値も指標も沈黙の扱いも別母音側の F0 も未宣言」と
//! 独立監査に批判されたので、全部先に決める。
//!
//!   F0 値      : 100 / 150 / 200 / 250 Hz（実験者が決める）
//!   指標       : 発火帯域集合のコサイン（場所符号）
//!   沈黙の扱い : どれか 1 条件でも無音なら **FAIL**（沈黙で「似ている」にしない）
//!   別母音側   : 片方に固定せず**同じ F0 集合すべて**で比較する
//!
//!   G53 F0 が表現されるか: F0 を変えたら応答が変わる
//!       （正解 = F0 を変えたのは実験者。違う刺激には違って応じるべき）
//!   G54 音程不変性       : 「同一母音・F0 違い」の**最小**類似度 >
//!                          「別母音・同 F0」の**最大**類似度
//!       （正解 = どれが同じ母音でどれが違う母音かは実験者が決めた）
//!
//! **G54 が要**: 母音同定は絶対的な類似度でなく**順序**で決まる。
//!
//! CLI: pitch_probe

use spiking_brain::phase2_f::cochlea::{
    erb_q_factor, erb_spaced_freqs, BandpassBiquad, Cochlea, EnvelopeDetector, FireGenerator,
    ENV_LEAK_SHIFT, FIRE_REFRACTORY_STEPS, FIRE_THRESHOLD, F_MAX_HZ, F_MIN_HZ, N_BANDS,
    Q_SHARPENING, SAMPLES_PER_STEP,
};
use spiking_brain::phase2_f::phoneme_synth::SAMPLE_RATE_HZ;
use spiking_brain::phase2_f::phoneme_synth::{synth_vowel, synth_vowel_f0, vowels};

const VOWEL_MS: f64 = 170.0;
const F0S: [f64; 4] = [100.0, 150.0, 200.0, 250.0];

fn cochlea_q(q_mul: f64) -> Cochlea {
    let center_freqs = erb_spaced_freqs(F_MIN_HZ, F_MAX_HZ, N_BANDS);
    let bands = center_freqs
        .iter()
        .map(|&fc| BandpassBiquad::new(fc, erb_q_factor(fc) * q_mul, SAMPLE_RATE_HZ))
        .collect();
    let envelopes = (0..N_BANDS).map(|_| EnvelopeDetector::new(ENV_LEAK_SHIFT)).collect();
    let fire_gens = (0..N_BANDS)
        .map(|_| FireGenerator::new(FIRE_THRESHOLD, FIRE_REFRACTORY_STEPS))
        .collect();
    Cochlea { bands, envelopes, fire_gens, center_freqs, ..Cochlea::new() }
}

fn band_spikes_q(wave: &[i32], q_mul: f64) -> [u32; N_BANDS] {
    let mut c = cochlea_q(q_mul);
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

fn band_spikes(wave: &[i32]) -> [u32; N_BANDS] {
    let mut c = Cochlea::new();
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
    if na == 0.0 || nb == 0.0 { 0.0 } else { dot / (na * nb) }
}

fn nearest_band(freqs: &[f64], f_hz: f64) -> usize {
    freqs
        .iter()
        .enumerate()
        .min_by(|a, b| (a.1 - f_hz).abs().partial_cmp(&(b.1 - f_hz).abs()).unwrap())
        .unwrap()
        .0
}

fn main() {
    let vs = vowels();
    let names = ["a", "i", "u", "e", "o"];
    let freqs = Cochlea::new().center_freqs.clone();

    println!("=== F0（音程）は表現されるか / 音程不変性 ===");
    println!("F0 = {:?} Hz ・ 指標 = 発火帯域集合のコサイン", F0S);
    println!();

    // --- 対照: 純音3本版 (F0 の概念が無い旧実装) ---
    println!("--- 対照: 旧 synth_vowel（純音3本・F0 の概念なし）---");
    println!("音素  発火帯域数  被覆/3（F1,F2,F3 の最寄り帯域）");
    for (k, v) in vs.iter().enumerate() {
        let c = band_spikes(&synth_vowel(v, VOWEL_MS));
        let cov = (0..3).filter(|&f| c[nearest_band(&freqs, v.formants_hz[f])] > 0).count();
        println!("{:>4}  {:>10}  {:>3}/3", names[k],
                 c.iter().filter(|&&x| x > 0).count(), cov);
    }

    // --- F0 つき ---
    println!();
    println!("--- 新 synth_vowel_f0（声帯パルス列 → 全極共鳴器）---");
    println!("音素   F0    発火帯域数  被覆/3  総スパイク");
    let mut profiles: Vec<Vec<[u32; N_BANDS]>> = Vec::new();
    let mut any_silent = false;
    for (k, v) in vs.iter().enumerate() {
        let mut row = Vec::new();
        for &f0 in F0S.iter() {
            let c = band_spikes(&synth_vowel_f0(v, f0, VOWEL_MS));
            let nb = c.iter().filter(|&&x| x > 0).count();
            let cov = (0..3).filter(|&f| c[nearest_band(&freqs, v.formants_hz[f])] > 0).count();
            let total: u32 = c.iter().sum();
            if total == 0 {
                any_silent = true;
            }
            println!("{:>4}  {:>4.0}  {:>10}  {:>5}/3  {:>10}", names[k], f0, nb, cov, total);
            row.push(c);
        }
        profiles.push(row);
        let _ = k;
    }

    // --- G53: F0 を変えたら応答が変わるか ---
    println!();
    println!("--- G53 F0 が表現されるか ---");
    let mut g53 = true;
    for (k, row) in profiles.iter().enumerate() {
        let mut all_same = true;
        for i in 1..row.len() {
            if row[i] != row[0] {
                all_same = false;
            }
        }
        if all_same {
            g53 = false;
            println!("  /{}/: F0 を変えても応答が完全に同一 → **F0 が表現されていない**", names[k]);
        }
    }
    if g53 {
        println!("  全母音で F0 を変えると応答が変わる → PASS");
    }

    // --- G54: 音程不変性（順序） ---
    println!();
    println!("--- G54 音程不変性（順序）---");
    let mut min_same = f64::INFINITY;
    let mut min_desc = String::new();
    for (k, row) in profiles.iter().enumerate() {
        for i in 0..row.len() {
            for j in (i + 1)..row.len() {
                let c = cosine(&row[i], &row[j]);
                if c < min_same {
                    min_same = c;
                    min_desc = format!("/{}/ {:.0}Hz vs {:.0}Hz", names[k], F0S[i], F0S[j]);
                }
            }
        }
    }
    let mut max_diff = f64::NEG_INFINITY;
    let mut max_desc = String::new();
    for fi in 0..F0S.len() {
        for a in 0..vs.len() {
            for b in (a + 1)..vs.len() {
                let c = cosine(&profiles[a][fi], &profiles[b][fi]);
                if c > max_diff {
                    max_diff = c;
                    max_desc = format!("/{}/ vs /{}/ @{:.0}Hz", names[a], names[b], F0S[fi]);
                }
            }
        }
    }
    println!("同一母音・F0 違いの**最小**類似度: {:.4}  ({})", min_same, min_desc);
    println!("別母音・同 F0 の**最大**類似度: {:.4}  ({})", max_diff, max_desc);

    let g54 = !any_silent && min_same > max_diff;
    println!();
    println!("--- 判定 ---");
    println!("沈黙した条件: {}", if any_silent { "**あり → FAIL**" } else { "なし" });
    println!("G53 F0 が表現される : {}", if g53 { "PASS" } else { "**FAIL**" });
    println!("G54 音程不変性       : {}", if g54 { "**PASS**" } else { "**FAIL**" });
    if g54 {
        println!();
        println!("**「低いあ」と「高いあ」が、別の母音より互いに近い。**");
        println!("  = 音程を変えても同じ母音として読める表現になっている。");
    }

    // --- Q と音程不変性の関係 ---
    //
    // 仮説: Q が鋭いほど**倍音が分解**され、応答が包絡でなく倍音を追う。
    // F0 を変えると倍音の位置が全部変わるので、音程不変性が壊れる。
    // 本物の蝸牛では低次倍音だけが分解され、高次は非分解になって包絡を追う。
    // 正解の出どころ = どれが同じ母音でどれが違う母音かは実験者が決めた。
    println!();
    println!("--- Q と音程不変性の関係 ---");
    println!(" Q   同一母音F0違いの最小  別母音同F0の最大  G54  無音条件");
    for &q in [1.0f64, 2.0, 3.0, 4.0, 6.0].iter() {
        let mut profs: Vec<Vec<[u32; N_BANDS]>> = Vec::new();
        let mut silent = 0usize;
        for v in vs.iter() {
            let mut row = Vec::new();
            for &f0 in F0S.iter() {
                let c = band_spikes_q(&synth_vowel_f0(v, f0, VOWEL_MS), q);
                if c.iter().all(|&x| x == 0) {
                    silent += 1;
                }
                row.push(c);
            }
            profs.push(row);
        }
        let mut mn = f64::INFINITY;
        for row in profs.iter() {
            for i in 0..row.len() {
                for j in (i + 1)..row.len() {
                    mn = mn.min(cosine(&row[i], &row[j]));
                }
            }
        }
        let mut mx = f64::NEG_INFINITY;
        for fi in 0..F0S.len() {
            for a in 0..vs.len() {
                for b in (a + 1)..vs.len() {
                    mx = mx.max(cosine(&profs[a][fi], &profs[b][fi]));
                }
            }
        }
        let ok = silent == 0 && mn > mx;
        println!("{:>2}x  {:>20.4}  {:>16.4}  {:<4} {:>8}",
                 q, mn, mx, if ok { "PASS" } else { "FAIL" }, silent);
    }
    println!("(出荷値は Q ×{:.0})", Q_SHARPENING);
}
