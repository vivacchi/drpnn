//! 子音は同じゲートを通るか (2026-08-25・母音の採用候補を子音で検証)
//!
//! ## 経緯
//!
//! 母音側で「Q×3・FIRE_THRESHOLD 160・刺激の絶対スケール ×4」が
//! 5 ゲート (被覆15/15・場所符号10/10・無音0・穴0・非減衰0) を通った。
//! **同じ設定で子音が壊れないかを確認する。**
//!
//! ## 構造的に到達不能な目標を先に潰す
//!
//! `se` の指定帯域は 3000-8000Hz だが蝸牛の上限は F_MAX_HZ = 4000Hz。
//! 指定帯域全体を被覆の分母にすると se は**原理的に満点を取れない**
//! (棄却域が空になるのと逆の罠)。よって:
//!   **被覆は「指定帯域 ∩ 蝸牛の可測域」に対して測り、
//!   表現できない範囲は別途報告する。**
//!
//! ## ゲート (実測前に固定)
//!
//!   G36 被覆(recall)    : 指定帯域 ∩ 可測域 に中心を持つ帯域のうち応答した割合。
//!                         鼻音は f1/f2 の最寄り帯域が応答するか。
//!                         正解の出どころ = 帯域を指定したのは実験者。
//!   G37 精度(precision) : 発火帯域のうち指定帯域 (±TOLERANCE_BANDS) に入る割合。
//!   G38 場所符号の相異  : 5 子音の発火帯域集合が全 10 対で相異なるか。
//!   G39 無音の子音なし  : 5 子音すべてが発火する。
//!
//! 従来合成 (`synth_consonant`・帯域指定を無視する既定) と
//! 帯域版 (`synth_consonant_banded`・S1) の**両方**を測る。
//! 前者は指定を使っていないので、被覆はそのまま S1 の欠陥の大きさになる。
//!
//! ⚠️ **この probe の精度は「±1 帯域」基準**（旧定義）。
//! 1 本の幅は帯域数で変わるので、**帯域数をまたいだ比較には使えない**。
//! 帯域数を含む比較・設計点の決定には `m0_design`（ERB 基準）を使うこと。
//! 単一の N_BANDS 内での比較には引き続き有効。
//!
//! CLI: consonant_gate

use spiking_brain::phase2_f::cochlea::{
    erb_q_factor, erb_spaced_freqs, BandpassBiquad, Cochlea, EnvelopeDetector, FireGenerator,
    ENV_LEAK_SHIFT, FIRE_REFRACTORY_STEPS, F_MAX_HZ, F_MIN_HZ, N_BANDS, SAMPLES_PER_STEP,
};
use spiking_brain::phase2_f::phoneme_synth::{F0_DEFAULT_HZ, 
    standard_syllables, synth_consonant, synth_consonant_banded, Consonant, LfsrNoise,
    SAMPLE_RATE_HZ,
};

const CONSONANT_MS: f64 = 30.0;
const TOLERANCE_BANDS: usize = 1;
const SEEDS: [u16; 8] = [0xACE1, 0x1234, 0xBEEF, 0x7FFF, 0x0001, 0xF0F0, 0x5A5A, 0x2468];

/// (ラベル, Q倍率, 閾値, 提示ゲイン)
const SETTINGS: [(&str, f64, i32, i32); 7] = [
    ("旧構成 (Q×1/閾200) ※刺激は現行", 1.0, 200, 1),
    ("**出荷構成** (Q×3/閾160/×1)", 3.0, 160, 1),
    ("Q×3/閾160/ ×2", 3.0, 160, 2),
    ("Q×3/閾160/ ×3", 3.0, 160, 3),
    ("Q×3/閾160/ ×4 (母音の採用候補)", 3.0, 160, 4),
    ("Q×3/閾120/ ×1", 3.0, 120, 1),
    ("Q×3/閾120/ ×2", 3.0, 120, 2),
];

fn cochlea_of(threshold: i32, q_mul: f64) -> Cochlea {
    let center_freqs = erb_spaced_freqs(F_MIN_HZ, F_MAX_HZ, N_BANDS);
    let bands = center_freqs
        .iter()
        .map(|&fc| BandpassBiquad::new(fc, erb_q_factor(fc) * q_mul, SAMPLE_RATE_HZ))
        .collect();
    let envelopes = (0..N_BANDS).map(|_| EnvelopeDetector::new(ENV_LEAK_SHIFT)).collect();
    let fire_gens = (0..N_BANDS)
        .map(|_| FireGenerator::new(threshold, FIRE_REFRACTORY_STEPS))
        .collect();
    Cochlea { bands, envelopes, fire_gens, center_freqs, ..Cochlea::new() }
}

fn band_spikes(wave: &[i32], threshold: i32, gain: i32, q_mul: f64) -> [u32; N_BANDS] {
    let mut c = cochlea_of(threshold, q_mul);
    let mut counts = [0u32; N_BANDS];
    for chunk in wave.chunks(SAMPLES_PER_STEP) {
        if chunk.len() < SAMPLES_PER_STEP {
            break;
        }
        let amp: Vec<i32> = chunk.iter().map(|&x| x.saturating_mul(gain)).collect();
        let out = c.process_step(&amp);
        for ch in 0..N_BANDS {
            if out[ch] != 0 {
                counts[ch] += 1;
            }
        }
    }
    counts
}

fn nearest_band(freqs: &[f64], f_hz: f64) -> usize {
    freqs
        .iter()
        .enumerate()
        .min_by(|a, b| (a.1 - f_hz).abs().partial_cmp(&(b.1 - f_hz).abs()).unwrap())
        .unwrap()
        .0
}

/// 子音の「指定帯域」に対応する蝸牛帯域の集合と、
/// 指定帯域のうち蝸牛が表現できない割合を返す。
fn target_bands(c: Consonant, freqs: &[f64]) -> (Vec<usize>, f64) {
    match c {
        Consonant::Plosive { burst_freq_low: lo, burst_freq_high: hi, .. }
        | Consonant::Fricative { freq_low: lo, freq_high: hi, .. } => {
            let inside: Vec<usize> = (0..N_BANDS)
                .filter(|&i| freqs[i] >= lo && freqs[i] <= hi)
                .collect();
            // 指定帯域のうち蝸牛の可測域 [F_MIN, F_MAX] の外にある割合
            let unrepresentable = ((hi - F_MAX_HZ).max(0.0) + (F_MIN_HZ - lo).max(0.0)) / (hi - lo);
            (inside, unrepresentable)
        }
        Consonant::Nasal { f1, f2, .. } => {
            let mut v = vec![nearest_band(freqs, f1), nearest_band(freqs, f2)];
            v.sort_unstable();
            v.dedup();
            (v, 0.0)
        }
        // 2026-08-26 に追加された variant。この probe は
        // standard_syllables() の 5 音素専用なので到達しない。
        Consonant::None | Consonant::Approximant { .. } | Consonant::Affricate { .. } => {
            (Vec::new(), 0.0)
        }
    }
}

fn widen(target: &[usize]) -> std::collections::HashSet<usize> {
    let mut set = std::collections::HashSet::new();
    for &b in target.iter() {
        for d in 0..=TOLERANCE_BANDS {
            if b >= d {
                set.insert(b - d);
            }
            if b + d < N_BANDS {
                set.insert(b + d);
            }
        }
    }
    set
}

fn main() {
    let freqs = erb_spaced_freqs(F_MIN_HZ, F_MAX_HZ, N_BANDS);
    let syls = standard_syllables();

    println!("=== 子音は同じゲートを通るか ===");
    println!("被覆の分母 = 指定帯域 ∩ 蝸牛の可測域 [{:.0}, {:.0}] Hz", F_MIN_HZ, F_MAX_HZ);
    println!("8 シードの中央値で報告 (単一シードの上振れを避ける)");
    println!();

    // 指定帯域のうち蝸牛が表現できない割合を先に出す
    println!("--- 各子音の指定帯域と、蝸牛が表現できない割合 ---");
    for s in syls.iter() {
        let (t, unrep) = target_bands(s.consonant, &freqs);
        let spec = match s.consonant {
            Consonant::Plosive { burst_freq_low: lo, burst_freq_high: hi, .. }
            | Consonant::Fricative { freq_low: lo, freq_high: hi, .. } => {
                format!("{:.0}-{:.0}Hz", lo, hi)
            }
            Consonant::Nasal { f1, f2, .. } => format!("f1={:.0} f2={:.0}Hz", f1, f2),
            Consonant::None => "なし".to_string(),
            Consonant::Approximant { f1, f2 } => format!("接近音 f1={:.0} f2={:.0}Hz", f1, f2),
            Consonant::Affricate { burst_freq_low, burst_freq_high, fric_freq_low, fric_freq_high } => {
                format!("破擦 {:.0}-{:.0}/{:.0}-{:.0}Hz",
                        burst_freq_low, burst_freq_high, fric_freq_low, fric_freq_high)
            }
        };
        println!(
            "  {:>2}: 指定 {:<16} 可測域内の帯域 {:>2} 本  表現できない割合 {:>5.1}%",
            s.label, spec, t.len(), unrep * 100.0
        );
    }

    for (banded, arm) in [(false, "従来合成 (帯域指定を無視・既定)"), (true, "帯域版 (S1)")] {
        println!();
        println!("=== {} ===", arm);
        println!("{:<32} G36被覆   G37精度  G38相異/10 無音  発火帯域計  波形ピーク  子音ごと帯域数", "設定");
        for (label, q, thr, gain) in SETTINGS.iter() {
            // シードごとに測り中央値を取る
            let mut recalls = Vec::new();
            let mut precs = Vec::new();
            let mut distincts = Vec::new();
            let mut silents = Vec::new();
            let mut totals = Vec::new();
            let mut per_cons = Vec::new();
            for &seed in SEEDS.iter() {
                let profiles: Vec<[u32; N_BANDS]> = syls
                    .iter()
                    .map(|s| {
                        let mut n = LfsrNoise::new(seed);
                        let w = if banded {
                            synth_consonant_banded(s.consonant, CONSONANT_MS, F0_DEFAULT_HZ, &mut n)
                        } else {
                            synth_consonant(s.consonant, CONSONANT_MS, &mut n)
                        };
                        band_spikes(&w, *thr, *gain, *q)
                    })
                    .collect();

                let mut hit = 0usize;
                let mut denom = 0usize;
                let mut phit = 0usize;
                let mut ptotal = 0usize;
                for (k, s) in syls.iter().enumerate() {
                    let (t, _) = target_bands(s.consonant, &freqs);
                    denom += t.len();
                    hit += t.iter().filter(|&&b| profiles[k][b] > 0).count();
                    let wide = widen(&t);
                    for ch in 0..N_BANDS {
                        if profiles[k][ch] > 0 {
                            ptotal += 1;
                            if wide.contains(&ch) {
                                phit += 1;
                            }
                        }
                    }
                }
                let sets: Vec<Vec<usize>> = profiles
                    .iter()
                    .map(|p| (0..N_BANDS).filter(|&i| p[i] > 0).collect())
                    .collect();
                let mut d = 0usize;
                for i in 0..sets.len() {
                    for j in (i + 1)..sets.len() {
                        if sets[i] != sets[j] {
                            d += 1;
                        }
                    }
                }
                recalls.push(if denom == 0 { 0.0 } else { hit as f64 / denom as f64 });
                precs.push(if ptotal == 0 { 0.0 } else { phit as f64 / ptotal as f64 });
                distincts.push(d as f64);
                silents.push(sets.iter().filter(|s| s.is_empty()).count() as f64);
                totals.push(ptotal as f64);
                per_cons.push(sets.iter().map(|s| s.len()).collect::<Vec<usize>>());
            }
            let med = |mut v: Vec<f64>| {
                v.sort_by(|a, b| a.partial_cmp(b).unwrap());
                v[v.len() / 2]
            };
            // 波形ピーク (i16 想定を外れていないかの確認)
            let peak = syls
                .iter()
                .map(|s2| {
                    let mut n = LfsrNoise::new(SEEDS[0]);
                    let w = if banded {
                        synth_consonant_banded(s2.consonant, CONSONANT_MS, F0_DEFAULT_HZ, &mut n)
                    } else {
                        synth_consonant(s2.consonant, CONSONANT_MS, &mut n)
                    };
                    w.iter().map(|v| v.abs()).max().unwrap_or(0) * gain
                })
                .max()
                .unwrap_or(0);
            println!(
                "{:<32} {:>6.1}%  {:>6.1}%  {:>8.0}  {:>4.0}  {:>9.0}  ピーク{:>6}  {:?}",
                label,
                med(recalls) * 100.0,
                med(precs) * 100.0,
                med(distincts),
                med(silents),
                med(totals),
                peak,
                per_cons[0]
            );
        }
    }

    println!();
    println!("--- 読み方 ---");
    println!("従来合成は帯域指定を使っていないので、G36 被覆はそのまま S1 の欠陥の大きさ。");
    println!("母音の採用候補 (Q×3/閾160/×4) で子音の精度が崩れるなら、");
    println!("**母音と子音で必要な絶対スケールが違う**ということ (別々に校正する必要がある)。");
}
