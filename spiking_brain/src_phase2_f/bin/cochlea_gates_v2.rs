//! 蝸牛ゲート 計器修正版 (S5・2026-08-25)
//!
//! ## なぜ作り直したか
//!
//! 第 1 ラウンド (`fmax_sweep` / `cochlea_calibration`) のゲートは、
//! 独立レビューで**4 件の致命的欠陥**が指摘され、いずれも実測で裏づけられた:
//!
//! 1. G5 のコサインは母音プロファイルが 1-hot なので {0.0, 1.0} の二値しか取らない。
//!    「紛らわしさの度合い」ではなく「F1 が同じ ERB 帯域に落ちたか」の真偽値だった。
//! 2. その 1.0000 は e と o が同じ F1=500Hz を持つことから来る**恒等式**で、
//!    F_MAX を何にしても変わらない (棄却域が構造的に空)。
//! 3. G5 が 1.0 を下回る唯一の経路は**母音が無音になること**。
//!    つまり防ぐはずの劣化によって PASS 方向へ動く。
//! 4. 採用規則の反例: 基準値が厳密に 1.0000 でさえなければ、規則は
//!    F_MAX=7000 (母音 3/5 が無音) を自動採用していた。
//!    **安全側に倒れたのは幸運であって設計ではない。**
//! 5. G6 の `is_finite()` は i32→f64 キャストゆえ恒真 (棄却域ゼロ)。
//!    `energy > 0` は減衰しないリミットサイクルほど高得点になる。
//! 6. G4 の「正解」はモデル側の量だった。実験者が指定したのは /s/ = 3000-8000Hz という
//!    **帯域**であって、スパイクが何発出るべきかではない。
//!
//! ## この版のゲート
//!
//! すべて「正解が実験の設計側にある量」だけで書き、**マジックナンバーの閾値を持たない**
//! (比較形か、被覆率の満点かのどちらか)。
//!
//!   G4"  サ行     : /s/ に指定した帯域 [3000,8000]Hz に中心を持つ蝸牛帯域のうち
//!                   応答した割合。正解 = 実験者が指定した帯域そのもの。
//!   G5a" 被覆     : 各母音の指定 3 フォルマントの最寄り帯域が応答するか。
//!                   5 母音 × 3 = 15 点満点。正解 = 実験者が置いたフォルマント。
//!   G5b" 場所符号 : 母音 10 ペアで**発火帯域の集合**が相異なるか。
//!                   集合で見るので「同じ 1 本を 47 発 vs 44 発」の抜け道が塞がる。
//!   G6"  減衰     : インパルス応答の後半 max|y| < 前半 max|y|。
//!                   正解 = 安定な帯域通過フィルタは減衰する (フィルタ理論)。
//!
//! **8 シードで中央値と範囲を報告する** (第 1 ラウンドの単一シード 0xACE1 は
//! 5 条件中 4 条件で 8 シード中の最大値を与える上振れだったとレビューで判明)。
//!
//! 注意: これらのゲートは第 1 ラウンドの結果を見た**後**に設計されている。
//! したがって「事前登録」ではなく「計器修正後の再測」である。この区別は記録に残す。
//!
//! 既知の交絡 (レビュー指摘・修正せず報告する): 子音は S1 で RMS を固定して正規化した。
//! 総電力が固定なので、F_MAX を上げて /s/ 帯域を覆う帯域数が増えるほど
//! **1 帯域あたりの電力は薄まる**。G4" の被覆率はこの影響を受ける。
//!
//! CLI: cochlea_gates_v2

use spiking_brain::phase2_f::cochlea::{
    erb_q_factor, erb_spaced_freqs, BandpassBiquad, Cochlea, EnvelopeDetector, FireGenerator,
    ENV_LEAK_SHIFT, FIRE_REFRACTORY_STEPS, FIRE_THRESHOLD, F_MIN_HZ, N_BANDS, SAMPLES_PER_STEP,
};
use spiking_brain::phase2_f::phoneme_synth::{
    standard_syllables, synth_consonant_banded, synth_vowel, vowels, Consonant, LfsrNoise,
    SAMPLE_RATE_HZ,
};

const CONSONANT_MS: f64 = 30.0;
const VOWEL_MS: f64 = 170.0;
const SEEDS: [u16; 8] = [0xACE1, 0x1234, 0xBEEF, 0x7FFF, 0x0001, 0xF0F0, 0x5A5A, 0x2468];
const F_MAX_CANDIDATES: [f64; 6] = [4000.0, 5000.0, 6000.0, 7000.0, 7500.0, 8000.0];
const GAIN_CANDIDATES: [i32; 2] = [1, 2];

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

fn band_spikes(wave: &[i32], f_max: f64, gain: i32) -> [u32; N_BANDS] {
    let mut c = cochlea_with_fmax(f_max);
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

/// G4": /s/ の指定帯域に中心を持つ蝸牛帯域のうち応答した割合。
/// 併せて指定帯域**外**で応答した帯域数も返す (誤応答の可視化)。
fn fricative_coverage(f_max: f64, gain: i32, seed: u16) -> (usize, usize, usize) {
    let (lo, hi) = match standard_syllables()[3].consonant {
        Consonant::Fricative { freq_low, freq_high } => (freq_low, freq_high),
        _ => unreachable!("se は摩擦音のはず"),
    };
    let mut n = LfsrNoise::new(seed);
    let wave = synth_consonant_banded(standard_syllables()[3].consonant, CONSONANT_MS, &mut n);
    let counts = band_spikes(&wave, f_max, gain);
    let freqs = erb_spaced_freqs(F_MIN_HZ, f_max, N_BANDS);
    let in_band: Vec<usize> = (0..N_BANDS).filter(|&i| freqs[i] >= lo && freqs[i] <= hi).collect();
    let hit = in_band.iter().filter(|&&i| counts[i] > 0).count();
    let out_of_band = (0..N_BANDS)
        .filter(|&i| counts[i] > 0 && !(freqs[i] >= lo && freqs[i] <= hi))
        .count();
    (hit, in_band.len(), out_of_band)
}

/// G5a" 被覆 (15点満点) と G5b" 場所符号の相異ペア数 (10点満点)、無音母音数。
fn vowel_gates(f_max: f64, gain: i32) -> (usize, usize, usize, Vec<usize>) {
    let vs = vowels();
    let freqs = erb_spaced_freqs(F_MIN_HZ, f_max, N_BANDS);
    let profiles: Vec<[u32; N_BANDS]> = vs
        .iter()
        .map(|v| band_spikes(&synth_vowel(v, VOWEL_MS), f_max, gain))
        .collect();

    // G5a": 指定 3 フォルマントの最寄り帯域が応答するか
    let mut covered = 0usize;
    let mut per_vowel = Vec::new();
    for (k, v) in vs.iter().enumerate() {
        let mut c = 0usize;
        for f in 0..3 {
            let fhz = v.formants_hz[f];
            let (bi, _) = freqs
                .iter()
                .enumerate()
                .min_by(|a, b| (a.1 - fhz).abs().partial_cmp(&(b.1 - fhz).abs()).unwrap())
                .unwrap();
            if profiles[k][bi] > 0 {
                c += 1;
            }
        }
        covered += c;
        per_vowel.push(c);
    }

    // G5b": 発火帯域の**集合**が相異なるペア数
    let sets: Vec<Vec<usize>> = profiles
        .iter()
        .map(|p| (0..N_BANDS).filter(|&i| p[i] > 0).collect())
        .collect();
    let mut distinct = 0usize;
    for i in 0..sets.len() {
        for j in (i + 1)..sets.len() {
            if sets[i] != sets[j] {
                distinct += 1;
            }
        }
    }
    let silent = sets.iter().filter(|s| s.is_empty()).count();
    (covered, distinct, silent, per_vowel)
}

/// G6": インパルス応答が本当に減衰しきるか。
///
/// `late < early` では不十分だった: 整数 biquad は小振幅のリミットサイクル
/// (例 [-3,6,-8,8,-7,4,...] が延々続く) に入りうるが、それでも late < early は成立する。
/// **安定な整数 biquad はゼロ入力に対し厳密にゼロ状態へ落ちる**ので、
/// 末尾 100 サンプルが全部 0 かどうかで切る (フィルタ理論＋整数演算からくる正解)。
///
/// 戻り値: (死んだ帯域数, リミットサイクル帯域数, 内訳)
fn band_health(f_max: f64) -> (usize, usize, Vec<(usize, f64, i32, i32)>) {
    let freqs = erb_spaced_freqs(F_MIN_HZ, f_max, N_BANDS);
    let (mut dead, mut cyclic) = (0usize, 0usize);
    let mut detail = Vec::new();
    for (i, &fc) in freqs.iter().enumerate() {
        let mut bp = BandpassBiquad::new(fc, erb_q_factor(fc), SAMPLE_RATE_HZ);
        let mut early = 0i32;
        let mut tail = 0i32;
        for n in 0..2000 {
            let y = bp.process(if n == 0 { 10000 } else { 0 }).abs();
            if n < 500 {
                early = early.max(y);
            } else if n >= 1900 {
                tail = tail.max(y);
            }
        }
        if early == 0 {
            dead += 1;
            detail.push((i, fc, early, tail));
        } else if tail != 0 {
            cyclic += 1;
            detail.push((i, fc, early, tail));
        }
    }
    (dead, cyclic, detail)
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = v.len();
    if n % 2 == 1 { v[n / 2] } else { (v[n / 2 - 1] + v[n / 2]) / 2.0 }
}

fn dump_band_health(f_max: f64) {
    let (dead, cyclic, detail) = band_health(f_max);
    println!();
    println!("--- 帯域健全性の内訳 (F_MAX = {:.0}): 死 {} / リミットサイクル {} ---", f_max, dead, cyclic);
    for (i, fc, early, tail) in detail.iter() {
        println!(
            "  帯域{:>2} fc={:>7.1}Hz  前半max|y|={:>6}  末尾max|y|={:>5}  {}",
            i, fc, early, tail,
            if *early == 0 { "死 (応答なし)" } else { "リミットサイクル (減衰しきらない)" }
        );
    }
}

fn main() {
    dump_band_health(4000.0);
    println!("=== 蝸牛ゲート 計器修正版 (N_BANDS={} 固定・FIRE_THRESHOLD={}) ===", N_BANDS, FIRE_THRESHOLD);
    println!("※ 第1ラウンドの結果を見た後に設計した計器。事前登録ではなく「計器修正後の再測」。");
    println!("※ 既知の交絡: 子音は RMS 固定正規化なので、覆う帯域が増えるほど1帯域あたりの電力は薄まる。");

    println!();
    println!("ゲイン F_MAX  G4\"サ行被覆(中央値/範囲)  帯域外誤応答  G5a\"被覆/15  G5b\"相異/10  無音母音  母音ごと被覆  G6\"死/循環");
    struct R {
        gain: i32,
        f_max: f64,
        cov_med: f64,
        cov_lo: f64,
        cov_hi: f64,
        n_in_band: usize,
        oob_med: f64,
        formant_cov: usize,
        distinct: usize,
        silent: usize,
        per_vowel: Vec<usize>,
        dead_bands: usize,
        cyclic_bands: usize,
    }
    let mut rows: Vec<R> = Vec::new();
    for &gain in GAIN_CANDIDATES.iter() {
        for &f_max in F_MAX_CANDIDATES.iter() {
            let per_seed: Vec<(usize, usize, usize)> =
                SEEDS.iter().map(|&s| fricative_coverage(f_max, gain, s)).collect();
            let n_in_band = per_seed[0].1;
            let ratios: Vec<f64> = per_seed
                .iter()
                .map(|&(h, t, _)| if t == 0 { 0.0 } else { h as f64 / t as f64 })
                .collect();
            let oob: Vec<f64> = per_seed.iter().map(|&(_, _, o)| o as f64).collect();
            let (fc, dis, sil, pv) = vowel_gates(f_max, gain);
            rows.push(R {
                gain,
                f_max,
                cov_med: median(ratios.clone()),
                cov_lo: ratios.iter().cloned().fold(f64::INFINITY, f64::min),
                cov_hi: ratios.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
                n_in_band,
                oob_med: median(oob),
                formant_cov: fc,
                distinct: dis,
                silent: sil,
                per_vowel: pv,
                dead_bands: band_health(f_max).0,
                cyclic_bands: band_health(f_max).1,
            });
        }
    }
    for r in rows.iter() {
        println!(
            "{:>4}x {:>5.0}   {:>5.1}% [{:>5.1}-{:>5.1}] /{:>2}本   {:>10.1}  {:>10}  {:>10}  {:>8}  {:<16} {:>12}",
            r.gain,
            r.f_max,
            r.cov_med * 100.0,
            r.cov_lo * 100.0,
            r.cov_hi * 100.0,
            r.n_in_band,
            r.oob_med,
            format!("{}/15", r.formant_cov),
            format!("{}/10", r.distinct),
            r.silent,
            format!("{:?}", r.per_vowel),
            format!("死{}/循環{}", r.dead_bands, r.cyclic_bands)
        );
    }

    println!();
    println!("--- ゲート判定 ---");
    println!("ゲイン F_MAX   G4\"(被覆>0かつ帯域外誤応答なし)  G5a\"(15/15)  G5b\"(10/10)  G6\"(非減衰0)  総合");
    let mut passing: Vec<&R> = Vec::new();
    for r in rows.iter() {
        let g4 = r.cov_lo > 0.0 && r.oob_med == 0.0;
        let g5a = r.formant_cov == 15;
        let g5b = r.distinct == 10 && r.silent == 0;
        let g6 = r.dead_bands == 0 && r.cyclic_bands == 0;
        let all = g4 && g5a && g5b && g6;
        println!(
            "{:>4}x {:>5.0}   {:<4} ({:>5.1}%, 外{:.0})              {:<4} ({:>2}/15)  {:<4} ({:>2}/10)  {:<4} ({})   {}",
            r.gain,
            r.f_max,
            if g4 { "PASS" } else { "FAIL" },
            r.cov_med * 100.0,
            r.oob_med,
            if g5a { "PASS" } else { "FAIL" },
            r.formant_cov,
            if g5b { "PASS" } else { "FAIL" },
            r.distinct,
            if g6 { "PASS" } else { "FAIL" },
            format!("死{}/循環{}", r.dead_bands, r.cyclic_bands),
            if all { "採用候補" } else { "-" }
        );
        if all {
            passing.push(r);
        }
    }

    println!();
    if passing.is_empty() {
        println!("採用候補なし。**F_MAX_HZ は変えない。**");
    } else {
        // 採用規則: ゲイン最小 → 次に G4" 被覆が最大 (F_MAX の大きさ自体は目的でない)
        let best = passing
            .iter()
            .min_by(|a, b| {
                a.gain
                    .cmp(&b.gain)
                    .then(b.cov_med.partial_cmp(&a.cov_med).unwrap())
            })
            .unwrap();
        println!(
            "採用候補: ゲイン {}x ・ F_MAX = {:.0} Hz (サ行被覆 {:.1}%・フォルマント被覆 {}/15)",
            best.gain, best.f_max, best.cov_med * 100.0, best.formant_cov
        );
        println!("※ 採用規則は「F_MAX が大きいほど良い」を捨て、G4\" 被覆が最大のものを採る。");
        println!("   第1ラウンドの規則はどのゲートも測っていない軸 (F_MAX の大きさ) を最大化していた。");
    }
}
