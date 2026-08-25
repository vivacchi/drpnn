//! M0 の設計点を詰める — 帯域数まで含めた掃引 (2026-08-25)
//!
//! ## なぜ
//!
//! ユーザー指示「M0 をちゃんと詰めよう」。
//! `formant_probe` / `band_coverage` で分かったこと:
//!   - 現行 M0 は「最大精度・最小被覆」の端 (被覆 5/15・母音は 1 帯域しか鳴らない)
//!   - 足りないのは選択性 Q (設計書 §1.5 の OHC・未実装)
//!   - Q を上げるとスペクトルに穴が空く
//!   - 穴の残差は幾何でなく**量子化**だった → biquad の状態を Q8 に高精度化して解消
//!   - 解消した結果、穴は**純粋な幾何の問題**に戻った = **帯域数で買える**
//!
//! そこで `N_BANDS` まで含めて掃引し、全ゲートを通す設計点を出す。
//! `N_BANDS` は配列サイズなので、Vec ベースの可変版で測る。
//!
//! ## ゲート (実測前に固定)
//!
//! 母音と子音の**両方**に、正解が完全に実験者側にある量だけで:
//!   G29 母音の被覆 15/15 / G30 母音の場所符号 10/10 / G31 無音母音なし
//!   G36 子音の被覆 (指定帯域 ∩ 可測域) / G38 子音の場所符号 10/10 / G39 無音の子音なし
//!   G32/G37 精度 (発火帯域のうち指定周波数から **1 ERB 以内**の割合)
//!       ※ 旧版は「±1 帯域」で測っていたが、1 本の幅は帯域数で変わるので
//!         帯域数をまたいだ比較にならなかった (2026-08-25 訂正)。
//!   G34' 穴 = 0 (純音を 3 レベル・**最弱フォルマント基準**で掃引した最悪値)
//!   G35 非減衰帯域 = 0
//!
//! **採用規則 (先に宣言)**: 全ゲート通過のうち**母音の精度が最大**。
//! ただし **N_BANDS ごとの最良精度の表も出す** —
//! 帯域数は M1 への波及があるので、膝の位置は人間が判断するため。
//!
//! ## 高速化 (数学的に妥当)
//!
//! 穴の検査で純音 f を入れたとき、遠い帯域が近い帯域より大きく応答することは
//! バンドパス列では起こりえない。よって**最寄り 7 本だけ**を回す。
//!
//! CLI: m0_design

use spiking_brain::phase2_f::cochlea::{
    compress_sqrt, erb_q_factor, erb_spaced_freqs, BandpassBiquad, EnvelopeDetector,
    FireGenerator, ENV_LEAK_SHIFT, FIRE_REFRACTORY_STEPS, F_MAX_HZ, F_MIN_HZ, SAMPLES_PER_STEP,
};
use spiking_brain::phase2_f::phoneme_synth::{
    freq_to_phase_step, sin_lookup, standard_syllables, synth_consonant_banded, synth_vowel,
    vowels, Consonant, LfsrNoise, SAMPLE_RATE_HZ,
};

const VOWEL_MS: f64 = 170.0;
const CONSONANT_MS: f64 = 30.0;
const PROBE_MS: f64 = 170.0;
const N_PROBE: usize = 120;
/// 「近傍」の定義は ERB 単位 (帯域数に依存しない)。`near_freqs` を参照。
const NEAREST_FOR_HOLES: usize = 7;
const SEED: u16 = 0xACE1;

const BAND_COUNTS: [usize; 5] = [40, 60, 80, 120, 160];
const Q_MULS: [f64; 5] = [1.0, 2.0, 3.0, 4.0, 6.0];
const THRESHOLDS: [i32; 6] = [80, 120, 160, 200, 240, 280];

struct Bank {
    bands: Vec<BandpassBiquad>,
    envs: Vec<EnvelopeDetector>,
    fires: Vec<FireGenerator>,
}

fn make_bank(freqs: &[f64], q_mul: f64, threshold: i32) -> Bank {
    Bank {
        bands: freqs
            .iter()
            .map(|&fc| BandpassBiquad::new(fc, erb_q_factor(fc) * q_mul, SAMPLE_RATE_HZ))
            .collect(),
        envs: (0..freqs.len()).map(|_| EnvelopeDetector::new(ENV_LEAK_SHIFT)).collect(),
        fires: (0..freqs.len())
            .map(|_| FireGenerator::new(threshold, FIRE_REFRACTORY_STEPS))
            .collect(),
    }
}

/// 波形を通し、帯域ごとに発火したかを返す。
fn fired_bands(wave: &[i32], freqs: &[f64], q_mul: f64, threshold: i32) -> Vec<bool> {
    let n = freqs.len();
    let mut bank = make_bank(freqs, q_mul, threshold);
    let mut out = vec![false; n];
    for chunk in wave.chunks(SAMPLES_PER_STEP) {
        if chunk.len() < SAMPLES_PER_STEP {
            break;
        }
        for &x in chunk {
            for ch in 0..n {
                let y = bank.bands[ch].process(x);
                bank.envs[ch].process(y);
            }
        }
        for ch in 0..n {
            if bank.fires[ch].process(compress_sqrt(bank.envs[ch].env)) {
                out[ch] = true;
            }
        }
    }
    out
}

fn nearest_band(freqs: &[f64], f_hz: f64) -> usize {
    freqs
        .iter()
        .enumerate()
        .min_by(|a, b| (a.1 - f_hz).abs().partial_cmp(&(b.1 - f_hz).abs()).unwrap())
        .unwrap()
        .0
}

/// 純音 f を最寄り数本だけに通し、1 本でも発火するか。
fn tone_heard(f_hz: f64, amp: i32, freqs: &[f64], q_mul: f64, threshold: i32) -> bool {
    let c = nearest_band(freqs, f_hz);
    let lo = c.saturating_sub(NEAREST_FOR_HOLES / 2);
    let hi = (c + NEAREST_FOR_HOLES / 2 + 1).min(freqs.len());
    let sub: Vec<f64> = freqs[lo..hi].to_vec();
    let mut bank = make_bank(&sub, q_mul, threshold);
    let n = sub.len();
    let n_samples = (PROBE_MS * SAMPLE_RATE_HZ / 1000.0) as usize;
    let step = freq_to_phase_step(f_hz);
    let mut phase = 0u32;
    let mut i = 0usize;
    while i + SAMPLES_PER_STEP <= n_samples {
        for _ in 0..SAMPLES_PER_STEP {
            let x = (sin_lookup(phase) * amp) >> 14;
            phase = phase.wrapping_add(step);
            for ch in 0..n {
                let y = bank.bands[ch].process(x);
                bank.envs[ch].process(y);
            }
        }
        for ch in 0..n {
            if bank.fires[ch].process(compress_sqrt(bank.envs[ch].env)) {
                return true;
            }
        }
        i += SAMPLES_PER_STEP;
    }
    false
}

fn holes(freqs: &[f64], q_mul: f64, threshold: i32, levels: &[i32]) -> usize {
    let mut worst = 0usize;
    for &amp in levels.iter() {
        let mut dead = 0usize;
        for k in 0..N_PROBE {
            let t = k as f64 / (N_PROBE - 1) as f64;
            let f = F_MIN_HZ * (F_MAX_HZ / F_MIN_HZ).powf(t);
            if !tone_heard(f, amp, freqs, q_mul, threshold) {
                dead += 1;
            }
        }
        worst = worst.max(dead);
    }
    worst
}

fn unstable(freqs: &[f64], q_mul: f64) -> usize {
    let mut bad = 0usize;
    for &fc in freqs.iter() {
        let mut bp = BandpassBiquad::new(fc, erb_q_factor(fc) * q_mul, SAMPLE_RATE_HZ);
        let mut tail = 0i32;
        for n in 0..4000 {
            let y = bp.process(if n == 0 { 10000 } else { 0 }).abs();
            if n >= 3900 {
                tail = tail.max(y);
            }
        }
        if tail != 0 {
            bad += 1;
        }
    }
    bad
}

/// 「指定周波数の近傍」を**帯域数に依存しない単位**で定義する (2026-08-25 訂正)。
///
/// 旧実装は「±TOLERANCE_BANDS 本」だったが、1 本の幅は帯域数で変わる
/// (N=40 で ±0.63 ERB、N=160 で ±0.16 ERB)。
/// **帯域数をまたいだ精度の比較になっていなかった。**
/// 蝸牛の自然な帯域幅単位である **ERB** で測る:
/// 中心周波数が指定周波数から 1 ERB 以内なら「近傍」。
fn near_freqs(freqs: &[f64], targets: &[f64]) -> std::collections::HashSet<usize> {
    let mut set = std::collections::HashSet::new();
    for &tf in targets {
        // ERB(f) = f / Q_erb(f)
        let erb = tf / erb_q_factor(tf);
        for (i, &fc) in freqs.iter().enumerate() {
            if (fc - tf).abs() <= erb {
                set.insert(i);
            }
        }
    }
    set
}

struct R {
    n_bands: usize,
    q: f64,
    thr: i32,
    v_recall: usize,
    v_distinct: usize,
    v_silent: usize,
    v_prec: f64,
    c_recall: f64,
    c_distinct: usize,
    c_silent: usize,
    c_prec: f64,
    holes: usize,
    unstable: usize,
}

/// 母音・子音の被覆/精度/場所符号を測る (穴は別・重いので後で)
fn measure_cheap(freqs: &[f64], q: f64, thr: i32) -> R {
    let n = freqs.len();
    let vs = vowels();
    let syls = standard_syllables();

    // --- 母音 ---
    let vp: Vec<Vec<bool>> = vs
        .iter()
        .map(|v| fired_bands(&synth_vowel(v, VOWEL_MS), freqs, q, thr))
        .collect();
    let mut v_recall = 0usize;
    let (mut vh, mut vt) = (0usize, 0usize);
    for (k, v) in vs.iter().enumerate() {
        let tgt: Vec<usize> = (0..3).map(|f| nearest_band(freqs, v.formants_hz[f])).collect();
        v_recall += tgt.iter().filter(|&&b| vp[k][b]).count();
        let wide = near_freqs(freqs, &v.formants_hz);
        for ch in 0..n {
            if vp[k][ch] {
                vt += 1;
                if wide.contains(&ch) {
                    vh += 1;
                }
            }
        }
    }
    let v_sets: Vec<Vec<usize>> =
        vp.iter().map(|p| (0..n).filter(|&i| p[i]).collect()).collect();
    let mut v_distinct = 0usize;
    for i in 0..v_sets.len() {
        for j in (i + 1)..v_sets.len() {
            if v_sets[i] != v_sets[j] {
                v_distinct += 1;
            }
        }
    }

    // --- 子音 ---
    let cp: Vec<Vec<bool>> = syls
        .iter()
        .map(|s| {
            let mut no = LfsrNoise::new(SEED);
            fired_bands(&synth_consonant_banded(s.consonant, CONSONANT_MS, &mut no), freqs, q, thr)
        })
        .collect();
    let (mut ch_hit, mut ch_den, mut cph, mut cpt) = (0usize, 0usize, 0usize, 0usize);
    for (k, s) in syls.iter().enumerate() {
        let tgt: Vec<usize> = match s.consonant {
            Consonant::Plosive { burst_freq_low: lo, burst_freq_high: hi }
            | Consonant::Fricative { freq_low: lo, freq_high: hi } => {
                (0..n).filter(|&i| freqs[i] >= lo && freqs[i] <= hi).collect()
            }
            Consonant::Nasal { f1, f2 } => {
                let mut v = vec![nearest_band(freqs, f1), nearest_band(freqs, f2)];
                v.sort_unstable();
                v.dedup();
                v
            }
            Consonant::None => Vec::new(),
        };
        ch_den += tgt.len();
        ch_hit += tgt.iter().filter(|&&b| cp[k][b]).count();
        // 指定帯域 (または鼻音のフォルマント) の周波数を近傍判定の基準にする
        let tgt_freqs: Vec<f64> = tgt.iter().map(|&b| freqs[b]).collect();
        let wide = near_freqs(freqs, &tgt_freqs);
        for c in 0..n {
            if cp[k][c] {
                cpt += 1;
                if wide.contains(&c) {
                    cph += 1;
                }
            }
        }
    }
    let c_sets: Vec<Vec<usize>> =
        cp.iter().map(|p| (0..n).filter(|&i| p[i]).collect()).collect();
    let mut c_distinct = 0usize;
    for i in 0..c_sets.len() {
        for j in (i + 1)..c_sets.len() {
            if c_sets[i] != c_sets[j] {
                c_distinct += 1;
            }
        }
    }

    R {
        n_bands: n,
        q,
        thr,
        v_recall,
        v_distinct,
        v_silent: v_sets.iter().filter(|s| s.is_empty()).count(),
        v_prec: if vt == 0 { 0.0 } else { vh as f64 / vt as f64 },
        c_recall: if ch_den == 0 { 0.0 } else { ch_hit as f64 / ch_den as f64 },
        c_distinct,
        c_silent: c_sets.iter().filter(|s| s.is_empty()).count(),
        c_prec: if cpt == 0 { 0.0 } else { cph as f64 / cpt as f64 },
        holes: usize::MAX, // 後で埋める
        unstable: usize::MAX,
    }
}

fn main() {
    let vs = vowels();
    let weakest = vs.iter().flat_map(|v| v.amplitudes.iter()).cloned().min().unwrap();
    let strongest = vs.iter().flat_map(|v| v.amplitudes.iter()).cloned().max().unwrap();
    let levels = [weakest, weakest * 2, strongest];

    println!("=== M0 の設計点を詰める ===");
    println!("掃引: N_BANDS {:?} × Q {:?} × 閾値 {:?}", BAND_COUNTS, Q_MULS, THRESHOLDS);
    println!("穴の検査レベル: {} / {} / {} (最弱フォルマント基準)", levels[0], levels[1], levels[2]);
    println!("採用規則 (実測前に固定): 全ゲート通過のうち**母音の精度が最大**。");
    println!();

    // 安いゲートで先に篩う
    let mut survivors: Vec<R> = Vec::new();
    for &nb in BAND_COUNTS.iter() {
        let freqs = erb_spaced_freqs(F_MIN_HZ, F_MAX_HZ, nb);
        for &q in Q_MULS.iter() {
            for &thr in THRESHOLDS.iter() {
                let r = measure_cheap(&freqs, q, thr);
                if r.v_recall == 15 && r.v_distinct == 10 && r.v_silent == 0
                    && r.c_distinct == 10 && r.c_silent == 0
                {
                    survivors.push(r);
                }
            }
        }
    }
    println!("被覆/場所符号/無音のゲートを通った設定: {} 通り", survivors.len());
    println!("穴と安定性を測る (重いので通過分のみ)…");
    println!();

    for r in survivors.iter_mut() {
        let freqs = erb_spaced_freqs(F_MIN_HZ, F_MAX_HZ, r.n_bands);
        r.unstable = unstable(&freqs, r.q);
        r.holes = holes(&freqs, r.q, r.thr, &levels);
    }

    println!("帯域  Q  閾値  母音精度  子音被覆 子音精度  穴/{}  非減衰  判定", N_PROBE);
    let mut pass: Vec<&R> = Vec::new();
    for r in survivors.iter() {
        let ok = r.holes == 0 && r.unstable == 0;
        println!(
            "{:>4} ×{:.0} {:>5}  {:>7.1}%  {:>7.1}% {:>7.1}%  {:>5}  {:>6}  {}",
            r.n_bands, r.q, r.thr, r.v_prec * 100.0, r.c_recall * 100.0, r.c_prec * 100.0,
            r.holes, r.unstable, if ok { "通過" } else { "-" }
        );
        if ok {
            pass.push(r);
        }
    }

    println!();
    println!("--- N_BANDS ごとの最良 (全ゲート通過のうち母音精度が最大) ---");
    println!("帯域   母音精度  Q  閾値  子音被覆 子音精度");
    for &nb in BAND_COUNTS.iter() {
        match pass
            .iter()
            .filter(|r| r.n_bands == nb)
            .max_by(|a, b| a.v_prec.partial_cmp(&b.v_prec).unwrap())
        {
            Some(b) => println!("{:>4}   {:>7.1}%  ×{:.0} {:>5}  {:>7.1}% {:>7.1}%",
                                nb, b.v_prec * 100.0, b.q, b.thr,
                                b.c_recall * 100.0, b.c_prec * 100.0),
            None => println!("{:>4}   通過なし", nb),
        }
    }

    println!();
    match pass.iter().max_by(|a, b| a.v_prec.partial_cmp(&b.v_prec).unwrap()) {
        Some(b) => {
            println!("採用規則の結果: **N_BANDS = {} ・ Q ×{:.0} ・ FIRE_THRESHOLD = {}**",
                     b.n_bands, b.q, b.thr);
            println!("  母音: 被覆 15/15 ・ 場所符号 10/10 ・ 精度 {:.1}%", b.v_prec * 100.0);
            println!("  子音: 被覆 {:.1}% ・ 場所符号 10/10 ・ 精度 {:.1}%",
                     b.c_recall * 100.0, b.c_prec * 100.0);
            println!("  スペクトルの穴 0 ・ 非減衰帯域 0");
        }
        None => println!("**全ゲートを通る設定なし — 不成立。**"),
    }
}
