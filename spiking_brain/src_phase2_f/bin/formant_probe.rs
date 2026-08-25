//! M0 はフォルマントを区別できるか (2026-08-25・ユーザー指示で M0 優先)
//!
//! ## 問題
//!
//! 発火の床が閉じた式で出ている:
//!   包絡線の平衡 env = 2^ENV_LEAK_SHIFT × mean|x| = 16 × (2/π)A = 10.19A
//!   圧縮後 sqrt(10.19A) が FIRE_THRESHOLD を超える条件 → A >= threshold^2 / 10.19
//!   threshold=200 のとき **A >= 3927**
//!
//! 母音の振幅は F1=4000 / F2=2000-3200 / F3=800-1600。
//! **F2/F3 は構造的に床の下**なので一度も鳴らない。
//! 結果、どの母音も発火帯域は 1 本 (F1 のみ)、/e/ と /o/ は F1 が同じ 500Hz で衝突。
//! **「フォルマントで母音を区別する」という設計が動いていない。**
//!
//! ## ゲート (実測前に固定)
//!
//!   G29 被覆(recall)   : 各母音の指定 3 フォルマントの最寄り帯域が応答するか。
//!                        5母音 × 3 = **15/15 が満点**。
//!                        正解の出どころ = 実験者が 3 つのフォルマントを置いた。
//!   G30 場所符号の相異 : 5 母音の**発火帯域の集合**が全 10 対で相異なるか (10/10)。
//!                        正解の出どころ = 実験者が 5 組の別々のフォルマントを与えた。
//!   G31 無音母音なし   : 5 母音すべてが発火する。
//!   G32 精度(precision): 発火した帯域のうち、いずれかの指定フォルマントの
//!                        近傍 (±TOLERANCE_BANDS) に属するものの割合。
//!
//! **G32 が要**: これが無いと「ゲインを上げて全帯域を鳴らす」で G29 を満点にできる。
//! 被覆だけ上げると精度が落ちるので、両方同時に要求すれば
//! **沈黙でもゲイン殴りでも通らない**。
//!
//! 採用規則 (先に宣言): G29=15/15 かつ G30=10/10 かつ G31 を満たすもののうち、
//! **G32(精度) が最大**のものを採る。無ければ不成立と報告し、定数は変えない。
//!
//! ## 掃引する軸
//!
//! 床 A >= threshold^2 / 10.19 に対して F2/F3 を届かせる手段は 3 つある:
//!   (1) 刺激の絶対スケール (提示ゲイン)
//!   (2) 発火閾値 FIRE_THRESHOLD
//!   (3) 圧縮の形 (sqrt → log。設計書 §3.4 が指定しながら未実装)
//! ここでは (1)(2) を掃引する。両方が「同じ比を動かす」だけなのか、
//! それとも別の効き方をするのかを実測で分ける。
//!
//! ⚠️ **この probe の精度は「±1 帯域」基準**（旧定義）。
//! 1 本の幅は帯域数で変わるので、**帯域数をまたいだ比較には使えない**。
//! 帯域数を含む比較・設計点の決定には `m0_design`（ERB 基準）を使うこと。
//! 単一の N_BANDS 内での比較には引き続き有効。
//!
//! CLI: formant_probe

use spiking_brain::phase2_f::cochlea::{
    erb_q_factor, erb_spaced_freqs, BandpassBiquad, Cochlea, EnvelopeDetector, FireGenerator,
    ENV_LEAK_SHIFT, FIRE_REFRACTORY_STEPS, FIRE_THRESHOLD, F_MAX_HZ, F_MIN_HZ, N_BANDS,
    SAMPLES_PER_STEP,
};
use spiking_brain::phase2_f::phoneme_synth::{synth_vowel, vowels, SAMPLE_RATE_HZ};

const VOWEL_MS: f64 = 170.0;
/// 指定フォルマントの「近傍」とみなす帯域幅 (最寄り帯域 ±この本数)。
/// 蝸牛の帯域はフォルマントより狭いので、1 本の余裕を持たせる。
const TOLERANCE_BANDS: usize = 1;
const THRESHOLDS: [i32; 9] = [40, 60, 80, 100, 120, 140, 160, 180, 200];
const GAINS: [i32; 4] = [1, 2, 3, 4];
/// Q 倍率 (1 = 現行の erb_q_factor)。
///
/// 設計書 §1.5: 外有毛細胞 (OHC) は「周波数選択性を 1/3 oct → 1/10 oct まで鋭くする。
/// これがないと『補聴器をつけても言葉が聞き取れない』(sensorineural hearing loss の
/// 典型症状)」。**未実装**。ERB は 800Hz で約 1/6 oct 相当なので、
/// 1/10 oct には約 2 倍の鋭さが要る。
const Q_MULTIPLIERS: [f64; 5] = [1.0, 2.0, 3.0, 4.0, 6.0];

fn cochlea_with_threshold(threshold: i32, q_mul: f64) -> Cochlea {
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
    let mut c = cochlea_with_threshold(threshold, q_mul);
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

struct Row {
    threshold: i32,
    gain: i32,
    q_mul: f64,
    recall: usize,          // /15
    distinct: usize,        // /10
    silent: usize,
    precision: f64,
    total_firing: usize,
    per_vowel_recall: Vec<usize>,
    per_vowel_bands: Vec<usize>,
    floor_amplitude: f64,
}

fn measure(threshold: i32, gain: i32, q_mul: f64) -> Row {
    let freqs = erb_spaced_freqs(F_MIN_HZ, F_MAX_HZ, N_BANDS);
    let vs = vowels();

    // 各母音の「指定フォルマント近傍」帯域の集合
    let mut target: Vec<std::collections::HashSet<usize>> = Vec::new();
    for v in vs.iter() {
        let mut set = std::collections::HashSet::new();
        for f in 0..3 {
            let bi = nearest_band(&freqs, v.formants_hz[f]);
            for d in 0..=TOLERANCE_BANDS {
                if bi >= d {
                    set.insert(bi - d);
                }
                if bi + d < N_BANDS {
                    set.insert(bi + d);
                }
            }
        }
        target.push(set);
    }

    let profiles: Vec<[u32; N_BANDS]> = vs
        .iter()
        .map(|v| band_spikes(&synth_vowel(v, VOWEL_MS), threshold, gain, q_mul))
        .collect();

    // G29 被覆: 指定フォルマントの最寄り帯域そのものが発火したか
    let mut recall = 0usize;
    let mut per_vowel_recall = Vec::new();
    for (k, v) in vs.iter().enumerate() {
        let mut c = 0usize;
        for f in 0..3 {
            if profiles[k][nearest_band(&freqs, v.formants_hz[f])] > 0 {
                c += 1;
            }
        }
        recall += c;
        per_vowel_recall.push(c);
    }

    // G32 精度: 発火帯域のうち、その母音の指定フォルマント近傍に入る割合
    let mut hit = 0usize;
    let mut total = 0usize;
    for (k, p) in profiles.iter().enumerate() {
        for ch in 0..N_BANDS {
            if p[ch] > 0 {
                total += 1;
                if target[k].contains(&ch) {
                    hit += 1;
                }
            }
        }
    }
    let precision = if total == 0 { 0.0 } else { hit as f64 / total as f64 };

    // G30 場所符号の相異 (集合として)
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

    Row {
        threshold,
        gain,
        q_mul,
        recall,
        distinct,
        silent: sets.iter().filter(|s| s.is_empty()).count(),
        precision,
        total_firing: total,
        per_vowel_recall,
        per_vowel_bands: sets.iter().map(|s| s.len()).collect(),
        // 床 A >= threshold^2 / 10.19
        floor_amplitude: (threshold as f64) * (threshold as f64) / 10.19,
    }
}

/// G34': 純音を**複数のレベルで**細かく掃引し、応答しない周波数があるか。
///
/// Q を上げると帯域が狭くなるが 40 個の中心は動かないので、**スペクトルに穴が空く**。
/// 穴があれば、そこに落ちたフォルマントは原理的に拾えない。
///
/// **1 レベルだけでは足りない** (2026-08-25 に一度やった): 大きい音なら帯域中心から
/// 外れても床を超えるので穴が埋まって見える。「大きい音では穴がない」は
/// 「穴がない」ではない。狭い帯域は**小さい音**を帯域間で聞き逃す。
/// よって刺激スケールの ×1/4・×1/2・×1 の 3 レベルで測り、**最悪値**を採る。
///
/// 正解の出どころ: どの周波数・どのレベルを入力したかは実験者が決めた。
fn spectral_holes(q_mul: f64, threshold: i32, gain: i32) -> (usize, usize, Vec<f64>) {
    use spiking_brain::phase2_f::phoneme_synth::{freq_to_phase_step, sin_lookup};
    let n_probe = 200usize;
    // 検査レベルは**実際の刺激に含まれる最弱フォルマント**を下端にする。
    // 「システムが検出すべき一番小さいもの」が正解の基準であり、
    // 最強の 1/4 では実際の最弱 (F3) より大きくなって甘くなる (2026-08-25 に一度やった)。
    let vs = vowels();
    let weakest = vs.iter().flat_map(|v| v.amplitudes.iter()).cloned().min().unwrap_or(800);
    let strongest = vs.iter().flat_map(|v| v.amplitudes.iter()).cloned().max().unwrap_or(4000);
    let levels = [weakest, weakest * 2, strongest];
    let mut worst_dead: Vec<f64> = Vec::new();
    let mut worst_n = 0usize;
    for &amp_level in levels.iter() {
        let mut dead = Vec::new();
        for k in 0..n_probe {
            let t = k as f64 / (n_probe - 1) as f64;
            let f = 50.0 * (4000.0f64 / 50.0).powf(t);
            let n_samples = (170.0 * SAMPLE_RATE_HZ / 1000.0) as usize;
            let step = freq_to_phase_step(f);
            let mut phase = 0u32;
            let wave: Vec<i32> = (0..n_samples)
                .map(|_| {
                    let v = (sin_lookup(phase) * amp_level) >> 14;
                    phase = phase.wrapping_add(step);
                    v
                })
                .collect();
            if band_spikes(&wave, threshold, gain, q_mul).iter().all(|&c| c == 0) {
                dead.push(f);
            }
        }
        if dead.len() > worst_n {
            worst_n = dead.len();
            worst_dead = dead;
        }
    }
    (worst_n, n_probe, worst_dead)
}

/// G35: Q を上げた全帯域でインパルス応答がゼロに落ちきるか。
fn unstable_bands(q_mul: f64) -> usize {
    let freqs = erb_spaced_freqs(F_MIN_HZ, F_MAX_HZ, N_BANDS);
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

fn main() {
    println!("=== M0 はフォルマントを区別できるか ===");
    println!("床の式: A >= threshold^2 / 10.19  (env = 16×(2/π)A, 圧縮 = sqrt)");
    println!("母音の振幅: F1=4000 / F2=2000-3200 / F3=800-1600");
    println!("近傍の許容 = 最寄り帯域 ±{} 本", TOLERANCE_BANDS);
    println!();
    println!("採用規則 (実測前に固定): G29=15/15 かつ G30=10/10 かつ G31(無音0) かつ");
    println!("  G34'(3レベルすべてでスペクトルの穴=0) かつ G35(非減衰帯域=0) を満たすもののうち G32(精度) 最大。");
    println!();

    let mut rows: Vec<Row> = Vec::new();
    for &q in Q_MULTIPLIERS.iter() {
        for &g in GAINS.iter() {
            for &t in THRESHOLDS.iter() {
                rows.push(measure(t, g, q));
            }
        }
    }

    // 通過候補すべてについて穴と安定性を測る
    let unstable: std::collections::HashMap<i64, usize> = Q_MULTIPLIERS
        .iter()
        .map(|&q| ((q * 1000.0) as i64, unstable_bands(q)))
        .collect();

    let mut best: Option<(&Row, usize)> = None;
    println!("  Q ゲイン 閾値  G32精度  発火帯域計  穴/200  非減衰  母音ごと帯域数");
    for r in rows.iter().filter(|r| r.recall == 15 && r.distinct == 10 && r.silent == 0) {
        let (holes, tested, _) = spectral_holes(r.q_mul, r.threshold, r.gain);
        let unst = unstable[&((r.q_mul * 1000.0) as i64)];
        let ok = holes == 0 && unst == 0;
        println!(
            "{:>3.0}x {:>4}x {:>4}  {:>6.1}%  {:>9}  {:>3}/{:<3}  {:>6}  {:<15} {}",
            r.q_mul, r.gain, r.threshold, r.precision * 100.0, r.total_firing,
            holes, tested, unst,
            format!("{:?}", r.per_vowel_bands),
            if ok { "採用候補" } else { "-" }
        );
        if ok {
            match best {
                Some((b, _)) if b.precision >= r.precision => {}
                _ => best = Some((r, holes)),
            }
        }
    }

    println!();
    println!("--- 採用規則の結果 ---");
    match best {
        Some((r, _)) => {
            println!("**Q ×{:.0} ・ 提示ゲイン {}x ・ FIRE_THRESHOLD {}**", r.q_mul, r.gain, r.threshold);
            println!("  被覆 {}/15 ・ 場所符号 {}/10 ・ 無音 {} ・ 精度 {:.1}%",
                     r.recall, r.distinct, r.silent, r.precision * 100.0);
            println!("  発火帯域計 {} (母音ごと {:?}) ・ スペクトルの穴 0 ・ 非減衰帯域 0",
                     r.total_firing, r.per_vowel_bands);
            println!("  床 A >= {:.0} (最弱フォルマント F3=800 との関係を確認すること)",
                     r.floor_amplitude);
            println!();
            println!("  現行 (Q×1・ゲイン1x・閾値200) は 被覆 5/15・場所符号 9/10・精度 100%");
            println!("  = 「鳴るものは全部本物だが、15本中5本しか鳴らない」最大精度・最小被覆の端。");
        }
        None => println!("**該当なし — 不成立。定数は変えない。**"),
    }
}
