//! 自発発火の床を引けば M0 のパターンは戻るか — 指標の再定義と、床引きは誰の仕事か (2026-08-26)
//!
//! ## 発端 — ユーザーの指摘
//!
//! > **M0 においては識別できるパターンが獲得できれば「鳴ったかどうか」は影響しない。
//! > それを判断するのは M0.5 もしくは M1 以降の話ではないか。
//! > 小さな音を聞くために、周囲が静かなとき耳鳴りのように聞こえてしまうのは人間も同じで、
//! > その処理はもっと高次で行っているのではないか。**
//!
//! これは正しい。そして**私の指標が嘘をついていた**ことが、これではっきりする。
//!
//! 同じ変更 (自発発火 ON) について:
//! - **二値の指標** (`level_axis` の場所符号 = 発火集合の相異): 全レベル **0/10 の壊滅**
//! - **発火数の指標** (`kana_identify` のコサイン): レベル軸 0.7% → **13.0% (18.6倍)**
//!
//! M0 の仕事が「識別できるパターンを出すこと」なら、**後者が正しく、前者が壊れた指標**である。
//!
//! ## 一点だけ補足 — 一律の利得では分離できない
//!
//! 「利得調整が一律に自発発火を閾値外にすれば OK」は、そのままでは成り立たない。
//! **一律の利得では自発発火も信号も同じだけ縮む**ので、静かな信号は一緒に消える。
//!
//! 生体がやっているのは**基準線からのずれの検出**である。静かな音への応答は
//! 「閾値を越えた」ではなく「**自発率より少し増えた**」であり、これは減算 (適応) にあたる。
//! そして**その機構は M0.5 が既に持っている** (Octopus=オンセット/変化・Stellate=持続レート・
//! 局所エントロピーによる適応)。
//!
//! **つまり「指標を発火数ベースに直す」と「床を引く」は同じ操作である。**
//!
//! ## 測ること
//!
//! 1. **床** = 無音を同じ長さ流したときの帯域ごとの発火数 (実験者が無音を与えた)
//! 2. **床引き後** = 刺激時の発火数 − 床 (0 でクランプ)
//! 3. 指標を「鳴ったか」から「**床を超えたか / 床の上でいくつ鳴ったか**」に置き換える
//! 4. **M0 出力**と **M0.5 出力**の両方で測る → **M0.5 が既に床を引いているか**が分かる
//!
//! ## ゲート (実測前に固定・以後動かさない)
//!
//! 定義は `level_axis` の G40-42 と同じだが、**「鳴った」を「床を超えた」に置き換える**。
//! レベルも同じ 9 点。
//!
//! - **G77a 被覆 (床引き後)**: 全レベルで指定 3 フォルマントが**床を超える** (15/15)。
//! - **G77b 場所符号 (床引き後)**: 全レベルで 5 母音の**床超えチャネル集合**が全 10 対で相異なる。
//! - **G77c 順序 (床引き後)**: 同一母音・別レベルの最小コサイン > 別母音・同レベルの最大。
//!   **床引き後のベクトルで計算する。**
//! - **G77d 子音の重心 (床引き後)**: 自発 OFF のときの重心に戻るか。
//!   *判定は「順序 pa < tu < ki < se が保たれ、かつ se の重心が 2500Hz を超える」。*
//!   2500Hz は OFF のとき 3153Hz・ON のとき 1795Hz なので、その中間に置いた。
//!   **実測前に固定する。**
//! - **G77e M0.5 は既に床を引いているか**: M0.5 出力の重心が M0 出力より OFF に近いか。
//!   *近ければ「床引きは M0.5 の仕事」というユーザーの読みが支持される。*
//!
//! ## 予測
//!
//! **数値は置かない。** 構造のみ:
//! - **床を引けば戻るはず。** 自発発火は刺激と独立な加算なので、引けば消えるはず。
//! - **M0.5 は部分的にしか引いていないはず。** 適応はあるが床引き専用の機構ではない。
//!
//! CLI: baseline_subtraction

use spiking_brain::phase2_f::cochlea::{Cochlea, N_BANDS, SAMPLES_PER_STEP};
use spiking_brain::phase2_f::cochlear_nucleus::{CochlearNucleus, N_CN_OUTPUT};
use spiking_brain::phase2_f::phoneme_synth::{F0_DEFAULT_HZ, 
    synth_consonant_banded, synth_vowel, vowels, Consonant, LfsrNoise,
};

const LEVELS_DB: [f64; 9] = [0.0, -3.0, -6.0, -9.0, -12.0, -15.0, -18.0, -21.0, -24.0];
const VOWEL_MS: f64 = 170.0;
const CONSONANT_MS: f64 = 30.0;
const SEED: u16 = 0xACE1;
/// G77d の判定閾値。**実測前に固定。** OFF 3153Hz / ON 1795Hz の中間。
const SE_CENTROID_MIN_HZ: f64 = 2500.0;

fn gain_of(db: f64) -> (i32, i32) {
    (((4096.0) * 10f64.powf(db / 20.0)).round() as i32, 4096)
}

/// M0 出力 (40 帯域) のスパイク数
fn m0_counts(wave: &[i32], gn: i32, gd: i32) -> Vec<f64> {
    let mut c = Cochlea::new();
    let mut counts = vec![0f64; N_BANDS];
    for chunk in wave.chunks(SAMPLES_PER_STEP) {
        if chunk.len() < SAMPLES_PER_STEP { break; }
        let amp: Vec<i32> = chunk.iter()
            .map(|&x| ((x as i64 * gn as i64) / gd as i64) as i32).collect();
        for (i, &v) in c.process_step(&amp).iter().enumerate() {
            if v != 0 { counts[i] += 1.0; }
        }
    }
    counts
}

/// M0.5 出力 (84ch) のスパイク数
fn m05_counts(wave: &[i32], gn: i32, gd: i32) -> Vec<f64> {
    let mut c = Cochlea::new();
    let mut cn = CochlearNucleus::new();
    let mut counts = vec![0f64; N_CN_OUTPUT];
    for chunk in wave.chunks(SAMPLES_PER_STEP) {
        if chunk.len() < SAMPLES_PER_STEP { break; }
        let amp: Vec<i32> = chunk.iter()
            .map(|&x| ((x as i64 * gn as i64) / gd as i64) as i32).collect();
        let m0 = c.process_step(&amp);
        for (i, &v) in cn.process_step(&m0).iter().enumerate() {
            if v != 0 { counts[i] += 1.0; }
        }
    }
    counts
}

fn subtract(x: &[f64], base: &[f64]) -> Vec<f64> {
    x.iter().zip(base.iter()).map(|(&a, &b)| (a - b).max(0.0)).collect()
}

fn cosine(a: &[f64], b: &[f64]) -> f64 {
    let dot: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let na: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
    let nb: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();
    if na == 0.0 || nb == 0.0 { 0.0 } else { dot / (na * nb) }
}

fn nearest_band(freqs: &[f64], f_hz: f64) -> usize {
    freqs.iter().enumerate()
        .min_by(|a, b| (a.1 - f_hz).abs().partial_cmp(&(b.1 - f_hz).abs()).unwrap())
        .unwrap().0
}

fn centroid(counts: &[f64], freqs: &[f64]) -> f64 {
    let tot: f64 = counts.iter().sum();
    if tot == 0.0 { return 0.0; }
    counts.iter().zip(freqs.iter()).map(|(&c, &f)| c * f).sum::<f64>() / tot
}

fn consonants() -> Vec<(&'static str, Consonant)> {
    vec![
        ("pa", Consonant::Plosive { burst_freq_low: 500.0, burst_freq_high: 2000.0, voiced: false }),
        ("tu", Consonant::Plosive { burst_freq_low: 1500.0, burst_freq_high: 3500.0, voiced: false }),
        ("ki", Consonant::Plosive { burst_freq_low: 2000.0, burst_freq_high: 4000.0, voiced: false }),
        ("se", Consonant::Fricative { freq_low: 3000.0, freq_high: 8000.0, voiced: false }),
    ]
}

fn main() {
    println!("=== 自発発火の床を引けば M0 のパターンは戻るか ===");
    println!();
    println!("【発端】ユーザーの指摘: M0 の仕事は識別できるパターンを出すことで、");
    println!("『鳴ったかどうか』を判断するのは M0.5 以降。静かなとき耳鳴りのように");
    println!("聞こえるのは人間も同じで、その処理はもっと高次。");
    println!();
    println!("同じ変更について 二値の指標は『場所符号 0/10 の壊滅』、発火数の指標は");
    println!("『レベル軸 18.6 倍の改善』と言っていた。**後者が正しく前者が壊れた指標。**");
    println!();
    println!("【補足】一律の利得では分離できない (自発も信号も同じだけ縮む)。");
    println!("生体がやっているのは**基準線からのずれの検出** = 減算。");
    println!("**『指標を発火数ベースに直す』と『床を引く』は同じ操作。**");
    println!();
    println!("【ゲート・実測前に固定】level_axis の G40-42 の『鳴った』を『床を超えた』に置換");
    println!("  G77a 被覆(床引き後) 全レベル 15/15   G77b 場所符号(床引き後) 全レベル 10/10");
    println!("  G77c 順序(床引き後)   G77d 子音の重心が戻るか (se > {:.0}Hz・順序保持)",
             SE_CENTROID_MIN_HZ);
    println!("  G77e M0.5 は既に床を引いているか");
    println!();
    println!("【予測】数値は置かない。構造のみ: 床を引けば戻るはず。");
    println!("M0.5 は部分的にしか引いていないはず (適応はあるが床引き専用ではない)。");

    let freqs = Cochlea::new().center_freqs.clone();
    let vs = vowels();
    let names = ["a", "i", "u", "e", "o"];

    // --- 床の測定 (無音・母音と同じ長さ) ---
    let n_samples = (VOWEL_MS * 16000.0 / 1000.0) as usize;
    let silence = vec![0i32; n_samples];
    let base_m0 = m0_counts(&silence, 4096, 4096);
    let base_total: f64 = base_m0.iter().sum();
    println!();
    println!("--- 自発発火の床 (無音 {:.0}ms) ---", VOWEL_MS);
    println!("  M0 総スパイク {:.0} ・ 帯域あたり平均 {:.1} ・ 最小 {:.0} ・ 最大 {:.0}",
             base_total, base_total / N_BANDS as f64,
             base_m0.iter().cloned().fold(f64::INFINITY, f64::min),
             base_m0.iter().cloned().fold(0f64, f64::max));

    // --- 母音: 床引き前後 ---
    let mut raw: Vec<Vec<Vec<f64>>> = Vec::new();
    let mut sub: Vec<Vec<Vec<f64>>> = Vec::new();
    for &db in LEVELS_DB.iter() {
        let (gn, gd) = gain_of(db);
        let r: Vec<Vec<f64>> = vs.iter().map(|v| m0_counts(&synth_vowel(v, VOWEL_MS), gn, gd)).collect();
        let s: Vec<Vec<f64>> = r.iter().map(|c| subtract(c, &base_m0)).collect();
        raw.push(r);
        sub.push(s);
    }

    println!();
    println!("--- 母音: 床引き前後 (M0 出力) ---");
    println!("  レベル   床超えチャネル数           被覆/15  場所符号/10");
    let mut g77a = true;
    let mut g77b = true;
    for (li, &db) in LEVELS_DB.iter().enumerate() {
        let above: Vec<usize> = sub[li].iter().map(|c| c.iter().filter(|&&x| x > 0.0).count()).collect();
        let mut cov = 0usize;
        for (vi, v) in vs.iter().enumerate() {
            for fi in 0..3 {
                if sub[li][vi][nearest_band(&freqs, v.formants_hz[fi])] > 0.0 { cov += 1; }
            }
        }
        let sets: Vec<Vec<bool>> = sub[li].iter().map(|c| c.iter().map(|&x| x > 0.0).collect()).collect();
        let mut dist = 0usize;
        for i in 0..5 { for j in (i + 1)..5 { if sets[i] != sets[j] { dist += 1; } } }
        if cov != 15 { g77a = false; }
        if dist != 10 { g77b = false; }
        println!("  {:>5.0}dB  {:?}   {:>5}/15   {:>7}/10", db, above, cov, dist);
    }

    // --- G77c 順序 (床引き後) ---
    let mut min_same = f64::INFINITY;
    let mut min_d = String::new();
    for vi in 0..5 {
        for li in 0..LEVELS_DB.len() {
            for lj in (li + 1)..LEVELS_DB.len() {
                let c = cosine(&sub[li][vi], &sub[lj][vi]);
                if c < min_same { min_same = c; min_d = format!("/{}/ {:.0}dB vs {:.0}dB", names[vi], LEVELS_DB[li], LEVELS_DB[lj]); }
            }
        }
    }
    let mut max_diff = f64::NEG_INFINITY;
    let mut max_d = String::new();
    for li in 0..LEVELS_DB.len() {
        for i in 0..5 { for j in (i + 1)..5 {
            let c = cosine(&sub[li][i], &sub[li][j]);
            if c > max_diff { max_diff = c; max_d = format!("/{}/ vs /{}/ @{:.0}dB", names[i], names[j], LEVELS_DB[li]); }
        }}
    }
    // 参考: 床引き前
    let mut min_same_raw = f64::INFINITY;
    let mut max_diff_raw = f64::NEG_INFINITY;
    for vi in 0..5 { for li in 0..LEVELS_DB.len() { for lj in (li + 1)..LEVELS_DB.len() {
        min_same_raw = min_same_raw.min(cosine(&raw[li][vi], &raw[lj][vi])); }}}
    for li in 0..LEVELS_DB.len() { for i in 0..5 { for j in (i + 1)..5 {
        max_diff_raw = max_diff_raw.max(cosine(&raw[li][i], &raw[li][j])); }}}

    println!();
    println!("--- G77c 順序 ---");
    println!("  床引き前: 同一母音最小 {:.4} / 別母音最大 {:.4} -> {}",
             min_same_raw, max_diff_raw, if min_same_raw > max_diff_raw { "PASS" } else { "**FAIL**" });
    println!("  床引き後: 同一母音最小 {:.4} ({}) / 別母音最大 {:.4} ({}) -> {}",
             min_same, min_d, max_diff, max_d,
             if min_same > max_diff { "**PASS**" } else { "**FAIL**" });

    // --- G77d/G77e 子音の重心 ---
    println!();
    println!("--- G77d/G77e 子音の重心 ---");
    println!("  子音    M0 床引き前   M0 床引き後   M0.5 床引き前  M0.5 床引き後   (参考 自発OFF)");
    let base_c_m0 = {
        let ns = (CONSONANT_MS * 16000.0 / 1000.0) as usize;
        m0_counts(&vec![0i32; ns], 4096, 4096)
    };
    let base_c_m05 = {
        let ns = (CONSONANT_MS * 16000.0 / 1000.0) as usize;
        m05_counts(&vec![0i32; ns], 4096, 4096)
    };
    // M0.5 の 84ch のうち Bushy 部分 (4..4+N_BANDS) を帯域として扱う
    let mut cent_raw = Vec::new();
    let mut cent_sub = Vec::new();
    let mut cent5_sub = Vec::new();
    for (nm, cons) in consonants() {
        let mut noise = LfsrNoise::new(SEED);
        let w = synth_consonant_banded(cons, CONSONANT_MS, F0_DEFAULT_HZ, &mut noise);
        let c0 = m0_counts(&w, 4096, 4096);
        let c0s = subtract(&c0, &base_c_m0);
        let c5 = m05_counts(&w, 4096, 4096);
        let c5s = subtract(&c5, &base_c_m05);
        let bushy: Vec<f64> = (0..N_BANDS).map(|i| c5[4 + i]).collect();
        let bushy_s: Vec<f64> = (0..N_BANDS).map(|i| c5s[4 + i]).collect();
        let (a, b, c, d) = (centroid(&c0, &freqs), centroid(&c0s, &freqs),
                            centroid(&bushy, &freqs), centroid(&bushy_s, &freqs));
        cent_raw.push(a); cent_sub.push(b); cent5_sub.push(d);
        println!("  {:<6} {:>10.0}Hz {:>12.0}Hz {:>13.0}Hz {:>13.0}Hz", nm, a, b, c, d);
    }
    println!("  (参考 自発OFF の記録: pa 1787 / tu 2360 / ki 2601 / se 3153 Hz)");

    let order_ok = cent_sub[0] < cent_sub[1] && cent_sub[1] < cent_sub[2] && cent_sub[2] < cent_sub[3];
    let se_ok = cent_sub[3] > SE_CENTROID_MIN_HZ;
    println!();
    println!("  G77d 子音の重心 (床引き後・M0): 順序 {} / se {:.0}Hz > {:.0}Hz {} -> {}",
             if order_ok { "保持" } else { "**崩れた**" },
             cent_sub[3], SE_CENTROID_MIN_HZ, if se_ok { "○" } else { "×" },
             if order_ok && se_ok { "**PASS**" } else { "**FAIL**" });
    let m05_closer = (cent5_sub[3] - 3153.0).abs() < (cent_raw[3] - 3153.0).abs();
    println!("  G77e M0.5 は既に床を引いているか: M0 床引き前 {:.0}Hz / M0.5 床引き後 {:.0}Hz",
             cent_raw[3], cent5_sub[3]);
    println!("       (OFF の 3153Hz に近いのは {}) -> {}",
             if m05_closer { "M0.5 側" } else { "M0 側" },
             if m05_closer { "M0.5 が寄与している" } else { "M0.5 単独では戻していない" });

    println!();
    println!("=== 判定 (規則は実測前に固定) ===");
    println!("  G77a 被覆 (床引き後・全レベル 15/15) -> {}", if g77a { "**PASS**" } else { "**FAIL**" });
    println!("  G77b 場所符号 (床引き後・全レベル 10/10) -> {}", if g77b { "**PASS**" } else { "**FAIL**" });
    println!("  G77c 順序 (床引き後) -> {}", if min_same > max_diff { "**PASS**" } else { "**FAIL**" });
    println!("  G77d 子音の重心 -> {}", if order_ok && se_ok { "**PASS**" } else { "**FAIL**" });
    println!();
    println!("  【この測定が答えないこと】床引きを M0.5/M1 の**どの機構**が担うべきかは決めない。");
    println!("  ここで測ったのは『引けば戻るか』だけ。**既定は変えていない。**");
}
