//! 閾値をずらした線維集団は、レベル不変性の壁を壊せるか (2026-08-26)
//!
//! ## 発端
//!
//! ユーザーの記憶「有毛細胞は自発放電をする」から辿った生理の事実:
//!
//! 聴神経線維は自発発火率で 3 クラスに分かれ (Liberman 1978)、
//! **自発率と閾値が逆相関している**。
//!
//! | クラス | 自発率 | 割合 | 閾値 |
//! |---|---|---|---|
//! | 高 (HSR) | >18/s (典型 60・最大 ~120) | 約 60% | **低い** (小さい音) |
//! | 中 (MSR) | 0.5〜18/s | 約 25% | 中 |
//! | 低 (LSR) | **<0.5/s (ほぼ沈黙)** | 約 15% | **高い** (大きい音) |
//!
//! そして **1 つの内有毛細胞に 10〜30 本の線維**が付き、閾値をずらして配置されている。
//! 線維 1 本の動的レンジは約 20〜30 dB だが、集団で **約 120 dB** を覆う。
//!
//! ## 数字が符合する
//!
//! | | 値 |
//! |---|---|
//! | `rate_code` が測った動的レンジ (G43) | **26.50 dB** |
//! | 聴神経線維 1 本の動的レンジ | 約 20〜30 dB |
//! | かな同定が生きているレベル範囲 (§14.6) | 0〜−18 dB ≈ **18〜21 dB** |
//!
//! **いまの M0 は 1 つの場所につき線維 1 本しか持っていない。**
//! だから系全体の動的レンジが線維 1 本ぶんしかない、というのが仮説。
//!
//! ## この機構は 6 原理に全部合う
//!
//! - 各線維は独立 → **局所性**
//! - 閾値の違いは物理的性質 → **物理性** (判断機構なし)
//! - 決定論的・整数
//! - **AGC が要らない** — 監査が塞いだ 2 つの道
//!   (教科書 AGC は原理 1/2 違反 / チャネル別除算は場所符号を壊す) を**どちらも通らない**
//!
//! ## 閾値の並べ方 (実測前に宣言・以後動かさない)
//!
//! 1 本の動的レンジは実測 26.50 dB。振幅で 26.50 dB = ×21.1。
//! 発火条件は `isqrt(env) >= threshold` で `env ∝ A` なので `threshold ∝ sqrt(A)`。
//! よって隣り合う線維の動的レンジがちょうど接する閾値比は **sqrt(21.1) = 4.59**。
//!
//! **K 本の閾値は、現行の 120 を含む公比 4.59 の等比列。**
//! K を増やすごとに 下・上・下 の順に足す (生物は低閾値も高閾値も持つ)。
//!
//! - K=1: {120}          ← **対照** (現行と同じはず)
//! - K=2: {26, 120}
//! - K=3: {26, 120, 551}
//! - K=4: {6, 26, 120, 551}
//!
//! ## ゲート (実測前に固定・`level_axis` の G40-42 と**同一の定義**)
//!
//! レベルも同じ 9 点 (0 〜 −24 dB) を使う。ゲートを難しくしないため増やさない。
//!
//! - **G76a レベル横断の場所符号**: 全レベルで 5 母音の発火チャネル集合が
//!   全 10 対で相異なり、かつ無音母音が 0 (`level_axis` の G40 と同一)。
//! - **G76b レベル横断の被覆**: 全レベルで指定 3 フォルマントが応答 (15/15)。G41 と同一。
//! - **G76c 順序 (要)**: 「同一母音・別レベル」の最小類似度が
//!   「別母音・同レベル」の最大類似度を**上回る**。G42 と同一。
//!   *現行は 0.0000 < 0.9613 で大きく逆転している。*
//! - **G76d 射程**: 無音母音が出ないレベル範囲 [dB]。**記述であって判定ではない。**
//! - **G76e 対照の健全性**: K=1 が `level_axis` の記録値 (§14.5.2) を再現するか。
//!   *再現しなければこのプローブ自体が信用できない。*
//!
//! ## 予測
//!
//! **数値は置かない** (§14.6.4 / §14.7 / §14.9.7 で 3 連続、§14.10.4 で構造予測も外した)。
//!
//! 構造:
//! - **射程 (G76d) は広がるはず。** 低閾値の線維を足すのだからほぼ構成上そうなる。
//!   **これは当たっても意味のない予測である。**
//! - **G76c (順序) が通るかは分からない。** 低閾値線維が高レベルで飽和して全部鳴れば、
//!   場所符号が潰れて逆に悪化しうる。**そこが本当の問い。**
//!
//! ## 既定は変えない
//!
//! これは**測定であって採用ではない**。ゲートを通ってから採否を決める。
//!
//! CLI: fiber_population

use spiking_brain::phase2_f::cochlea::{
    compress_sqrt, Cochlea, FireGenerator, FIRE_REFRACTORY_STEPS, N_BANDS, SAMPLES_PER_STEP,
};
use spiking_brain::phase2_f::phoneme_synth::{synth_vowel, vowels};

const LEVELS_DB: [f64; 9] = [0.0, -3.0, -6.0, -9.0, -12.0, -15.0, -18.0, -21.0, -24.0];
/// 射程の記述用に、ゲートとは別に下まで伸ばす (判定には使わない)
const EXTENDED_DB: [f64; 5] = [-27.0, -30.0, -33.0, -36.0, -42.0];
const VOWEL_MS: f64 = 170.0;

/// 宣言した閾値の並べ方
fn thresholds(k: usize) -> Vec<i32> {
    match k {
        1 => vec![120],
        2 => vec![26, 120],
        3 => vec![26, 120, 551],
        4 => vec![6, 26, 120, 551],
        _ => unreachable!(),
    }
}

/// K 本の線維で走らせて、チャネルごとのスパイク数を返す (長さ N_BANDS*K)
fn fiber_spikes(wave: &[i32], gain_num: i32, gain_den: i32, k: usize) -> Vec<u32> {
    let th = thresholds(k);
    let mut c = Cochlea::new(); // 出荷構成の前段 (自発発火 ON を含む)
    let mut fibers: Vec<FireGenerator> = (0..N_BANDS * k)
        .map(|i| FireGenerator::new(th[i % k], FIRE_REFRACTORY_STEPS))
        .collect();
    let mut counts = vec![0u32; N_BANDS * k];
    for chunk in wave.chunks(SAMPLES_PER_STEP) {
        if chunk.len() < SAMPLES_PER_STEP { break; }
        let amp: Vec<i32> = chunk.iter()
            .map(|&x| ((x as i64 * gain_num as i64) / gain_den as i64) as i32)
            .collect();
        // 前段 (biquad → 自発発火 → 包絡) を出荷構成のまま回す。
        // 戻り値は使わない。包絡の値を直接読んで自前の線維に掛ける。
        let _ = c.process_step(&amp);
        for ch in 0..N_BANDS {
            let compressed = compress_sqrt(c.envelopes[ch].env);
            for f in 0..k {
                if fibers[ch * k + f].process(compressed) {
                    counts[ch * k + f] += 1;
                }
            }
        }
    }
    counts
}


/// 表現B: 各帯域で K 本の発火数を合算し、N 次元に畳む。
///
/// 生体でも蝸牛神経核は多数の聴神経線維の収束を受ける。
/// **表現A (N*K の生ベクトル) は同一性とレベルが絡んだままだが、
/// 合算すると パターン=同一性 / 大きさ=レベル に分離される。
/// コサインは大きさに不変なので、自動的にレベル不変になりうる。**
fn fold_bands(c: &[u32], k: usize) -> Vec<u32> {
    (0..N_BANDS).map(|ch| (0..k).map(|f| c[ch * k + f]).sum()).collect()
}

fn cosine(a: &[u32], b: &[u32]) -> f64 {
    let dot: f64 = a.iter().zip(b.iter()).map(|(&x, &y)| x as f64 * y as f64).sum();
    let na: f64 = a.iter().map(|&x| (x as f64).powi(2)).sum::<f64>().sqrt();
    let nb: f64 = b.iter().map(|&x| (x as f64).powi(2)).sum::<f64>().sqrt();
    if na == 0.0 || nb == 0.0 { 0.0 } else { dot / (na * nb) }
}

fn nearest_band(freqs: &[f64], f_hz: f64) -> usize {
    freqs.iter().enumerate()
        .min_by(|a, b| (a.1 - f_hz).abs().partial_cmp(&(b.1 - f_hz).abs()).unwrap())
        .unwrap().0
}

fn gain_of(db: f64) -> (i32, i32) {
    (((4096.0) * 10f64.powf(db / 20.0)).round() as i32, 4096)
}

struct LevelRow { db: f64, bands: Vec<usize>, coverage: usize, distinct: usize, silent: usize }

fn measure(k: usize, levels: &[f64], fold: bool) -> (Vec<LevelRow>, Vec<Vec<Vec<u32>>>) {
    let vs = vowels();
    let freqs = Cochlea::new().center_freqs.clone();
    let mut rows = Vec::new();
    let mut all: Vec<Vec<Vec<u32>>> = Vec::new(); // [level][vowel] -> counts
    for &db in levels.iter() {
        let (gn, gd) = gain_of(db);
        let raw: Vec<Vec<u32>> = vs.iter()
            .map(|v| fiber_spikes(&synth_vowel(v, VOWEL_MS), gn, gd, k))
            .collect();
        // 被覆と発火帯域は raw (線維単位) で見る。分布の比較だけ表現を切り替える。
        let per_vowel: Vec<Vec<u32>> = if fold {
            raw.iter().map(|c| fold_bands(c, k)).collect()
        } else { raw.clone() };
        // 発火した「帯域」の数 (どれか1本でも鳴れば その帯域は鳴っている)
        let bands: Vec<usize> = raw.iter().map(|c| {
            (0..N_BANDS).filter(|&ch| (0..k).any(|f| c[ch * k + f] > 0)).count()
        }).collect();
        // 被覆: 各母音の指定3フォルマントの最近傍帯域が鳴っているか
        let mut coverage = 0usize;
        for (vi, v) in vs.iter().enumerate() {
            for fi in 0..3 {
                let ch = nearest_band(&freqs, v.formants_hz[fi]);
                if (0..k).any(|f| raw[vi][ch * k + f] > 0) { coverage += 1; }
            }
        }
        // 場所符号の相異: 発火チャネル集合が全10対で相異なるか
        let sets: Vec<Vec<bool>> = per_vowel.iter()
            .map(|c| c.iter().map(|&x| x > 0).collect()).collect();
        let mut distinct = 0usize;
        for i in 0..5 { for j in (i + 1)..5 { if sets[i] != sets[j] { distinct += 1; } } }
        let silent = per_vowel.iter().filter(|c| c.iter().all(|&x| x == 0)).count();
        rows.push(LevelRow { db, bands, coverage, distinct, silent });
        all.push(per_vowel);
    }
    (rows, all)
}

fn main() {
    println!("=== 閾値をずらした線維集団は、レベル不変性の壁を壊せるか ===");
    println!();
    println!("【発端】聴神経線維は自発率と閾値が逆相関し (Liberman 1978)、");
    println!("1つの内有毛細胞に 10-30 本が閾値をずらして付く。1本 20-30dB、集団で約 120dB。");
    println!("いまの M0 は 1 場所につき 1 本しか持たない。");
    println!("実測の符合: rate_code の動的レンジ 26.50dB / かな同定が生きる範囲 18-21dB。");
    println!();
    println!("【閾値の並べ方・実測前に宣言】1本 26.50dB = 振幅×21.1、threshold ∝ sqrt(A) より");
    println!("公比 sqrt(21.1) = 4.59。現行の 120 を含む等比列。K を増やすごとに 下・上・下。");
    for k in 1..=4 { println!("  K={}: {:?}", k, thresholds(k)); }
    println!();
    println!("【ゲート】level_axis の G40-42 と**同一定義**・レベルも同じ 9 点");
    println!("  G76a 全レベルで場所符号 10/10 かつ無音母音 0");
    println!("  G76b 全レベルで被覆 15/15");
    println!("  G76c **要**: 同一母音・別レベルの最小 > 別母音・同レベルの最大");
    println!("  G76d 射程 (記述・判定ではない)   G76e K=1 が現行を再現するか");
    println!();
    println!("【予測】射程は広がるはず (構成上ほぼ自明・当たっても意味がない)。");
    println!("**G76c が通るかは分からない。低閾値線維が高レベルで飽和して場所符号を潰しうる。**");
    println!("**既定は変えない。これは測定であって採用ではない。**");

    let mut summary: Vec<(usize, f64, f64, bool, bool, bool)> = Vec::new();
    println!();
    println!("【表現の比較・実測前に宣言】表現A は同一性とレベルが絡む。表現B は");
    println!("パターン=同一性 / 大きさ=レベル に分離され、コサインが大きさ不変なので");
    println!("レベル不変になりうる。**表現B が G76c を通すなら M0 の出力幅を変えずに済む。**");

    for &(fold, rep) in [(false, "表現A: N*K の生ベクトル"), (true, "表現B: 帯域ごとに K 本を合算 (N 次元)")].iter() {
    println!();
    println!("################################################################");
    println!("#### {} ####", rep);
    println!("################################################################");
    for k in 1..=4usize {
        println!();
        println!("################ K = {} 本/帯域  閾値 {:?} ################", k, thresholds(k));
        let (rows, all) = measure(k, &LEVELS_DB, fold);

        println!();
        println!("  レベル  母音ごとの発火帯域数    被覆/15  場所符号/10  無音母音");
        for r in rows.iter() {
            println!("  {:>5.0}dB  {:?}   {:>5}/15  {:>7}/10  {:>6}",
                     r.db, r.bands, r.coverage, r.distinct, r.silent);
        }

        // G76c: 同一母音・別レベル の最小 vs 別母音・同レベル の最大
        let names = ["a", "i", "u", "e", "o"];
        let mut min_same = f64::INFINITY;
        let mut min_desc = String::new();
        for vi in 0..5 {
            for li in 0..LEVELS_DB.len() {
                for lj in (li + 1)..LEVELS_DB.len() {
                    let c = cosine(&all[li][vi], &all[lj][vi]);
                    if c < min_same {
                        min_same = c;
                        min_desc = format!("/{}/ {:.0}dB vs {:.0}dB", names[vi], LEVELS_DB[li], LEVELS_DB[lj]);
                    }
                }
            }
        }
        let mut max_diff = f64::NEG_INFINITY;
        let mut max_desc = String::new();
        for li in 0..LEVELS_DB.len() {
            for i in 0..5 {
                for j in (i + 1)..5 {
                    let c = cosine(&all[li][i], &all[li][j]);
                    if c > max_diff {
                        max_diff = c;
                        max_desc = format!("/{}/ vs /{}/ @{:.0}dB", names[i], names[j], LEVELS_DB[li]);
                    }
                }
            }
        }

        let g76a = rows.iter().all(|r| r.distinct == 10 && r.silent == 0);
        let g76b = rows.iter().all(|r| r.coverage == 15);
        let g76c = min_same > max_diff;

        println!();
        println!("  同一母音・別レベルの**最小**: {:.4}  ({})", min_same, min_desc);
        println!("  別母音・同レベルの**最大**  : {:.4}  ({})", max_diff, max_desc);
        println!();
        println!("  G76a 場所符号 (全レベル 10/10 かつ無音0) -> {}", if g76a { "PASS" } else { "**FAIL**" });
        println!("  G76b 被覆 (全レベル 15/15)              -> {}", if g76b { "PASS" } else { "**FAIL**" });
        println!("  G76c 順序 (要)                          -> {}", if g76c { "**PASS**" } else { "**FAIL**" });

        // G76d 射程 (記述)
        let mut ext_levels: Vec<f64> = LEVELS_DB.to_vec();
        ext_levels.extend_from_slice(&EXTENDED_DB);
        let (ext_rows, _) = measure(k, &ext_levels, fold);
        let alive: Vec<f64> = ext_rows.iter().filter(|r| r.silent == 0).map(|r| r.db).collect();
        let span = if alive.is_empty() { 0.0 } else {
            alive.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
          - alive.iter().cloned().fold(f64::INFINITY, f64::min) };
        println!("  G76d 射程 (無音母音が出ないレベル範囲・記述) -> {:.0} dB  (下限 {:.0} dB)",
                 span, alive.iter().cloned().fold(f64::INFINITY, f64::min));

        summary.push((k, min_same, max_diff, g76a, g76b, g76c));
    }
    println!();
    println!("=== {} のまとめ ===", rep);
    println!("  K  閾値                    同一母音最小  別母音最大   G76a  G76b  G76c");
    for &(k, mn, mx, a, b, c) in summary.iter() {
        println!("  {}  {:<22} {:>10.4} {:>11.4}   {:<5} {:<5} {}",
                 k, format!("{:?}", thresholds(k)), mn, mx,
                 if a { "PASS" } else { "FAIL" },
                 if b { "PASS" } else { "FAIL" },
                 if c { "**PASS**" } else { "FAIL" });
    }
    summary.clear();
    }

    println!();
    println!("  G76e 対照の健全性: K=1 の行が level_axis の記録 (§14.5.2) と一致するか。");
    println!("    記録: 0dB [35,40,36,39,37] ・ 被覆 0dB 15/15 ・ -21dB と -24dB で無音母音 5");
    println!("    同一母音・別レベル最小 0.0000 / 別母音・同レベル最大 0.9613");
    println!("    **一致しなければこのプローブ自体が信用できない。**");
    println!();
    println!("  【既定は変えていない】これは測定であって採用ではない。");
}
