//! 特徴ごとの伝達情報量 — 人間と同じ順序で壊れているか (2026-08-27)
//!
//! ## 発端 — ユーザーの指摘
//!
//! > **無線通信でもフォネティックコードを使うくらいだから、実際には人間の判別も
//! > そこまで正確ではないのではないか。視覚や知識で言語の音を補完している可能性もある。**
//!
//! **正しい。そしてこれは私への診断になる。**
//!
//! ## 人間側に既知の順序がある (外部の基準)
//!
//! **Miller & Nicely (1955)** が 16 子音の混同行列を雑音・帯域制限の下で測り、
//! **特徴ごとの伝達情報量**を出した。結果:
//!
//! **有声性と鼻音性は最後まで残り、調音位置が真っ先に失われる。**
//!
//! フォネティックコードはその工学的な追認である。無線は 300-3400Hz に帯域制限され、
//! B/C/D/E/G/P/T/V/Z がまとめて潰れる (英語の "E-set" 問題)。
//! だから Bravo・Charlie・Delta という冗長な語に置き換える。
//!
//! 補完も実在する: McGurk 効果 (1976) で視覚が聴覚を上書きし、
//! Ganong 効果 (1980) で語彙知識が曖昧な音素を単語になる方に寄せ、
//! Warren (1970) の音素修復では咳で置き換えた音素が「聞こえて」しまう。
//! **そして視覚が最もよく供給するのは調音位置であり、聴覚が真っ先に失うのも調音位置**
//! という相補になっている。
//!
//! ## なぜこれが良い測定か
//!
//! 1. **正解の出どころが実験者側にある** — どの特徴を与えたかは私が決めた
//! 2. **人間側に既知の順序があるので、比較の基準が外にある** — 私が決めた閾値ではない
//! 3. **合成器の重み付けが正しいかを直接診断できる**
//!
//! ## ゲート (実測前に固定・以後動かさない)
//!
//! - **G85a 特徴ごとの相対伝達情報量**: 有声性 / 調音方法 / 調音位置 の T/H。*記録。*
//! - **G85b 人間との順序**: 人間は **有声性 > 調音位置**。**私の系でもそうなるか。**
//! - **G85c 退化ベースライン**: 全条件が同一ベクトルなら全特徴で T ≈ 0。
//! - **G85d 決定論性**。
//!
//! ## 予測 (実測前・機構つき)
//!
//! **順序は人間と逆になるはず。つまり 調音位置 > 有声性。**
//!
//! 根拠: 私の合成器では**調音位置は帯域指定として強く符号化**されている一方、
//! **有声性は voice bar としてベクトルの約 2.5% しかない** (§14.23.4)。
//! 実際 §14.24 で **合成子音 25.0% に対し濁音は 23 音中 2 音**だった。
//!
//! **これは外れることを期待していない予測である。**
//! 当たれば「合成器の重み付けが生体と逆」の確定になる。
//!
//! ## 測る範囲
//!
//! **母音 5 音は除く** (子音を持たないので子音特徴が定義できない)。64 クラスで測る。
//! Miller & Nicely も子音のみを扱った。
//!
//! ## 留保 (先に書く)
//!
//! - M&N は**英語**の子音を**雑音下**で測った。日本語・無雑音の本測定と
//!   絶対値は比較できない。**比較できるのは順序だけ。**
//! - 同点棄却で「判定不能」になった条件は除く (M&N は強制選択だった)。件数を印字する。
//!
//! CLI: feature_information

use spiking_brain::phase2_f::cochlea::{Cochlea, SAMPLES_PER_STEP};
use spiking_brain::phase2_f::cochlear_nucleus::{CochlearNucleus, N_CN_OUTPUT};
use spiking_brain::phase2_f::kana::{moras_from_kana, synth_utterance};
use spiking_brain::phase2_f::phoneme_synth::LfsrNoise;
use std::collections::HashMap;

/// (かな, 合成子音の記号)。§14.23 の LABELS と同じ 69 クラスから母音 5 を除いた 64。
const KANA: &[(&str, &str)] = &[
    ("か","k"),("き","k"),("く","k"),("け","k"),("こ","k"),
    ("さ","s"),("し","S"),("す","s"),("せ","s"),("そ","s"),
    ("た","t"),("ち","C"),("つ","c"),("て","t"),("と","t"),
    ("な","n"),("に","n"),("ぬ","n"),("ね","n"),("の","n"),
    ("は","h"),("ひ","h"),("ふ","h"),("へ","h"),("ほ","h"),
    ("ま","m"),("み","m"),("む","m"),("め","m"),("も","m"),
    ("や","y"),("ゆ","y"),("よ","y"),
    ("ら","r"),("り","r"),("る","r"),("れ","r"),("ろ","r"),
    ("わ","w"),("を","w"),
    ("ん","N"),
    ("が","g"),("ぎ","g"),("ぐ","g"),("げ","g"),("ご","g"),
    ("ざ","z"),("じ","Z"),("ず","z"),("ぜ","z"),("ぞ","z"),
    ("だ","d"),("で","d"),("ど","d"),
    ("ば","b"),("び","b"),("ぶ","b"),("べ","b"),("ぼ","b"),
    ("ぱ","p"),("ぴ","p"),("ぷ","p"),("ぺ","p"),("ぽ","p"),
];

/// 合成子音の記号 → (有声性, 調音方法, 調音位置)
///
/// **実験者が与えるラベルである。** 合成器がどの帯域・どの機構で作ったかから決まる。
fn features(sym: &str) -> (&'static str, &'static str, &'static str) {
    match sym {
        "k" => ("無声", "破裂", "軟口蓋"),
        "g" => ("有声", "破裂", "軟口蓋"),
        "t" => ("無声", "破裂", "歯茎"),
        "d" => ("有声", "破裂", "歯茎"),
        "p" => ("無声", "破裂", "両唇"),
        "b" => ("有声", "破裂", "両唇"),
        "s" => ("無声", "摩擦", "歯茎"),
        "z" => ("有声", "摩擦", "歯茎"),
        "S" => ("無声", "摩擦", "硬口蓋"),
        "Z" => ("有声", "摩擦", "硬口蓋"),
        "C" => ("無声", "破擦", "硬口蓋"),
        "c" => ("無声", "破擦", "歯茎"),
        "h" => ("無声", "摩擦", "声門"),
        "n" => ("有声", "鼻音", "歯茎"),
        "m" => ("有声", "鼻音", "両唇"),
        "N" => ("有声", "鼻音", "口蓋垂"),
        "r" => ("有声", "弾き", "歯茎"),
        "y" => ("有声", "接近", "硬口蓋"),
        "w" => ("有声", "接近", "両唇"),
        _ => unreachable!("未定義の子音記号: {}", sym),
    }
}

const F0S: [f64; 4] = [100.0, 130.0, 160.0, 200.0];
const CONSONANT_STEPS: usize = 60;

fn utterance_seed(k: usize, v: usize) -> u16 {
    ((k as u16).wrapping_mul(97).wrapping_add(v as u16).wrapping_mul(2851)) | 1
}

fn wave_of(text: &str, f0: f64, seed: u16) -> Vec<i32> {
    let mut n = LfsrNoise::new(seed);
    let (m, sk) = moras_from_kana(text);
    assert_eq!(sk, 0, "未対応: {}", text);
    synth_utterance(&m, f0, &mut n)
}

fn counts(wave: &[i32], windowed: bool) -> Vec<f64> {
    let dim = if windowed { 2 * N_CN_OUTPUT } else { N_CN_OUTPUT };
    let mut co = Cochlea::new();
    let mut cn = CochlearNucleus::new();
    let mut c = vec![0f64; dim];
    for (step, chunk) in wave.chunks(SAMPLES_PER_STEP).enumerate() {
        if chunk.len() < SAMPLES_PER_STEP { break; }
        let m0 = co.process_step(chunk);
        let w = if windowed && step >= CONSONANT_STEPS { 1 } else { 0 };
        for (i, &v) in cn.process_step(&m0).iter().enumerate() {
            if v != 0 { c[w * N_CN_OUTPUT + i] += 1.0; }
        }
    }
    c
}

fn cosine(a: &[f64], b: &[f64]) -> f64 {
    let d: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let na: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
    let nb: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();
    if na == 0.0 || nb == 0.0 { 0.0 } else { d / (na * nb) }
}

/// 同点棄却つき 1-NN。返り値 = (真のクラス, 予測クラス) の列と、判定不能の数。
fn confuse(conds: &[(usize, Vec<f64>)]) -> (Vec<(usize, usize)>, usize) {
    let n = conds.len();
    let mut out = Vec::new();
    let mut undec = 0usize;
    for i in 0..n {
        let mut best = f64::NEG_INFINITY;
        for j in 0..n { if j != i { let c = cosine(&conds[i].1, &conds[j].1); if c > best { best = c; } } }
        let tied: Vec<usize> = (0..n).filter(|&j| j != i && cosine(&conds[i].1, &conds[j].1) == best)
            .map(|j| conds[j].0).collect();
        // 同点でも「同じ子音記号なら一意」とみなす (特徴の分析なのでかなの同一性は問わない)
        let first = KANA[tied[0]].1;
        if tied.iter().all(|&t| KANA[t].1 == first) {
            out.push((conds[i].0, tied[0]));
        } else { undec += 1; }
    }
    (out, undec)
}

/// 特徴ごとの相対伝達情報量 T/H(x)。
///
/// Miller & Nicely (1955) の方法: 混同行列を特徴で畳んで
/// T = H(x) + H(y) − H(x,y) を出し、H(x) で正規化する。
fn transmitted(pairs: &[(usize, usize)], feat: &dyn Fn(&str) -> &'static str) -> (f64, f64, usize) {
    let mut joint: HashMap<(&str, &str), f64> = HashMap::new();
    let mut px: HashMap<&str, f64> = HashMap::new();
    let mut py: HashMap<&str, f64> = HashMap::new();
    let n = pairs.len() as f64;
    for &(t, p) in pairs {
        let (a, b) = (feat(KANA[t].1), feat(KANA[p].1));
        *joint.entry((a, b)).or_insert(0.0) += 1.0 / n;
        *px.entry(a).or_insert(0.0) += 1.0 / n;
        *py.entry(b).or_insert(0.0) += 1.0 / n;
    }
    let h = |m: &HashMap<&str, f64>| -m.values().map(|&p| if p > 0.0 { p * p.log2() } else { 0.0 }).sum::<f64>();
    let hx = h(&px);
    let hy = h(&py);
    let hxy = -joint.values().map(|&p| if p > 0.0 { p * p.log2() } else { 0.0 }).sum::<f64>();
    let t = hx + hy - hxy;
    (if hx > 0.0 { t / hx } else { 0.0 }, hx, px.len())
}

fn main() {
    println!("=== 特徴ごとの伝達情報量 — 人間と同じ順序で壊れているか ===");
    println!();
    println!("【発端】ユーザーの指摘: フォネティックコードを使うくらいだから人間の判別も");
    println!("そこまで正確ではないのでは。視覚や知識で補完している可能性もある。");
    println!();
    println!("【人間側の既知の順序 (外部の基準)】Miller & Nicely (1955):");
    println!("  **有声性と鼻音性は最後まで残り、調音位置が真っ先に失われる。**");
    println!("  フォネティックコードはその工学的追認 (無線は 300-3400Hz で E-set が潰れる)。");
    println!("  補完も実在: McGurk(1976) 視覚が聴覚を上書き / Ganong(1980) 語彙 /");
    println!("  Warren(1970) 音素修復。**視覚が最もよく供給するのも、聴覚が真っ先に失うのも");
    println!("  調音位置**という相補になっている。");
    println!();
    println!("【ゲート・実測前に固定】正解の出どころ = どの特徴を与えたかは実験者が決めた");
    println!("  G85a 特徴ごとの相対伝達情報量 T/H (記録)");
    println!("  G85b **人間との順序**: 人間は 有声性 > 調音位置。私の系でもそうなるか");
    println!("  G85c 退化ベースライン (全同一なら T≈0)   G85d 決定論性");
    println!();
    println!("【予測・事前】**順序は人間と逆になるはず (調音位置 > 有声性)。**");
    println!("  合成器では調音位置は帯域指定として強く符号化され、有声性は voice bar として");
    println!("  ベクトルの約 2.5% しかない。§14.24 で合成子音 25.0% に対し濁音は 23音中2音。");
    println!("  **外れることを期待していない予測。当たれば『重み付けが生体と逆』の確定。**");
    println!();
    println!("【留保】M&N は英語・雑音下。**比較できるのは順序だけで絶対値ではない。**");
    println!("母音 5 音は除く (子音特徴が定義できない)。64 クラス × 4 変種 = 256 条件。");

    for &(windowed, name) in [(false, "時間平均"), (true, "2窓 (子音区間/母音区間)")].iter() {
        let mut conds: Vec<(usize, Vec<f64>)> = Vec::new();
        for (k, &(kana, _)) in KANA.iter().enumerate() {
            for (v, &f0) in F0S.iter().enumerate() {
                conds.push((k, counts(&wave_of(kana, f0, utterance_seed(k, v)), windowed)));
            }
        }
        let (pairs, undec) = confuse(&conds);
        let correct = pairs.iter().filter(|(t, p)| KANA[*t].1 == KANA[*p].1).count();

        println!();
        println!("################ {} ################", name);
        println!("  条件 {} / 判定不能 {} / 子音記号が一致 {} = {:.1}%",
                 conds.len(), undec, correct, correct as f64 / pairs.len() as f64 * 100.0);
        println!();
        println!("  {:<10} {:>10} {:>12} {:>10}", "特徴", "H(x) [bit]", "T/H(x)", "値の数");
        let mut results = Vec::new();
        for &(fname, f) in [
            ("有声性", &(|s: &str| features(s).0) as &dyn Fn(&str) -> &'static str),
            ("調音方法", &(|s: &str| features(s).1) as &dyn Fn(&str) -> &'static str),
            ("調音位置", &(|s: &str| features(s).2) as &dyn Fn(&str) -> &'static str),
        ].iter() {
            let (rel, hx, nv) = transmitted(&pairs, f);
            println!("  {:<10} {:>10.3} {:>11.1}% {:>10}", fname, hx, rel * 100.0, nv);
            results.push((fname, rel));
        }
        let voi = results.iter().find(|(n, _)| *n == "有声性").unwrap().1;
        let pla = results.iter().find(|(n, _)| *n == "調音位置").unwrap().1;
        println!();
        println!("  **G85b 順序**: 有声性 {:.1}% vs 調音位置 {:.1}% -> {}",
                 voi * 100.0, pla * 100.0,
                 if voi > pla { "**人間と同じ (有声性 > 位置)**" }
                 else { "**人間と逆 (位置 > 有声性) — 予測どおり**" });
    }

    // --- G85c 退化ベースライン ---
    let degen: Vec<(usize, Vec<f64>)> = (0..KANA.len())
        .flat_map(|k| (0..F0S.len()).map(move |_| (k, vec![1f64; N_CN_OUTPUT]))).collect();
    let (dp, du) = confuse(&degen);
    println!();
    println!("--- G85c 退化ベースライン (全条件が同一ベクトル) ---");
    if dp.is_empty() {
        println!("  全条件が判定不能 ({} 件) -> **PASS** (同点棄却が効いている)", du);
    } else {
        let (v, _, _) = transmitted(&dp, &|s: &str| features(s).0);
        println!("  有声性の T/H = {:.3}% (判定不能 {}) -> {}",
                 v * 100.0, du, if v.abs() < 0.01 { "**PASS**" } else { "**FAIL**" });
    }

    println!();
    println!("  【この測定が答えないこと】人間の絶対値との比較はしていない (条件が違う)。");
    println!("  視覚・語彙による補完もモデルに入っていない。**既定は変えていない。**");
}
