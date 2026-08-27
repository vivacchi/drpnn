//! 文脈 × 適応 × 時間分解 を全部組み合わせる (2026-08-27)
//!
//! ## なぜ
//!
//! 3 つの効果を**別々に**測ってきたが、**一度も同時に測っていない**。
//!
//! | 効果 | 出典 | 合成子音の同定 |
//! |---|---|---|
//! | **文脈** (連続発話) | §14.19 | 8.9% → **21.7%** |
//! | **時間分解** (2窓) | §14.24 | 7.6% → **25.0%** |
//! | 適応 (平衡) | §14.19 | 21.7% → 10.3% (**害した**) |
//!
//! **掛け算になるのか、食い合うのかが分かっていない。**
//!
//! ## 条件 (6 アーム)
//!
//! §14.19 で「**リセットしなければ孤立させたつもりでも自動的に連続になる**」
//! (孤立と平衡は排他的) と分かったので、文脈は 3 通り:
//!
//! - **孤立** (= 必然的に冷開始)
//! - **連続・1 パス目**
//! - **連続・平衡** (ウォームアップ後)
//!
//! これに 時間平均 / 2窓 を掛けて 6 アーム。
//!
//! ## ゲート (実測前に固定・以後動かさない)
//!
//! **正解の出どころ**: どのかなをどの順で合成したかは実験者が決めた。
//!
//! - **G86a 最良の組み合わせ**: 6 アームで合成子音の同定が最高になるのはどれか。*記録。*
//! - **G86b 掛け算か食い合いか**: 文脈だけの効果 (③−①) と 2窓だけの効果 (②−①) の
//!   **和** に対して、**両方** (④−①) がどうなるか。
//!   *和 ≈ 両方 → 加算的 / 両方 > 和 → 相乗 / 両方 < 和 → **食い合い***
//! - **G86c 特徴ごとの順序**: **全アームで 有声性 > 調音位置** が保たれるか
//!   (§14.25 で人間 (Miller & Nicely) と一致した順序)。
//! - **G86d 決定論性**。
//!
//! ## 予測 (実測前・機構つき・事前)
//!
//! **食い合うはず。** 文脈も 2窓も「**子音を際立たせて母音を薄める**」という
//! 同じ機構に依っている:
//! - 文脈: 適応が直前の母音の帯域を抑える (§14.17)
//! - 2窓: 子音区間を母音区間から切り離す (§14.24)
//!
//! どちらも**子音が得をする分だけ母音が損をする**という同じ形を示した
//! (§14.19: 母音列 92.8% → 65.2% / §14.24: 94.6% → 82.6%)。
//! **二重には効かないはず。数値は置かない。**
//!
//! CLI: kana_full_factorial

use spiking_brain::phase2_f::cochlea::{Cochlea, SAMPLES_PER_STEP};
use spiking_brain::phase2_f::cochlear_nucleus::{CochlearNucleus, N_CN_OUTPUT};
use spiking_brain::phase2_f::kana::{moras_from_kana, synth_utterance, MORA_MS};
use spiking_brain::phase2_f::phoneme_synth::LfsrNoise;
use std::collections::HashMap;

/// (かな, 母音列, 合成子音の記号) — §14.23 の 69 クラス
const LABELS: &[(&str, &str, &str)] = &[
    ("あ","あ","-"),("い","い","-"),("う","う","-"),("え","え","-"),("お","お","-"),
    ("か","あ","k"),("き","い","k"),("く","う","k"),("け","え","k"),("こ","お","k"),
    ("さ","あ","s"),("し","い","S"),("す","う","s"),("せ","え","s"),("そ","お","s"),
    ("た","あ","t"),("ち","い","C"),("つ","う","c"),("て","え","t"),("と","お","t"),
    ("な","あ","n"),("に","い","n"),("ぬ","う","n"),("ね","え","n"),("の","お","n"),
    ("は","あ","h"),("ひ","い","h"),("ふ","う","h"),("へ","え","h"),("ほ","お","h"),
    ("ま","あ","m"),("み","い","m"),("む","う","m"),("め","え","m"),("も","お","m"),
    ("や","あ","y"),("ゆ","う","y"),("よ","お","y"),
    ("ら","あ","r"),("り","い","r"),("る","う","r"),("れ","え","r"),("ろ","お","r"),
    ("わ","あ","w"),("を","お","w"),
    ("ん","ん","N"),
    ("が","あ","g"),("ぎ","い","g"),("ぐ","う","g"),("げ","え","g"),("ご","お","g"),
    ("ざ","あ","z"),("じ","い","Z"),("ず","う","z"),("ぜ","え","z"),("ぞ","お","z"),
    ("だ","あ","d"),("で","え","d"),("ど","お","d"),
    ("ば","あ","b"),("び","い","b"),("ぶ","う","b"),("べ","え","b"),("ぼ","お","b"),
    ("ぱ","あ","p"),("ぴ","い","p"),("ぷ","う","p"),("ぺ","え","p"),("ぽ","お","p"),
];

/// 合成子音の記号 → (有声性, 調音方法, 調音位置) — §14.25 と同一
fn feat3(sym: &str) -> (&'static str, &'static str, &'static str) {
    match sym {
        "-" => ("有声", "母音", "なし"),
        "k" => ("無声", "破裂", "軟口蓋"), "g" => ("有声", "破裂", "軟口蓋"),
        "t" => ("無声", "破裂", "歯茎"),   "d" => ("有声", "破裂", "歯茎"),
        "p" => ("無声", "破裂", "両唇"),   "b" => ("有声", "破裂", "両唇"),
        "s" => ("無声", "摩擦", "歯茎"),   "z" => ("有声", "摩擦", "歯茎"),
        "S" => ("無声", "摩擦", "硬口蓋"), "Z" => ("有声", "摩擦", "硬口蓋"),
        "C" => ("無声", "破擦", "硬口蓋"), "c" => ("無声", "破擦", "歯茎"),
        "h" => ("無声", "摩擦", "声門"),
        "n" => ("有声", "鼻音", "歯茎"),   "m" => ("有声", "鼻音", "両唇"),
        "N" => ("有声", "鼻音", "口蓋垂"),
        "r" => ("有声", "弾き", "歯茎"),
        "y" => ("有声", "接近", "硬口蓋"), "w" => ("有声", "接近", "両唇"),
        _ => unreachable!("未定義: {}", sym),
    }
}

const F0S: [f64; 4] = [100.0, 130.0, 160.0, 200.0];
const N_VAR: usize = 4;
const WARMUP: usize = 3;
const STEPS_PER_MORA: usize = (MORA_MS as usize) * 16 / SAMPLES_PER_STEP;
/// 子音区間の終わり (CONSONANT_MS=30ms ÷ 0.5ms)。
/// **DRPNN_WINDOW_STEPS で動かせる** — §14.38 で「2窓の利得は合成器の谷を見ているだけか」を
/// 切り分けるために追加。谷由来なら 60 で鋭いピークになるはず。
fn consonant_steps() -> usize {
    std::env::var("DRPNN_WINDOW_STEPS").ok().and_then(|v| v.parse().ok()).unwrap_or(60)
}

fn utterance_seed(k: usize, v: usize) -> u16 {
    ((k as u16).wrapping_mul(97).wrapping_add(v as u16).wrapping_mul(2851)) | 1
}

fn order_for(v: usize) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..LABELS.len()).collect();
    let mut s = 0xC0FF_EE00_1234_5678u64 ^ ((v as u64) << 32);
    for i in (1..idx.len()).rev() {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        idx.swap(i, ((s >> 33) as usize) % (i + 1));
    }
    idx
}

fn mora_wave(k: usize, f0: f64, seed: u16) -> Vec<i32> {
    let mut n = LfsrNoise::new(seed);
    let (m, sk) = moras_from_kana(LABELS[k].0);
    assert_eq!(sk, 0);
    synth_utterance(&m, f0, &mut n)
}

#[derive(Clone, Copy, PartialEq)]
enum Ctx { Isolated, ContFirst, ContEq }

/// 1 アーム分の条件ベクトル。`windowed` で 時間平均 / 2窓 を切り替える。
fn build(ctx: Ctx, windowed: bool) -> Vec<(usize, Vec<f64>)> {
    let dim = if windowed { 2 * N_CN_OUTPUT } else { N_CN_OUTPUT };
    let mut out: Vec<(usize, Vec<f64>)> = Vec::new();
    for v in 0..N_VAR {
        let ord = order_for(v);
        let f0 = F0S[v];
        if ctx == Ctx::Isolated {
            // 孤立 = かなごとに新しい蝸牛と神経核 (= 必然的に冷開始)
            for &k in ord.iter() {
                let w = mora_wave(k, f0, utterance_seed(k, v));
                let mut co = Cochlea::new();
                let mut cn = CochlearNucleus::new();
                let mut c = vec![0f64; dim];
                for (step, chunk) in w.chunks(SAMPLES_PER_STEP).enumerate() {
                    if chunk.len() < SAMPLES_PER_STEP { break; }
                    let m0 = co.process_step(chunk);
                    let win = if windowed && step >= consonant_steps() { 1 } else { 0 };
                    for (i, &x) in cn.process_step(&m0).iter().enumerate() {
                        if x != 0 { c[win * N_CN_OUTPUT + i] += 1.0; }
                    }
                }
                out.push((k, c));
            }
        } else {
            // 連続: 46+ かなを 1 本の発話につなぎ、各モーラの窓だけ数える
            let mut wave: Vec<i32> = Vec::new();
            for &k in ord.iter() { wave.extend_from_slice(&mora_wave(k, f0, utterance_seed(k, v))); }
            let mut co = Cochlea::new();
            let mut cn = CochlearNucleus::new();
            if ctx == Ctx::ContEq {
                for _ in 0..WARMUP {
                    for chunk in wave.chunks(SAMPLES_PER_STEP) {
                        if chunk.len() < SAMPLES_PER_STEP { break; }
                        let m0 = co.process_step(chunk);
                        let _ = cn.process_step(&m0);
                    }
                }
            }
            let mut c = vec![vec![0f64; dim]; ord.len()];
            for (step, chunk) in wave.chunks(SAMPLES_PER_STEP).enumerate() {
                if chunk.len() < SAMPLES_PER_STEP { break; }
                let m0 = co.process_step(chunk);
                let mi = step / STEPS_PER_MORA;
                if mi >= ord.len() { continue; }
                let inner = step % STEPS_PER_MORA;
                let win = if windowed && inner >= consonant_steps() { 1 } else { 0 };
                for (i, &x) in cn.process_step(&m0).iter().enumerate() {
                    if x != 0 { c[mi][win * N_CN_OUTPUT + i] += 1.0; }
                }
            }
            for (mi, &k) in ord.iter().enumerate() { out.push((k, c[mi].clone())); }
        }
    }
    out
}

fn cosine(a: &[f64], b: &[f64]) -> f64 {
    let d: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let na: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
    let nb: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();
    if na == 0.0 || nb == 0.0 { 0.0 } else { d / (na * nb) }
}

/// 同点棄却つき 1-NN。返り値 = (真, 予測) の列と判定不能数。
fn nn(conds: &[(usize, Vec<f64>)], label: &dyn Fn(usize) -> &'static str) -> (Vec<(usize, usize)>, usize) {
    let n = conds.len();
    let (mut out, mut undec) = (Vec::new(), 0usize);
    for i in 0..n {
        let mut best = f64::NEG_INFINITY;
        for j in 0..n { if j != i { let c = cosine(&conds[i].1, &conds[j].1); if c > best { best = c; } } }
        let tied: Vec<usize> = (0..n).filter(|&j| j != i && cosine(&conds[i].1, &conds[j].1) == best)
            .map(|j| conds[j].0).collect();
        let f = label(tied[0]);
        if tied.iter().all(|&t| label(t) == f) { out.push((conds[i].0, tied[0])); } else { undec += 1; }
    }
    (out, undec)
}

fn l_kana(t: usize) -> &'static str { LABELS[t].0 }
fn l_vowel(t: usize) -> &'static str { LABELS[t].1 }
fn l_cons(t: usize) -> &'static str { LABELS[t].2 }

fn acc(conds: &[(usize, Vec<f64>)], label: &dyn Fn(usize) -> &'static str) -> f64 {
    let (p, _) = nn(conds, label);
    p.iter().filter(|(t, q)| label(*t) == label(*q)).count() as f64 / conds.len() as f64 * 100.0
}

/// 特徴ごとの相対伝達情報量 T/H(x) (Miller & Nicely の方法)
fn transmitted(pairs: &[(usize, usize)], f: &dyn Fn(&str) -> &'static str) -> f64 {
    let mut joint: HashMap<(&str, &str), f64> = HashMap::new();
    let (mut px, mut py): (HashMap<&str, f64>, HashMap<&str, f64>) = (HashMap::new(), HashMap::new());
    let n = pairs.len() as f64;
    for &(t, p) in pairs {
        let (a, b) = (f(LABELS[t].2), f(LABELS[p].2));
        *joint.entry((a, b)).or_insert(0.0) += 1.0 / n;
        *px.entry(a).or_insert(0.0) += 1.0 / n;
        *py.entry(b).or_insert(0.0) += 1.0 / n;
    }
    let h = |m: &HashMap<&str, f64>| -m.values().map(|&p| if p > 0.0 { p * p.log2() } else { 0.0 }).sum::<f64>();
    let hxy = -joint.values().map(|&p| if p > 0.0 { p * p.log2() } else { 0.0 }).sum::<f64>();
    let hx = h(&px);
    if hx > 0.0 { (hx + h(&py) - hxy) / hx } else { 0.0 }
}

struct R { kana: f64, vowel: f64, cons: f64, voi: f64, man: f64, pla: f64, undec: usize }

fn eval(ctx: Ctx, windowed: bool) -> R {
    let c = build(ctx, windowed);
    let (pairs, undec) = nn(&c, &l_cons);
    R {
        kana: acc(&c, &l_kana), vowel: acc(&c, &l_vowel), cons: acc(&c, &l_cons),
        voi: transmitted(&pairs, &|s| feat3(s).0),
        man: transmitted(&pairs, &|s| feat3(s).1),
        pla: transmitted(&pairs, &|s| feat3(s).2),
        undec,
    }
}

fn main() {
    println!("=== 文脈 × 適応 × 時間分解 を全部組み合わせる ===");
    println!();
    println!("【なぜ】3 つの効果を別々に測ってきたが**一度も同時に測っていない**。");
    println!("  文脈(連続発話) §14.19: 合成子音 8.9% -> 21.7%");
    println!("  時間分解(2窓)  §14.24: 合成子音 7.6% -> 25.0%");
    println!("  適応(平衡)     §14.19: 21.7% -> 10.3% (**害した**)");
    println!("**掛け算になるのか食い合うのかが分かっていない。**");
    println!();
    println!("【条件】§14.19 で「リセットしなければ孤立させたつもりでも自動的に連続になる」");
    println!("(孤立と平衡は排他的) と分かったので文脈は 3 通り × 時間 2 通り = 6 アーム。");
    println!();
    println!("【ゲート・実測前に固定】");
    println!("  G86a 最良の組み合わせ (記録)");
    println!("  G86b **掛け算か食い合いか**: (③−①)+(②−①) に対して (④−①) がどうか");
    println!("  G86c 全アームで **有声性 > 調音位置** が保たれるか (§14.25 の人間の順序)");
    println!("  G86d 決定論性");
    println!();
    println!("【予測・事前・機構つき】**食い合うはず。**");
    println!("  文脈も 2窓も「**子音を際立たせて母音を薄める**」という同じ機構に依っている。");
    println!("  どちらも「子音が得をする分だけ母音が損をする」を示した");
    println!("  (§14.19 母音 92.8%->65.2% / §14.24 94.6%->82.6%)。**二重には効かないはず。**");

    let arms = [
        ("① 孤立 × 時間平均", Ctx::Isolated, false),
        ("② 孤立 × 2窓", Ctx::Isolated, true),
        ("③ 連続1パス × 時間平均", Ctx::ContFirst, false),
        ("④ 連続1パス × 2窓", Ctx::ContFirst, true),
        ("⑤ 連続平衡 × 時間平均", Ctx::ContEq, false),
        ("⑥ 連続平衡 × 2窓", Ctx::ContEq, true),
    ];

    println!();
    println!("  {:<24} {:>8} {:>8} {:>10} | {:>8} {:>8} {:>8} {:>6}",
             "アーム", "かな", "母音列", "**合成子音**", "有声性", "方法", "位置", "不能");
    let mut res = Vec::new();
    for &(nm, ctx, w) in arms.iter() {
        let r = eval(ctx, w);
        println!("  {:<24} {:>7.1}% {:>7.1}% {:>9.1}% | {:>7.1}% {:>7.1}% {:>7.1}% {:>6}",
                 nm, r.kana, r.vowel, r.cons, r.voi * 100.0, r.man * 100.0, r.pla * 100.0, r.undec);
        res.push(r);
    }
    println!("  (チャンス: かな 1.09% / 母音列 約19% / 合成子音 5.81%)");

    // --- G86a ---
    let best = res.iter().enumerate().max_by(|a, b| a.1.cons.partial_cmp(&b.1.cons).unwrap()).unwrap().0;
    println!();
    println!("  G86a 合成子音が最高のアーム -> **{}** ({:.1}%)", arms[best].0, res[best].cons);

    // --- G86b 掛け算か食い合いか ---
    let base = res[0].cons;
    let d_win = res[1].cons - base;   // 2窓だけ
    let d_ctx = res[2].cons - base;   // 文脈だけ
    let d_both = res[3].cons - base;  // 両方
    println!();
    println!("  G86b 掛け算か食い合いか:");
    println!("    ① 孤立×時間平均 (基準)       {:.1}%", base);
    println!("    ② 2窓だけの効果   (②−①)     {:+.1}pt", d_win);
    println!("    ③ 文脈だけの効果  (③−①)     {:+.1}pt", d_ctx);
    println!("    **和 (②−①)+(③−①)**         {:+.1}pt", d_win + d_ctx);
    println!("    **④ 両方 (④−①)**            {:+.1}pt", d_both);
    let verdict = if d_both > (d_win + d_ctx) * 1.1 { "**相乗 (両方 > 和)**" }
        else if d_both < (d_win + d_ctx) * 0.9 { "**食い合い (両方 < 和) — 予測どおり**" }
        else { "**加算的 (両方 ≈ 和)**" };
    println!("    -> {}", verdict);

    // --- G86c 特徴の順序 ---
    println!();
    let all_ok = res.iter().all(|r| r.voi > r.pla);
    println!("  G86c 全アームで 有声性 > 調音位置 -> {}",
             if all_ok { "**PASS — 人間 (Miller & Nicely) の順序を保つ**" } else { "**FAIL**" });

    // --- G86d 決定論性 ---
    let a = eval(Ctx::ContFirst, true);
    let b = eval(Ctx::ContFirst, true);
    println!("  G86d 決定論性 -> {}",
             if (a.cons - b.cons).abs() < 1e-12 && (a.voi - b.voi).abs() < 1e-12 { "PASS" } else { "**FAIL**" });

    println!();
    println!("  【留保】単一シード。窓の境界は合成器の定数由来だが **M1 は持てない分節知識**。");
    println!("  **既定は変えていない。**");
}
