//! 鼻音・接近音を声帯源で駆動したら悪化した — 正しい物理か、壊したのか (2026-08-27)
//!
//! ## 何が起きたか
//!
//! 旧経路は鼻音・接近音を `sin_lookup` の**純音 2 本**で作っており、
//! **声帯源を通らず F0 を一切使っていなかった** (§14.32 で出荷コード確認)。
//! **鼻音も接近音も有声音なので、これは欠陥である。** 声帯源で駆動するように直した。
//!
//! **すると同定が大きく下がった** (② 孤立×2窓 の合成子音 51.1% → 36.6%)。
//!
//! ## 切り分け — 合理化しないために
//!
//! 事前に挙げた機構は 2 つあった:
//! 1. 倍音が入って な/ま の F2 差 (1700 vs 1500) が場所符号に出る → **良くなるはず**
//! 2. これまで鼻音の子音区間は **F0 で完全に同一**だったので、
//!    **4 変種を跨いだ照合に人工的に有利**だった → **それが失われる**
//!
//! **「②が効いた」で片付けるのは合理化である。** 決定的な切り分けはこれ:
//!
//! - **下がったのが有声共鳴音 16 かなに集中しているなら**、正しい物理が出ただけ。
//!   (実音声でも鼻音・接近音は母音に似ており、混同されやすい。
//!    Miller & Nicely も鼻音性は頑健だが鼻音**内**の位置 m/n は脆いと報告している。)
//! - **残り 53 かなも下がっているなら**、共鳴音が他を巻き込んでいる = 何かを壊した。
//!
//! ## ゲート (実測前に固定・以後動かさない)
//!
//! - **G93a 下がり方は共鳴音に集中しているか**: 16 かなの低下 vs 53 かなの低下。
//! - **G93b 共鳴音は母音に似すぎていないか**: 共鳴音と母音単独のコサイン (OFF/ON)。
//!   *実音声でも似ているが、「似すぎ」なら区別できない。*
//! - **G93c 鼻音どうしの区別は良くなったか**: な/ま/ん の対のコサイン (OFF/ON)。
//!   *機構①が効いたかを直接見る。*
//! - **G93d 決定論性**。
//!
//! ## 予測 (実測前)
//!
//! **G93c は良くなるはず** (倍音が入るので F2 の違いが出る)。
//! **G93a は分からない。** これが本題。
//! **もし 53 かなも同じくらい下がっていたら、この修正は取り下げるべき。**
//!
//! CLI: sonorant_check

use spiking_brain::phase2_f::cochlea::{Cochlea, SAMPLES_PER_STEP};
use spiking_brain::phase2_f::cochlear_nucleus::{CochlearNucleus, N_CN_OUTPUT};
use spiking_brain::phase2_f::kana::{moras_from_kana, synth_utterance};
use spiking_brain::phase2_f::phoneme_synth::{set_voiced_sonorant, LfsrNoise};

const F0S: [f64; 4] = [100.0, 130.0, 160.0, 200.0];
const CONSONANT_STEPS: usize = 60;

/// (かな, 有声共鳴音か, 母音単独か) — §14.23 の 69 クラス
const LABELS: &[(&str, bool, bool)] = &[
    ("あ",false,true),("い",false,true),("う",false,true),("え",false,true),("お",false,true),
    ("か",false,false),("き",false,false),("く",false,false),("け",false,false),("こ",false,false),
    ("さ",false,false),("し",false,false),("す",false,false),("せ",false,false),("そ",false,false),
    ("た",false,false),("ち",false,false),("つ",false,false),("て",false,false),("と",false,false),
    ("な",true,false),("に",true,false),("ぬ",true,false),("ね",true,false),("の",true,false),
    ("は",false,false),("ひ",false,false),("ふ",false,false),("へ",false,false),("ほ",false,false),
    ("ま",true,false),("み",true,false),("む",true,false),("め",true,false),("も",true,false),
    ("や",true,false),("ゆ",true,false),("よ",true,false),
    ("ら",false,false),("り",false,false),("る",false,false),("れ",false,false),("ろ",false,false),
    ("わ",true,false),("を",true,false),
    ("ん",true,false),
    ("が",false,false),("ぎ",false,false),("ぐ",false,false),("げ",false,false),("ご",false,false),
    ("ざ",false,false),("じ",false,false),("ず",false,false),("ぜ",false,false),("ぞ",false,false),
    ("だ",false,false),("で",false,false),("ど",false,false),
    ("ば",false,false),("び",false,false),("ぶ",false,false),("べ",false,false),("ぼ",false,false),
    ("ぱ",false,false),("ぴ",false,false),("ぷ",false,false),("ぺ",false,false),("ぽ",false,false),
];

fn utterance_seed(k: usize, v: usize) -> u16 {
    ((k as u16).wrapping_mul(97).wrapping_add(v as u16).wrapping_mul(2851)) | 1
}

/// 孤立 × 2窓 (最良アーム) の条件ベクトル
fn build() -> Vec<(usize, Vec<f64>)> {
    let dim = 2 * N_CN_OUTPUT;
    let mut out = Vec::new();
    for v in 0..4 {
        for k in 0..LABELS.len() {
            let mut n = LfsrNoise::new(utterance_seed(k, v));
            let (m, sk) = moras_from_kana(LABELS[k].0);
            assert_eq!(sk, 0);
            let w = synth_utterance(&m, F0S[v], &mut n);
            let (mut co, mut cn) = (Cochlea::new(), CochlearNucleus::new());
            let mut c = vec![0f64; dim];
            for (step, chunk) in w.chunks(SAMPLES_PER_STEP).enumerate() {
                if chunk.len() < SAMPLES_PER_STEP { break; }
                let m0 = co.process_step(chunk);
                let win = if step >= CONSONANT_STEPS { 1 } else { 0 };
                for (i, &x) in cn.process_step(&m0).iter().enumerate() {
                    if x != 0 { c[win * N_CN_OUTPUT + i] += 1.0; }
                }
            }
            out.push((k, c));
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

/// 同点棄却つき 1-NN。かな単位の正誤を返す (棄却は不正解)。
fn correct(v: &[(usize, Vec<f64>)]) -> Vec<(usize, bool)> {
    let n = v.len();
    let mut out = Vec::new();
    for i in 0..n {
        let mut best = f64::NEG_INFINITY;
        for j in 0..n { if j != i { let c = cosine(&v[i].1, &v[j].1); if c > best { best = c; } } }
        let tied: Vec<usize> = (0..n).filter(|&j| j != i && cosine(&v[i].1, &v[j].1) == best)
            .map(|j| v[j].0).collect();
        let ok = !tied.is_empty() && tied.iter().all(|&t| LABELS[t].0 == LABELS[tied[0]].0)
                 && tied[0] == v[i].0;
        out.push((v[i].0, ok));
    }
    out
}

struct R { son: f64, oth: f64, son_vs_vowel: f64, nasal_pair: f64 }

fn eval() -> R {
    let v = build();
    let c = correct(&v);
    let f = |sel: &dyn Fn(usize) -> bool| {
        let sub: Vec<&(usize, bool)> = c.iter().filter(|(k, _)| sel(*k)).collect();
        sub.iter().filter(|(_, ok)| *ok).count() as f64 / sub.len().max(1) as f64 * 100.0
    };
    // 共鳴音 vs 母音単独 の平均コサイン
    let (mut sv, mut nsv) = (0f64, 0usize);
    let (mut np, mut nnp) = (0f64, 0usize);
    for i in 0..v.len() {
        for j in (i + 1)..v.len() {
            let (a, b) = (v[i].0, v[j].0);
            if LABELS[a].1 && LABELS[b].2 { sv += cosine(&v[i].1, &v[j].1); nsv += 1; }
            if LABELS[b].1 && LABELS[a].2 { sv += cosine(&v[i].1, &v[j].1); nsv += 1; }
            // 鼻音・接近音どうし (別のかな)
            if LABELS[a].1 && LABELS[b].1 && a != b { np += cosine(&v[i].1, &v[j].1); nnp += 1; }
        }
    }
    R {
        son: f(&|k| LABELS[k].1),
        oth: f(&|k| !LABELS[k].1),
        son_vs_vowel: sv / nsv.max(1) as f64,
        nasal_pair: np / nnp.max(1) as f64,
    }
}

fn main() {
    println!("=== 鼻音・接近音を声帯源で駆動したら悪化した — 正しい物理か、壊したのか ===");
    println!();
    println!("【何が起きたか】旧経路は鼻音・接近音を **純音2本** で作り声帯源を通らず");
    println!("F0 を一切使っていなかった。**有声音なのでこれは欠陥**。声帯源で駆動するよう直した。");
    println!("**すると同定が大きく下がった** (② 孤立×2窓 の合成子音 51.1% -> 36.6%)。");
    println!();
    println!("【切り分け・合理化しないために】事前に挙げた機構は2つ:");
    println!("  ① 倍音が入って な/ま の F2差(1700 vs 1500) が出る -> **良くなるはず**");
    println!("  ② これまで鼻音の子音区間は **F0で完全に同一** -> **4変種を跨いだ照合に");
    println!("     人工的に有利だった** -> それが失われる");
    println!("**「②が効いた」で片付けるのは合理化。**決定的な切り分けはこれ:");
    println!("  **下がったのが共鳴音16かなに集中していれば** 正しい物理が出ただけ。");
    println!("  **残り53かなも下がっているなら** 共鳴音が他を巻き込んでいる = 壊した。");
    println!();
    println!("【ゲート・実測前に固定】");
    println!("  **G93a 下がり方は共鳴音に集中しているか** (16かな vs 53かな)");
    println!("  G93b 共鳴音は母音に似すぎていないか / G93c 鼻音どうしの区別は良くなったか");
    println!("  G93d 決定論性");
    println!();
    println!("【予測・事前】**G93c は良くなるはず**(倍音でF2差が出る)。**G93a は分からない。これが本題。**");
    println!("**もし53かなも同じくらい下がっていたら、この修正は取り下げるべき。**");

    set_voiced_sonorant(false);
    let off = eval();
    set_voiced_sonorant(true);
    let on = eval();

    println!();
    println!("--- 孤立 × 2窓 (最良アーム)・かな同定 ---");
    println!("  {:<28} {:>10} {:>10} {:>10}", "", "OFF", "ON", "差");
    println!("  {:<28} {:>9.1}% {:>9.1}% {:>+9.1}", "**有声共鳴音 16 かな**", off.son, on.son, on.son - off.son);
    println!("  {:<28} {:>9.1}% {:>9.1}% {:>+9.1}", "残り 53 かな", off.oth, on.oth, on.oth - off.oth);
    println!();
    let concentrated = (on.son - off.son) < (on.oth - off.oth) - 5.0;
    println!("  **G93a 下がり方は共鳴音に集中しているか** -> {}",
             if concentrated { "**はい — 正しい物理が出たと読める**" }
             else { "**いいえ — 共鳴音が残り53かなも巻き込んでいる**" });

    println!();
    println!("--- G93b/G93c 何が似ているか (平均コサイン) ---");
    println!("  {:<28} {:>10} {:>10} {:>10}", "", "OFF", "ON", "差");
    println!("  {:<28} {:>10.4} {:>10.4} {:>+10.4}", "共鳴音 vs 母音単独", off.son_vs_vowel, on.son_vs_vowel, on.son_vs_vowel - off.son_vs_vowel);
    println!("  {:<28} {:>10.4} {:>10.4} {:>+10.4}", "共鳴音どうし", off.nasal_pair, on.nasal_pair, on.nasal_pair - off.nasal_pair);
    println!();
    println!("  G93b 共鳴音が母音に近づいたか -> {}",
             if on.son_vs_vowel > off.son_vs_vowel { "**近づいた** (実音声でも共鳴音は母音に似ている)" } else { "遠ざかった" });
    println!("  **G93c 共鳴音どうしの区別は良くなったか** -> {}",
             if on.nasal_pair < off.nasal_pair { "**良くなった (コサインが下がった) — 予測どおり**" }
             else { "**良くなっていない — 予測が外れた**" });

    let a = eval();
    println!();
    println!("  G93d 決定論性 -> {}", if (a.son - on.son).abs() < 1e-12 { "PASS" } else { "**FAIL**" });
    println!();
    println!("  【この測定が答えないこと】**どちらを採るべきかは決めない。**");
    println!("  「同定率が高い」と「刺激が音声として正しい」は**別の軸**である。");
}
