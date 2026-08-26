//! 母音の後に置いた子音は強調されるか — 物理信号は同じで、応答だけが変わるか (2026-08-26)
//!
//! ## 発端 — ユーザーの指摘
//!
//! > **母音と子音の認識は切り離せないのではないか。母音のすぐ後に続く子音信号は、
//! > どこかで物理層での信号は変わらないけど、入力に対して信号処理部位で
//! > 強調される気がしないか。**
//!
//! **これは実在する現象で、心理音響では「聴覚の強調効果 (auditory enhancement)」と呼ばれる。**
//!
//! 機構: **適応は周波数ごとに起きる。** 母音が鳴っている間、そのフォルマント帯域の
//! 応答は適応して落ちる。次に来る子音は**違う帯域**を使うので、そちらは適応していない。
//! **物理信号が同じでも、適応済みの背景に対する相対的な応答が上がる。**
//!
//! ## 私の測定は、この効果が原理的に出ない条件だった
//!
//! `consonant_probe` は**孤立した 30ms・冷開始**で測っている。適応がゼロなので
//! 強調が起きようがない。**§14.6.7 で「孤立 1 モーラの冷開始」と自分で留保に書いたのに、
//! §14.15.5 の子音劣化 (重心が中央に寄る) をその条件のまま結論していた。**
//!
//! ## 機構は既にあるか
//!
//! - **M0 (蝸牛)**: `EnvelopeDetector` は漏れ積分で、適応 (疲労) は持たない。
//!   `FireGenerator` の `spike_cost` 蓄積が弱い適応にあたる。
//! - **M0.5 (蝸牛神経核)**: `local_entropy` が**チャネルごとに**実効閾値を上げる。
//!   **これが周波数固有の適応そのもの。** §12.14-15 で累積失聴として調整した機構。
//!
//! **よって強調が起きるなら M0.5 のはず。** M0 でも起きるなら `spike_cost` が効いている。
//!
//! ## 対照 — 物理信号が同一であることを確かめる
//!
//! **3 文脈で子音の波形バイトが完全に同一であることを assert する。**
//! 変えるのは「前に何があったか」だけ。ここが崩れたら測定の意味が消える。
//!
//! ## ゲート (実測前に固定・以後動かさない)
//!
//! **正解の出どころ**: どの子音を合成したかも、前に母音を置いたかも実験者が決めた。
//!
//! - **G78a 強調 (識別余裕)**: 母音の後に置いた子音の**相互コサインの最大値**が、
//!   孤立のときより**低い** (= より区別できる)。M0.5 出力で判定する。
//! - **G78b 重心**: 母音の後に置いた子音の重心が、孤立のときより**指定帯域に近い**
//!   (順序 pa < tu < ki < se が保たれ、se の重心が上がる)。
//! - **G78c 段の切り分け**: M0 と M0.5 の両方で測る。
//!   *M0 に適応機構は弱いので、M0 では変わらず M0.5 で変わるはず。*
//!   **両方で起きなければ、いまの実装に強調機構が無いということ。**
//! - **G78d 決定論性**: 2 回実行して完全一致。
//!
//! ## 予測
//!
//! **数値は置かない。** 構造のみ:
//! - **M0.5 で強調が起きるはず** (`local_entropy` が周波数固有の適応)。
//! - **M0 では起きないか小さいはず** (適応機構が弱い)。
//! - **これは事前の予測である** (§14.11 の反省を受けて、事後観察でないことを明記する)。
//!
//! CLI: context_enhancement

use spiking_brain::phase2_f::cochlea::{Cochlea, N_BANDS, SAMPLES_PER_STEP};
use spiking_brain::phase2_f::cochlear_nucleus::{CochlearNucleus, N_CN_OUTPUT};
use spiking_brain::phase2_f::phoneme_synth::{F0_DEFAULT_HZ, 
    synth_consonant_banded, synth_vowel, vowels, Consonant, LfsrNoise,
};

const VOWEL_MS: f64 = 170.0;
const CONSONANT_MS: f64 = 30.0;
const SEED: u16 = 0xACE1;
const SR: f64 = 16000.0;

fn consonants() -> Vec<(&'static str, Consonant)> {
    vec![
        ("pa", Consonant::Plosive { burst_freq_low: 500.0, burst_freq_high: 2000.0, voiced: false }),
        ("tu", Consonant::Plosive { burst_freq_low: 1500.0, burst_freq_high: 3500.0, voiced: false }),
        ("ki", Consonant::Plosive { burst_freq_low: 2000.0, burst_freq_high: 4000.0, voiced: false }),
        ("se", Consonant::Fricative { freq_low: 3000.0, freq_high: 8000.0, voiced: false }),
        ("mo", Consonant::Nasal { f1: 250.0, f2: 1500.0 }),
    ]
}

#[derive(Clone, Copy, PartialEq)]
enum Ctx { Isolated, AfterVowel, BeforeVowel }
impl Ctx {
    fn name(&self) -> &'static str {
        match self {
            Ctx::Isolated => "孤立 (冷開始)",
            Ctx::AfterVowel => "母音の後 (VC)",
            Ctx::BeforeVowel => "母音の前 (CV)",
        }
    }
}

/// 子音の波形 (全文脈で同一であるべきもの)
fn consonant_wave(c: Consonant) -> Vec<i32> {
    let mut n = LfsrNoise::new(SEED);
    synth_consonant_banded(c, CONSONANT_MS, F0_DEFAULT_HZ, &mut n)
}

/// 文脈つきの波形と、その中で子音が占める step 範囲を返す
fn wave_with_context(c: Consonant, ctx: Ctx) -> (Vec<i32>, usize, usize) {
    let cons = consonant_wave(c);
    let steps_of = |n: usize| n / SAMPLES_PER_STEP;
    match ctx {
        Ctx::Isolated => {
            let n = cons.len();
            (cons, 0, steps_of(n))
        }
        Ctx::AfterVowel => {
            let v = synth_vowel(&vowels()[0], VOWEL_MS); // /a/
            let start = steps_of(v.len());
            let mut w = v;
            w.extend_from_slice(&cons);
            let end = steps_of(w.len());
            (w, start, end)
        }
        Ctx::BeforeVowel => {
            let v = synth_vowel(&vowels()[0], VOWEL_MS);
            let mut w = cons.clone();
            w.extend_from_slice(&v);
            (w, 0, steps_of(cons.len()))
        }
    }
}

/// 指定 step 範囲での発火数。`use_cn` で M0 / M0.5 を切り替える。
/// 蝸牛と神経核は波形の先頭から通し、**窓の中だけ数える** (文脈の影響を残すため)。
fn window_counts(wave: &[i32], s0: usize, s1: usize, use_cn: bool) -> Vec<f64> {
    let c_len = if use_cn { N_CN_OUTPUT } else { N_BANDS };
    let mut co = Cochlea::new();
    let mut cn = CochlearNucleus::new();
    let mut counts = vec![0f64; c_len];
    for (step, chunk) in wave.chunks(SAMPLES_PER_STEP).enumerate() {
        if chunk.len() < SAMPLES_PER_STEP { break; }
        let m0 = co.process_step(chunk);
        if use_cn {
            let out = cn.process_step(&m0);
            if step >= s0 && step < s1 {
                for (i, &v) in out.iter().enumerate() { if v != 0 { counts[i] += 1.0; } }
            }
        } else if step >= s0 && step < s1 {
            for (i, &v) in m0.iter().enumerate() { if v != 0 { counts[i] += 1.0; } }
        }
    }
    counts
}

fn cosine(a: &[f64], b: &[f64]) -> f64 {
    let dot: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let na: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
    let nb: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();
    if na == 0.0 || nb == 0.0 { 0.0 } else { dot / (na * nb) }
}

fn centroid(counts: &[f64], freqs: &[f64]) -> f64 {
    let tot: f64 = counts.iter().take(freqs.len()).sum();
    if tot == 0.0 { return 0.0; }
    counts.iter().take(freqs.len()).zip(freqs.iter()).map(|(&c, &f)| c * f).sum::<f64>() / tot
}

struct Row { max_cos: f64, worst: String, centroids: Vec<f64>, totals: Vec<f64> }

fn measure(ctx: Ctx, use_cn: bool, freqs: &[f64]) -> Row {
    let cs = consonants();
    let vecs: Vec<Vec<f64>> = cs.iter()
        .map(|&(_, c)| { let (w, s0, s1) = wave_with_context(c, ctx); window_counts(&w, s0, s1, use_cn) })
        .collect();
    let mut max_cos = f64::NEG_INFINITY;
    let mut worst = String::new();
    for i in 0..cs.len() {
        for j in (i + 1)..cs.len() {
            let c = cosine(&vecs[i], &vecs[j]);
            if c > max_cos { max_cos = c; worst = format!("{}-{}", cs[i].0, cs[j].0); }
        }
    }
    // M0.5 のときは Bushy 部分 (4..4+N_BANDS) を帯域として重心を取る
    let centroids: Vec<f64> = vecs.iter().map(|v| {
        if use_cn { centroid(&v[4..4 + N_BANDS], freqs) } else { centroid(v, freqs) }
    }).collect();
    let totals: Vec<f64> = vecs.iter().map(|v| v.iter().sum()).collect();
    Row { max_cos, worst, centroids, totals }
}

fn main() {
    println!("=== 母音の後に置いた子音は強調されるか ===");
    println!();
    println!("【発端】ユーザーの指摘: 母音のすぐ後に続く子音は、物理層の信号は変わらないが");
    println!("信号処理部位で強調されるのではないか。");
    println!("→ 心理音響の「聴覚の強調効果」。適応は周波数ごとに起きるので、");
    println!("  母音のフォルマント帯域が適応し、子音の帯域は適応していない = 相対的に上がる。");
    println!();
    println!("【私の測定はこの効果が原理的に出ない条件だった】consonant_probe は");
    println!("孤立 30ms・冷開始。§14.6.7 で自分で留保に書いたのに、§14.15.5 の子音劣化を");
    println!("その条件のまま結論していた。");
    println!();
    println!("【機構】M0 の EnvelopeDetector は漏れ積分で適応を持たない (spike_cost が弱い適応)。");
    println!("M0.5 の local_entropy がチャネルごとに閾値を上げる = 周波数固有の適応そのもの。");
    println!();
    println!("【ゲート・実測前に固定】正解の出どころ = 子音も文脈も実験者が決めた");
    println!("  G78a 強調: 母音の後の相互コサイン最大 < 孤立のとき (M0.5 で判定)");
    println!("  G78b 重心: 母音の後の重心が指定帯域に近い (順序保持・se が上がる)");
    println!("  G78c 段の切り分け: M0 と M0.5 の両方で測る");
    println!("  G78d 決定論性");
    println!();
    println!("【予測・事前】M0.5 で強調が起きるはず。M0 では起きないか小さいはず。");
    println!("**両方で起きなければ、いまの実装に強調機構が無いということ。**");

    // --- 対照: 物理信号が同一か ---
    println!();
    println!("--- 対照: 3 文脈で子音の波形バイトが同一か ---");
    let mut same = true;
    for (nm, c) in consonants() {
        let base = consonant_wave(c);
        for ctx in [Ctx::Isolated, Ctx::AfterVowel, Ctx::BeforeVowel] {
            let (w, s0, s1) = wave_with_context(c, ctx);
            let seg = &w[s0 * SAMPLES_PER_STEP..(s1 * SAMPLES_PER_STEP).min(w.len())];
            let n = seg.len().min(base.len());
            if seg[..n] != base[..n] { same = false; println!("  **{} の {} で不一致**", nm, ctx.name()); }
        }
    }
    println!("  {}", if same { "全子音・全文脈でバイト同一 → **物理信号は変わっていない**" }
                     else { "**不一致あり — 測定の前提が崩れている**" });
    assert!(same, "文脈で子音の波形が変わってしまっている");

    let freqs = Cochlea::new().center_freqs.clone();
    let cs = consonants();

    for &(use_cn, stage) in [(false, "M0 (40帯域)"), (true, "M0.5 (84ch)")].iter() {
        println!();
        println!("################ 段: {} ################", stage);
        let rows: Vec<Row> = [Ctx::Isolated, Ctx::AfterVowel, Ctx::BeforeVowel].iter()
            .map(|&c| measure(c, use_cn, &freqs)).collect();

        println!();
        println!("  文脈              相互コサイン最大 (最悪の対)    子音ごとの重心 [Hz]");
        for (i, ctx) in [Ctx::Isolated, Ctx::AfterVowel, Ctx::BeforeVowel].iter().enumerate() {
            let cents: Vec<String> = rows[i].centroids.iter().map(|c| format!("{:.0}", c)).collect();
            println!("  {:<16} {:.4} ({:<6})   [{}]",
                     ctx.name(), rows[i].max_cos, rows[i].worst, cents.join(", "));
        }
        println!("  (子音の順: {})", cs.iter().map(|c| c.0).collect::<Vec<_>>().join(", "));

        println!();
        println!("  文脈              子音ごとの総スパイク数");
        for (i, ctx) in [Ctx::Isolated, Ctx::AfterVowel, Ctx::BeforeVowel].iter().enumerate() {
            let t: Vec<String> = rows[i].totals.iter().map(|c| format!("{:.0}", c)).collect();
            println!("  {:<16} [{}]", ctx.name(), t.join(", "));
        }

        let iso = &rows[0];
        let aft = &rows[1];
        let enhanced = aft.max_cos < iso.max_cos;
        let order = |c: &Vec<f64>| c[0] < c[1] && c[1] < c[2] && c[2] < c[3];
        println!();
        println!("  識別余裕: 孤立 {:.4} → 母音の後 {:.4}  ({}{:.4}) -> {}",
                 iso.max_cos, aft.max_cos,
                 if enhanced { "改善 -" } else { "悪化 +" }, (aft.max_cos - iso.max_cos).abs(),
                 if enhanced { "**強調あり**" } else { "強調なし" });
        println!("  se の重心: 孤立 {:.0}Hz → 母音の後 {:.0}Hz ({}{:.0}Hz)",
                 iso.centroids[3], aft.centroids[3],
                 if aft.centroids[3] > iso.centroids[3] { "+" } else { "" },
                 aft.centroids[3] - iso.centroids[3]);
        println!("  重心の順序 pa<tu<ki<se: 孤立 {} / 母音の後 {}",
                 if order(&iso.centroids) { "保持" } else { "崩れ" },
                 if order(&aft.centroids) { "保持" } else { "崩れ" });

        if use_cn {
            println!();
            println!("  === G78a 判定 (M0.5) -> {} ===",
                     if enhanced { "**PASS — 強調が起きている**" } else { "**FAIL — 強調が起きていない**" });
            println!("  === G78b 判定 -> {} ===",
                     if order(&aft.centroids) && aft.centroids[3] > iso.centroids[3] {
                         "**PASS**" } else { "**FAIL**" });
        }
    }

    // --- G78d 決定論性 ---
    println!();
    println!("--- G78d 決定論性 ---");
    let a = measure(Ctx::AfterVowel, true, &freqs);
    let b = measure(Ctx::AfterVowel, true, &freqs);
    println!("  2 回実行: {}",
             if (a.max_cos - b.max_cos).abs() < 1e-12 && a.centroids == b.centroids {
                 "完全一致 PASS" } else { "**不一致 FAIL**" });

    println!();
    println!("  【この測定が答えないこと】強調が起きたとして、それが かな同定を上げるかは");
    println!("  測っていない。ここで測ったのは子音どうしの識別余裕と重心だけ。");
    println!("  **既定は変えていない。**");
}
