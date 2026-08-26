//! 蝸牛は有声/無声を聞き分けられるか (2026-08-27)
//!
//! ## 問い
//!
//! §14.22 で voice bar (前有声) を実装し、**波形は違うものになった**
//! (75 かな → 51 通り が **69 通り**へ、頻度加重の縮退 62.29% → **20.59%**)。
//!
//! **だが波形が違うことと、蝸牛が聞き分けられることは別である。**
//! voice bar は F0 (150Hz) 付近の低域にあるので、**低域の帯域が効いていなければ届かない。**
//!
//! ## 対照 — 音量では当てられないことを確かめる
//!
//! `normalize_rms` は最後に掛かるので、**有声も無声も総 RMS は同じ**はず。
//! **それを assert する。** 音量で当てられるなら、この測定は
//! 「有声性を聞き分けた」ことの証拠にならない。
//!
//! ## ゲート (実測前に固定・以後動かさない)
//!
//! **正解の出どころ**: どちらを有声として合成したかは実験者が決めた。
//!
//! - **G84a 波形は違うか**: 18 対すべてでバイト非同一。*ここが崩れたら実装が効いていない。*
//! - **G84b 音量は同じか**: 各対の RMS 比が 1.00 ± 0.02。
//!   *崩れたら音量で当てられるので測定が無意味になる。*
//! - **G84c M0 で聞き分けられるか**: 各対の M0 出力のコサインが、
//!   **同じ有声性の別の子音どうしのコサインの最大値より低い**。
//!   *対照を同じ量の中で取る (閾値を後から置かないため)。*
//! - **G84d M0.5 で聞き分けられるか**: 同上を M0.5 出力で。
//! - **G84e どの帯域が効いているか**: 有声−無声の差が最大の帯域の中心周波数。
//!   *F0 (150Hz) 付近なら voice bar が届いている。記述であって判定ではない。*
//!
//! ## 予測 (実測前・機構つき)
//!
//! - **波形は違う** (G84a) — 実装したので当然。**当たっても意味がない。**
//! - **差が最大の帯域は F0 の 150Hz 付近のはず** — voice bar の周波数。
//! - **聞き分けられるかは分からない。** voice bar の振幅は子音本体の 1/4 で、
//!   しかも 40 帯域中の低域数本にしか乗らない。**そこが本当の問い。**
//!
//! CLI: voicing_check

use spiking_brain::phase2_f::cochlea::{Cochlea, N_BANDS, SAMPLES_PER_STEP};
use spiking_brain::phase2_f::cochlear_nucleus::{CochlearNucleus, N_CN_OUTPUT};
use spiking_brain::phase2_f::kana::{moras_from_kana, synth_utterance};
use spiking_brain::phase2_f::phoneme_synth::LfsrNoise;

const F0: f64 = 150.0;
const SEED: u16 = 0xACE1;

/// 有声/無声の 18 対 (清音, 濁音・半濁音)
const PAIRS: &[(&str, &str)] = &[
    ("か","が"),("き","ぎ"),("く","ぐ"),("け","げ"),("こ","ご"),
    ("さ","ざ"),("し","じ"),("す","ず"),("せ","ぜ"),("そ","ぞ"),
    ("た","だ"),("て","で"),("と","ど"),
    ("は","ば"),("ひ","び"),("ふ","ぶ"),("へ","べ"),("ほ","ぼ"),
];

fn wave(s: &str) -> Vec<i32> {
    let mut n = LfsrNoise::new(SEED);
    let (m, sk) = moras_from_kana(s);
    assert_eq!(sk, 0, "未対応: {}", s);
    synth_utterance(&m, F0, &mut n)
}

fn rms(w: &[i32]) -> f64 {
    (w.iter().map(|&s| (s as f64) * (s as f64)).sum::<f64>() / w.len().max(1) as f64).sqrt()
}

fn counts(w: &[i32], use_cn: bool) -> Vec<f64> {
    let c = if use_cn { N_CN_OUTPUT } else { N_BANDS };
    let mut co = Cochlea::new();
    let mut cn = CochlearNucleus::new();
    let mut out = vec![0f64; c];
    for chunk in w.chunks(SAMPLES_PER_STEP) {
        if chunk.len() < SAMPLES_PER_STEP { break; }
        let m0 = co.process_step(chunk);
        if use_cn {
            for (i, &v) in cn.process_step(&m0).iter().enumerate() { if v != 0 { out[i] += 1.0; } }
        } else {
            for (i, &v) in m0.iter().enumerate() { if v != 0 { out[i] += 1.0; } }
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

fn main() {
    println!("=== 蝸牛は有声/無声を聞き分けられるか ===");
    println!();
    println!("【問い】voice bar を実装して波形は違うものになった (縮退 62.29% -> 20.59%)。");
    println!("**だが波形が違うことと蝸牛が聞き分けられることは別。**");
    println!("voice bar は F0 (150Hz) 付近の低域にあるので、低域が効いていなければ届かない。");
    println!();
    println!("【ゲート・実測前に固定】正解の出どころ = どちらを有声にしたかは実験者が決めた");
    println!("  G84a 波形は違うか (18対すべてバイト非同一)");
    println!("  G84b **音量は同じか** (RMS比 1.00±0.02) ← 崩れたら測定が無意味");
    println!("  G84c M0 で聞き分け / G84d M0.5 で聞き分け");
    println!("       *対照: 同じ有声性の別子音どうしのコサインの最大より低いか*");
    println!("  G84e どの帯域が効いているか (記述)");
    println!();
    println!("【予測・事前】波形は違う(当然・意味なし) / 差が最大の帯域は F0 150Hz 付近のはず /");
    println!("**聞き分けられるかは分からない。voice bar は子音本体の 1/4 で低域数本にしか乗らない。**");

    let freqs = Cochlea::new().center_freqs.clone();

    // --- G84a / G84b ---
    let mut all_diff = true;
    let mut worst_rms = 0f64;
    for (u, v) in PAIRS.iter() {
        let (wu, wv) = (wave(u), wave(v));
        if wu == wv { all_diff = false; println!("  **{}={} が同一波形**", u, v); }
        let r = rms(&wv) / rms(&wu).max(1e-9);
        worst_rms = worst_rms.max((r - 1.0).abs());
    }
    println!();
    println!("  G84a 波形は違うか -> {}", if all_diff { "**PASS** (18対すべて非同一)" } else { "**FAIL**" });
    println!("  G84b 音量は同じか -> RMS比の最大ずれ {:.4} -> {}",
             worst_rms, if worst_rms < 0.02 { "**PASS**" } else { "**FAIL — 音量で当てられる**" });

    // --- G84c / G84d ---
    for &(use_cn, stage) in [(false, "M0 (40帯域)"), (true, "M0.5 (84ch)")].iter() {
        let vecs: Vec<(String, Vec<f64>, Vec<f64>)> = PAIRS.iter()
            .map(|(u, v)| (format!("{}/{}", u, v), counts(&wave(u), use_cn), counts(&wave(v), use_cn)))
            .collect();
        // 対内 (有声 vs 無声) のコサイン
        let within: Vec<(String, f64)> = vecs.iter()
            .map(|(n, a, b)| (n.clone(), cosine(a, b))).collect();
        // 対照: 同じ有声性どうし (無声 vs 別の無声) の最大コサイン
        let mut ctrl = f64::NEG_INFINITY;
        let mut ctrl_name = String::new();
        for i in 0..vecs.len() {
            for j in (i + 1)..vecs.len() {
                let c = cosine(&vecs[i].1, &vecs[j].1);   // 無声どうし
                if c > ctrl { ctrl = c; ctrl_name = format!("{} vs {}", vecs[i].0, vecs[j].0); }
            }
        }
        let max_within = within.iter().map(|(_, c)| *c).fold(f64::NEG_INFINITY, f64::max);
        let min_within = within.iter().map(|(_, c)| *c).fold(f64::INFINITY, f64::min);
        let n_ok = within.iter().filter(|(_, c)| *c < ctrl).count();

        println!();
        println!("--- {} ---", stage);
        println!("  有声/無声 対のコサイン: 最小 {:.4} / 最大 {:.4}", min_within, max_within);
        println!("  対照 (無声どうしの最大): {:.4}  ({})", ctrl, ctrl_name);
        println!("  対照より低い対: {}/{}", n_ok, PAIRS.len());
        let gate = max_within < ctrl;
        println!("  {} -> {}", if use_cn { "G84d" } else { "G84c" },
                 if gate { "**PASS — 全対が対照より低い**" } else { "**FAIL**" });
        // 上位/下位を少し出す
        let mut sorted = within.clone();
        sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        println!("  最も似ている対: {} {:.4} / {} {:.4} / {} {:.4}",
                 sorted[0].0, sorted[0].1, sorted[1].0, sorted[1].1, sorted[2].0, sorted[2].1);
    }

    // --- G84e どの帯域が効いているか ---
    println!();
    println!("--- G84e どの帯域が効いているか (記述) ---");
    let mut diff = vec![0f64; N_BANDS];
    for (u, v) in PAIRS.iter() {
        let (a, b) = (counts(&wave(u), false), counts(&wave(v), false));
        for i in 0..N_BANDS { diff[i] += b[i] - a[i]; }
    }
    let mut idx: Vec<usize> = (0..N_BANDS).collect();
    idx.sort_by(|&a, &b| diff[b].abs().partial_cmp(&diff[a].abs()).unwrap());
    println!("  有声−無声の差が大きい帯域 (18対の合計):");
    for &i in idx.iter().take(6) {
        println!("    帯域{:>2} fc={:>7.1}Hz  差 {:+.0} 発", i, freqs[i], diff[i]);
    }
    println!("  (F0 = {:.0}Hz。voice bar が届いていれば その付近が上位に来るはず)", F0);
}
