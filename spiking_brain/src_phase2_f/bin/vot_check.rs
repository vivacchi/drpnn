//! VOT は主張どおりのことをしているか (2026-08-27)
//!
//! ## なぜ
//!
//! §14.26 で「連続にすると有声性の伝達情報量が 76.8% → 3.3% に崩壊する」が出た。
//! 原因は **本実装の有声性の手がかりが voice bar (閉鎖中の声帯振動) 1 本しか無く、
//! それが最も文脈に弱い手がかりだった**こと。連続では直前の母音の低域に埋もれる。
//!
//! 実音声で有声/無声を分ける最も強い手がかりは **VOT** である。
//! 無声破裂音は解放のあと声帯振動が始まるまで**気音**が入り、有声破裂音には入らない。
//! **VOT は解放「後」の手がかりなので、フォルマント遷移と同じく連続でも生き残る。**
//! これが voice bar との決定的な違いである。
//!
//! ## これは何を測るか
//!
//! **同定率ではない。** 実装が主張どおりのことをしているかだけを検算する。
//!
//! ## ゲート (実測前に固定・以後動かさない)
//!
//! **正解の出どころ**: どの子音に何 ms の VOT を与えたかは実験者が決めた。
//!
//! - **G90a 波形は変わるか**: 無声破裂音・破擦音で ON/OFF が非同一。
//! - **G90b 意図した所だけが変わったか**: **有声破裂音・摩擦音・鼻音・接近音・弾き音・
//!   母音単独は ON/OFF でバイト同一**。VOT を 0 にした音は変わってはいけない。
//! - **G90c 同じアームの中で、有声性によって音量が違わないか**:
//!   無声破裂音の平均 RMS ÷ 有声破裂音の平均 RMS が 1.00 ± 0.02。
//!   *§14.27 の G87c で「ON/OFF 間の音量差」という**的を外したゲート**を書いた。
//!    その教訓を反映し、**最初から本当の交絡 (同じアーム内の相関) を見る。***
//! - **G90d VOT は有声性を運ぶか**: **床引きした**母音区間の M0 で、
//!   **同じ有声性どうしのコサイン − 違う有声性どうしのコサイン** が OFF より ON で大きい。
//!   同じ母音の中だけで比べる (母音で勝たないため)。
//!   *床引きは §14.27 の G87d で生コサインが 0.9999 に飽和したため最初から使う。*
//! - **G90e 決定論性**。
//!
//! ## 予測 (実測前・機構つき)
//!
//! - G90a/b/c は**通って当然**。落ちたら実装の欠陥。
//! - **G90d が本題。** 上がらなければ、気音が母音区間の中で埋もれているか、
//!   RMS を揃えたことで手がかりごと消してしまったか、どちらか。
//!
//! CLI: vot_check

use spiking_brain::phase2_f::cochlea::{Cochlea, N_BANDS, SAMPLES_PER_STEP};
use spiking_brain::phase2_f::kana::{moras_from_kana, set_vot, synth_utterance};
use spiking_brain::phase2_f::phoneme_synth::LfsrNoise;

const F0: f64 = 150.0;
const SEED: u16 = 0xACE1;
const CONSONANT_STEPS: usize = 60;

/// VOT を持つ (= 変わるはず) かな: 無声破裂音・無声破擦音
const VOICELESS_STOPS: &[&str] = &[
    "か","き","く","け","こ","た","て","と","つ","ち",
    "ぱ","ぴ","ぷ","ぺ","ぽ",
];

/// VOT が 0 (= バイト同一のはず) かな: 有声破裂音・摩擦・鼻音・接近・弾き・母音単独
const NO_VOT: &[&str] = &[
    "が","ぎ","ぐ","げ","ご","だ","で","ど","ば","び","ぶ","べ","ぼ",
    "さ","し","す","せ","そ","ざ","じ","ず","ぜ","ぞ","は","ひ","ふ","へ","ほ",
    "な","に","ぬ","ね","の","ま","み","む","め","も",
    "や","ゆ","よ","わ","を","ら","り","る","れ","ろ",
    "あ","い","う","え","お",
];

/// (かな, 有声か) — 破裂音のみ。同じ母音の中で有声性だけを比べるため。
const PLOSIVES: &[(&str, bool)] = &[
    ("か",false),("き",false),("く",false),("け",false),("こ",false),
    ("た",false),("て",false),("と",false),
    ("ぱ",false),("ぴ",false),("ぷ",false),("ぺ",false),("ぽ",false),
    ("が",true),("ぎ",true),("ぐ",true),("げ",true),("ご",true),
    ("だ",true),("で",true),("ど",true),
    ("ば",true),("び",true),("ぶ",true),("べ",true),("ぼ",true),
];

fn vowel_of(k: &str) -> char {
    const A: &str = "かがさざただなはばぱまやらわ";
    const I: &str = "きぎしじちぢにひびぴみり";
    const U: &str = "くぐすずつづぬふぶぷむゆる";
    const E: &str = "けげせぜてでねへべぺめれ";
    const O: &str = "こごそぞとどのほぼぽもよろを";
    let c = k.chars().next().unwrap();
    if A.contains(c) { 'あ' } else if I.contains(c) { 'い' }
    else if U.contains(c) { 'う' } else if E.contains(c) { 'え' }
    else if O.contains(c) { 'お' } else { '?' }
}

fn wave(k: &str) -> Vec<i32> {
    let mut n = LfsrNoise::new(SEED);
    let (m, sk) = moras_from_kana(k);
    assert_eq!(sk, 0, "未対応: {}", k);
    synth_utterance(&m, F0, &mut n)
}

fn rms(w: &[i32]) -> f64 {
    (w.iter().map(|&s| (s as f64) * (s as f64)).sum::<f64>() / w.len().max(1) as f64).sqrt()
}

fn vowel_region_m0(w: &[i32]) -> Vec<f64> {
    let mut co = Cochlea::new();
    let mut out = vec![0f64; N_BANDS];
    for (step, chunk) in w.chunks(SAMPLES_PER_STEP).enumerate() {
        if chunk.len() < SAMPLES_PER_STEP { break; }
        let m0 = co.process_step(chunk);
        if step < CONSONANT_STEPS { continue; }
        for (i, &v) in m0.iter().enumerate() { if v != 0 { out[i] += 1.0; } }
    }
    out
}

fn cosine(a: &[f64], b: &[f64]) -> f64 {
    let d: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let na: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
    let nb: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();
    if na == 0.0 || nb == 0.0 { 0.0 } else { d / (na * nb) }
}

/// 同じ母音の中で「同じ有声性どうし」と「違う有声性どうし」の平均コサイン。**床引き済み。**
fn voicing_separation() -> (f64, f64, usize, usize) {
    let mut v: Vec<(bool, char, Vec<f64>)> = PLOSIVES.iter()
        .map(|&(k, vd)| (vd, vowel_of(k), vowel_region_m0(&wave(k)))).collect();
    // 床引き: 同じ母音の群の平均を引く (生コサインは 0.9999 で飽和するため)
    for vw in ['あ', 'い', 'う', 'え', 'お'] {
        let idx: Vec<usize> = (0..v.len()).filter(|&i| v[i].1 == vw).collect();
        if idx.len() < 2 { continue; }
        let mut mean = vec![0f64; N_BANDS];
        for &i in idx.iter() { for b in 0..N_BANDS { mean[b] += v[i].2[b]; } }
        for m in mean.iter_mut() { *m /= idx.len() as f64; }
        for &i in idx.iter() { for b in 0..N_BANDS { v[i].2[b] -= mean[b]; } }
    }
    let (mut win, mut nw, mut bet, mut nb) = (0f64, 0usize, 0f64, 0usize);
    for i in 0..v.len() {
        for j in (i + 1)..v.len() {
            if v[i].1 != v[j].1 { continue; }
            let c = cosine(&v[i].2, &v[j].2);
            if v[i].0 == v[j].0 { win += c; nw += 1; } else { bet += c; nb += 1; }
        }
    }
    (win / nw.max(1) as f64, bet / nb.max(1) as f64, nw, nb)
}

/// **同じアームの中で、有声性によって音量が違うか。** G90c が見るべき本当の量。
fn rms_ratio_by_voicing() -> (f64, f64, f64) {
    let vl: Vec<f64> = PLOSIVES.iter().filter(|(_, v)| !v).map(|&(k, _)| rms(&wave(k))).collect();
    let vd: Vec<f64> = PLOSIVES.iter().filter(|(_, v)| *v).map(|&(k, _)| rms(&wave(k))).collect();
    let (a, b) = (vl.iter().sum::<f64>() / vl.len() as f64, vd.iter().sum::<f64>() / vd.len() as f64);
    (a / b, a, b)
}

fn main() {
    println!("=== VOT は主張どおりのことをしているか ===");
    println!();
    println!("【なぜ】§14.26 で連続だと有声性が 76.8%->3.3% に崩壊。原因は");
    println!("**有声性の手がかりが voice bar 1本しか無く、それが最も文脈に弱かった**こと。");
    println!("実音声で有声/無声を分ける最強の手がかりは **VOT**。無声破裂音は解放後に");
    println!("気音が入り、有声破裂音には入らない。**VOT は解放『後』なので連続でも生き残る。**");
    println!();
    println!("【ゲート・実測前に固定】正解の出どころ = どの子音に何ms与えたかは実験者が決めた");
    println!("  G90a 波形は変わるか (無声破裂音・破擦音)");
    println!("  G90b **意図した所だけか** (有声破裂音・摩擦・鼻音・接近・弾き・母音単独はバイト同一)");
    println!("  G90c **同じアーム内で有声性によって音量が違わないか** (無声RMS/有声RMS = 1.00±0.02)");
    println!("       *§14.27 の G87c で的を外したゲートを書いた。**今度は最初から本当の交絡を見る。***");
    println!("  G90d **VOT は有声性を運ぶか** (床引き・同母音内で 同有声性−異有声性 が OFF<ON)");
    println!("       *床引きは G87d が 0.9999 で飽和したので最初から使う。*");
    println!("  G90e 決定論性");
    println!();
    println!("【予測・事前】G90a/b/c は**通って当然**。**G90d が本題。**");
    println!("上がらなければ、気音が母音区間で埋もれているか、RMS を揃えて手がかりごと");
    println!("消してしまったか、どちらか。");

    set_vot(false);
    let off_vl: Vec<Vec<i32>> = VOICELESS_STOPS.iter().map(|&k| wave(k)).collect();
    let off_nv: Vec<Vec<i32>> = NO_VOT.iter().map(|&k| wave(k)).collect();
    let (off_w, off_b, nw, nb) = voicing_separation();
    let off_r = rms_ratio_by_voicing();

    set_vot(true);
    let on_vl: Vec<Vec<i32>> = VOICELESS_STOPS.iter().map(|&k| wave(k)).collect();
    let on_nv: Vec<Vec<i32>> = NO_VOT.iter().map(|&k| wave(k)).collect();
    let (on_w, on_b, _, _) = voicing_separation();
    let on_r = rms_ratio_by_voicing();

    // --- G90a ---
    let same: Vec<&str> = VOICELESS_STOPS.iter().zip(off_vl.iter()).zip(on_vl.iter())
        .filter(|((_, o), n)| o == n).map(|((&k, _), _)| k).collect();
    println!();
    println!("  G90a 波形は変わるか ({} かな) -> {}", VOICELESS_STOPS.len(),
             if same.is_empty() { "**PASS** (すべて非同一)".to_string() }
             else { format!("**FAIL** — 変わらなかった: {:?}", same) });

    // --- G90b ---
    let diff: Vec<&str> = NO_VOT.iter().zip(off_nv.iter()).zip(on_nv.iter())
        .filter(|((_, o), n)| o != n).map(|((&k, _), _)| k).collect();
    println!("  G90b 意図した所だけか ({} かな) -> {}", NO_VOT.len(),
             if diff.is_empty() { "**PASS** (すべてバイト同一)".to_string() }
             else { format!("**FAIL** — 変わってしまった: {:?}", diff) });

    // --- G90c ---
    println!();
    println!("  G90c 同じアーム内で有声性によって音量が違うか (無声RMS ÷ 有声RMS):");
    println!("     OFF {:.4}   (無声 {:.0} / 有声 {:.0})", off_r.0, off_r.1, off_r.2);
    println!("     ON  {:.4}   (無声 {:.0} / 有声 {:.0})", on_r.0, on_r.1, on_r.2);
    println!("     -> {}", if (on_r.0 - 1.0).abs() < 0.02 { "**PASS — 音量では有声性を当てられない**" }
             else { "**FAIL — 音量で当てられてしまう**" });

    // --- G90d ---
    println!();
    println!("--- G90d VOT は有声性を運ぶか (床引き・母音区間の M0・同じ母音の中だけ) ---");
    println!("  対 (同有声性 {} / 異有声性 {})", nw, nb);
    println!("  {:<8} {:>12} {:>12} {:>12}", "", "同有声性", "異有声性", "**差**");
    println!("  {:<8} {:>12.4} {:>12.4} {:>12.4}", "OFF", off_w, off_b, off_w - off_b);
    println!("  {:<8} {:>12.4} {:>12.4} {:>12.4}", "ON ", on_w, on_b, on_w - on_b);
    let gate = (on_w - on_b) > (off_w - off_b);
    println!("  G90d -> {}", if gate {
        format!("**PASS — 差が {:+.4} から {:+.4} へ拡大**", off_w - off_b, on_w - on_b)
    } else {
        format!("**FAIL — 差が {:+.4} から {:+.4} へ (拡大しない)**", off_w - off_b, on_w - on_b)
    });

    // --- G90e ---
    let (a, _, _, _) = voicing_separation();
    let (b, _, _, _) = voicing_separation();
    println!();
    println!("  G90e 決定論性 -> {}", if (a - b).abs() < 1e-12 { "PASS" } else { "**FAIL**" });
    println!();
    println!("  【この測定が答えないこと】**同定率も、連続での挙動も測っていない。**");
    println!("  §14.25 の教訓: 特徴を運ぶことと同定できることは別。**次の再測定で見る。**");
}
