//! フォルマント遷移は主張どおりのことをしているか (2026-08-27)
//!
//! ## なぜ
//!
//! §14.26 で「連続にすると有声性が崩壊する (76.8% → 3.3%)」が出た。
//! 本実装の子音の手がかりは**子音区間の中にしか無い**ので、直前の母音に埋もれると
//! 何も残らない。実音声では子音の調音位置は**後続母音のフォルマントの動き**に
//! 転写される (locus theory: Delattre, Liberman & Cooper 1955)。
//! **遷移は文脈でしか存在しない手がかり**で、本実装に無かった
//! (旧コメント「遷移 (formant transition) は未実装」がそれを認めていた)。
//!
//! ## これは何を測るか
//!
//! **同定率ではない。** 実装が主張どおりのことをしているかだけを検算する。
//! 同定率は次の再測定 (§14.26 の全組み合わせをもう一度) で見る。
//!
//! ## ゲート (実測前に固定・以後動かさない)
//!
//! **正解の出どころ**: どの子音にどの locus を与えたかは実験者が決めた。
//!
//! - **G87a 波形は変わるか**: locus を持つかなすべてで ON/OFF が非同一。
//!   *崩れたら実装が効いていない。*
//! - **G87b 意図した所だけが変わったか**: **は行と母音単独は ON/OFF でバイト同一**。
//!   /h/ は声門摩擦音で locus を持たない (音声学的に正しい) ので変わってはいけない。
//!   *崩れたら遷移が無関係な所まで動かしている。*
//! - **G87c 音量は変わらないか**: RMS 比 1.00 ± 0.02。
//!   *崩れたら音量で当てられるので、以後の測定が無意味になる。*
//! - **G87d 遷移は調音位置を運ぶか**: 母音区間の M0 出力で、
//!   **同じ調音位置どうしのコサイン − 違う位置どうしのコサイン** が OFF より ON で大きい。
//!   *これが本題。対照を同じ量の中で取る (閾値を後から置かないため)。*
//! - **G87e 決定論性**。
//!
//! ## 予測 (実測前・機構つき)
//!
//! - G87a/G87b/G87c は**通って当然**。通っても何も分からない。**落ちたら実装の欠陥。**
//! - **G87d が本題。** ここが上がらなければ、locus の値が悪いか、
//!   40ms の遷移が M0 の時間分解能に対して速すぎるか、どちらか。
//!
//! CLI: transition_check

use spiking_brain::phase2_f::cochlea::{Cochlea, N_BANDS, SAMPLES_PER_STEP};
use spiking_brain::phase2_f::kana::{moras_from_kana, set_formant_transition, synth_utterance};
use spiking_brain::phase2_f::phoneme_synth::LfsrNoise;

const F0: f64 = 150.0;
const SEED: u16 = 0xACE1;
/// 子音区間の終わり (CONSONANT_MS=30ms ÷ 0.5ms)。ここから先が母音区間。
const CONSONANT_STEPS: usize = 60;

/// (かな, 調音位置)。locus を持つ CV かなのみ。母音単独と は行は別枠。
const PLACED: &[(&str, &str)] = &[
    ("か","軟口蓋"),("き","軟口蓋"),("く","軟口蓋"),("け","軟口蓋"),("こ","軟口蓋"),
    ("が","軟口蓋"),("ぎ","軟口蓋"),("ぐ","軟口蓋"),("げ","軟口蓋"),("ご","軟口蓋"),
    ("さ","歯茎"),("す","歯茎"),("せ","歯茎"),("そ","歯茎"),
    ("ざ","歯茎"),("ず","歯茎"),("ぜ","歯茎"),("ぞ","歯茎"),
    ("た","歯茎"),("て","歯茎"),("と","歯茎"),("つ","歯茎"),
    ("だ","歯茎"),("で","歯茎"),("ど","歯茎"),
    ("な","歯茎"),("に","歯茎"),("ぬ","歯茎"),("ね","歯茎"),("の","歯茎"),
    ("ら","歯茎"),("り","歯茎"),("る","歯茎"),("れ","歯茎"),("ろ","歯茎"),
    ("ま","両唇"),("み","両唇"),("む","両唇"),("め","両唇"),("も","両唇"),
    ("ば","両唇"),("び","両唇"),("ぶ","両唇"),("べ","両唇"),("ぼ","両唇"),
    ("ぱ","両唇"),("ぴ","両唇"),("ぷ","両唇"),("ぺ","両唇"),("ぽ","両唇"),
    ("わ","両唇"),("を","両唇"),
    ("し","硬口蓋"),("じ","硬口蓋"),("ち","硬口蓋"),
    ("や","硬口蓋"),("ゆ","硬口蓋"),("よ","硬口蓋"),
];

/// locus を持たないはずのかな (は行 = 声門・母音単独)
const UNPLACED: &[&str] = &["は","ひ","ふ","へ","ほ","あ","い","う","え","お"];

/// 母音の行 (同じ母音どうしでしか比べない — 母音の違いで勝ってしまわないように)
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

/// 母音区間だけの M0 発火数 (40 帯域)
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

/// 同じ母音の中で「同じ位置どうし」と「違う位置どうし」の平均コサイン
fn place_separation(center: bool) -> (f64, f64, usize, usize) {
    let mut v: Vec<(&str, &str, char, Vec<f64>)> = PLACED.iter()
        .map(|&(k, p)| (k, p, vowel_of(k), vowel_region_m0(&wave(k)))).collect();
    if center {
        // **床引き**: 同じ母音の群の平均を引く。
        // 発火が密なので生のコサインは 0.9999 で飽和する (今セッションで登録済みの失敗型)。
        // 全条件が共有する床を取り除くと、**残るのは遷移による差だけ**になる。
        // **判定基準 (ON > OFF) は変えていない。統計を替えただけで、理由は飽和である。**
        for vw in ['あ', 'い', 'う', 'え', 'お'] {
            let idx: Vec<usize> = (0..v.len()).filter(|&i| v[i].2 == vw).collect();
            if idx.len() < 2 { continue; }
            let mut mean = vec![0f64; N_BANDS];
            for &i in idx.iter() { for b in 0..N_BANDS { mean[b] += v[i].3[b]; } }
            for m in mean.iter_mut() { *m /= idx.len() as f64; }
            for &i in idx.iter() { for b in 0..N_BANDS { v[i].3[b] -= mean[b]; } }
        }
    }
    let (mut win, mut nw, mut bet, mut nb) = (0f64, 0usize, 0f64, 0usize);
    for i in 0..v.len() {
        for j in (i + 1)..v.len() {
            if v[i].2 != v[j].2 { continue; }          // 母音が違うものは比べない
            let c = cosine(&v[i].3, &v[j].3);
            if v[i].1 == v[j].1 { win += c; nw += 1; } else { bet += c; nb += 1; }
        }
    }
    (win / nw.max(1) as f64, bet / nb.max(1) as f64, nw, nb)
}

/// **同じアームの中で、調音位置によって音量が違うか。** G87c が本来見るべきだった量。
/// 位置ごとの平均 RMS の、最大と最小の比。1.00 に近ければ音量では位置を当てられない。
fn rms_spread_by_place() -> (f64, String, String) {
    let places = ["軟口蓋", "歯茎", "両唇", "硬口蓋"];
    let mut stats: Vec<(&str, f64)> = Vec::new();
    for p in places.iter() {
        let rs: Vec<f64> = PLACED.iter().filter(|(_, q)| q == p).map(|&(k, _)| rms(&wave(k))).collect();
        if !rs.is_empty() { stats.push((p, rs.iter().sum::<f64>() / rs.len() as f64)); }
    }
    stats.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    let (lo, hi) = (stats[0], stats[stats.len() - 1]);
    (hi.1 / lo.1, format!("{} {:.0}", lo.0, lo.1), format!("{} {:.0}", hi.0, hi.1))
}

fn main() {
    println!("=== フォルマント遷移は主張どおりのことをしているか ===");
    println!();
    println!("【なぜ】§14.26 で連続だと有声性が崩壊 (76.8%->3.3%)。本実装の子音の手がかりは");
    println!("**子音区間の中にしか無い**ので直前の母音に埋もれると何も残らない。");
    println!("実音声では調音位置は**後続母音のフォルマントの動き**に転写される (locus theory)。");
    println!("**遷移は文脈でしか存在しない手がかり**で、本実装に無かった。");
    println!();
    println!("【これは何を測るか】**同定率ではない。**実装が主張どおりか検算するだけ。");
    println!();
    println!("【ゲート・実測前に固定】正解の出どころ = どの子音にどの locus を与えたかは実験者が決めた");
    println!("  G87a 波形は変わるか / G87b **意図した所だけか** (は行・母音単独はバイト同一)");
    println!("  G87c **音量は変わらないか** (崩れたら以後の測定が無意味)");
    println!("  G87d **遷移は調音位置を運ぶか** (同位置−異位置のコサイン差が OFF より ON で大)");
    println!("  G87e 決定論性");
    println!();
    println!("【予測・事前】G87a/b/c は**通って当然・通っても何も分からない**。");
    println!("**G87d が本題。**上がらなければ locus の値が悪いか、40ms の遷移が");
    println!("M0 の時間分解能に対して速すぎるか、どちらか。");

    // --- OFF 側を先に取る (§14.21 の教訓: 比較したいものは壊す前に測る) ---
    set_formant_transition(false);
    let off_placed: Vec<Vec<i32>> = PLACED.iter().map(|&(k, _)| wave(k)).collect();
    let off_unplaced: Vec<Vec<i32>> = UNPLACED.iter().map(|&k| wave(k)).collect();
    let (off_w, off_b, nw, nb) = place_separation(false);
    let (off_cw, off_cb, _, _) = place_separation(true);
    let off_spread = rms_spread_by_place();

    set_formant_transition(true);
    let on_placed: Vec<Vec<i32>> = PLACED.iter().map(|&(k, _)| wave(k)).collect();
    let on_unplaced: Vec<Vec<i32>> = UNPLACED.iter().map(|&k| wave(k)).collect();
    let (on_w, on_b, _, _) = place_separation(false);
    let (on_cw, on_cb, _, _) = place_separation(true);
    let on_spread = rms_spread_by_place();

    // --- G87a ---
    let same: Vec<&str> = PLACED.iter().zip(off_placed.iter()).zip(on_placed.iter())
        .filter(|((_, o), n)| o == n).map(|((&(k, _), _), _)| k).collect();
    println!();
    println!("  G87a 波形は変わるか ({} かな) -> {}", PLACED.len(),
             if same.is_empty() { "**PASS** (すべて非同一)".to_string() }
             else { format!("**FAIL** — 変わらなかった: {:?}", same) });

    // --- G87b ---
    let diff: Vec<&str> = UNPLACED.iter().zip(off_unplaced.iter()).zip(on_unplaced.iter())
        .filter(|((_, o), n)| o != n).map(|((&k, _), _)| k).collect();
    println!("  G87b 意図した所だけか (は行・母音単独 {} かな) -> {}", UNPLACED.len(),
             if diff.is_empty() { "**PASS** (すべてバイト同一)".to_string() }
             else { format!("**FAIL** — 変わってしまった: {:?}", diff) });

    // --- G87c ---
    let worst = PLACED.iter().zip(off_placed.iter()).zip(on_placed.iter())
        .map(|((_, o), n)| (rms(n) / rms(o).max(1e-9) - 1.0).abs())
        .fold(0f64, f64::max);
    println!("  G87c 音量は変わらないか -> RMS比の最大ずれ {:.4} -> {}",
             worst, if worst < 0.02 { "**PASS**" } else { "**FAIL — 宣言どおり落ちた**" });
    println!();
    println!("  【G87c は私のゲート設計の誤りだった】ON/OFF 間の音量差は交絡にならない。");
    println!("  本当に危ないのは **同じアームの中で調音位置と音量が相関すること**。");
    println!("  **宣言した基準では落ちたので落ちたと記録し、本来見るべき量を別に出す。**");
    println!("  G87c' 同じアーム内の位置ごとの平均 RMS の比 (1.00 なら音量では当てられない):");
    println!("     OFF {:.4}  ({} 〜 {})", off_spread.0, off_spread.1, off_spread.2);
    println!("     ON  {:.4}  ({} 〜 {})", on_spread.0, on_spread.1, on_spread.2);

    // --- G87d ---
    println!();
    println!("--- G87d 遷移は調音位置を運ぶか (母音区間の M0・同じ母音の中だけで比較) ---");
    println!("  対 (同位置 {} / 異位置 {})", nw, nb);
    println!("  {:<8} {:>10} {:>10} {:>12}", "", "同位置", "異位置", "**差**");
    println!("  {:<8} {:>10.4} {:>10.4} {:>12.4}", "OFF", off_w, off_b, off_w - off_b);
    println!("  {:<8} {:>10.4} {:>10.4} {:>12.4}", "ON ", on_w, on_b, on_w - on_b);
    let gate = (on_w - on_b) > (off_w - off_b);
    println!("  G87d (生) -> {}", if gate {
        format!("PASS だが **統計が 0.9999 で飽和している = 何も証明していない**")
    } else {
        format!("FAIL (差 {:.4})", (on_w - on_b) - (off_w - off_b))
    });
    println!();
    println!("  【床引き版】発火が密で生のコサインは飽和する (登録済みの失敗型)。");
    println!("  同じ母音の群の平均を引くと、**残るのは遷移による差だけ**になる。");
    println!("  **判定基準 (ON > OFF) は変えていない。統計を替えただけで、理由は飽和である。**");
    println!("  {:<8} {:>10} {:>10} {:>12}", "", "同位置", "異位置", "**差**");
    println!("  {:<8} {:>10.4} {:>10.4} {:>12.4}", "OFF", off_cw, off_cb, off_cw - off_cb);
    println!("  {:<8} {:>10.4} {:>10.4} {:>12.4}", "ON ", on_cw, on_cb, on_cw - on_cb);
    let gate2 = (on_cw - on_cb) > (off_cw - off_cb);
    println!("  G87d' (床引き) -> {}", if gate2 {
        format!("**PASS — 差が {:+.4} から {:+.4} へ拡大**", off_cw - off_cb, on_cw - on_cb)
    } else {
        format!("**FAIL — 差が {:+.4} から {:+.4} へ (拡大しない)**", off_cw - off_cb, on_cw - on_cb)
    });

    // --- G87e ---
    set_formant_transition(true);
    let (a, _, _, _) = place_separation(true);
    let (b, _, _, _) = place_separation(true);
    println!();
    println!("  G87e 決定論性 -> {}", if (a - b).abs() < 1e-12 { "PASS" } else { "**FAIL**" });

    println!();
    println!("  【この測定が答えないこと】**同定率は測っていない。**");
    println!("  遷移が『位置を運ぶ』ことと『同定できる』ことは別 (§14.25 で学んだ)。");
    println!("  **既定は ON にしたが、次の再測定で OFF/ON を並べる。**");
}
