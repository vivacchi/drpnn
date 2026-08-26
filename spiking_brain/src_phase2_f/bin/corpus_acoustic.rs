//! コーパスのかなは、音として何通りあるか (2026-08-27)
//!
//! ## 問い
//!
//! §14.20 で、発音変換後のコーパスに **75 の異なりかな**があると測った。
//! だが §14.6.6 では **71 かな → 51 通りの波形**だった。
//! **合成器に有声/無声の区別が無く、拗音は既存かなとバイト同一**だからである。
//!
//! **75 は文字上の数であって、音としての数ではない。** それを測る。
//!
//! ## 頻度で重みを付けるのが本命
//!
//! 潰れる対が**稀なかな**どうしなら実害は小さい。**よく出るかな**が潰れるなら大きい。
//! よって **「一意でない波形を持つモーラが、コーパスの何 % を占めるか」** を出す。
//! これが実際に効く量である。
//!
//! ## ゲート (実測前に固定)
//!
//! **正解の出どころ**: どのかなを合成したかは実験者が決めた。
//! 波形が同一かはバイト比較で、判断機構は入らない。
//!
//! - **G83a 異なり波形数**: 75 かな (実際は入力ファイルにある全種) が何通りの波形になるか。
//!   *記録のみ。閾値は置かない。*
//! - **G83b 頻度加重の潰れ**: **一意でない波形を持つモーラがコーパスに占める割合。**
//!   *これが本命。記録のみ。閾値は置かない (置く根拠が無い)。*
//! - **G83c 現行コーパスとの比較**: 46 項目は全部一意か。
//! - **G83d 決定論性**: 2 回実行して完全一致。
//!
//! ## 予測 (実測前・機構つき)
//!
//! - **異なり波形数は 75 より減るはず。** 濁音 20 と半濁音 5 は清音と同一のはず
//!   (§14.6.6 で確認済み)。減る量は 25 前後。
//! - **頻度加重の潰れは大きいはず。** 濁音 (ガ行・ザ行・ダ行・バ行) は日本語で頻出する。
//!   **これが大きければ、コーパスの多様性は見かけより小さい。**
//!
//! ## 入力
//!
//! `<かな頻度ファイル>` = 1 行 `かな<TAB>回数<TAB>割合` の表。
//! **本文は読まない。かな 1 文字と回数だけ。**
//!
//! CLI: corpus_acoustic <かな頻度ファイル>

use spiking_brain::phase2_f::kana::{moras_from_kana, synth_utterance};
use spiking_brain::phase2_f::phoneme_synth::LfsrNoise;
use std::collections::HashMap;
use std::io::{BufRead, BufReader};

const F0: f64 = 150.0;
const SEED: u16 = 0xACE1;

/// 現行コーパス (46 項目)
const CURRENT: &[&str] = &[
    "あ","い","う","え","お","か","き","く","け","こ","さ","し","す","せ","そ",
    "た","ち","つ","て","と","な","に","ぬ","ね","の","は","ひ","ふ","へ","ほ",
    "ま","み","む","め","も","や","ゆ","よ","ら","り","る","れ","ろ","わ","を","ん",
];

fn wave_of(s: &str) -> Option<Vec<i32>> {
    let (m, skipped) = moras_from_kana(s);
    if m.is_empty() || skipped > 0 { return None; }
    let mut n = LfsrNoise::new(SEED);
    Some(synth_utterance(&m, F0, &mut n))
}

/// バイト同一で同値類に分ける。返り値 = (代表 -> 仲間たち)
fn classes(items: &[(String, u64)]) -> Vec<(Vec<String>, u64)> {
    let mut reps: Vec<(Vec<i32>, Vec<String>, u64)> = Vec::new();
    for (k, c) in items.iter() {
        let w = match wave_of(k) { Some(w) => w, None => continue };
        match reps.iter_mut().find(|(rw, _, _)| *rw == w) {
            Some((_, names, cnt)) => { names.push(k.clone()); *cnt += c; }
            None => reps.push((w, vec![k.clone()], *c)),
        }
    }
    reps.into_iter().map(|(_, n, c)| (n, c)).collect()
}

fn main() {
    let path = std::env::args().nth(1).expect("使い方: corpus_acoustic <かな頻度ファイル>");
    println!("=== コーパスのかなは、音として何通りあるか ===");
    println!();
    println!("【問い】§14.20 で発音変換後に 75 の異なりかなと測ったが、§14.6.6 では");
    println!("**71 かな → 51 通りの波形**だった。有声/無声の区別が無く拗音は既存かなと同一。");
    println!("**75 は文字上の数であって音としての数ではない。**");
    println!();
    println!("【本命】潰れる対が稀なら実害は小さい。よく出るかなが潰れるなら大きい。");
    println!("**一意でない波形を持つモーラがコーパスの何 % を占めるか**を出す。");
    println!();
    println!("【予測・事前】異なり波形数は 75 より 25 前後減るはず (濁音20+半濁音5)。");
    println!("**頻度加重の潰れは大きいはず** — 濁音は日本語で頻出する。");
    println!("**大きければ、コーパスの多様性は見かけより小さい。**");
    println!();
    println!("【原則】本文は読まない。かな 1 文字と回数だけ。");

    // --- 入力: かな 1 文字と回数 ---
    let f = std::fs::File::open(&path).expect("開けない");
    let mut items: Vec<(String, u64)> = Vec::new();
    let mut skipped_chars: Vec<String> = Vec::new();
    let mut total: u64 = 0;
    for line in BufReader::new(f).lines() {
        let line = match line { Ok(l) => l, Err(_) => continue };
        let mut it = line.split('\t');
        let (k, c) = match (it.next(), it.next()) { (Some(a), Some(b)) => (a, b), _ => continue };
        let c: u64 = match c.trim().parse() { Ok(v) => v, Err(_) => continue };
        if k.chars().count() != 1 { continue; }
        total += c;
        if wave_of(k).is_some() { items.push((k.to_string(), c)); }
        else { skipped_chars.push(k.to_string()); }
    }

    let usable: u64 = items.iter().map(|(_, c)| c).sum();
    println!();
    println!("--- 入力 ---");
    println!("  総モーラ (回数の合計)   : {:>12}", total);
    println!("  合成できたかな          : {:>12} 種  ({} モーラ・{:.2}%)",
             items.len(), usable, usable as f64 / total as f64 * 100.0);
    println!("  合成できなかったかな    : {:>12} 種  ({:.2}%)",
             skipped_chars.len(), (total - usable) as f64 / total as f64 * 100.0);
    if !skipped_chars.is_empty() {
        println!("    (単独ではモーラにならないもの: {})", skipped_chars.join(""));
    }

    // --- G83a/G83b ---
    let cls = classes(&items);
    let collapsed: Vec<&(Vec<String>, u64)> = cls.iter().filter(|(n, _)| n.len() > 1).collect();
    let collapsed_moras: u64 = collapsed.iter().map(|(_, c)| c).sum();
    let collapsed_kinds: usize = collapsed.iter().map(|(n, _)| n.len()).sum();

    println!();
    println!("--- G83a 異なり波形数 ---");
    println!("  合成できたかな {} 種 -> **{} 通りの波形**", items.len(), cls.len());
    println!("  一意でないかな : {} 種 ({} 個の同値類にまとまる)", collapsed_kinds, collapsed.len());
    println!();
    for (n, c) in collapsed.iter() {
        println!("    {{{}}}  {:>10} モーラ ({:.3}%)",
                 n.join(" = "), c, *c as f64 / total as f64 * 100.0);
    }

    println!();
    println!("--- G83b 頻度加重の潰れ (**本命**) ---");
    println!("  一意でない波形を持つモーラ: {} / {} = **{:.2}%**",
             collapsed_moras, total, collapsed_moras as f64 / total as f64 * 100.0);
    println!("  (このぶんは、蝸牛が何をしても区別できない)");

    // --- G83c 現行コーパスとの比較 ---
    let cur: Vec<(String, u64)> = CURRENT.iter().map(|s| (s.to_string(), 1u64)).collect();
    let cur_cls = classes(&cur);
    let cur_collapsed: usize = cur_cls.iter().filter(|(n, _)| n.len() > 1).map(|(n, _)| n.len()).sum();
    println!();
    println!("--- G83c 現行コーパス (46 項目) との比較 ---");
    println!("  46 項目 -> {} 通りの波形 (一意でない {} 種)", cur_cls.len(), cur_collapsed);
    println!();
    println!("  {:<22} {:>10} {:>12}", "", "異なり波形", "一意でない割合");
    println!("  {:<22} {:>10} {:>12}", "現行 (46項目・一様)", cur_cls.len(),
             format!("{:.1}%", cur_collapsed as f64 / CURRENT.len() as f64 * 100.0));
    println!("  {:<22} {:>10} {:>12}", "実コーパス (頻度加重)", cls.len(),
             format!("{:.1}%", collapsed_moras as f64 / total as f64 * 100.0));

    // --- G83d ---
    let c2 = classes(&items);
    let same = cls.len() == c2.len()
        && cls.iter().zip(c2.iter()).all(|(a, b)| a.0 == b.0 && a.1 == b.1);
    println!();
    println!("  G83d 決定論性 -> {}", if same { "PASS" } else { "**FAIL**" });
    println!();
    println!("  【この測定が答えないこと】潰れが同定率をどれだけ下げるかは測っていない。");
    println!("  ここで測ったのは**合成器の縮退がコーパスに占める割合だけ**。");
}
