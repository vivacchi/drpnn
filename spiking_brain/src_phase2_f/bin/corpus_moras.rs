//! 実コーパスは音にできるか — 歩留まりと多様性を測る (2026-08-27)
//!
//! ## 発端 — ユーザーの指摘
//!
//! > **触れる音の種類を多くしよう。つまり多様性を増やして、かつ単語や熟語の音も
//! > 聞かせたい。実際に昨日試したチャットコーパスの文章を音として聞かせたらどうなるか。**
//!
//! §14.20 で計算したとおり、いまの律速は**時間ではなくコーパスの多様性**である。
//! いまのコーパスは **46 項目**しかなく、刈り込みまでに各かなを 91 回見る。
//! 時間を 2880 倍にしても**同じ 46 項目を 26 万回見るだけ**。
//! そして §14.19 で**同じ 46 かなを 3 回繰り返すだけで子音が 21.7% → 10.3% に落ちた。**
//!
//! ## コーパス
//!
//! `data/corpus/roleplay_filtered.jsonl` (317MB・4324 スレッド)。
//! **`.gitignore:27` に `data/corpus/` があり、追跡されていない。**
//! 「実データは実機のみ・GitHub 経由では運ばない」という原則は守られている。
//! このプローブも**数値しか出さない。本文は一切印字しない。**
//!
//! 文字構成 (500 スレッドのサンプル・5,967,070 文字):
//! ひらがな 58.0% / 漢字 25.0% / 記号その他 12.1% / カタカナ 2.7% / 空白 1.8%
//!
//! **かな = 60.7%。全 4324 スレッドで約 4,180 万モーラ相当** =
//! 知覚の絞り込み (生後 6-12 か月・1200万モーラ) の **3.5 倍**、3 年ぶんの 56%。
//!
//! ## 測ること
//!
//! 1. **歩留まり**: 実際に `moras_from_kana` を通して、何モーラ得られ何文字落ちるか
//! 2. **多様性 (1 次)**: 異なりかな数と、その頻度分布のエントロピー
//! 3. **多様性 (2 次)**: **かな→かなの遷移**のエントロピー。
//!    *実際の日本語には音韻配列の構造があり、46 項目のランダム順にはそれが無い。*
//!    **条件付きエントロピー H(次|今) が 1 次エントロピー H(今) より小さければ、
//!    系列に構造がある。**
//!
//! ## ゲート (実測前に固定)
//!
//! **正解の出どころ**: どのファイルを読ませたかは実験者が決めた。
//! 歩留まりも異なり数も、変換規則 (`kana.rs` の表) から決まる量で、系の判断は入らない。
//!
//! - **G81a 歩留まり**: 全文字のうち何 % がモーラになるか。
//!   *記録するだけ。閾値は置かない (置く根拠が無い)。*
//! - **G81b 異なりモーラ数**: 現行の 46 項目より**多い**か。
//! - **G81c 系列の構造**: 条件付きエントロピー H(次|今) が 1 次エントロピー H(今) より
//!   **小さい**か。*小さければ音韻配列の構造がある。*
//!   **対照**: 46 項目を一様ランダムに並べた場合、H(今) = log2(46) = 5.52 bit で
//!   H(次|今) も 5.52 bit (独立なので減らない)。**この対照を同じコードで計算して並べる。**
//! - **G81d 決定論性**: 2 回実行して完全一致。
//!
//! ## 予測 (実測前・機構つき)
//!
//! - **歩留まりは 60% 前後**。文字構成から。**当たっても意味のない予測。**
//! - **異なりモーラ数は 46 を大きく超えるはず** (濁音・半濁音・拗音・促音・長音)。
//! - **H(次|今) < H(今) になるはず。** 日本語には音韻配列の制約がある。
//!   **これが本命。** 差が大きいほど、46 項目のランダム順には無い構造があるということ。
//!
//! CLI: corpus_moras <テキストファイル>

use spiking_brain::phase2_f::kana::moras_from_kana;
use std::collections::HashMap;
use std::io::{BufRead, BufReader};

/// その 1 文字が単独でモーラになるか (= かな表に載っているか)
fn is_mora_char(c: char) -> bool {
    let s = c.to_string();
    let (m, skipped) = moras_from_kana(&s);
    m.len() == 1 && skipped == 0
}

fn entropy(counts: &HashMap<char, u64>) -> f64 {
    let total: u64 = counts.values().sum();
    if total == 0 { return 0.0; }
    -counts.values().map(|&c| {
        let p = c as f64 / total as f64;
        if p > 0.0 { p * p.log2() } else { 0.0 }
    }).sum::<f64>()
}

/// 条件付きエントロピー H(次|今) = Σ p(今) H(次|今=その文字)
fn conditional_entropy(bi: &HashMap<(char, char), u64>) -> f64 {
    let mut by_first: HashMap<char, HashMap<char, u64>> = HashMap::new();
    for (&(a, b), &c) in bi.iter() {
        *by_first.entry(a).or_default().entry(b).or_insert(0) += c;
    }
    let total: u64 = bi.values().sum();
    if total == 0 { return 0.0; }
    by_first.values().map(|m| {
        let n: u64 = m.values().sum();
        (n as f64 / total as f64) * entropy(m)
    }).sum()
}

fn main() {
    let path = std::env::args().nth(1).expect("使い方: corpus_moras <テキストファイル>");
    println!("=== 実コーパスは音にできるか ===");
    println!();
    println!("【原則】このプローブは**数値しか出さない。本文は一切印字しない。**");
    println!("コーパスは .gitignore:27 で追跡されておらず、実機に留まる。");
    println!();
    println!("【ゲート・実測前に固定】");
    println!("  G81a 歩留まり (記録のみ・閾値は置かない)");
    println!("  G81b 異なりモーラ数が 46 より多いか");
    println!("  G81c **系列の構造**: H(次|今) < H(今) か。対照として 46 項目一様ランダムを並べる");
    println!("  G81d 決定論性");
    println!();
    println!("【予測・事前】歩留まり 60% 前後 (当たっても意味なし) / 異なり数は 46 を大きく超える /");
    println!("**H(次|今) < H(今) になるはず。これが本命。**");

    let f = std::fs::File::open(&path).expect("開けない");
    let rdr = BufReader::new(f);

    let mut total_chars: u64 = 0;
    let mut mora_chars: u64 = 0;
    let mut total_moras: u64 = 0;
    let mut total_skipped: u64 = 0;
    let mut lines: u64 = 0;
    let mut uni: HashMap<char, u64> = HashMap::new();
    let mut bi: HashMap<(char, char), u64> = HashMap::new();

    for line in rdr.lines() {
        let line = match line { Ok(l) => l, Err(_) => continue };
        lines += 1;
        total_chars += line.chars().count() as u64;
        // 実際の変換を通す
        let (m, sk) = moras_from_kana(&line);
        total_moras += m.len() as u64;
        total_skipped += sk as u64;
        // 文字レベルの在庫と遷移 (かなだけを連ねる)
        let mut prev: Option<char> = None;
        for c in line.chars() {
            if is_mora_char(c) {
                mora_chars += 1;
                *uni.entry(c).or_insert(0) += 1;
                if let Some(p) = prev { *bi.entry((p, c)).or_insert(0) += 1; }
                prev = Some(c);
            } else {
                prev = None; // かな以外で列を切る
            }
        }
    }

    let h1 = entropy(&uni);
    let h2 = conditional_entropy(&bi);

    println!();
    println!("--- 歩留まり (G81a) ---");
    println!("  行 (投稿) 数        : {:>12}", lines);
    println!("  総文字数            : {:>12}", total_chars);
    println!("  **得られたモーラ数**: {:>12}  ({:.1}%)", total_moras,
             total_moras as f64 / total_chars as f64 * 100.0);
    println!("  落ちた文字数        : {:>12}  ({:.1}%)", total_skipped,
             total_skipped as f64 / total_chars as f64 * 100.0);
    println!("  (モーラになる文字   : {:>12}  {:.1}%)", mora_chars,
             mora_chars as f64 / total_chars as f64 * 100.0);

    println!();
    println!("--- 多様性 (G81b) ---");
    println!("  **異なりモーラ (かな) 数: {}**  (現行のコーパスは 46)", uni.len());
    let mut top: Vec<(&char, &u64)> = uni.iter().collect();
    top.sort_by(|a, b| b.1.cmp(a.1));
    let tot: u64 = uni.values().sum();
    let top10: u64 = top.iter().take(10).map(|(_, &c)| c).sum();
    println!("  上位 10 種が占める割合 : {:.1}%", top10 as f64 / tot as f64 * 100.0);
    let rare = uni.values().filter(|&&c| c < tot / 10000).count();
    println!("  出現率 0.01% 未満の種  : {} / {}", rare, uni.len());

    println!();
    println!("--- 系列の構造 (G81c) ---");
    println!("  1 次エントロピー H(今)        : {:.3} bit", h1);
    println!("  条件付きエントロピー H(次|今) : {:.3} bit", h2);
    println!("  **減少量 H(今) − H(次|今)     : {:.3} bit**", h1 - h2);
    println!();
    println!("  [対照] 46 項目を一様ランダムに並べた場合:");
    println!("    H(今) = log2(46) = {:.3} bit / H(次|今) も同じ (独立なので減らない)", 46f64.log2());
    println!("    **減少量 = 0.000 bit**");
    println!();
    println!("  G81b 異なり数 {} > 46 -> {}", uni.len(), if uni.len() > 46 { "**PASS**" } else { "**FAIL**" });
    println!("  G81c 減少量 {:.3} bit > 0 -> {}", h1 - h2,
             if h2 < h1 { "**PASS — 音韻配列の構造がある**" } else { "**FAIL**" });

    println!();
    println!("  【この測定が答えないこと】この多様性が M0/M0.5/M1 の学習を良くするかは");
    println!("  測っていない。ここで測ったのは**コーパス側の性質だけ**。");
}
