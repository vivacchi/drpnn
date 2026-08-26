//! かな 69 音のうちどこまで「どのかなか」を当てられるか (2026-08-26 / 濁音追加 2026-08-27)
//!
//! ## 2026-08-27: 濁音・半濁音を追加した
//!
//! §14.22 で有声/無声を実装するまで、濁音は清音と**完全に同一の波形**だった。
//! だから 46 清音+ん でしか測っておらず、**出口の指標に濁音が入っていなかった。**
//! 実コーパスでは濁音がモーラの **41.7%** を占める (§14.21)。
//!
//! **69 クラス** = 45 清音 + ん + 濁音 18 (が行5・ざ行5・だ行3・ば行5) + 半濁音 5。
//! **ぢ と づ は入れない** — 現代日本語で ぢ=じ・づ=ず は同音で、合成器でもバイト同一。
//! **正しい縮退なので別クラスとして数えるのは不公平である。**
//!
//! **これが「蝸牛は有声/無声を聞き分けられるか」の出口の指標である。**
//! §14.22.3 の G84c/G84d はコサインの比較であって同定率ではなかった。
//!
//! ## なぜ新しいプローブが要るか
//!
//! 既存の kana_probe は「完全に同一の応答を持つ対が 0 組」しか見ていない。
//! これは**区別**の測定であって**同定**ではない。各かな 1 例ずつしか無いので、
//! 「どのかなか当てる」ことは原理的に測れない
//! (kana_probe の「自分自身が最近傍 = 0/45」はその副作用で、意味の無い数字)。
//!
//! 同定を測るには、**同じかなの別の例**が要る。同じかなを「違う言い方」で複数回出し、
//! そのうち 1 つを取り除いて残りから当てられるかを見る (leave-one-out 1-NN)。
//!
//! ## 事前監査で潰した欠陥 (2026-08-26・実測前)
//!
//! 初版を書いたあと、実測する前に 9 レンズの敵対監査に掛けた。6 件の欠陥が出た。
//! すべて**実測してから直したのでは数字が嘘になる**ものだった。
//!
//! 1. **棄却域が空だった (致命的)**。同点を index 昇順・strict `>` で解決していたため、
//!    蝸牛が完全に死んで全条件が同一ベクトルになっても、全 184 条件が index 0 =「あ」を
//!    予測し、かな 4/184 = 2.17% > チャンス 1.64% で G68a が PASS してしまう。
//!    → **同点棄却**にした。最大コサインを与える候補に 2 つ以上のラベルが混じったら
//!    「判定不能」とし、正解に数えない。これで退化系はちょうど 0.0% になる。
//! 2. **雑音軸で不変なのは 6 条件でなく 21 かな 84 条件**。Nasal は純正弦で雑音を
//!    一切消費せず (phoneme_synth.rs:467-493)、Approximant は Nasal に委譲する。
//!    → 注記を訂正し、実際に雑音を消費する 25 かなの部分集合を別に出す。
//! 3. **ん は F0 を受け取らない** (kana.rs の Mora::Moraic は f0_hz を渡さない)。
//!    → 自動正解になる条件数を数えて印字する。
//! 4. **G68e が真の縮退を見ていなかった**。ば行 ≡ ぱ行、ぢ ≡ じ ≡ し、づ ≡ ず ≡ す。
//!    → 手書きの対リストをやめ、71 かなをバイト一致で同値類に分割する。
//! 5. **主軸で行内の子音 30ms がバイト同一だった**。wave_of は毎回 LfsrNoise を
//!    巻き戻し、子音は f0 に依存しないので、か/き/く/け/こ の先頭 30ms が同じ波形になる。
//!    → 主軸を「話者の言い直し」軸にし、条件ごとに異なる雑音実現を与える。
//!    かな単位で seed を固定すると**かなごとの指紋**ができて指紋照合で解けてしまうので、
//!    seed は (かな, 変種) の組ごとに全 184 通り異なる値にする。
//! 6. **子音行ラベルが合成器と一致していない**。し='S'・ち='C'・つ='c' で、
//!    行の中が合成器レベルで別物。しかも初版の宣言は「12 択」と書いたが実際は 11 種類。
//!    → LABELS に合成器の子音記号を第 4 列として持たせ、
//!    「かな行 (言語ラベル)」と「合成子音」の 2 本を別々に出す。
//!    **系についての断定は合成子音の側にだけ掛ける。**
//!
//! ## 正解の出どころ
//!
//! どのかなを合成したかは**実験者が決めた**。母音列・かな行・合成子音のラベルも
//! 実験者が与える。よって「正解が分からないものは計量できない」に抵触しない。
//!
//! ## 測る段
//!
//! **M0.5 出力 (84ch = Octopus 4 + Bushy 40 + Stellate 40)** のスパイク数ベクトル。
//! M0 出力ではない。M1 には触れない。
//!
//! **限定**: 240 step を 1 本の 84 次元カウントに畳んでいるので、Octopus のオンセット
//! 時刻・Bushy の立ち上がり強調という M0.5 の**時間符号は消えている**。
//! 出る値は「M0.5 の」天井ではなく「**M0.5 の時間平均レートベクトルの**」天井。
//!
//! **限定**: 条件ごとに Cochlea/CochlearNucleus を作り直す (reset() は entropy を
//! 持ち越すので冷えた初期状態にならない)。よって出る値は
//! **孤立した 1 モーラの冷開始**の値で、連続発話の値ではない。累積失聴は意図的に除外。
//!
//! ## 天井は蝸牛ではなく合成器が決めている (重要)
//!
//! 合成器には既知の近似が残っている。以下はすべて**蝸牛の性能ではなく合成の欠落**:
//!
//! - **有声/無声の区別が無い**。kana.rs は k と g、t と d、p と b、s と z を
//!   同じ子音にマップする。G68e で同値類として機械的に出す。
//! - **ラ行を破裂音で近似**している。
//! - **フォルマント遷移が無い**。子音から母音への渡りが手がかりにならない。
//! - **拗音 33 音**は 'ゃ'|'ゅ'|'ょ' が直前 CV の母音を差し替えるだけで子音を触らないので、
//!   きゃ = か のように既存かなとバイト同一になる。「未測定」ではなく
//!   **この合成器では原理的に 0/33**。
//!
//! **単独 1 モーラの数字は、連続音声に対する上限でも下限でもない。**
//! フォルマント遷移の欠落は下向きに、分節・縮約・隣接モーラのマスキング・韻律変動の
//! 欠落は上向きに偏る。結果を見てから「下限だ」「上限だ」と選べてしまうので、
//! **片側の境界としては引用しない。**
//!
//! ## ゲート (実測前に固定・以後動かさない)
//!
//! - **G68a かなの同定**: 46 かなの同定率が (i) 退化ベースライン (同点棄却下で 0.0%) と
//!   (ii) チャンス (約 1.64%) の**両方**を超える。
//!   *帰無 = 話し方が変われば別物として扱う → チャンス付近に落ちる。*
//!   **予測: 20〜40%**。根拠 = 母音 5 択が M0.5 で 55% (§14.5.3)、
//!   子音がその半分当たるとして 0.55 × 0.5 = 約 27%。
//!   **この根拠は弱い** (55% は distractor 19 の母音単独課題、本プローブは distractor 183 の
//!   46 クラス。距離空間が違う値を条件数の効果を無視して掛けている)。
//!   予測が外れたら「予測が外れた」と記録し、根拠の弱さを言い訳にしない。
//! - **G68b 母音列の同定**: 母音列 (6 択: あ〜お段 + ん) の同定率がチャンスを超える。
//! - **G68c 合成子音の同定**: 合成器の子音記号 (14 種) の同定率がチャンスを超える。
//!   *ここがチャンス付近なら「子音は聞こえていない」と確定する。*
//!   かな行 (言語ラベル・11 種) も出すが、行は合成器の音響カテゴリではないので
//!   **行が当たらないことを「子音が聞こえない」の根拠にはしない。**
//! - **G68d 決定論性**: 全条件のカウントのハッシュを印字する。プロセスを跨いで比較できる。
//! - **G68e 同値類**: 71 かな (45 清音 + ん + 濁音 20 + 半濁音 5) をバイト一致で
//!   同値類に分割し、一意でないかなが何音あるかを出す。
//!
//! **G68a / G68b / G68c は独立な 3 測定ではない。** (行, 母音) がかなを一意に決めるので
//! かな正解 ⊆ 母音正解 ∩ 行正解 が恒等的に成立する。3 つを独立な証拠として読まない。
//!
//! CLI: kana_identify

use spiking_brain::phase2_f::cochlea::{Cochlea, SAMPLES_PER_STEP};
use spiking_brain::phase2_f::cochlear_nucleus::{CochlearNucleus, N_CN_OUTPUT};
use spiking_brain::phase2_f::kana::{moras_from_kana, synth_utterance};
use spiking_brain::phase2_f::phoneme_synth::LfsrNoise;

/// 実験者が与える正解ラベル: (かな, かな行, 母音列, 合成器の子音記号)
///
/// 第 4 列は kana.rs の kana_to_cv が実際に割り当てる記号。
/// し='S'・ち='C'・つ='c' のように、かな行とは一致しない。
/// ん は Mora::Moraic で Nasal(250,1700) を 1 モーラ丸ごと使うので 'N' と別扱いにする
/// (な行の 'n' と同じパラメータだが長さが違う)。
const LABELS: &[(&str, &str, &str, &str)] = &[
    ("あ", "母音", "あ", "-"), ("い", "母音", "い", "-"), ("う", "母音", "う", "-"),
    ("え", "母音", "え", "-"), ("お", "母音", "お", "-"),
    ("か", "か行", "あ", "k"), ("き", "か行", "い", "k"), ("く", "か行", "う", "k"),
    ("け", "か行", "え", "k"), ("こ", "か行", "お", "k"),
    ("さ", "さ行", "あ", "s"), ("し", "さ行", "い", "S"), ("す", "さ行", "う", "s"),
    ("せ", "さ行", "え", "s"), ("そ", "さ行", "お", "s"),
    ("た", "た行", "あ", "t"), ("ち", "た行", "い", "C"), ("つ", "た行", "う", "c"),
    ("て", "た行", "え", "t"), ("と", "た行", "お", "t"),
    ("な", "な行", "あ", "n"), ("に", "な行", "い", "n"), ("ぬ", "な行", "う", "n"),
    ("ね", "な行", "え", "n"), ("の", "な行", "お", "n"),
    ("は", "は行", "あ", "h"), ("ひ", "は行", "い", "h"), ("ふ", "は行", "う", "h"),
    ("へ", "は行", "え", "h"), ("ほ", "は行", "お", "h"),
    ("ま", "ま行", "あ", "m"), ("み", "ま行", "い", "m"), ("む", "ま行", "う", "m"),
    ("め", "ま行", "え", "m"), ("も", "ま行", "お", "m"),
    ("や", "や行", "あ", "y"), ("ゆ", "や行", "う", "y"), ("よ", "や行", "お", "y"),
    ("ら", "ら行", "あ", "r"), ("り", "ら行", "い", "r"), ("る", "ら行", "う", "r"),
    ("れ", "ら行", "え", "r"), ("ろ", "ら行", "お", "r"),
    ("わ", "わ行", "あ", "w"), ("を", "わ行", "お", "w"),
    ("ん", "ん", "ん", "N"),
    // 2026-08-27: **濁音・半濁音を追加した (23 音・合計 69)。**
    //
    // §14.22 で有声/無声を実装するまで、これらは清音と**完全に同一の波形**だった。
    // だから 46 清音+ん でしか測っておらず、**出口の指標に濁音が入っていなかった。**
    // 実コーパスでは濁音がモーラの 41.7% を占める (§14.21)。
    //
    // **ぢ と づ は入れない。** 現代日本語で ぢ=じ・づ=ず は同音であり、
    // 合成器でもバイト同一になる (§14.22.2 で確認)。**正しい縮退なので、
    // 別クラスとして数えるのは不公平である。**
    ("が", "が行", "あ", "g"), ("ぎ", "が行", "い", "g"), ("ぐ", "が行", "う", "g"),
    ("げ", "が行", "え", "g"), ("ご", "が行", "お", "g"),
    ("ざ", "ざ行", "あ", "z"), ("じ", "ざ行", "い", "Z"), ("ず", "ざ行", "う", "z"),
    ("ぜ", "ざ行", "え", "z"), ("ぞ", "ざ行", "お", "z"),
    ("だ", "だ行", "あ", "d"), ("で", "だ行", "え", "d"), ("ど", "だ行", "お", "d"),
    ("ば", "ば行", "あ", "b"), ("び", "ば行", "い", "b"), ("ぶ", "ば行", "う", "b"),
    ("べ", "ば行", "え", "b"), ("ぼ", "ば行", "お", "b"),
    ("ぱ", "ぱ行", "あ", "p"), ("ぴ", "ぱ行", "い", "p"), ("ぷ", "ぱ行", "う", "p"),
    ("ぺ", "ぱ行", "え", "p"), ("ぽ", "ぱ行", "お", "p"),
];

/// G68e 用: 71 かな (45 清音 + ん + 濁音 20 + 半濁音 5)
const ALL_71: &[&str] = &[
    "あ","い","う","え","お","か","き","く","け","こ","さ","し","す","せ","そ",
    "た","ち","つ","て","と","な","に","ぬ","ね","の","は","ひ","ふ","へ","ほ",
    "ま","み","む","め","も","や","ゆ","よ","ら","り","る","れ","ろ","わ","を","ん",
    "が","ぎ","ぐ","げ","ご","ざ","じ","ず","ぜ","ぞ","だ","ぢ","づ","で","ど",
    "ば","び","ぶ","べ","ぼ","ぱ","ぴ","ぷ","ぺ","ぽ",
];

const F0S: [f64; 4] = [100.0, 130.0, 160.0, 200.0];
const LEVELS: [(i32, i32); 3] = [(1, 1), (1, 2), (1, 4)]; // 0 / -6 / -12 dB
const SEEDS: [u16; 4] = [0xACE1, 0x1234, 0x7FFF, 0x0BAD];

/// 主軸 (話者の言い直し) の seed。(かな, 変種) ごとに全 184 通り異なる。
///
/// かな単位で固定すると**かなごとの指紋**ができて、G68a が指紋照合で解けてしまう。
/// 全条件で異ならせれば、雑音実現の一致という近道が一つも残らない。
fn utterance_seed(kana_idx: usize, variant: usize) -> u16 {
    // 0 は LFSR の吸収状態なので避ける
    ((kana_idx as u16).wrapping_mul(97).wrapping_add(variant as u16).wrapping_mul(2851)) | 1
}

fn wave_of(text: &str, f0: f64, seed: u16, gain_num: i32, gain_den: i32) -> Vec<i32> {
    let mut noise = LfsrNoise::new(seed);
    let (moras, skipped) = moras_from_kana(text);
    // debug_assert は release で消えるので、実行時に必ず落とす
    assert_eq!(skipped, 0, "未対応のかな: {}", text);
    let w = synth_utterance(&moras, f0, &mut noise);
    if gain_num == gain_den {
        w
    } else {
        w.iter().map(|&s| s * gain_num / gain_den).collect()
    }
}

fn cn_counts(wave: &[i32]) -> Vec<u32> {
    let mut co = Cochlea::new();
    let mut cn = CochlearNucleus::new();
    let mut counts = vec![0u32; N_CN_OUTPUT];
    for chunk in wave.chunks(SAMPLES_PER_STEP) {
        if chunk.len() < SAMPLES_PER_STEP {
            break;
        }
        let out = co.process_step(chunk);
        for (i, &v) in cn.process_step(&out).iter().enumerate() {
            if v != 0 {
                counts[i] += 1;
            }
        }
    }
    counts
}

fn cosine(a: &[u32], b: &[u32]) -> f64 {
    let dot: f64 = a.iter().zip(b.iter()).map(|(&x, &y)| x as f64 * y as f64).sum();
    let na: f64 = a.iter().map(|&x| (x as f64).powi(2)).sum::<f64>().sqrt();
    let nb: f64 = b.iter().map(|&x| (x as f64).powi(2)).sum::<f64>().sqrt();
    if na == 0.0 || nb == 0.0 { 0.0 } else { dot / (na * nb) }
}

/// 1-NN の予測。**同点棄却**。
///
/// 最大コサインを与える候補を全部集め、その中にラベルが 2 種類以上あれば None
/// (判定不能) を返す。これが無いと、全条件が同一ベクトルになった退化系が
/// index 最小のラベルを総取りして、チャンスを超えてしまう。
fn predict(conds: &[(usize, Vec<u32>)], i: usize, label: &dyn Fn(usize) -> &'static str) -> Option<usize> {
    let mut best = f64::NEG_INFINITY;
    for j in 0..conds.len() {
        if j == i {
            continue;
        }
        let c = cosine(&conds[i].1, &conds[j].1);
        if c > best {
            best = c;
        }
    }
    let tied: Vec<usize> = (0..conds.len())
        .filter(|&j| j != i && cosine(&conds[i].1, &conds[j].1) == best)
        .map(|j| conds[j].0)
        .collect();
    let first = label(tied[0]);
    if tied.iter().all(|&t| label(t) == first) {
        Some(tied[0])
    } else {
        None
    }
}

fn pct(n: usize, d: usize) -> f64 {
    n as f64 / d as f64 * 100.0
}

/// ラベル関数 f に対する 1-NN のチャンスレベル。
/// 「i を除いた残り n-1 個のうち、i と同じラベルを持つものの割合」の平均。
/// 実験者が与えたラベル分布だけから決まる (データを見ていない)。
fn chance_for(conds: &[(usize, Vec<u32>)], f: &dyn Fn(usize) -> &'static str) -> f64 {
    let n = conds.len();
    let mut acc = 0.0f64;
    for i in 0..n {
        let same = conds.iter().enumerate()
            .filter(|(j, c)| *j != i && f(c.0) == f(conds[i].0))
            .count();
        acc += same as f64 / (n - 1) as f64;
    }
    acc / n as f64 * 100.0
}

fn l_kana(t: usize) -> &'static str { LABELS[t].0 }
fn l_row(t: usize) -> &'static str { LABELS[t].1 }
fn l_vowel(t: usize) -> &'static str { LABELS[t].2 }
fn l_cons(t: usize) -> &'static str { LABELS[t].3 }

struct AxisResult {
    kana: f64,
    vowel: f64,
    cons: f64,
    ch_kana: f64,
    ch_vowel: f64,
    ch_cons: f64,
    degenerate: f64,
}

fn report(axis: &str, n_variants: usize, conds: &[(usize, Vec<u32>)]) -> AxisResult {
    let n = conds.len();

    // --- 健全性: 沈黙・重複ベクトル・同点 ---
    let silent: Vec<usize> = (0..n).filter(|&i| conds[i].1.iter().all(|&x| x == 0)).collect();
    let mut twin = 0usize;
    for i in 0..n {
        if (0..n).any(|j| j != i && conds[j].1 == conds[i].1) {
            twin += 1;
        }
    }

    // --- 同定 ---
    let mut kana_hit = 0usize;
    let mut vowel_hit = 0usize;
    let mut cons_hit = 0usize;
    let mut row_hit = 0usize;
    let mut undecidable = 0usize;
    let mut conf: Vec<(usize, usize)> = Vec::new();
    for i in 0..n {
        match predict(conds, i, &l_kana) {
            None => undecidable += 1,
            Some(p) => {
                let t = conds[i].0;
                if t == p { kana_hit += 1; } else { conf.push((t, p)); }
            }
        }
        if let Some(p) = predict(conds, i, &l_vowel) {
            if l_vowel(conds[i].0) == l_vowel(p) { vowel_hit += 1; }
        }
        if let Some(p) = predict(conds, i, &l_cons) {
            if l_cons(conds[i].0) == l_cons(p) { cons_hit += 1; }
        }
        if let Some(p) = predict(conds, i, &l_row) {
            if l_row(conds[i].0) == l_row(p) { row_hit += 1; }
        }
    }

    let ch_k = chance_for(conds, &l_kana);
    let ch_v = chance_for(conds, &l_vowel);
    let ch_c = chance_for(conds, &l_cons);
    let ch_r = chance_for(conds, &l_row);

    println!();
    println!("--- 軸: {} ({} 条件 = {} かな × {} 通り) ---", axis, n, LABELS.len(), n_variants);
    println!("  [健全性] 無音の条件 {} / 他と完全に同一のベクトルを持つ条件 {} / 判定不能(同点) {}",
             silent.len(), twin, undecidable);
    println!("  かなの同定     : {:>3}/{} = {:>5.1}%   (チャンス {:>5.2}%)", kana_hit, n, pct(kana_hit, n), ch_k);
    println!("  母音列の同定   : {:>3}/{} = {:>5.1}%   (チャンス {:>5.2}%)", vowel_hit, n, pct(vowel_hit, n), ch_v);
    println!("  合成子音の同定 : {:>3}/{} = {:>5.1}%   (チャンス {:>5.2}%)  <- 系についての断定はこちら", cons_hit, n, pct(cons_hit, n), ch_c);
    println!("  かな行の同定   : {:>3}/{} = {:>5.1}%   (チャンス {:>5.2}%)  (言語ラベル・参考)", row_hit, n, pct(row_hit, n), ch_r);

    let mut tally: std::collections::HashMap<(usize, usize), usize> = std::collections::HashMap::new();
    for &c in conf.iter() {
        *tally.entry(c).or_insert(0) += 1;
    }
    let mut top: Vec<_> = tally.into_iter().collect();
    top.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    print!("  混同の上位     : ");
    for (i, ((t, p), c)) in top.iter().take(8).enumerate() {
        if i > 0 { print!(" / "); }
        print!("{}->{}x{}", LABELS[*t].0, LABELS[*p].0, c);
    }
    println!();

    let vowel_err = conf.iter().filter(|(t, p)| l_vowel(*t) != l_vowel(*p)).count();
    let cons_err = conf.iter().filter(|(t, p)| l_cons(*t) != l_cons(*p)).count();
    let both = conf.iter()
        .filter(|(t, p)| l_vowel(*t) != l_vowel(*p) && l_cons(*t) != l_cons(*p))
        .count();
    println!("  誤り {} 件の内訳: 母音を取り違え {} / 合成子音を取り違え {} / 両方 {}",
             conf.len(), vowel_err, cons_err, both);

    AxisResult {
        kana: pct(kana_hit, n), vowel: pct(vowel_hit, n), cons: pct(cons_hit, n),
        ch_kana: ch_k, ch_vowel: ch_v, ch_cons: ch_c, degenerate: 0.0,
    }
}

/// 退化ベースライン: 全条件が同一ベクトルだったときの得点。
/// **データを一切見ずに**、ラベル構成だけから決まる。
fn degenerate_baseline(n_variants: usize) -> (f64, f64, f64) {
    let conds: Vec<(usize, Vec<u32>)> = (0..LABELS.len())
        .flat_map(|k| (0..n_variants).map(move |_| (k, vec![1u32; N_CN_OUTPUT])))
        .collect();
    let n = conds.len();
    let mut k = 0usize;
    let mut v = 0usize;
    let mut c = 0usize;
    for i in 0..n {
        if let Some(p) = predict(&conds, i, &l_kana) { if conds[i].0 == p { k += 1; } }
        if let Some(p) = predict(&conds, i, &l_vowel) { if l_vowel(conds[i].0) == l_vowel(p) { v += 1; } }
        if let Some(p) = predict(&conds, i, &l_cons) { if l_cons(conds[i].0) == l_cons(p) { c += 1; } }
    }
    (pct(k, n), pct(v, n), pct(c, n))
}

fn fnv(data: &[Vec<u32>]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for v in data.iter() {
        for &x in v.iter() {
            for b in x.to_le_bytes().iter() {
                h ^= *b as u64;
                h = h.wrapping_mul(0x100000001b3);
            }
        }
    }
    h
}

fn main() {
    println!("=== 五十音のうちどこまで「どのかなか」を当てられるか ===");
    println!("段: M0.5 出力 ({} ch) の**時間平均レートベクトル** ・ {} かな (45 清音 + ん)",
             N_CN_OUTPUT, LABELS.len());
    println!("正解の出どころ: どのかなを合成したかは実験者が決めた");
    println!();
    println!("【天井は蝸牛でなく合成器】有声/無声なし・ラ行は破裂音で近似・フォルマント遷移なし。");
    println!("【限定】孤立 1 モーラの冷開始。連続発話ではない (累積失聴は意図的に除外)。");
    println!("【限定】時間符号は畳んで消してある。連続音声に対する上限でも下限でもない。");

    // --- 退化ベースライン (データを見る前に確定する) ---
    let (dk, dv, dc) = degenerate_baseline(F0S.len());
    println!();
    println!("--- 退化ベースライン (蝸牛が全条件で同じ応答を返した場合) ---");
    println!("  かな {:.2}% / 母音列 {:.2}% / 合成子音 {:.2}%", dk, dv, dc);
    println!("  同点棄却により 0.00% になるのが正しい。ここが 0 でなければ棄却域が空である。");

    // --- G68e: 71 かなの同値類 ---
    println!();
    println!("--- G68e 71 かなは何通りの波形になるか (合成器の縮退) ---");
    let waves_71: Vec<Vec<i32>> = ALL_71.iter().map(|k| wave_of(k, F0S[0], SEEDS[0], 1, 1)).collect();
    let mut classes: Vec<Vec<&str>> = Vec::new();
    for (i, w) in waves_71.iter().enumerate() {
        match classes.iter_mut().find(|c| waves_71[ALL_71.iter().position(|k| *k == c[0]).unwrap()] == *w) {
            Some(c) => c.push(ALL_71[i]),
            None => classes.push(vec![ALL_71[i]]),
        }
    }
    let collapsed: Vec<&Vec<&str>> = classes.iter().filter(|c| c.len() > 1).collect();
    let n_collapsed_kana: usize = collapsed.iter().map(|c| c.len()).sum();
    println!("  71 かな -> {} 通りの波形 (同値類 {} 個)", classes.len(), classes.len());
    println!("  一意でないかな: {} 音 ({} 個の同値類にまとまっている)", n_collapsed_kana, collapsed.len());
    for c in collapsed.iter() {
        println!("    {{{}}}", c.join(" = "));
    }
    println!("  -> 同値類の中は、蝸牛が何をしようと**原理的に区別できない**(合成器の欠落)。");

    // --- 主軸: 話者の言い直し (F0 と雑音実現が同時に変わる) ---
    let mut main_conds: Vec<(usize, Vec<u32>)> = Vec::new();
    for (k, &(kana, _, _, _)) in LABELS.iter().enumerate() {
        for (v, &f0) in F0S.iter().enumerate() {
            main_conds.push((k, cn_counts(&wave_of(kana, f0, utterance_seed(k, v), 1, 1))));
        }
    }
    let main_result = report("話者の言い直し (F0 100-200Hz + 雑音実現が条件ごとに全て異なる) [主軸]",
                             F0S.len(), &main_conds);

    // --- 副軸: F0 のみ (雑音実現は固定) ---
    let mut f0_conds: Vec<(usize, Vec<u32>)> = Vec::new();
    for (k, &(kana, _, _, _)) in LABELS.iter().enumerate() {
        for &f0 in F0S.iter() {
            f0_conds.push((k, cn_counts(&wave_of(kana, f0, SEEDS[0], 1, 1))));
        }
    }
    println!();
    println!("(注) この軸は雑音実現を固定するので、**同じ行のかなの先頭 30ms がバイト同一**になる");
    println!("     (子音は f0 に依存せず、LfsrNoise は呼び出しごとに巻き戻る)。");
    println!("     行の同定が高く出てもそれは同じ雑音標本の再照合かもしれない。主軸と比べること。");
    report("F0 のみ (100/130/160/200 Hz・雑音実現は固定)", F0S.len(), &f0_conds);

    // --- 副軸: レベル ---
    println!();
    println!("(注) レベル軸は「小さい声の同じかな」ではなく「閾値を跨いだ帯域の生き残り」を測る。");
    println!("     level_axis の記録では -21dB で全母音が無音になる。-12dB は崖の手前。");
    let mut lv_conds: Vec<(usize, Vec<u32>)> = Vec::new();
    for (k, &(kana, _, _, _)) in LABELS.iter().enumerate() {
        for &(gn, gd) in LEVELS.iter() {
            lv_conds.push((k, cn_counts(&wave_of(kana, F0S[0], SEEDS[0], gn, gd))));
        }
    }
    report("レベル (0 / -6 / -12 dB)", LEVELS.len(), &lv_conds);

    // --- 副軸: 雑音実現のみ ---
    println!();
    println!("(注) Nasal は純正弦で雑音を一切消費せず (phoneme_synth.rs:467-493)、");
    println!("     Approximant は Nasal に委譲する。よって雑音で変わらないのは");
    println!("     母音5 + な行5 + ま行5 + や行3 + わ行2 + ん1 = **21 かな 84 条件**。");
    println!("     実際に雑音実現が変わるのは か/さ/た/は/ら行の 25 かなだけ。");
    let mut sd_conds: Vec<(usize, Vec<u32>)> = Vec::new();
    for (k, &(kana, _, _, _)) in LABELS.iter().enumerate() {
        for &s in SEEDS.iter() {
            sd_conds.push((k, cn_counts(&wave_of(kana, F0S[0], s, 1, 1))));
        }
    }
    report("雑音実現のみ (4 seed・全 46 かな)", SEEDS.len(), &sd_conds);

    // --- かなごとの成否 (主軸) ---
    println!();
    println!("--- 主軸でかなごとに何回当たったか (4 回中) ---");
    let mut per_kana = vec![0usize; LABELS.len()];
    for i in 0..main_conds.len() {
        if let Some(p) = predict(&main_conds, i, &l_kana) {
            if p == main_conds[i].0 {
                per_kana[main_conds[i].0] += 1;
            }
        }
    }
    for (k, &(kana, row, _, _)) in LABELS.iter().enumerate() {
        let mark = match per_kana[k] { 4 => "4", 3 => "3", 2 => "2", 1 => "1", _ => "." };
        print!("{}{} ", kana, mark);
        if row == "ん" || (k + 1) % 5 == 0 {
            println!();
        }
    }
    println!("  (数字 = 4 回中の正解数・. = 0 回)");
    let full: Vec<&str> = LABELS.iter().enumerate().filter(|(k, _)| per_kana[*k] == 4).map(|(_, l)| l.0).collect();
    let some: Vec<&str> = LABELS.iter().enumerate().filter(|(k, _)| per_kana[*k] > 0).map(|(_, l)| l.0).collect();
    let zero: Vec<&str> = LABELS.iter().enumerate().filter(|(k, _)| per_kana[*k] == 0).map(|(_, l)| l.0).collect();
    println!("  全問正解 ({} 音): {}", full.len(), full.join(""));
    println!("  1 回でも当たった ({} 音): {}", some.len(), some.join(""));
    println!("  一度も当たらない ({} 音): {}", zero.len(), zero.join(""));

    // --- G68d 決定論性 ---
    println!();
    println!("--- G68d 決定論性 ---");
    let vecs: Vec<Vec<u32>> = main_conds.iter().map(|(_, v)| v.clone()).collect();
    println!("  主軸 全 {} 条件のカウントのハッシュ: {:016x}", vecs.len(), fnv(&vecs));
    println!("  (このプローブを 2 回起動して、この行が一致すれば決定論)");

    // --- 判定 ---
    let r = main_result;
    println!();
    println!("=== 判定 (ゲートは実測前に固定・動かさない) ===");
    println!("  G68a かなの同定     {:>5.1}%  vs 退化 {:.2}% / チャンス {:>5.2}%  -> {}",
             r.kana, dk, r.ch_kana,
             if r.kana > dk && r.kana > r.ch_kana { "PASS" } else { "**FAIL**" });
    println!("       (予測は 20-40%。外れていれば予測が外れたと記録する)");
    println!("  G68b 母音列の同定   {:>5.1}%  vs 退化 {:.2}% / チャンス {:>5.2}%  -> {}",
             r.vowel, dv, r.ch_vowel,
             if r.vowel > dv && r.vowel > r.ch_vowel { "PASS" } else { "**FAIL**" });
    println!("  G68c 合成子音の同定 {:>5.1}%  vs 退化 {:.2}% / チャンス {:>5.2}%  -> {}",
             r.cons, dc, r.ch_cons,
             if r.cons > dc && r.cons > r.ch_cons { "PASS" } else { "**FAIL**" });
    println!();
    println!("  G68a/b/c は独立でない: (行, 母音) がかなを一意に決めるので");
    println!("  かな正解 ⊆ 母音正解 ∩ 子音正解 が恒等的に成立する。独立な 3 証拠として読まない。");
}
