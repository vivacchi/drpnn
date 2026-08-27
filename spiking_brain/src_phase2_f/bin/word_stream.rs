//! 単語をストリームとして聞き比べる — 再現性と弁別性 (2026-08-27)
//!
//! ## なぜ — ユーザーの指摘による目標の変更
//!
//! > 「単語や熟語会話を認識するには**かなの発音聞き分けじゃなくて、文体全体の学習**が
//! >  必要なのかも。…人間も『あーーーー』とか『ーーーーーー』は最終的に『あーーーー』だから
//! >  分からなくなるし、**ストリームで考えているのだから、ここもかなごとに区切って
//! >  非線形にしたら意味がない**のだと思う。なので、例えば**単語を複数用意して聞き比べ、
//! >  再現性を確認する**のが優先なんだと思う。」
//!
//! §14.38 (V6) がこの指摘の裏を取っている: **いまの刺激はストリームではない。**
//! 連続合成と個別連結が**バイト同一**で、貼り目にちょうど −35 dB の切れ目がある。
//! **「かなごとに区切ったら意味がない」どころか、合成器の方が先に区切っていた。**
//!
//! 「定常部には同一性が無く、情報は変化の側にある」も具体的な帰結を持つ。
//! 遷移 (§14.28) と VOT (§14.31) があれだけ効いたのはそのためである。
//!
//! ## この計器の作法
//!
//! - **窓で切らない。** `CONSONANT_STEPS` のような分節知識を一切使わない。
//! - **整列アルゴリズムを持ち込まない。** DTW も**系が持てない知識**なので使わない。
//!   代わりに**モーラ数を揃えた単語群**を使い、フレーム列をそのまま比べる。
//! - **時間平均とフレーム列を並べる。** これが本題。
//!
//! ## ゲート (実測前に固定・以後動かさない)
//!
//! **正解の出どころ**: どの単語をどう合成したかは実験者が決めた。
//!
//! - **G94a 時間平均で単語を区別できるか** (チャンスと比べる)
//! - **G94b フレーム列で区別できるか**
//! - **G94c 予測の判定: フレーム列 > 時間平均か**
//! - **G94d 再現性と弁別性**: 同一単語どうし / 異単語どうし の平均コサインとその差
//! - **G94e 退化ベースライン**: 全条件が同一ベクトルなら 0.0%
//! - **G94f 決定論性**
//!
//! ## 予測 (**実測前・ユーザーの洞察からそのまま出る**)
//!
//! > **時間平均では単語が区別できず、フレーム列なら区別できる。**
//!
//! 機構: 単語の同一性は**フォルマントがどう動いたか**にあり、
//! 時間平均はその動きを潰す。定常部だけを見れば「あーーーー」と「ーーーーーー」が
//! 区別できないのと同じことである。
//!
//! **これが外れたら「情報は変化にある」という読みが間違っている。外れ方が情報になる。**
//!
//! CLI: word_stream

use spiking_brain::phase2_f::cochlea::{Cochlea, N_BANDS, SAMPLES_PER_STEP};
use spiking_brain::phase2_f::cochlear_nucleus::{CochlearNucleus, N_CN_OUTPUT};
use spiking_brain::phase2_f::kana::{moras_from_kana, synth_utterance};
use spiking_brain::phase2_f::phoneme_synth::LfsrNoise;

/// **最小対 (1 モーラだけ違う 3 モーラの実在語) 16 組 = 32 語。**
///
/// ## なぜ最小対にしたか (2026-08-27・**理由を明記する**)
///
/// 最初は無関係な 16 語で測り、**フレーム列で 100.0% に張り付いた** (§14.39)。
/// **天井に張り付いた計器では、連続合成への作り直しが何を変えたかを測れない。**
///
/// **難しくしたのは「勝つため」ではなく「測れるようにするため」である。**
/// §14.39 の 16 語の結果はそのまま記録に残してある。
///
/// モーラ数は揃えたままなので、**整列アルゴリズムは依然として要らない。**
const PAIRS: &[(&str, &str)] = &[
    ("こころ", "ところ"),   // 1 モーラ目
    ("からだ", "かなだ"),   // 2 モーラ目
    ("たまご", "たなご"),   // 2
    ("てがみ", "てあみ"),   // 2
    ("せかい", "せたい"),   // 2
    ("みどり", "みのり"),   // 2
    ("かたち", "かたな"),   // 3
    ("ひかり", "ひかる"),   // 3
    ("さかな", "さかや"),   // 3
    ("くるま", "くるみ"),   // 3
    ("なまえ", "なまり"),   // 3
    ("ちから", "ちかく"),   // 3
    ("いのち", "いのり"),   // 3
    ("みかん", "みかた"),   // 3
    ("あたま", "あたり"),   // 3
    ("からす", "からて"),   // 3
];

/// 平坦化した単語列。`WORDS[2k]` と `WORDS[2k+1]` が最小対をなす。
fn words() -> Vec<&'static str> {
    PAIRS.iter().flat_map(|&(a, b)| [a, b]).collect()
}

const F0S: [f64; 4] = [100.0, 130.0, 160.0, 200.0];
const N_VAR: usize = 4;
/// 1 フレーム = 10ms = 20 step (DT_MS=0.5)
const STEPS_PER_FRAME: usize = 20;

/// **変種ごとに単語の並び順を変える** (2026-08-27 追加)。
///
/// これが無いと**平衡アームで単語 w が常に単語 w−1 の直後に来る**ので、
/// 4 変種すべてで文脈が同じになり、**持ち越された適応状態が単語の同一性と相関する**。
/// 「直前に何が来たか」が手がかりになってしまう (登録簿「測定条件が結論を作る」)。
/// 冷開始アームは単語ごとにリセットするので影響を受けない = **対照になる。**
fn order_for(v: usize) -> Vec<usize> {
    let n = PAIRS.len() * 2;
    let mut idx: Vec<usize> = (0..n).collect();
    let mut s = 0xC0FF_EE00_1234_5678u64 ^ ((v as u64) << 32);
    for i in (1..n).rev() {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        idx.swap(i, ((s >> 33) as usize) % (i + 1));
    }
    idx
}

fn utterance_seed(w: usize, v: usize) -> u16 {
    ((w as u16).wrapping_mul(131).wrapping_add(v as u16).wrapping_mul(4099)) | 1
}

/// 単語を鳴らして (M0 のフレーム列, M0.5 のフレーム列) を返す。
/// `reset` が false なら蝸牛・神経核の状態を持ち越す (= 平衡アーム)。
fn frames(
    w: usize, v: usize, co: &mut Cochlea, cn: &mut CochlearNucleus,
) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
    let mut n = LfsrNoise::new(utterance_seed(w, v));
    let (m, sk) = moras_from_kana(words()[w]);
    assert_eq!(sk, 0, "未対応の単語: {}", words()[w]);
    let wave = synth_utterance(&m, F0S[v], &mut n);
    let (mut m0f, mut cnf) = (Vec::new(), Vec::new());
    let (mut m0a, mut cna) = (vec![0f64; N_BANDS], vec![0f64; N_CN_OUTPUT]);
    let mut in_frame = 0usize;
    for chunk in wave.chunks(SAMPLES_PER_STEP) {
        if chunk.len() < SAMPLES_PER_STEP { break; }
        let m0 = co.process_step(chunk);
        let cno = cn.process_step(&m0);
        for (i, &x) in m0.iter().enumerate() { if x != 0 { m0a[i] += 1.0; } }
        for (i, &x) in cno.iter().enumerate() { if x != 0 { cna[i] += 1.0; } }
        in_frame += 1;
        if in_frame == STEPS_PER_FRAME {
            m0f.push(std::mem::replace(&mut m0a, vec![0f64; N_BANDS]));
            cnf.push(std::mem::replace(&mut cna, vec![0f64; N_CN_OUTPUT]));
            in_frame = 0;
        }
    }
    (m0f, cnf)
}

/// フレーム列 → 時間平均 (1 フレーム分の次元に潰す)
fn averaged(f: &[Vec<f64>]) -> Vec<f64> {
    let d = f.first().map(|x| x.len()).unwrap_or(0);
    let mut out = vec![0f64; d];
    for fr in f { for i in 0..d { out[i] += fr[i]; } }
    for x in out.iter_mut() { *x /= f.len().max(1) as f64; }
    out
}

/// フレーム列 → 連結 (時間を保つ)
fn flattened(f: &[Vec<f64>]) -> Vec<f64> {
    f.iter().flat_map(|x| x.iter().cloned()).collect()
}

fn cosine(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len().min(b.len());
    let d: f64 = (0..n).map(|i| a[i] * b[i]).sum();
    let na: f64 = a[..n].iter().map(|x| x * x).sum::<f64>().sqrt();
    let nb: f64 = b[..n].iter().map(|x| x * x).sum::<f64>().sqrt();
    if na == 0.0 || nb == 0.0 { 0.0 } else { d / (na * nb) }
}

/// 同点棄却つき 1-NN。棄却は不正解として数える (保守側)。
fn accuracy(v: &[(usize, Vec<f64>)]) -> f64 {
    let n = v.len();
    let mut ok = 0usize;
    for i in 0..n {
        let mut best = f64::NEG_INFINITY;
        for j in 0..n { if j != i { let c = cosine(&v[i].1, &v[j].1); if c > best { best = c; } } }
        let tied: Vec<usize> = (0..n).filter(|&j| j != i && cosine(&v[i].1, &v[j].1) == best)
            .map(|j| v[j].0).collect();
        if tied.is_empty() { continue; }
        let first = tied[0];
        if tied.iter().all(|&t| t == first) && first == v[i].0 { ok += 1; }
    }
    ok as f64 / n as f64 * 100.0
}

/// 1-NN の誤答のうち、**最小対の相手**だった割合。誤りが原理的かを見る。
/// `WORDS[2k]` と `WORDS[2k+1]` が対なので、`w ^ 1` が相手。
fn partner_share(v: &[(usize, Vec<f64>)]) -> (usize, usize) {
    let n = v.len();
    let (mut wrong, mut partner) = (0usize, 0usize);
    for i in 0..n {
        let mut best = f64::NEG_INFINITY;
        let mut arg = 0usize;
        for j in 0..n {
            if j == i { continue; }
            let c = cosine(&v[i].1, &v[j].1);
            if c > best { best = c; arg = v[j].0; }
        }
        if arg != v[i].0 {
            wrong += 1;
            if arg == (v[i].0 ^ 1) { partner += 1; }
        }
    }
    (partner, wrong)
}

/// 同一クラスどうし / 異クラスどうし の平均コサイン
fn within_between(v: &[(usize, Vec<f64>)]) -> (f64, f64) {
    let (mut w, mut nw, mut b, mut nb) = (0f64, 0usize, 0f64, 0usize);
    for i in 0..v.len() {
        for j in (i + 1)..v.len() {
            let c = cosine(&v[i].1, &v[j].1);
            if v[i].0 == v[j].0 { w += c; nw += 1; } else { b += c; nb += 1; }
        }
    }
    (w / nw.max(1) as f64, b / nb.max(1) as f64)
}

fn chance(n_class: usize, per_class: usize) -> f64 {
    let n = n_class * per_class;
    (per_class - 1) as f64 / (n - 1) as f64 * 100.0
}

struct Arm { m0_avg: f64, m0_seq: f64, cn_avg: f64, cn_seq: f64,
             seq_w: f64, seq_b: f64, avg_w: f64, avg_b: f64, partner: usize, wrong: usize }

/// `warm` が true なら単語をまたいで状態を持ち越す (平衡アーム)
fn eval(warm: bool) -> Arm {
    let (mut m0_avg, mut m0_seq) = (Vec::new(), Vec::new());
    let (mut cn_avg, mut cn_seq) = (Vec::new(), Vec::new());
    let (mut co, mut cn) = (Cochlea::new(), CochlearNucleus::new());
    if warm {
        // ウォームアップ: 全単語を 1 周流してから測る
        for v in 0..N_VAR { for &w in order_for(v).iter() { let _ = frames(w, v, &mut co, &mut cn); } }
    }
    for v in 0..N_VAR {
        for &w in order_for(v).iter() {
            let (m0f, cnf) = if warm {
                frames(w, v, &mut co, &mut cn)
            } else {
                let (mut c1, mut c2) = (Cochlea::new(), CochlearNucleus::new());
                frames(w, v, &mut c1, &mut c2)
            };
            m0_avg.push((w, averaged(&m0f)));
            m0_seq.push((w, flattened(&m0f)));
            cn_avg.push((w, averaged(&cnf)));
            cn_seq.push((w, flattened(&cnf)));
        }
    }
    let ps = partner_share(&cn_seq);
    let (sw, sb) = within_between(&cn_seq);
    let (aw, ab) = within_between(&cn_avg);
    Arm {
        m0_avg: accuracy(&m0_avg), m0_seq: accuracy(&m0_seq),
        cn_avg: accuracy(&cn_avg), cn_seq: accuracy(&cn_seq),
        seq_w: sw, seq_b: sb, avg_w: aw, avg_b: ab, partner: ps.0, wrong: ps.1,
    }
}

fn main() {
    println!("=== 単語をストリームとして聞き比べる — 再現性と弁別性 ===");
    println!();
    println!("【なぜ・ユーザーの指摘による目標の変更】");
    println!("「単語や熟語会話を認識するには**かなの聞き分けじゃなくて文体全体の学習**が必要。");
    println!(" **ストリームで考えているのだから、かなごとに区切って非線形にしたら意味がない。**");
    println!(" **単語を複数用意して聞き比べ、再現性を確認する**のが優先」");
    println!();
    println!("§14.38(V6) がこの指摘の裏を取っている: **いまの刺激はストリームではない。**");
    println!("連続合成と個別連結が**バイト同一**で、貼り目に −35dB の切れ目がある。");
    println!("**『かなごとに区切ったら意味がない』どころか、合成器の方が先に区切っていた。**");
    println!();
    println!("【この計器の作法】**窓で切らない**(CONSONANT_STEPS のような分節知識を使わない)。");
    println!("**整列アルゴリズムを持ち込まない**(DTW も系が持てない知識)。");
    println!("代わりに**モーラ数を揃えた実在語**を使い、フレーム列をそのまま比べる。");
    println!();
    println!("【予測・実測前・ユーザーの洞察からそのまま出る】");
    println!("  **時間平均では単語が区別できず、フレーム列なら区別できる。**");
    println!("  機構: 単語の同一性は**フォルマントがどう動いたか**にあり、時間平均はそれを潰す。");
    println!("  定常部だけ見れば「あーーーー」と「ーーーーーー」が区別できないのと同じこと。");
    println!("  **外れたら『情報は変化にある』という読みが間違っている。外れ方が情報になる。**");

    let ch = chance(words().len(), N_VAR);
    println!();
    println!("  単語 {} 語 × F0 {} 変種 = {} 条件。1 フレーム = 10ms。",
             words().len(), N_VAR, words().len() * N_VAR);
    println!("  **チャンス = {:.2}%** (同点棄却つき 1-NN・棄却は不正解として計上)", ch);

    let cold = eval(false);
    let warm = eval(true);

    println!();
    println!("--- 単語の同定率 ---");
    println!("  {:<22} {:>12} {:>14} {:>10}", "", "**時間平均**", "**フレーム列**", "差");
    println!("  {:<22} {:>11.1}% {:>13.1}% {:>+10.1}", "冷開始 · M0 (40帯域)", cold.m0_avg, cold.m0_seq, cold.m0_seq - cold.m0_avg);
    println!("  {:<22} {:>11.1}% {:>13.1}% {:>+10.1}", "冷開始 · M0.5 (84ch)", cold.cn_avg, cold.cn_seq, cold.cn_seq - cold.cn_avg);
    println!("  {:<22} {:>11.1}% {:>13.1}% {:>+10.1}", "平衡 · M0 (40帯域)", warm.m0_avg, warm.m0_seq, warm.m0_seq - warm.m0_avg);
    println!("  {:<22} {:>11.1}% {:>13.1}% {:>+10.1}", "平衡 · M0.5 (84ch)", warm.cn_avg, warm.cn_seq, warm.cn_seq - warm.cn_avg);

    println!();
    println!("  **G94a 時間平均で区別できるか** (M0.5 冷開始 {:.1}% vs チャンス {:.2}%) -> {}",
             cold.cn_avg, ch, if cold.cn_avg > ch * 2.0 { "**できる**" } else { "**ほぼできない**" });
    println!("  **G94b フレーム列で区別できるか** ({:.1}%) -> {}",
             cold.cn_seq, if cold.cn_seq > ch * 2.0 { "**できる**" } else { "**ほぼできない**" });
    println!("  **G94c 予測の判定: フレーム列 > 時間平均か** -> {}",
             if cold.cn_seq > cold.cn_avg { "**当たり — フレーム列が上**" }
             else if cold.cn_seq < cold.cn_avg { "**外れ — 時間平均が上**" } else { "**同じ**" });

    println!();
    println!("--- G94d 再現性と弁別性 (M0.5・冷開始・平均コサイン) ---");
    println!("  {:<16} {:>12} {:>12} {:>12}", "", "同一単語", "異単語", "**差**");
    println!("  {:<16} {:>12.4} {:>12.4} {:>12.4}", "時間平均", cold.avg_w, cold.avg_b, cold.avg_w - cold.avg_b);
    println!("  {:<16} {:>12.4} {:>12.4} {:>12.4}", "**フレーム列**", cold.seq_w, cold.seq_b, cold.seq_w - cold.seq_b);
    println!("  *同一単語(=再現性)が高く、異単語(=弁別性)が低いほどよい。差が大きいほど分離している。*");

    // --- G94e 退化ベースライン ---
    let deg: Vec<(usize, Vec<f64>)> = (0..words().len() * N_VAR)
        .map(|i| (i / N_VAR, vec![1.0f64; 64])).collect();
    println!();
    println!("  G94e 退化ベースライン (全条件が同一ベクトル) -> {:.2}% -> {}",
             accuracy(&deg), if accuracy(&deg) == 0.0 { "**PASS** (同点棄却が効いている)" } else { "**FAIL**" });

    let again = eval(false);
    println!("  G94f 決定論性 -> {}",
             if (again.cn_seq - cold.cn_seq).abs() < 1e-12 { "PASS" } else { "**FAIL**" });

    println!();
    println!("  **誤答 {} 件のうち最小対の相手だったのは {} 件** (原理的な誤りか無関係な誤りか)",
             cold.wrong, cold.partner);
    println!();
    println!("  【この計器が答えないこと】**いまの刺激はストリームではない** (§14.38)。");
    println!("  **これはその上でのベースラインである。**連続合成に作り直したあと、");
    println!("  **同じ計器で測り直す**ことで、作り直しが何を変えたかが分かる。");
    println!("  (「比較したいものは壊す前に測る」= 今日 G67b で登録した教訓)");
}
