//! コーパスを聞かせた M1 は、何かを読み出せるようになるか (2026-08-27)
//!
//! ## なぜ — 一度も測っていない穴
//!
//! これまでの同定率は**すべて M0.5 出力の読み出し**である。
//! **M1 の出力で同定を測ったことが一度もない。**
//!
//! さらに、ユーザーの定義「**シナプスの動的平衡に至ることを学習と言う**」に対して、
//! **平衡に至った網が至っていない網より良くなっているか**を一度も測っていない。
//! §14.29 で平衡に達したことは示したが、**それが読み出しを良くしたかは別の問い**である。
//!
//! ## 帰無を先に置く
//!
//! **帰無: M1 の同定率は M0.5 以下である。**
//!
//! 理由は 2 つあり、どちらも強い:
//! 1. **段を 1 つ足せば普通は情報が減る** (データ処理不等式)。
//! 2. **M1 は教師なしであり、同定という目的を一切知らない。**
//!    M1 は局所的な物理過程 (LTP/LTD/vitality) だけで動いており、
//!    「かなを区別せよ」という圧力はどこにも無い。
//!
//! **面白いのは帰無が破れたときだけである。**
//!
//! ## 測り方 — 本線を乱さずに分岐する
//!
//! コーパスを 1 回流しながら、決めておいた地点で **(蝸牛・神経核・M1) を丸ごと複製**し、
//! **複製の側にだけ**テスト刺激を流す。本線はテストの影響を受けない。
//! これで「**いま何モーラ聞いた網か**」ごとの読み出しを、1 回の走行で並べられる。
//!
//! テスト刺激は **連続** (69 かなを 1 本につなぐ)。コーパスが連続なので、
//! 網は連続発話の平衡にいる。**孤立で試すと条件が食い違う。**
//! 窓は **時間平均のみ** — 2 窓は `CONSONANT_STEPS` という
//! **M1 が持てない分節知識**を使うので、ここでは使わない。
//!
//! ## ゲート (実測前に固定・以後動かさない)
//!
//! **正解の出どころ**: どのかなをどの順で合成したかは実験者が決めた。
//!
//! - **G92a (帰無) M1 ≤ M0.5 か**: 全地点で M1 の合成子音同定が M0.5 以下。
//!   *破れたら、それがこの測定の発見である。*
//! - **G92b M1 はチャンスを超えるか**: かな 1.09% / 合成子音 5.81% / 母音列 約19%。
//!   *超えなければ M1 の出力は何も表現していない。*
//! - **G92c 用量反応 — 聞くほど良くなるか**: 聞いたモーラ数に対して
//!   M1 の同定率が**単調に上がるか**。
//!   ***これが「動的平衡に至ることが学習である」ことの唯一の証拠になる量。***
//! - **G92d 決定論性**: 同じ地点の複製を 2 回測って一致。
//! - **G92e コーパスの内容は一切出力しない**: **数値のみ。**
//!
//! ## 予測 (実測前・機構つき・数値は置かない)
//!
//! 1. **G92a は保たれる (M1 < M0.5)。** 上の 2 つの理由が強い。
//! 2. **G92b は超える。** M1 は M0.5 の入力を受けているので、
//!    完全に情報を捨てていない限り何かは残る。
//! 3. **G92c は上がらない。** これが本命の予測である。
//!    M1 の可塑性は**同定を目的にしていない**ので、
//!    平衡に至ることと読み出しが良くなることは**別のこと**のはず。
//!    **上がったら、それは私の予測が外れたということであり、大きな発見である。**
//! 4. **冷開始 (0 モーラ) では M1 出力がほとんど発火せず、判定不能が多発するはず。**
//!
//! CLI: corpus_analysis   (DRPNN_CORPUS_MORAS でモーラ数・既定 12000)

use spiking_brain::phase2_f::cochlea::{Cochlea, SAMPLES_PER_STEP};
use spiking_brain::phase2_f::cochlear_nucleus::{CochlearNucleus, N_CN_OUTPUT};
use spiking_brain::phase2_f::kana::{moras_from_kana, synth_utterance, Mora, MORA_MS};
use spiking_brain::phase2_f::phoneme_synth::LfsrNoise;
use spiking_brain::phase2_f::thermo_network::{ThermoNetwork, ThermoNetworkConfig};
use std::collections::HashMap;
use std::io::Read;

const CORPUS: &str = "../data/corpus/roleplay_kana.txt";
const SEED: u16 = 0xACE1;
const STEPS_PER_MORA: usize = (MORA_MS as usize) * 16 / SAMPLES_PER_STEP;
const F0S: [f64; 4] = [100.0, 130.0, 160.0, 200.0];
const N_VAR: usize = 4;

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

/// コーパスの先頭から `n_moras` 分のモーラ。**内容は返さない・出力もしない。**
fn load_moras(n_moras: usize) -> (Vec<Mora>, usize) {
    let want = n_moras * 9 + 4096;
    let mut f = std::fs::File::open(CORPUS)
        .unwrap_or_else(|e| panic!("コーパスが開けない ({}): {}", CORPUS, e));
    let mut buf = vec![0u8; want];
    let mut filled = 0usize;
    while filled < want {
        match f.read(&mut buf[filled..]) {
            Ok(0) => break,
            Ok(n) => filled += n,
            Err(e) => panic!("読み込み失敗: {}", e),
        }
    }
    buf.truncate(filled);
    let text = loop {
        match std::str::from_utf8(&buf) {
            Ok(s) => break s.to_string(),
            Err(e) => { buf.truncate(e.valid_up_to()); }
        }
    };
    let mut out: Vec<Mora> = Vec::new();
    let mut kinds: std::collections::BTreeSet<char> = std::collections::BTreeSet::new();
    for c in text.chars() {
        if out.len() >= n_moras { break; }
        if c == '\n' || c == ' ' { continue; }
        let (m, _) = moras_from_kana(&c.to_string());
        if !m.is_empty() { kinds.insert(c); }
        out.extend(m);
    }
    out.truncate(n_moras);
    (out, kinds.len())
}

/// 1 条件分の読み出し。**M0.5 (84ch) と M1 出力 (40ch) を同じ走行で取る。**
struct Readout { label: usize, cn: Vec<f64>, m1: Vec<f64> }

/// **複製の側に**テスト刺激を連続で流す。本線は触らない。
fn test_battery(
    net: &ThermoNetwork, co: &Cochlea, cn: &CochlearNucleus,
) -> (Vec<Readout>, f64, f64) {
    let mut out: Vec<Readout> = Vec::new();
    let n_out = net.output_neurons.len();
    let (mut cn_sum, mut m1_sum) = (0f64, 0f64);
    for v in 0..N_VAR {
        // **地点ごとに複製する。** 変種どうしが互いを汚さないようにする。
        let (mut net, mut co, mut cn) = (net.clone(), co.clone(), cn.clone());
        let ord = order_for(v);
        let mut wave: Vec<i32> = Vec::new();
        for &k in ord.iter() {
            let mut n = LfsrNoise::new(utterance_seed(k, v));
            let (m, sk) = moras_from_kana(LABELS[k].0);
            assert_eq!(sk, 0, "未対応: {}", LABELS[k].0);
            wave.extend(synth_utterance(&m, F0S[v], &mut n));
        }
        let mut cnv = vec![vec![0f64; N_CN_OUTPUT]; ord.len()];
        let mut m1v = vec![vec![0f64; n_out]; ord.len()];
        for (step, chunk) in wave.chunks(SAMPLES_PER_STEP).enumerate() {
            if chunk.len() < SAMPLES_PER_STEP { break; }
            let mi = step / STEPS_PER_MORA;
            if mi >= ord.len() { break; }
            let m0 = co.process_step(chunk);
            let cn_out = cn.process_step(&m0);
            for (i, &x) in cn_out.iter().enumerate() { if x != 0 { cnv[mi][i] += 1.0; } }
            for nid in net.step(&cn_out) {
                if let Some(oi) = net.output_index_of(nid) { m1v[mi][oi] += 1.0; }
            }
        }
        for (mi, &k) in ord.iter().enumerate() {
            cn_sum += cnv[mi].iter().sum::<f64>();
            m1_sum += m1v[mi].iter().sum::<f64>();
            out.push(Readout { label: k, cn: cnv[mi].clone(), m1: m1v[mi].clone() });
        }
    }
    let n = out.len() as f64;
    (out, cn_sum / n, m1_sum / n)
}

fn cosine(a: &[f64], b: &[f64]) -> f64 {
    let d: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let na: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
    let nb: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();
    if na == 0.0 || nb == 0.0 { 0.0 } else { d / (na * nb) }
}

/// 同点棄却つき 1-NN。**棄却は「不正解」として数える (保守側)。**
/// 全ゼロベクトルどうしはコサイン 0 で全同点になるため、**退化は自動的に棄却される。**
fn nn(v: &[(usize, Vec<f64>)], label: &dyn Fn(usize) -> &'static str)
    -> (Vec<(usize, usize)>, usize) {
    let n = v.len();
    let (mut pairs, mut undec) = (Vec::new(), 0usize);
    for i in 0..n {
        let mut best = f64::NEG_INFINITY;
        for j in 0..n { if j != i { let c = cosine(&v[i].1, &v[j].1); if c > best { best = c; } } }
        let tied: Vec<usize> = (0..n).filter(|&j| j != i && cosine(&v[i].1, &v[j].1) == best)
            .map(|j| v[j].0).collect();
        if tied.is_empty() { undec += 1; continue; }
        let f = label(tied[0]);
        if tied.iter().all(|&t| label(t) == f) { pairs.push((v[i].0, tied[0])); } else { undec += 1; }
    }
    (pairs, undec)
}

fn l_kana(t: usize) -> &'static str { LABELS[t].0 }
fn l_vowel(t: usize) -> &'static str { LABELS[t].1 }
fn l_cons(t: usize) -> &'static str { LABELS[t].2 }

fn acc(v: &[(usize, Vec<f64>)], label: &dyn Fn(usize) -> &'static str) -> f64 {
    let (p, _) = nn(v, label);
    p.iter().filter(|(t, q)| label(*t) == label(*q)).count() as f64 / v.len() as f64 * 100.0
}

fn transmitted(pairs: &[(usize, usize)], f: &dyn Fn(&str) -> &'static str) -> f64 {
    let mut joint: HashMap<(&str, &str), f64> = HashMap::new();
    let (mut px, mut py): (HashMap<&str, f64>, HashMap<&str, f64>) = (HashMap::new(), HashMap::new());
    if pairs.is_empty() { return 0.0; }
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

struct Score { kana: f64, vowel: f64, cons: f64, voi: f64, pla: f64, undec: usize, density: f64 }

fn score(v: Vec<(usize, Vec<f64>)>, density: f64) -> Score {
    let (pairs, undec) = nn(&v, &l_cons);
    Score {
        kana: acc(&v, &l_kana), vowel: acc(&v, &l_vowel), cons: acc(&v, &l_cons),
        voi: transmitted(&pairs, &|s| feat3(s).0) * 100.0,
        pla: transmitted(&pairs, &|s| feat3(s).2) * 100.0,
        undec, density,
    }
}

fn main() {
    let n_moras: usize = std::env::var("DRPNN_CORPUS_MORAS").ok()
        .and_then(|v| v.parse().ok()).unwrap_or(12000);
    let checkpoints: Vec<usize> = vec![0, 250, 1000, 4000, 9000, n_moras]
        .into_iter().filter(|&c| c <= n_moras).collect();

    println!("=== コーパスを聞かせた M1 は、何かを読み出せるようになるか ===");
    println!();
    println!("【一度も測っていない穴】これまでの同定率は**すべて M0.5 出力の読み出し**。");
    println!("**M1 の出力で同定を測ったことが一度もない。**");
    println!("さらに「**シナプスの動的平衡に至ることを学習と言う**」という定義に対して、");
    println!("**平衡に至った網が至っていない網より良いか**を一度も測っていない。");
    println!();
    println!("【帰無を先に置く】**M1 の同定率は M0.5 以下である。**");
    println!("  ① 段を1つ足せば普通は情報が減る (データ処理不等式)");
    println!("  ② **M1 は教師なしで、同定という目的を一切知らない。**");
    println!("     局所的な物理過程だけで動いており「かなを区別せよ」という圧力はどこにも無い。");
    println!("**面白いのは帰無が破れたときだけである。**");
    println!();
    println!("【測り方】コーパスを1回流しながら決めた地点で **(蝸牛・神経核・M1) を丸ごと複製**し、");
    println!("**複製の側にだけ**テスト刺激を流す。本線はテストの影響を受けない。");
    println!("テストは **連続** (69かなを1本につなぐ)。窓は **時間平均のみ**");
    println!("(2窓は CONSONANT_STEPS という **M1 が持てない分節知識**を使うので使わない)。");
    println!();
    println!("【ゲート・実測前に固定】正解の出どころ = どのかなをどの順で合成したかは実験者が決めた");
    println!("  **G92a (帰無) 全地点で M1 ≤ M0.5 か**");
    println!("  G92b M1 はチャンスを超えるか (かな 1.09% / 合成子音 5.81% / 母音列 約19%)");
    println!("  **G92c 用量反応 — 聞くほど M1 が良くなるか**");
    println!("     ***これが「動的平衡に至ることが学習である」ことの唯一の証拠になる量。***");
    println!("  G92d 決定論性 / **G92e コーパスの内容は一切出力しない**");
    println!();
    println!("【予測・事前・数値は置かない】");
    println!("  ① G92a は保たれる (M1 < M0.5)");
    println!("  ② G92b は超える (M1 は M0.5 を受けているので完全には捨てていないはず)");
    println!("  ③ **G92c は上がらない。これが本命の予測。**");
    println!("     M1 の可塑性は**同定を目的にしていない**ので、平衡に至ることと");
    println!("     読み出しが良くなることは**別のこと**のはず。");
    println!("     **上がったら私の予測が外れたということであり、大きな発見である。**");
    println!("  ④ 冷開始 (0モーラ) では M1 出力がほとんど発火せず判定不能が多発するはず");

    let t0 = std::time::Instant::now();
    let (moras, kinds) = load_moras(n_moras);
    println!();
    println!("  コーパス: {} モーラ / **{} 種類のかなが鳴った**。**内容は出力しない。**",
             moras.len(), kinds);

    let cfg = if N_CN_OUTPUT == 164 { ThermoNetworkConfig::for_m1_cn_80() }
              else { ThermoNetworkConfig::for_m1_cn_40() };
    assert_eq!(cfg.n_input, N_CN_OUTPUT,
        "M1 の入力数 {} と M0.5 の出力数 {} が一致しない", cfg.n_input, N_CN_OUTPUT);
    let mut net = ThermoNetwork::new(cfg);
    let mut co = Cochlea::new();
    let mut cn = CochlearNucleus::new();
    let mut noise = LfsrNoise::new(SEED);

    let mut rows: Vec<(usize, Score, Score, usize)> = Vec::new();
    let mut next = 0usize;
    for i in 0..=moras.len() {
        if next < checkpoints.len() && checkpoints[next] == i {
            let (r, cn_d, m1_d) = test_battery(&net, &co, &cn);
            let cnv: Vec<(usize, Vec<f64>)> = r.iter().map(|x| (x.label, x.cn.clone())).collect();
            let m1v: Vec<(usize, Vec<f64>)> = r.iter().map(|x| (x.label, x.m1.clone())).collect();
            rows.push((i, score(cnv, cn_d), score(m1v, m1_d), net.n_open_synapses()));
            next += 1;
        }
        if i == moras.len() { break; }
        let w = synth_utterance(std::slice::from_ref(&moras[i]), F0S[0], &mut noise);
        for chunk in w.chunks(SAMPLES_PER_STEP) {
            if chunk.len() < SAMPLES_PER_STEP { break; }
            let m0 = co.process_step(chunk);
            let cn_out = cn.process_step(&m0);
            let _ = net.step(&cn_out);
        }
    }

    println!();
    println!("--- 読み出し (連続・時間平均・同点棄却つき 1-NN・棄却は不正解として計上) ---");
    println!();
    println!("  {:>7} {:>8} | {:>7} {:>7} {:>7} {:>7} {:>7} | {:>7} {:>7} {:>7} {:>7} {:>7} {:>6}",
             "聞いた", "alive", "CN子音", "CNかな", "CN母音", "CN有声", "CN位置",
             "**M1子音**", "M1かな", "M1母音", "M1有声", "M1位置", "不能");
    for (m, c, k, alive) in rows.iter() {
        println!("  {:>7} {:>8} | {:>6.1}% {:>6.1}% {:>6.1}% {:>6.1}% {:>6.1}% | {:>6.1}% {:>6.1}% {:>6.1}% {:>6.1}% {:>6.1}% {:>6}",
                 m, alive, c.cons, c.kana, c.vowel, c.voi, c.pla,
                 k.cons, k.kana, k.vowel, k.voi, k.pla, k.undec);
    }
    println!("  (チャンス: かな 1.09% / 母音列 約19% / 合成子音 5.81%)");
    println!();
    println!("  発火密度 (1条件あたりの総発火数): CN {:.0} / M1出力 {:.1}",
             rows.last().unwrap().1.density, rows.last().unwrap().2.density);
    println!("  *密度が飽和していると発火数が情報を運ばない (今セッションで登録済みの失敗型)。*");

    // --- G92a ---
    println!();
    let violations: Vec<usize> = rows.iter().filter(|(_, c, k, _)| k.cons > c.cons)
        .map(|(m, _, _, _)| *m).collect();
    println!("  **G92a (帰無) 全地点で M1 ≤ M0.5 か** -> {}",
             if violations.is_empty() { "**帰無は保たれた** (M1 は M0.5 を超えなかった)".to_string() }
             else { format!("**帰無が破れた。M1 > M0.5 になった地点: {:?} モーラ**", violations) });

    // --- G92b ---
    let best_m1 = rows.iter().map(|(_, _, k, _)| k.cons).fold(f64::NEG_INFINITY, f64::max);
    println!("  G92b M1 はチャンス (合成子音 5.81%) を超えるか -> 最良 {:.1}% -> {}",
             best_m1, if best_m1 > 5.81 { "**超えた**" } else { "**超えない — M1 出力は何も表現していない**" });

    // --- G92c ---
    let first = rows.first().unwrap().2.cons;
    let last = rows.last().unwrap().2.cons;
    let mono = rows.windows(2).all(|w| w[1].2.cons >= w[0].2.cons);
    println!("  **G92c 用量反応 — 聞くほど M1 が良くなるか**");
    println!("     0 モーラ {:.1}% -> {} モーラ {:.1}% ({:+.1}pt) / 単調増加: {}",
             first, rows.last().unwrap().0, last, last - first, if mono { "**はい**" } else { "いいえ" });
    println!("     -> {}", if last > first {
        "**上がった。予測③が外れた = 平衡に至ることが読み出しを良くしている**"
    } else if last < first {
        "**下がった。平衡に至ることは読み出しを良くしない (むしろ悪くする)**"
    } else { "**変わらない。平衡に至ることと読み出しは別のこと (予測どおり)**" });

    // --- G92d ---
    let (r1, d1, _) = test_battery(&net, &co, &cn);
    let (r2, _, _) = test_battery(&net, &co, &cn);
    let s1 = score(r1.iter().map(|x| (x.label, x.m1.clone())).collect(), d1);
    let s2 = score(r2.iter().map(|x| (x.label, x.m1.clone())).collect(), d1);
    println!();
    println!("  G92d 決定論性 -> {}", if (s1.cons - s2.cons).abs() < 1e-12 { "PASS" } else { "**FAIL**" });
    println!("  G92e コーパスの内容 -> **一切出力していない (数値のみ)**");
    println!();
    println!("  所要 {:.1} 秒。", t0.elapsed().as_secs_f64());
    println!("  【この測定が答えないこと】M1 の出力層 40 個の**発火数**しか見ていない。");
    println!("  **時間パターンは捨てている。**M1 が時間で表現していれば、この計器では見えない。");
}
