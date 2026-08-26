//! 連続発話 × 平衡でかなを同定する — 子音の数字は二重に悲観側だった (2026-08-27)
//!
//! ## なぜ測り直すのか
//!
//! `kana_identify` の子音まわりの数字は**すべて二つの条件で測られていた**:
//!
//! 1. **孤立モーラ** — §14.17 で「母音の後に置くと M0.5 で重心が **+949Hz**」と判明。
//!    孤立では**文脈強調が原理的に起きない**。
//! 2. **真の冷開始** — §14.18 で「平衡では重心が **+139〜218Hz**」と判明。
//!    冷開始では**試行をまたぐ適応が効いていない**。
//!
//! **この 2 つは両方とも子音に不利な向きである。**
//!
//! 影響を受ける結論:
//! - 主軸 合成子音 20.7% (自発OFF) / 10.9% (自発ON)
//! - §14.9.4「当たったのはほぼ母音・子音はゼロ」
//! - §14.6.5「壁は不変性・落ちているのは子音」
//!
//! **これらは全部、子音にとって最悪の条件で測った値である。**
//!
//! ## 【2026-08-27 実測後の訂正】2×2 は成立しない
//!
//! ② (孤立×平衡) と ④ (連続×平衡) が**完全に同じ数字**を出した。理由は単純:
//! **リセットしなければ、孤立させたつもりでも自動的に連続になる。**
//! かなを続けて流して状態を持ち越せば、それは連続発話そのものである。
//!
//! **「孤立」と「平衡」は独立な軸ではなく、排他的だった。**
//! 実際にあったのは 3 条件:
//! ① 孤立 (= 必然的に冷開始) / ③ 連続・1パス目 / ②=④ 連続・ウォームアップ後
//!
//! ## 2×2 の対照 (当初の設計・上記のとおり成立しなかった)
//!
//! | | 冷開始 | 平衡 |
//! |---|---|---|
//! | **孤立** | ① 現行の `kana_identify` と同条件 (対照) | ② 適応だけ |
//! | **連続** | ③ 文脈だけ | ④ **両方** |
//!
//! **どちらが効いているかを分けられるようにする。**
//!
//! ## 連続発話の作り方
//!
//! 46 かなを決定論的な順序で 1 本の発話につなぎ、**各モーラの窓だけを数える**。
//! 変種ごとに**別の順序**を使うので、同じかなが毎回**違う直前のかな**を持つ。
//! これが「連続発話」の実体である (§14.17 で使った「/a/ を前に置く」より現実的)。
//!
//! 平衡アームは M0.5 を**リセットせず**、測る前に同じ列を `WARMUP` 回流す。
//!
//! ## ゲート (実測前に固定・`kana_identify` の G68 と同一定義)
//!
//! **正解の出どころ**: どのかなをどの順で合成したかは実験者が決めた。
//!
//! - **G80a かなの同定**: 4 アームで比較。**連続×平衡 > 孤立×冷開始** か。
//! - **G80b 合成子音の同定 (本命)**: 同上。**子音が救われるかの問い。**
//! - **G80c 母音列の同定**: 同上。
//! - **G80d 退化ベースライン**: 全アームで 0.00% (同点棄却が効いていること)。
//! - **G80e 決定論性**: 2 回実行してハッシュ一致。
//!
//! 変動の無いクラス (全条件がバイト同一) は §14.9 と同じ規則で除く。
//!
//! ## 予測 (実測前に固定・機構つき・事前)
//!
//! **合成子音の同定は 連続×平衡 で最も高くなるはず。**
//! §14.17 (文脈強調 M0.5 +949Hz) と §14.18 (平衡で +139〜218Hz) が**両方効く**ので。
//!
//! **母音列は大きくは変わらないはず。** 母音は元々よく取れている (83〜93%)。
//!
//! **数値は置かない。** 順位だけ。§14.6.4 / §14.7 / §14.9.7 で数値予測を 3 連続で外した。
//!
//! CLI: kana_continuous

use spiking_brain::phase2_f::cochlea::{Cochlea, SAMPLES_PER_STEP};
use spiking_brain::phase2_f::cochlear_nucleus::{CochlearNucleus, N_CN_OUTPUT};
use spiking_brain::phase2_f::kana::{moras_from_kana, synth_utterance, MORA_MS};
use spiking_brain::phase2_f::phoneme_synth::LfsrNoise;

/// (かな, かな行, 母音列, 合成器の子音記号) — §14.6 の LABELS と同一
const LABELS: &[(&str, &str, &str, &str)] = &[
    ("あ","母音","あ","-"),("い","母音","い","-"),("う","母音","う","-"),("え","母音","え","-"),("お","母音","お","-"),
    ("か","か行","あ","k"),("き","か行","い","k"),("く","か行","う","k"),("け","か行","え","k"),("こ","か行","お","k"),
    ("さ","さ行","あ","s"),("し","さ行","い","S"),("す","さ行","う","s"),("せ","さ行","え","s"),("そ","さ行","お","s"),
    ("た","た行","あ","t"),("ち","た行","い","C"),("つ","た行","う","c"),("て","た行","え","t"),("と","た行","お","t"),
    ("な","な行","あ","n"),("に","な行","い","n"),("ぬ","な行","う","n"),("ね","な行","え","n"),("の","な行","お","n"),
    ("は","は行","あ","h"),("ひ","は行","い","h"),("ふ","は行","う","h"),("へ","は行","え","h"),("ほ","は行","お","h"),
    ("ま","ま行","あ","m"),("み","ま行","い","m"),("む","ま行","う","m"),("め","ま行","え","m"),("も","ま行","お","m"),
    ("や","や行","あ","y"),("ゆ","や行","う","y"),("よ","や行","お","y"),
    ("ら","ら行","あ","r"),("り","ら行","い","r"),("る","ら行","う","r"),("れ","ら行","え","r"),("ろ","ら行","お","r"),
    ("わ","わ行","あ","w"),("を","わ行","お","w"),
    ("ん","ん","ん","N"),
];

const F0S: [f64; 4] = [100.0, 130.0, 160.0, 200.0];
const N_VAR: usize = 4;
const WARMUP: usize = 3;
/// 1 モーラの step 数 (MORA_MS=120ms ・ 16kHz ・ 8 サンプル/step)
const STEPS_PER_MORA: usize = (MORA_MS as usize) * 16 / SAMPLES_PER_STEP;

fn utterance_seed(k: usize, v: usize) -> u16 {
    ((k as u16).wrapping_mul(97).wrapping_add(v as u16).wrapping_mul(2851)) | 1
}

/// 変種 v での かなの並び順 (決定論・変種ごとに別)
fn order_for(v: usize) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..LABELS.len()).collect();
    let mut s = 0xC0FF_EE00_1234_5678u64 ^ ((v as u64) << 32);
    for i in (1..idx.len()).rev() {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        idx.swap(i, ((s >> 33) as usize) % (i + 1));
    }
    idx
}

fn mora_wave(k: usize, f0: f64, seed: u16) -> Vec<i32> {
    let mut n = LfsrNoise::new(seed);
    let (m, sk) = moras_from_kana(LABELS[k].0);
    assert_eq!(sk, 0);
    synth_utterance(&m, f0, &mut n)
}

/// 1 アーム分の条件ベクトルを作る。返り値 = [(クラス, 84次元カウント)]
fn build_arm(continuous: bool, equilibrium: bool) -> Vec<(usize, Vec<f64>)> {
    let mut out: Vec<(usize, Vec<f64>)> = Vec::new();
    for v in 0..N_VAR {
        let ord = order_for(v);
        let f0 = F0S[v];
        if continuous {
            // 46 かなを 1 本の発話につなぐ
            let mut wave: Vec<i32> = Vec::new();
            for &k in ord.iter() {
                wave.extend_from_slice(&mora_wave(k, f0, utterance_seed(k, v)));
            }
            let mut co = Cochlea::new();
            let mut cn = CochlearNucleus::new();
            if equilibrium {
                // **リセットせず**同じ列を WARMUP 回流してから測る
                for _ in 0..WARMUP {
                    for chunk in wave.chunks(SAMPLES_PER_STEP) {
                        if chunk.len() < SAMPLES_PER_STEP { break; }
                        let m0 = co.process_step(chunk);
                        let _ = cn.process_step(&m0);
                    }
                }
            }
            let mut counts = vec![vec![0f64; N_CN_OUTPUT]; ord.len()];
            for (step, chunk) in wave.chunks(SAMPLES_PER_STEP).enumerate() {
                if chunk.len() < SAMPLES_PER_STEP { break; }
                let m0 = co.process_step(chunk);
                let cnout = cn.process_step(&m0);
                let mi = step / STEPS_PER_MORA;
                if mi < ord.len() {
                    for (i, &x) in cnout.iter().enumerate() { if x != 0 { counts[mi][i] += 1.0; } }
                }
            }
            for (mi, &k) in ord.iter().enumerate() { out.push((k, counts[mi].clone())); }
        } else {
            // 孤立: かなごとに独立
            let mut co = Cochlea::new();
            let mut cn = CochlearNucleus::new();
            if equilibrium {
                // 平衡: **リセットせず** ウォームアップしてから、続けて各かなを測る
                for _ in 0..WARMUP {
                    for &k in ord.iter() {
                        let w = mora_wave(k, f0, utterance_seed(k, v));
                        for chunk in w.chunks(SAMPLES_PER_STEP) {
                            if chunk.len() < SAMPLES_PER_STEP { break; }
                            let m0 = co.process_step(chunk);
                            let _ = cn.process_step(&m0);
                        }
                    }
                }
            }
            for &k in ord.iter() {
                let w = mora_wave(k, f0, utterance_seed(k, v));
                let (mut c2, mut n2);
                let (co_r, cn_r): (&mut Cochlea, &mut CochlearNucleus) = if equilibrium {
                    (&mut co, &mut cn)
                } else {
                    c2 = Cochlea::new(); n2 = CochlearNucleus::new(); (&mut c2, &mut n2)
                };
                let mut counts = vec![0f64; N_CN_OUTPUT];
                for chunk in w.chunks(SAMPLES_PER_STEP) {
                    if chunk.len() < SAMPLES_PER_STEP { break; }
                    let m0 = co_r.process_step(chunk);
                    for (i, &x) in cn_r.process_step(&m0).iter().enumerate() {
                        if x != 0 { counts[i] += 1.0; }
                    }
                }
                out.push((k, counts));
            }
        }
    }
    out
}

fn cosine(a: &[f64], b: &[f64]) -> f64 {
    let dot: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let na: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
    let nb: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();
    if na == 0.0 || nb == 0.0 { 0.0 } else { dot / (na * nb) }
}

/// 同点棄却つき 1-NN (§14.6 と同じ規則)
fn identify(conds: &[(usize, Vec<f64>)], label: &dyn Fn(usize) -> &'static str) -> (usize, usize) {
    let n = conds.len();
    let (mut hit, mut undec) = (0usize, 0usize);
    for i in 0..n {
        let mut best = f64::NEG_INFINITY;
        for j in 0..n { if j != i { let c = cosine(&conds[i].1, &conds[j].1); if c > best { best = c; } } }
        let tied: Vec<usize> = (0..n).filter(|&j| j != i && cosine(&conds[i].1, &conds[j].1) == best)
            .map(|j| conds[j].0).collect();
        let f = label(tied[0]);
        if tied.iter().all(|&t| label(t) == f) {
            if f == label(conds[i].0) { hit += 1; }
        } else { undec += 1; }
    }
    (hit, undec)
}

fn l_kana(t: usize) -> &'static str { LABELS[t].0 }
fn l_vowel(t: usize) -> &'static str { LABELS[t].2 }
fn l_cons(t: usize) -> &'static str { LABELS[t].3 }

/// 変動の無いクラス (全条件がバイト同一) を除く (§14.9 と同じ規則)
fn drop_invariant(conds: Vec<(usize, Vec<f64>)>) -> (Vec<(usize, Vec<f64>)>, Vec<&'static str>) {
    let mut dropped = Vec::new();
    let mut keep = Vec::new();
    for k in 0..LABELS.len() {
        let mine: Vec<&Vec<f64>> = conds.iter().filter(|(c, _)| *c == k).map(|(_, v)| v).collect();
        if mine.len() > 1 && mine.iter().all(|v| **v == *mine[0]) { dropped.push(LABELS[k].0); }
        else { keep.push(k); }
    }
    (conds.into_iter().filter(|(c, _)| keep.contains(c)).collect(), dropped)
}

fn fnv(c: &[(usize, Vec<f64>)]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for (k, v) in c {
        for b in (*k as u64).to_le_bytes().iter() { h ^= *b as u64; h = h.wrapping_mul(0x100000001b3); }
        for x in v { for b in (*x as u64).to_le_bytes().iter() { h ^= *b as u64; h = h.wrapping_mul(0x100000001b3); } }
    }
    h
}

struct Res { kana: f64, vowel: f64, cons: f64, n: usize, undec: usize, dropped: Vec<&'static str> }

fn eval(continuous: bool, equilibrium: bool) -> Res {
    let (conds, dropped) = drop_invariant(build_arm(continuous, equilibrium));
    let n = conds.len();
    let (k, u) = identify(&conds, &l_kana);
    let (v, _) = identify(&conds, &l_vowel);
    let (c, _) = identify(&conds, &l_cons);
    Res { kana: k as f64 / n as f64 * 100.0, vowel: v as f64 / n as f64 * 100.0,
          cons: c as f64 / n as f64 * 100.0, n, undec: u, dropped }
}

fn main() {
    println!("=== 連続発話 × 平衡でかなを同定する ===");
    println!();
    println!("【なぜ測り直すか】kana_identify の子音の数字は**二重に悲観側**だった:");
    println!("  1. 孤立モーラ — §14.17: 母音の後に置くと M0.5 で重心 **+949Hz**");
    println!("  2. 真の冷開始 — §14.18: 平衡では重心 **+139〜218Hz**");
    println!("  **両方とも子音に不利な向き。**");
    println!();
    println!("【2x2 の対照】どちらが効いているかを分ける");
    println!("  ① 孤立×冷開始 (現行と同条件)  ② 孤立×平衡 (適応だけ)");
    println!("  ③ 連続×冷開始 (文脈だけ)      ④ **連続×平衡 (両方)**");
    println!();
    println!("【連続発話】46 かなを決定論的な順序で 1 本につなぎ、各モーラの窓だけ数える。");
    println!("変種ごとに別の順序なので、同じかなが毎回**違う直前のかな**を持つ。");
    println!("平衡アームは M0.5 をリセットせず、測る前に同じ列を {} 回流す。", WARMUP);
    println!();
    println!("【ゲート・実測前に固定】kana_identify の G68 と同一定義");
    println!("  G80a かなの同定   G80b **合成子音の同定 (本命)**   G80c 母音列の同定");
    println!("  G80d 退化ベースライン 0.00%   G80e 決定論性");
    println!();
    println!("【予測・事前・機構つき】**合成子音は 連続×平衡 で最も高くなるはず。**");
    println!("§14.17 と §14.18 が両方効くので。**母音列は大きく変わらないはず** (元々 83-93%)。");
    println!("**数値は置かない。順位だけ。**");

    let arms = [
        ("① 孤立×冷開始 (対照)", false, false),
        ("② 【無効】孤立×平衡", false, true),
        ("③ 連続×冷開始", true, false),
        ("④ 連続×平衡 (=②と同一条件)", true, true),
    ];

    println!();
    println!("  アーム                    n   除いたクラス  判定不能  かな同定  母音列  **合成子音**");
    let mut res = Vec::new();
    for &(nm, c, e) in arms.iter() {
        let r = eval(c, e);
        println!("  {:<24} {:>3}   {:<8}  {:>6}  {:>7.1}% {:>6.1}% {:>10.1}%",
                 nm, r.n,
                 if r.dropped.is_empty() { "なし".to_string() } else { r.dropped.join("") },
                 r.undec, r.kana, r.vowel, r.cons);
        res.push(r);
    }
    println!("  (チャンス: かな 約2.2% / 母音列 約19% / 合成子音 約9%)");

    // --- G80d 退化ベースライン ---
    let n_cls = LABELS.len();
    let degen: Vec<(usize, Vec<f64>)> = (0..n_cls)
        .flat_map(|k| (0..N_VAR).map(move |_| (k, vec![1f64; N_CN_OUTPUT]))).collect();
    let (dk, _) = identify(&degen, &l_kana);
    println!();
    println!("  G80d 退化ベースライン (全条件が同一ベクトル): {:.2}% -> {}",
             dk as f64 / degen.len() as f64 * 100.0,
             if dk == 0 { "**PASS** (同点棄却が効いている)" } else { "**FAIL**" });

    // --- 判定 ---
    let base = &res[0];
    let best = &res[3];
    println!();
    println!("=== 判定 (規則は実測前に固定) ===");
    println!("  G80a かなの同定   ①{:.1}% → ④{:.1}%  -> {}",
             base.kana, best.kana, if best.kana > base.kana { "**改善**" } else { "改善せず" });
    println!("  G80b **合成子音**  ①{:.1}% → ④{:.1}%  -> {}",
             base.cons, best.cons, if best.cons > base.cons { "**改善**" } else { "**改善せず**" });
    println!("  G80c 母音列       ①{:.1}% → ④{:.1}%  -> {}",
             base.vowel, best.vowel, if best.vowel > base.vowel { "改善" } else { "改善せず" });
    println!();
    let order_ok = res[3].cons >= res[1].cons && res[3].cons >= res[2].cons;
    println!("  予測「合成子音は ④ で最も高い」-> {} (①{:.1} ②{:.1} ③{:.1} ④{:.1})",
             if order_ok { "**当たり**" } else { "**外れ**" },
             res[0].cons, res[1].cons, res[2].cons, res[3].cons);
    println!("  どちらが効いたか: 文脈だけ(③−①)={:+.1}pt / 適応だけ(②−①)={:+.1}pt / 両方(④−①)={:+.1}pt",
             res[2].cons - res[0].cons, res[1].cons - res[0].cons, res[3].cons - res[0].cons);

    // --- G80e 決定論性 ---
    let a = build_arm(true, true);
    let b = build_arm(true, true);
    println!();
    println!("  G80e 決定論性: {:016x} / {:016x} -> {}",
             fnv(&a), fnv(&b), if fnv(&a) == fnv(&b) { "PASS" } else { "**FAIL**" });
    println!();
    println!("  【既定は変えていない。】");
}
