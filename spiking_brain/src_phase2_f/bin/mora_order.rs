//! 順序課題 — M2 は「順序」を空間パターンに変換できているか (2026-08-28)
//!
//! ## なぜ — G100b「段を足せば情報は減る」を破る道
//!
//! §14.53 で M2 は「長い時間スケールの符号」と実証されたが、**情報量では M1 を
//! 一度も上回っていない** (G100b 帰無は 6/6 走行で維持)。
//! 現行の最小対は「**どのモーラが鳴ったか**」の課題であり、M0.5 の率符号でも解ける。
//! M2 の武器 (因果窓 320 step = 160ms ≈ モーラ 1.3 個) が効くのは
//! 「**どの順で鳴ったか**しか手がかりが無い」課題のはずである。
//!
//! ## 設計 (ユーザー承認: Q1-A 無意味語の円順列 / Q2-A 時間平均+語末窓)
//!
//! - **刺激**: 3 モーラの無意味語 10 組 × 円順列 3 通り (ABC/BCA/CAB) = 30 クラス × 4 F0
//!   = 120 条件。**同じ組の 3 順列はモーラ集合が同一** — 実在語の連接頻度の交絡が無い。
//! - **順序盲の読み出し** (これが本題):
//!   - **時間平均** — 順序を完全に捨てる
//!   - **語末窓** (最終モーラ 120ms のカウント) — 再帰状態の代理。
//!     「どんな順序でここまで辿り着いたか」の履歴が畳み込まれているはずの場所
//!   - 対照として**フレーム列** (10ms) — 順序が読み出し器側にあるので全段で高いはず (陽性対照)
//! - **判定は群内 1-NN**: 候補を同じ組の 12 条件 (3 順列 × 4 変種) に限る。
//!   内容は完全に統制され、**残る手がかりは順序だけ**。群内チャンス = 3/11 ≈ 27.3%。
//!
//! ## 設計上の正直な注記 (実測前に書く)
//!
//! **連続合成 (§14.41) の協調調音は、順序を部分的に「内容」に変換する** —
//! ABC の遷移集合 {A→B, B→C} と BCA の {B→C, C→A} は違う。
//! したがって「M0.5 の時間平均は構成的にチャンス」は**厳密には成立しない**
//! (実音声でも同じ)。帰無は構成でなく**群内置換帰無 + 段間比較**に置く。
//! **判定の核は「順序盲読み出しで M2 > M1 か」であり、これは遷移の存在に影響されない**
//! (遷移は全段に等しく届くので)。
//!
//! ## ゲート (実測前に固定・以後動かさない)
//!
//! - **G102a 陽性対照**: フレーム列で全段が群内置換帰無を超える
//!   (順序情報は信号に存在し、読み出し器側で組める)。*落ちたら計器の欠陥。*
//! - **G102b 記述**: M0.5 時間平均の群内成績 (遷移でどこまで持ち上がるかの床)。
//! - **G102c 本丸**: **順序盲読み出し (時間平均・語末窓) の群内成績で M2 > M1 か。**
//!   さらに M2 が M0.5 も上回れば、**G100b の壁は「次元」でなく「情報の質」だったことになる。**
//! - **G102d** 群内置換帰無 (ラベルを組の中でシャッフル・列×地点で Bonferroni)。
//! - **G102e** 決定論性 / **G102f** コーパスの内容は一切出力しない (学習はコーパス・刺激は合成)。
//!
//! ## 予測 (実測前・数値は置かない・②と G100c の予測が外れた履歴を銘記)
//!
//! 1. G102a は通るはず (陽性対照)。
//! 2. M0.5 の時間平均は遷移ぶんだけ群内チャンスを超えるが、控えめのはず。
//! 3. **G102c は賭けである。** M2 の順序盲読み出しが M1 を超えれば設計の勝ち。
//!    超えなければ「M2 はまだ順序を空間化できていない」— それも確定として価値がある。
//!
//! CLI: mora_order  (DRPNN_CORPUS_MORAS 既定 4000)

use spiking_brain::phase2_f::cochlea::{Cochlea, SAMPLES_PER_STEP};
use spiking_brain::phase2_f::cochlear_nucleus::{CochlearNucleus, N_CN_OUTPUT};
use spiking_brain::phase2_f::kana::{moras_from_kana, synth_utterance, Mora, MORA_MS};
use spiking_brain::phase2_f::phoneme_synth::LfsrNoise;
use spiking_brain::phase2_f::thermo_network::{ThermoNetwork, ThermoNetworkConfig, SIGNAL_SCALE_DIVISOR};
use std::io::Read;

const CORPUS: &str = "../data/corpus/roleplay_kana.txt";
const F0S: [f64; 4] = [100.0, 130.0, 160.0, 200.0];
const N_VAR: usize = 4;
const N_PERM: usize = 400;
const INPUT_CURRENT_M2: i32 = 60;
const FRAME: usize = 20;                       // 10ms (フレーム列・陽性対照用)
const MORA_STEPS: usize = (MORA_MS as usize) * 16 / SAMPLES_PER_STEP;   // 240

/// 3 モーラの無意味語 10 組。全て単純 CV・組内で重複なし。
/// 調音位置・有声性・母音を混ぜてある (特定の特徴に依存しないため)。
const TRIPLES: &[[&str; 3]] = &[
    ["か", "め", "そ"], ["に", "ろ", "た"], ["す", "べ", "や"], ["こ", "ぬ", "ぜ"],
    ["ま", "り", "ど"], ["は", "ぐ", "ね"], ["ぼ", "ち", "わ"], ["て", "む", "ざ"],
    ["き", "の", "ぶ"], ["ら", "ぱ", "せ"],
];

/// クラス = (組, 円順列)。ABC/BCA/CAB。
fn class_word(group: usize, rot: usize) -> String {
    let t = TRIPLES[group];
    (0..3).map(|k| t[(k + rot) % 3]).collect()
}

fn lcg(s: &mut u64) -> u64 {
    *s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    *s >> 33
}

fn shuffled(n: usize, seed: u64) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..n).collect();
    let mut s = seed | 1;
    for i in (1..n).rev() { let r = lcg(&mut s) as usize % (i + 1); idx.swap(i, r); }
    idx
}

fn utterance_seed(c: usize, v: usize) -> u16 {
    ((c as u16).wrapping_mul(151).wrapping_add(v as u16).wrapping_mul(3557)) | 1
}

fn load_moras(n: usize) -> Vec<Mora> {
    let want = n * 9 + 4096;
    let mut f = std::fs::File::open(CORPUS)
        .unwrap_or_else(|e| panic!("コーパスが開けない ({}): {}", CORPUS, e));
    let mut buf = vec![0u8; want];
    let mut filled = 0usize;
    while filled < want {
        match f.read(&mut buf[filled..]) { Ok(0) => break, Ok(k) => filled += k, Err(e) => panic!("{}", e) }
    }
    buf.truncate(filled);
    let text = loop {
        match std::str::from_utf8(&buf) { Ok(s) => break s.to_string(), Err(e) => buf.truncate(e.valid_up_to()) }
    };
    let mut out = Vec::new();
    for ch in text.chars() {
        if out.len() >= n { break; }
        if ch == '\n' || ch == ' ' { continue; }
        let (m, _) = moras_from_kana(&ch.to_string());
        out.extend(m);
    }
    out.truncate(n);
    out
}

fn m2_input(m1: &ThermoNetwork, fired: &[usize], n_in: usize) -> Vec<i32> {
    let mut v = vec![0i32; n_in];
    for &nid in fired {
        if let Some(oi) = m1.output_index_of(nid) {
            if oi < n_in { v[oi] = v[oi].saturating_add(INPUT_CURRENT_M2); }
        }
    }
    v
}

/// 1 条件の読み出し 9 本: 3 段 × {フレーム列, 時間平均, 語末窓}
struct Cond { class: usize, group: usize, v: [Vec<f64>; 9] }

fn battery(m1n: &ThermoNetwork, m2n: &ThermoNetwork, co: &Cochlea, cn: &CochlearNucleus) -> Vec<Cond> {
    let n_cls = TRIPLES.len() * 3;
    let n1 = m1n.output_neurons.len();
    let n2 = m2n.output_neurons.len();
    let n2_in = m2n.input_neurons.len();
    let mut out: Vec<Cond> = Vec::new();
    for var in 0..N_VAR {
        let (mut m1c, mut m2c, mut coc, mut cnc) = (m1n.clone(), m2n.clone(), co.clone(), cn.clone());
        for &c in shuffled(n_cls, 0x5EED ^ ((var as u64) << 24)).iter() {
            let (g, rot) = (c / 3, c % 3);
            let word = class_word(g, rot);
            let mut noise = LfsrNoise::new(utterance_seed(c, var));
            let (m, sk) = moras_from_kana(&word);
            assert_eq!(sk, 0, "未対応: {}", word);
            let wave = synth_utterance(&m, F0S[var], &mut noise);
            let n_steps = wave.len() / SAMPLES_PER_STEP;
            let n_frames = n_steps / FRAME;
            let last_start = n_steps.saturating_sub(MORA_STEPS);   // 語末窓 = 最終モーラ
            let dims = [N_CN_OUTPUT, n1, n2];
            let mut seq: Vec<Vec<f64>> = dims.iter().map(|&d| vec![0f64; n_frames * d]).collect();
            let mut avg: Vec<Vec<f64>> = dims.iter().map(|&d| vec![0f64; d]).collect();
            let mut end: Vec<Vec<f64>> = dims.iter().map(|&d| vec![0f64; d]).collect();
            for (step, chunk) in wave.chunks(SAMPLES_PER_STEP).enumerate() {
                if chunk.len() < SAMPLES_PER_STEP { break; }
                let m0 = coc.process_step(chunk);
                let cno = cnc.process_step(&m0);
                let fired1 = m1c.step(&cno);
                let inp2 = m2_input(&m1c, &fired1, n2_in);
                let fired2 = m2c.step(&inp2);
                let fr = step / FRAME;
                let tail = step >= last_start;
                // 段 0: M0.5
                for (i, &x) in cno.iter().enumerate() {
                    if x != 0 {
                        if fr < n_frames { seq[0][fr * N_CN_OUTPUT + i] += 1.0; }
                        avg[0][i] += 1.0;
                        if tail { end[0][i] += 1.0; }
                    }
                }
                // 段 1: M1 出力
                for &nid in &fired1 {
                    if let Some(oi) = m1c.output_index_of(nid) {
                        if fr < n_frames { seq[1][fr * n1 + oi] += 1.0; }
                        avg[1][oi] += 1.0;
                        if tail { end[1][oi] += 1.0; }
                    }
                }
                // 段 2: M2 出力
                for nid in fired2 {
                    if let Some(oi) = m2c.output_index_of(nid) {
                        if fr < n_frames { seq[2][fr * n2 + oi] += 1.0; }
                        avg[2][oi] += 1.0;
                        if tail { end[2][oi] += 1.0; }
                    }
                }
            }
            out.push(Cond {
                class: c, group: g,
                v: [seq[0].clone(), avg[0].clone(), end[0].clone(),
                    seq[1].clone(), avg[1].clone(), end[1].clone(),
                    seq[2].clone(), avg[2].clone(), end[2].clone()],
            });
        }
    }
    out
}

fn cosine(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len().min(b.len());
    let d: f64 = (0..n).map(|i| a[i] * b[i]).sum();
    let na: f64 = a[..n].iter().map(|x| x * x).sum::<f64>().sqrt();
    let nb: f64 = b[..n].iter().map(|x| x * x).sum::<f64>().sqrt();
    if na == 0.0 || nb == 0.0 { 0.0 } else { d / (na * nb) }
}

/// **群内 1-NN**: 候補を同じ組の条件に限る。返り値 = 近傍の条件 index (同点棄却は None)。
fn neighbors_within(conds: &[Cond], col: usize) -> Vec<Option<usize>> {
    let n = conds.len();
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let mut best = f64::NEG_INFINITY;
        for j in 0..n {
            if j == i || conds[j].group != conds[i].group { continue; }
            let c = cosine(&conds[i].v[col], &conds[j].v[col]);
            if c > best { best = c; }
        }
        let tied: Vec<usize> = (0..n)
            .filter(|&j| j != i && conds[j].group == conds[i].group
                    && cosine(&conds[i].v[col], &conds[j].v[col]) == best)
            .collect();
        if tied.is_empty() { out.push(None); }
        else if tied.iter().all(|&j| conds[j].class == conds[tied[0]].class) { out.push(Some(tied[0])); }
        else { out.push(None); }
    }
    out
}

fn acc_from(nb: &[Option<usize>], lab: &[usize]) -> f64 {
    let ok = nb.iter().enumerate()
        .filter(|(i, x)| x.map_or(false, |j| lab[j] == lab[*i])).count();
    ok as f64 / nb.len() as f64 * 100.0
}

/// 群内置換帰無: ラベルを**組の中で**シャッフルして同じ統計を計算。
fn perm_null(conds: &[Cond], nb: &[Option<usize>], n_tests: usize) -> f64 {
    let lab: Vec<usize> = conds.iter().map(|c| c.class).collect();
    let mut accs: Vec<f64> = Vec::with_capacity(N_PERM);
    for p in 0..N_PERM {
        let mut sl = lab.clone();
        for g in 0..TRIPLES.len() {
            let idx: Vec<usize> = (0..conds.len()).filter(|&i| conds[i].group == g).collect();
            let perm = shuffled(idx.len(), 0xFACE ^ ((p as u64) << 14) ^ ((g as u64) << 4));
            let vals: Vec<usize> = perm.iter().map(|&k| lab[idx[k]]).collect();
            for (t, &i) in idx.iter().enumerate() { sl[i] = vals[t]; }
        }
        accs.push(acc_from(nb, &sl));
    }
    accs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let q = 1.0 - 0.05 / n_tests as f64;
    accs[((N_PERM as f64 * q) as usize).min(N_PERM - 1)]
}

fn main() {
    let n_moras: usize = std::env::var("DRPNN_CORPUS_MORAS").ok()
        .and_then(|v| v.parse().ok()).unwrap_or(4000);
    let cps: Vec<usize> = vec![0, 1000, 4000].into_iter().filter(|&c| c <= n_moras).collect();
    const COLS: [&str; 9] = ["M0.5 フレーム列", "M0.5 時間平均", "M0.5 語末窓",
                             "M1 フレーム列", "M1 時間平均", "M1 語末窓",
                             "M2 フレーム列", "**M2 時間平均**", "**M2 語末窓**"];

    println!("=== 順序課題 — M2 は「順序」を空間パターンに変換できているか ===");
    println!();
    println!("【刺激】無意味語 10 組 × 円順列 3 (ABC/BCA/CAB) × F0 4 変種 = 120 条件。");
    println!("**組内はモーラ集合が同一 — 残る手がかりは順序 (と協調調音の遷移) だけ。**");
    println!("【判定】**群内 1-NN** (候補 = 同じ組の 11 条件・群内チャンス 3/11 ≈ 27.3%)");
    println!("【注記・実測前】連続合成の協調調音は順序を部分的に「内容」へ変換する");
    println!("(遷移集合が順列で違う)。帰無は構成でなく**群内置換帰無 + 段間比較**に置く。");
    println!("**判定の核 (M2 > M1 か) は遷移の存在に影響されない — 遷移は全段に等しく届く。**");
    println!();
    println!("【ゲート・実測前に固定】G102a 陽性対照 (フレーム列・全段) / G102b M0.5 時間平均の床 /");
    println!("**G102c 本丸: 順序盲読み出し (時間平均・語末窓) で M2 > M1 か** /");
    println!("G102d 群内置換帰無 (Bonferroni) / G102e 決定論性 / G102f 内容非出力");
    println!();
    println!("【予測・数値は置かない・②と G100c の予測が外れた履歴を銘記】");
    println!("  ①G102a は通るはず ②M0.5 時間平均は遷移ぶんだけ床を超えるが控えめのはず");
    println!("  ③**G102c は賭け。**超えなければ「M2 はまだ順序を空間化できていない」— それも確定。");

    let moras = load_moras(n_moras);
    let cfg1 = if N_CN_OUTPUT == 164 { ThermoNetworkConfig::for_m1_cn_80() }
               else { ThermoNetworkConfig::for_m1_cn_40() };
    let mut m1 = ThermoNetwork::new(cfg1);
    let mut m2 = ThermoNetwork::new(ThermoNetworkConfig::for_m2());
    let (mut co, mut cn) = (Cochlea::new(), CochlearNucleus::new());
    let mut noise = LfsrNoise::new(0xACE1);

    let mut table: Vec<(usize, Vec<f64>)> = Vec::new();
    let mut nulls: Vec<f64> = vec![0.0; 9];
    let mut next = 0usize;
    for i in 0..=moras.len() {
        if next < cps.len() && cps[next] == i {
            let conds = battery(&m1, &m2, &co, &cn);
            let mut row = Vec::with_capacity(9);
            for col in 0..9 {
                let nb = neighbors_within(&conds, col);
                let lab: Vec<usize> = conds.iter().map(|c| c.class).collect();
                row.push(acc_from(&nb, &lab));
                if next == 0 { nulls[col] = perm_null(&conds, &nb, cps.len()); }
            }
            table.push((i, row));
            next += 1;
        }
        if i == moras.len() { break; }
        let w = synth_utterance(std::slice::from_ref(&moras[i]), F0S[i % N_VAR], &mut noise);
        for chunk in w.chunks(SAMPLES_PER_STEP) {
            if chunk.len() < SAMPLES_PER_STEP { break; }
            let m0 = co.process_step(chunk);
            let cno = cn.process_step(&m0);
            let fired1 = m1.step(&cno);
            let inp2 = m2_input(&m1, &fired1, m2.input_neurons.len());
            let _ = m2.step(&inp2);
        }
    }

    println!();
    println!("--- 群内順序同定率 (群内チャンス 27.3%・列ごとの群内置換帰無を併記) ---");
    print!("  {:>7}", "聞いた");
    for c in COLS.iter() { print!(" | {:>14}", c); }
    println!();
    print!("  {:>7}", "帰無");
    for n in nulls.iter() { print!(" | {:>13.1}%", n); }
    println!();
    for (m, row) in table.iter() {
        print!("  {:>7}", m);
        for (v, n) in row.iter().zip(nulls.iter()) {
            print!(" | {:>12.1}%{}", v, if v > n { "*" } else { " " });
        }
        println!();
    }
    println!("  (* = 群内置換帰無 (Bonferroni {} 地点) 超え)", cps.len());

    let last = &table.last().unwrap().1;
    println!();
    println!("  **G102a 陽性対照 (フレーム列・全段)** -> M0.5 {:.1}% / M1 {:.1}% / M2 {:.1}% -> {}",
             last[0], last[3], last[6],
             if last[0] > nulls[0] && last[3] > nulls[3] && last[6] > nulls[6]
             { "**PASS — 順序情報は信号に存在する**" } else { "**FAIL — 計器の欠陥**" });
    println!("  G102b M0.5 時間平均の床 -> {:.1}% (帰無 {:.1}%)", last[1], nulls[1]);
    let m2_blind = last[7].max(last[8]);
    let m1_blind = last[4].max(last[5]);
    let m05_blind = last[1].max(last[2]);
    println!("  **G102c 本丸: 順序盲読み出し (時間平均/語末窓の良い方)**");
    println!("     M0.5 {:.1}% / M1 {:.1}% / **M2 {:.1}%**", m05_blind, m1_blind, m2_blind);
    println!("     -> {}", if m2_blind > m1_blind && m2_blind > nulls[7].min(nulls[8]) {
        if m2_blind > m05_blind {
            "**M2 が全段の上 — G100b の壁は「次元」でなく「情報の質」だった**"
        } else {
            "**M2 > M1 — 段は順序を足している (M0.5 の遷移床は未超え)**"
        }
    } else { "**M2 は M1 を超えない — M2 はまだ順序を空間化できていない (これも確定)**" });

    let again = battery(&m1, &m2, &co, &cn);
    let nb = neighbors_within(&again, 7);
    let lab: Vec<usize> = again.iter().map(|c| c.class).collect();
    println!("  G102e 決定論性 -> {}",
             if (acc_from(&nb, &lab) - last[7]).abs() < 1e-12 { "PASS" } else { "**FAIL**" });
    println!("  G102f コーパスの内容 -> **一切出力していない (学習のみに使用・数値のみ)**");
}
