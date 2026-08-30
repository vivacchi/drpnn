//! 順序課題・全開口版 — 正しい開口 (全皮質) で G102c を再判定する (2026-08-30)
//!
//! ## なぜ
//!
//! §14.60 で確定: **変種不変な順序符号は皮質集団に存在し、出力層 (40/436・20/400) という
//! 開口がそれを捨てていた。** 生体の文献規則は「**興奮性錐体細胞は事実上すべて投射する。
//! 局所限定は介在ニューロンだけ**」— 現行の開口は 1 桁狭い。
//!
//! §14.54 の順序課題は出力層の開口で測っていた。**正しい開口 (全皮質) で再判定する。**
//! 原版 `mora_order` は比較可能性のため不変のまま残す。**モデルには一切触れない (計器のみ)。**
//!
//! ## 列 (5 段 × 3 読み出し = 15)
//!
//! 段 = [M0.5(84) / M1出力(40) / **M1全皮質(436)** / M2出力(20) / **M2全皮質(400)**]
//! 読み出し = [フレーム列 (陽性対照) / 時間平均 (純粋な順序盲) / 語末窓 (末尾内容と混合)]
//!
//! ## ゲート (実測前に固定・以後動かさない)
//!
//! - **G106a 陽性対照**: 全段のフレーム列が群内置換帰無超え。
//! - **G106b 開口の効果**: 全皮質列の順序盲成績が出力層列を上回るか (M1・M2 それぞれ)。
//! - **G106c 本丸 (開口を正した G102c)**: 順序盲読み出しで **M2全皮質 > M1全皮質** か。
//!   これが「段は順序情報を足すか」の正しい開口での判定。
//! - **G106d** 決定論性 / **G106e** 内容非出力。
//!
//! ## 予測 (実測前・数値は置かない)
//!
//! 1. G106b は成立するはず (invariance §14.60 の帰結)。
//! 2. **G106c は賭け。** invariance では M2全 (0.357) > M1全 (0.265) だったが、
//!    順序同定は別の統計であり、予測しない。
//!
//! CLI: mora_order_full  (DRPNN_CORPUS_MORAS 既定 4000)

use spiking_brain::phase2_f::cochlea::{Cochlea, SAMPLES_PER_STEP};
use spiking_brain::phase2_f::cochlear_nucleus::{CochlearNucleus, N_CN_OUTPUT};
use spiking_brain::phase2_f::kana::{moras_from_kana, synth_utterance, Mora, MORA_MS};
use spiking_brain::phase2_f::phoneme_synth::LfsrNoise;
use spiking_brain::phase2_f::thermo_network::{ThermoNetwork, ThermoNetworkConfig};
use std::io::Read;

const CORPUS: &str = "../data/corpus/roleplay_kana.txt";
const F0S: [f64; 4] = [100.0, 130.0, 160.0, 200.0];
const N_VAR: usize = 4;
const N_PERM: usize = 400;
const INPUT_CURRENT_M2: i32 = 60;
const FRAME: usize = 20;
const MORA_STEPS: usize = (MORA_MS as usize) * 16 / SAMPLES_PER_STEP;
const N_COLS: usize = 15;

const TRIPLES: &[[&str; 3]] = &[
    ["か", "め", "そ"], ["に", "ろ", "た"], ["す", "べ", "や"], ["こ", "ぬ", "ぜ"],
    ["ま", "り", "ど"], ["は", "ぐ", "ね"], ["ぼ", "ち", "わ"], ["て", "む", "ざ"],
    ["き", "の", "ぶ"], ["ら", "ぱ", "せ"],
];

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
    // §14.62: 共有ヘルパへ委譲 (DRPNN_M2_PROJ で legacy / topo を切替)
    spiking_brain::phase2_f::thermo_network::project_m1_m2(m1, fired, n_in)
}

struct Cond { class: usize, group: usize, v: Vec<Vec<f64>> }

fn battery(m1n: &ThermoNetwork, m2n: &ThermoNetwork, co: &Cochlea, cn: &CochlearNucleus) -> Vec<Cond> {
    let n_cls = TRIPLES.len() * 3;
    let n1 = m1n.output_neurons.len();
    let n2 = m2n.output_neurons.len();
    let n1_in = m1n.input_neurons.len();
    let n2_in = m2n.input_neurons.len();
    let f1 = m1n.n_neurons() - n1_in;   // M1 全皮質
    let f2 = m2n.n_neurons() - n2_in;   // M2 全皮質
    let dims = [N_CN_OUTPUT, n1, f1, n2, f2];
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
            let last_start = n_steps.saturating_sub(MORA_STEPS);
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
                let mut hit = |s: usize, i: usize| {
                    if fr < n_frames { seq[s][fr * dims[s] + i] += 1.0; }
                    avg[s][i] += 1.0;
                    if tail { end[s][i] += 1.0; }
                };
                for (i, &x) in cno.iter().enumerate() { if x != 0 { hit(0, i); } }
                for &nid in &fired1 {
                    if let Some(oi) = m1c.output_index_of(nid) { hit(1, oi); }
                    if nid >= n1_in { hit(2, nid - n1_in); }
                }
                for nid in fired2 {
                    if let Some(oi) = m2c.output_index_of(nid) { hit(3, oi); }
                    if nid >= n2_in { hit(4, nid - n2_in); }
                }
            }
            let mut v: Vec<Vec<f64>> = Vec::with_capacity(N_COLS);
            for s in 0..5 {
                v.push(std::mem::take(&mut seq[s]));
                v.push(std::mem::take(&mut avg[s]));
                v.push(std::mem::take(&mut end[s]));
            }
            out.push(Cond { class: c, group: g, v });
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
    let cps: Vec<usize> = vec![0, 4000].into_iter().filter(|&c| c <= n_moras).collect();
    const STAGE: [&str; 5] = ["M0.5(84)", "M1出力(40)", "**M1全皮質**", "M2出力(20)", "**M2全皮質**"];
    const KIND: [&str; 3] = ["フレーム列", "時間平均", "語末窓"];

    println!("=== 順序課題・全開口版 — 正しい開口で G102c を再判定する ===");
    println!();
    println!("【なぜ】§14.60: 変種不変な順序符号は皮質集団に存在し、出力層という開口が捨てていた。");
    println!("生体規則「興奮性錐体細胞は事実上すべて投射・局所限定は介在ニューロンだけ」に照らし、");
    println!("**正しい開口 (全皮質) で順序課題を再判定する。**原版 mora_order は不変のまま。");
    println!();
    println!("【ゲート・実測前に固定】G106a 陽性対照 / **G106b 全皮質列 > 出力層列 (開口の効果)** /");
    println!("**G106c 本丸: 順序盲で M2全皮質 > M1全皮質 か (開口を正した G102c)** /");
    println!("G106d 決定論性 / G106e 内容非出力");
    println!();
    println!("【予測・数値は置かない】①G106b は成立するはず (§14.60 の帰結)");
    println!("②**G106c は賭け** (invariance では M2全 0.357 > M1全 0.265 だが順序同定は別の統計)");

    let moras = load_moras(n_moras);
    let cfg1 = if N_CN_OUTPUT == 164 { ThermoNetworkConfig::for_m1_cn_80() }
               else { ThermoNetworkConfig::for_m1_cn_40() };
    let mut m1 = ThermoNetwork::new(cfg1);
    let mut m2 = ThermoNetwork::new(ThermoNetworkConfig::for_m2());
    let (mut co, mut cn) = (Cochlea::new(), CochlearNucleus::new());
    let mut noise = LfsrNoise::new(0xACE1);

    let mut table: Vec<(usize, Vec<f64>)> = Vec::new();
    let mut nulls: Vec<f64> = vec![0.0; N_COLS];
    let mut next = 0usize;
    for i in 0..=moras.len() {
        if next < cps.len() && cps[next] == i {
            let conds = battery(&m1, &m2, &co, &cn);
            let mut row = Vec::with_capacity(N_COLS);
            for col in 0..N_COLS {
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
    println!("--- 群内順序同定率 (群内チャンス 27.3%・* = 群内置換帰無 Bonferroni 超え) ---");
    for (m, row) in table.iter() {
        println!("  【{} モーラ聞いた時点】", m);
        for s in 0..5 {
            print!("    {:<14}", STAGE[s]);
            for k in 0..3 {
                let c = s * 3 + k;
                print!("  {} {:>5.1}%{} (帰無{:.0})", KIND[k], row[c],
                       if row[c] > nulls[c] { "*" } else { " " }, nulls[c]);
            }
            println!();
        }
    }

    let last = &table.last().unwrap().1;
    let blind = |s: usize| last[s * 3 + 1].max(last[s * 3 + 2]);
    println!();
    println!("  G106a 陽性対照 (フレーム列・全段) -> {}",
             if (0..5).all(|s| last[s * 3] > nulls[s * 3]) { "**PASS**" } else { "**FAIL**" });
    println!("  **G106b 開口の効果 (順序盲の最良)**:");
    println!("     M1: 出力層 {:.1}% -> **全皮質 {:.1}%** ({})", blind(1), blind(2),
             if blind(2) > blind(1) { "**開口で改善**" } else { "改善せず" });
    println!("     M2: 出力層 {:.1}% -> **全皮質 {:.1}%** ({})", blind(3), blind(4),
             if blind(4) > blind(3) { "**開口で改善**" } else { "改善せず" });
    println!("  **G106c 本丸 (開口を正した G102c)**: M0.5 {:.1}% / M1全 {:.1}% / M2全 {:.1}%",
             blind(0), blind(2), blind(4));
    println!("     -> {}", if blind(4) > blind(2) {
        "**M2 > M1 — 段は順序情報を足している (正しい開口で初めて見えた)**"
    } else {
        "**M2 は M1 を超えない — 開口を正しても段は順序を足せていない**"
    });
    println!("  G106e コーパスの内容 -> **一切出力していない (学習のみに使用・数値のみ)**");
}
