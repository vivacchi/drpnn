//! M2 の投入 — 時刻符号を受けて音節スケールで統合できるか (2026-08-28)
//!
//! ## なぜ — 設計のパズルの最後のピース
//!
//! §14.52 で確定した: **M1 は疎な時刻符号への変換器**である (1 語あたり
//! ~1.4 発火/ニューロン・10ms 読み出しで対照の 2.1 倍・120ms では潰れる)。
//! **「音節スケールの統合」は M1 の読み出し窓の仕事ではなかった。**
//!
//! それを担うべく設計されているのが **M2** (`for_m2`・因果窓 320 step = 160ms・
//! M2_A2_DESIGN.md §2.4) である。**M1 の 10ms 時刻符号を受けて、
//! 音節スケールで統合する** — この設計仮説をここで初めて実測する。
//!
//! ## 設定の引き継ぎ (ユーザー承認・§14.53)
//!
//! - **E・F は自動的に引き継がれる** (共有コンストラクタに実装したので)。
//!   **リスク**: M2 の入力駆動は M1 の疎な出力なので、「E 単独の壊滅」(駆動が弱いと
//!   皮質が黙る・§14.46) が **M2 で再現する可能性**がある。発火率と伝達可で監視する。
//! - **D は明示的に引き継ぐ** (`conductance_decay_m2` 既定 100,000)。同じ文献是正。
//! - **因果窓 320 は引き継がない。** M2 の 320 は明示された設計判断であり、
//!   **M2 で検証する仮説そのもの**である。
//!
//! ## 配線 (既存の作法・発明なし)
//!
//! M1 出力スパイク → M2 入力 40 に 1:1 で **+60** (`m0_cn_m1_m2_pipeline.rs` の
//! `INPUT_CURRENT_M2 = 60` = CN の `FIRE_CURRENT` と同じ既存値)。
//!
//! ## ゲート (実測前に固定・以後動かさない)
//!
//! - **G100a** M2 は単語を区別できるか (M2 自身の幾何で置換帰無 + Bonferroni)
//! - **G100b** 帰無「段を足せば情報は減る」: M2 (20ch) ≤ M1 (40ch) か。
//!   *M2 は次元が半分なので不利側。勝てば強い破りになる。*
//! - **G100c (本命・設計仮説)** フレーム長との相互作用。
//!   M1 は 120ms で潰れる (§14.52)。**M2 が設計どおり音節スケールの統合器なら、
//!   M2 の読み出しは長いフレームで M1 より持ちこたえるはず。**
//! - **G100d** M2 の切断監視 (伝達可・発火率・LTP/LTD) — E リスクと D 病理の検出
//! - **G100e** 決定論性 / **G100f** コーパスの内容は一切出力しない
//!
//! ## 予測 (実測前・数値は置かない・**前回②の予測は真逆に外れたことを銘記**)
//!
//! 1. **G100b の帰無は保たれる方が自然** (段の追加 + 20ch)。破れたら大きい。
//! 2. **G100c が本丸**: M2@120ms が M1@120ms より高ければ、設計仮説は支持される。
//! 3. **E リスク**: M2 皮質発火率が ~0 なら壊滅の再現 — その場合も判定はできる
//!    (「M2 は現行の疎な入力では駆動できない」という結果として)。
//!
//! CLI: corpus_m2_learning  (DRPNN_CORPUS_MORAS 既定 4000 / DRPNN_FRAME_STEPS 既定 20)

use spiking_brain::phase2_f::cochlea::{Cochlea, SAMPLES_PER_STEP};
use spiking_brain::phase2_f::cochlear_nucleus::{CochlearNucleus, N_CN_OUTPUT};
use spiking_brain::phase2_f::kana::{moras_from_kana, synth_utterance, Mora};
use spiking_brain::phase2_f::phoneme_synth::LfsrNoise;
use spiking_brain::phase2_f::thermo_network::{ThermoNetwork, ThermoNetworkConfig, SIGNAL_SCALE_DIVISOR};
use std::io::Read;

const CORPUS: &str = "../data/corpus/roleplay_kana.txt";
const F0S: [f64; 4] = [100.0, 130.0, 160.0, 200.0];
const N_VAR: usize = 4;
const N_PERM: usize = 400;
/// M1→M2 の入力電流 (`m0_cn_m1_m2_pipeline.rs` の既存値 = CN の FIRE_CURRENT)
const INPUT_CURRENT_M2: i32 = 60;

const PAIRS: &[(&str, &str)] = &[
    ("こころ", "ところ"), ("からだ", "かなだ"), ("たまご", "たなご"), ("てがみ", "てあみ"),
    ("せかい", "せたい"), ("みどり", "みのり"), ("かたち", "かたな"), ("ひかり", "ひかる"),
    ("さかな", "さかや"), ("くるま", "くるみ"), ("なまえ", "なまり"), ("ちから", "ちかく"),
    ("いのち", "いのり"), ("みかん", "みかた"), ("あたま", "あたり"), ("からす", "からて"),
];

fn words() -> Vec<&'static str> { PAIRS.iter().flat_map(|&(a, b)| [a, b]).collect() }

fn frame_steps() -> usize {
    std::env::var("DRPNN_FRAME_STEPS").ok().and_then(|v| v.parse().ok()).unwrap_or(20).max(1)
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

fn utterance_seed(w: usize, v: usize) -> u16 {
    ((w as u16).wrapping_mul(131).wrapping_add(v as u16).wrapping_mul(4099)) | 1
}

fn load_moras(n: usize) -> (Vec<Mora>, usize) {
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
    let mut kinds = std::collections::BTreeSet::new();
    for c in text.chars() {
        if out.len() >= n { break; }
        if c == '\n' || c == ' ' { continue; }
        let (m, _) = moras_from_kana(&c.to_string());
        if !m.is_empty() { kinds.insert(c); }
        out.extend(m);
    }
    out.truncate(n);
    (out, kinds.len())
}

/// M1 のスパイクを M2 の入力電流へ (1:1・+60・既存の作法)
fn m2_input(m1: &ThermoNetwork, fired: &[usize], n_in: usize) -> Vec<i32> {
    // §14.62: 共有ヘルパへ委譲 (DRPNN_M2_PROJ で legacy / topo を切替)
    spiking_brain::phase2_f::thermo_network::project_m1_m2(m1, fired, n_in)
}

struct Readout { m1: Vec<(usize, Vec<f64>)>, m2: Vec<(usize, Vec<f64>)>,
                 m1_density: f64, m2_density: f64 }

fn battery(m1n: &ThermoNetwork, m2n: &ThermoNetwork, co: &Cochlea, cn: &CochlearNucleus) -> Readout {
    let ws = words();
    let fs = frame_steps();
    let n1 = m1n.output_neurons.len();
    let n2 = m2n.output_neurons.len();
    let n2_in = m2n.input_neurons.len();
    let (mut r1, mut r2) = (Vec::new(), Vec::new());
    let (mut d1, mut d2) = (0f64, 0f64);
    for v in 0..N_VAR {
        let (mut m1c, mut m2c, mut coc, mut cnc) = (m1n.clone(), m2n.clone(), co.clone(), cn.clone());
        for &w in shuffled(ws.len(), 0xA5A5 ^ ((v as u64) << 20)).iter() {
            let mut noise = LfsrNoise::new(utterance_seed(w, v));
            let (m, sk) = moras_from_kana(ws[w]);
            assert_eq!(sk, 0);
            let wave = synth_utterance(&m, F0S[v], &mut noise);
            let n_frames = (wave.len() / SAMPLES_PER_STEP) / fs;
            let mut f1 = vec![0f64; n_frames * n1];
            let mut f2 = vec![0f64; n_frames * n2];
            for (step, chunk) in wave.chunks(SAMPLES_PER_STEP).enumerate() {
                if chunk.len() < SAMPLES_PER_STEP { break; }
                let m0 = coc.process_step(chunk);
                let cno = cnc.process_step(&m0);
                let fired1 = m1c.step(&cno);
                let fr = step / fs;
                for &nid in &fired1 {
                    if let Some(oi) = m1c.output_index_of(nid) {
                        d1 += 1.0;
                        if fr < n_frames { f1[fr * n1 + oi] += 1.0; }
                    }
                }
                let inp2 = m2_input(&m1c, &fired1, n2_in);
                for nid in m2c.step(&inp2) {
                    if let Some(oi) = m2c.output_index_of(nid) {
                        d2 += 1.0;
                        if fr < n_frames { f2[fr * n2 + oi] += 1.0; }
                    }
                }
            }
            r1.push((w, f1));
            r2.push((w, f2));
        }
    }
    let n = r1.len() as f64;
    Readout { m1: r1, m2: r2, m1_density: d1 / n, m2_density: d2 / n }
}

fn cosine(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len().min(b.len());
    let d: f64 = (0..n).map(|i| a[i] * b[i]).sum();
    let na: f64 = a[..n].iter().map(|x| x * x).sum::<f64>().sqrt();
    let nb: f64 = b[..n].iter().map(|x| x * x).sum::<f64>().sqrt();
    if na == 0.0 || nb == 0.0 { 0.0 } else { d / (na * nb) }
}

fn neighbors(v: &[(usize, Vec<f64>)]) -> Vec<Option<usize>> {
    let n = v.len();
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let mut best = f64::NEG_INFINITY;
        for j in 0..n { if j != i { let c = cosine(&v[i].1, &v[j].1); if c > best { best = c; } } }
        let tied: Vec<usize> = (0..n).filter(|&j| j != i && cosine(&v[i].1, &v[j].1) == best).collect();
        if tied.is_empty() { out.push(None); }
        else if tied.iter().all(|&j| v[j].0 == v[tied[0]].0) { out.push(Some(tied[0])); }
        else { out.push(None); }
    }
    out
}

fn acc_from(nb: &[Option<usize>], lab: &[usize]) -> f64 {
    let ok = nb.iter().enumerate()
        .filter(|(i, x)| x.map_or(false, |j| lab[j] == lab[*i])).count();
    ok as f64 / nb.len() as f64 * 100.0
}

fn acc(v: &[(usize, Vec<f64>)]) -> f64 {
    let nb = neighbors(v);
    let lab: Vec<usize> = v.iter().map(|x| x.0).collect();
    acc_from(&nb, &lab)
}

fn perm_null_b(v: &[(usize, Vec<f64>)], n_points: usize) -> f64 {
    let nb = neighbors(v);
    let lab: Vec<usize> = v.iter().map(|x| x.0).collect();
    let mut accs: Vec<f64> = Vec::with_capacity(N_PERM);
    for p in 0..N_PERM {
        let perm = shuffled(lab.len(), 0xBEEF ^ ((p as u64) << 16));
        let sl: Vec<usize> = perm.iter().map(|&i| lab[i]).collect();
        accs.push(acc_from(&nb, &sl));
    }
    accs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let q = 1.0 - 0.05 / n_points as f64;
    accs[((N_PERM as f64 * q) as usize).min(N_PERM - 1)]
}

fn live_count(net: &ThermoNetwork) -> usize {
    net.synapses.iter().filter(|s| s.alive && s.conductance >= SIGNAL_SCALE_DIVISOR).count()
}

fn main() {
    let n_moras: usize = std::env::var("DRPNN_CORPUS_MORAS").ok()
        .and_then(|v| v.parse().ok()).unwrap_or(4000);
    let cps: Vec<usize> = vec![0, 250, 1000, 2000, 4000]
        .into_iter().filter(|&c| c <= n_moras).collect();

    println!("=== M2 の投入 — 時刻符号を受けて音節スケールで統合できるか ===");
    println!();
    println!("【なぜ】§14.52 で M1 は**疎な時刻符号への変換器**と確定 (120ms で潰れる)。");
    println!("音節スケールの統合を担うべく設計されているのが M2 (因果窓 320 = 160ms)。");
    println!("**その設計仮説をここで初めて実測する。**");
    println!();
    println!("【引き継ぎ】E・F は自動 / **D は明示的に引き継ぐ** (decay 100,000) /");
    println!("**因果窓 320 は引き継がない — M2 で検証する仮説そのもの。**");
    println!("【E リスク】M2 の入力は M1 の疎な出力 → 「E 単独の壊滅」(§14.46) が再現しうる。監視する。");
    println!();
    println!("【ゲート・実測前に固定】");
    println!("  G100a M2 は単語を区別できるか (M2 自身の幾何で置換帰無 + Bonferroni)");
    println!("  G100b 帰無「段を足せば情報は減る」: M2(20ch) <= M1(40ch) か (M2 は次元不利側)");
    println!("  **G100c (本命・設計仮説): M2 の読み出しは長いフレームで M1 より持ちこたえるか**");
    println!("  G100d 切断監視 / G100e 決定論性 / G100f 内容非出力");
    println!();
    println!("【予測・実測前・数値は置かない (前回②の予測は真逆に外れたことを銘記)】");
    println!("  ① G100b の帰無は保たれる方が自然 (段の追加 + 20ch)。破れたら大きい");
    println!("  ② **G100c が本丸**: M2@長フレーム > M1@長フレーム なら設計仮説を支持");
    println!("  ③ E リスクが出た場合も「M2 は現行の疎な入力では駆動できない」という判定になる");
    println!();
    println!("  フレーム長 = {} step ({} ms)", frame_steps(), frame_steps() as f64 * 0.5);

    let (moras, kinds) = load_moras(n_moras);
    println!("  コーパス {} モーラ / **{} 種類**。**内容は出力しない。**", moras.len(), kinds);

    let mut cfg1 = if N_CN_OUTPUT == 164 { ThermoNetworkConfig::for_m1_cn_80() }
               else { ThermoNetworkConfig::for_m1_cn_40() };
    // シード再現用 (§14.48.7 と同じ規律: 系は決定論なので意味のある再試行 = 別の初期網)。
    // M2 は +7 (固定オフセット・両網が同じ列にならないため)。
    if let Some(seed) = std::env::var("DRPNN_M1_SEED").ok().and_then(|v| v.parse::<u64>().ok()) {
        cfg1.seed = seed;
    }
    assert_eq!(cfg1.n_input, N_CN_OUTPUT);
    let mut m1 = ThermoNetwork::new(cfg1);
    let mut cfg2 = ThermoNetworkConfig::for_m2();
    if let Some(seed) = std::env::var("DRPNN_M1_SEED").ok().and_then(|v| v.parse::<u64>().ok()) {
        cfg2.seed = seed + 7;
    }
    let mut m2 = ThermoNetwork::new(cfg2);
    assert_eq!(m2.input_neurons.len(), m1.output_neurons.len(), "M1 出力と M2 入力が 1:1 でない");
    let (mut co, mut cn) = (Cochlea::new(), CochlearNucleus::new());
    let mut noise = LfsrNoise::new(0xACE1);

    let mut rows: Vec<(usize, f64, f64, f64, f64, usize, usize)> = Vec::new();
    let (mut null_m1, mut null_m2) = (0f64, 0f64);
    let mut next = 0usize;
    for i in 0..=moras.len() {
        if next < cps.len() && cps[next] == i {
            let r = battery(&m1, &m2, &co, &cn);
            if next == 0 {
                null_m1 = perm_null_b(&r.m1, cps.len());
                null_m2 = perm_null_b(&r.m2, cps.len());
            }
            rows.push((i, acc(&r.m1), acc(&r.m2), r.m1_density, r.m2_density,
                       live_count(&m1), live_count(&m2)));
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
    println!("--- 単語の同定率 (128 条件・フレーム列・窓なし) ---");
    println!("  {:>7} | {:>9} {:>9} | {:>10} {:>10} | {:>9} {:>9}",
             "聞いた", "M1(40)", "**M2(20)**", "M1発火/語", "M2発火/語", "M1伝達可", "M2伝達可");
    for (m, a1, a2, d1, d2, l1, l2) in rows.iter() {
        println!("  {:>7} | {:>8.1}% {:>8.1}% | {:>10.1} {:>10.1} | {:>9} {:>9}",
                 m, a1, a2, d1, d2, l1, l2);
    }
    println!("  **置換帰無 (Bonferroni {}地点): M1 {:.1}% / M2 {:.1}%**", cps.len(), null_m1, null_m2);

    let m2s: Vec<f64> = rows.iter().map(|r| r.2).collect();
    let m1s: Vec<f64> = rows.iter().map(|r| r.1).collect();
    let over2 = m2s.iter().filter(|&&x| x > null_m2).count();
    let (b1, b2) = (m1s.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
                    m2s.iter().cloned().fold(f64::NEG_INFINITY, f64::max));
    println!();
    println!("  **G100a M2 は単語を区別できるか** -> {}/{} 地点が帰無超え -> {}",
             over2, m2s.len(), if over2 > 0 { "**超えた**" } else { "**超えない**" });
    println!("  **G100b M2(20ch) vs M1(40ch)** -> M2 最良 {:.1}% / M1 最良 {:.1}% -> {}",
             b2, b1, if b2 > b1 { "**M2 が上 — 帰無が破れた**" } else { "**M1 が上 — 帰無は保たれた**" });
    println!("  (G100c はフレーム長を変えた別走行と並べて判定する)");

    let last = rows.last().unwrap();
    println!("  **G100d 切断監視** -> M2 発火/語 {:.1} / M2 伝達可 {} -> {}",
             last.4, last.6,
             if last.4 < 1.0 { "**M2 はほぼ沈黙 — E リスクの再現**" }
             else if last.6 < 100 { "**M2 の伝達可が枯渇 — D 病理の疑い**" }
             else { "駆動されている" });

    let r2 = battery(&m1, &m2, &co, &cn);
    println!("  G100e 決定論性 -> {}",
             if (acc(&r2.m2) - last.2).abs() < 1e-12 { "PASS" } else { "**FAIL**" });
    println!("  G100f コーパスの内容 -> **一切出力していない (数値のみ)**");
}
