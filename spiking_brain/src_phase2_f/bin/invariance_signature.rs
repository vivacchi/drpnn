//! 履歴署名の変種間一致度 — 符号か、カオス的分岐か (2026-08-29)
//!
//! ## なぜ — §14.56 の判定を受けて
//!
//! 状態分岐実験で確定した: **履歴は状態にあり、スパイクにも出ている** (M2 末尾スパイク
//! 分岐 0.29)。それでも順序同定 (§14.54) が低い。残る仮説は**不変性**:
//! **履歴署名が F0 変種を跨いで同じ形をしていない**のではないか。
//!
//! ## 何を測るか
//!
//! 各組 g の語対 (ABC, BAC) について、F0 変種 v ごとに
//! **署名 Δ_{g,v} = 末尾スパイク数(ABC) − 末尾スパイク数(BAC)** (共通応答は差で消える)。
//!
//! - **群内・変種間相関**: 同じ組の Δ を変種間で比べる (6 対/組)。
//!   **同じ「順序の違い」が変種を跨いで同じ形の署名を残すか。**
//! - **対照 = 群間相関**: 違う組の Δ どうし (本来無関係)。この分布の 95% 点を帰無帯とする。
//!
//! ## 判定 (実測前に固定)
//!
//! - **G104a 対照**: 群間相関の分布 (帰無帯)。
//! - **G104b 本丸**: 群内・変種間の平均相関が帰無帯を超えるか (段ごと・学習後)。
//!   **超える → 「弱いが符号」= 量・冗長化の道 / 超えない → 「カオス的分岐」=
//!   軌道を安定させる機構 (生体なら抑制による正規化) の是正が要る。**
//! - **G104c 学習効果**: 0 vs 4,000 モーラで一致度が上がるか。
//! - **G104d 決定論性 / G104e 内容非出力。**
//!
//! ## 予測 (実測前・数値は置かない)
//!
//! §14.54 で M2 語末窓は 58.3% (帰無 35 超え) だったので、**相関は正だが弱いはず**
//! (M1 も正のはず)。ゼロ (純カオス) なら §14.54 の語末窓の帰無超えと矛盾するので、
//! **完全なカオスは予測しない。問いは「どれだけ弱いか」と「学習で伸びるか」。**
//!
//! CLI: invariance_signature  (DRPNN_CORPUS_MORAS 既定 4000)

use spiking_brain::phase2_f::cochlea::{Cochlea, SAMPLES_PER_STEP};
use spiking_brain::phase2_f::cochlear_nucleus::{CochlearNucleus, N_CN_OUTPUT};
use spiking_brain::phase2_f::kana::{moras_from_kana, synth_utterance, Mora, MORA_MS};
use spiking_brain::phase2_f::phoneme_synth::LfsrNoise;
use spiking_brain::phase2_f::thermo_network::{ThermoNetwork, ThermoNetworkConfig};
use std::io::Read;

const CORPUS: &str = "../data/corpus/roleplay_kana.txt";
const F0S: [f64; 4] = [100.0, 130.0, 160.0, 200.0];
const N_VAR: usize = 4;
const INPUT_CURRENT_M2: i32 = 60;
const MORA_STEPS: usize = (MORA_MS as usize) * 16 / SAMPLES_PER_STEP;

const TRIPLES: &[[&str; 3]] = &[
    ["か", "め", "そ"], ["に", "ろ", "た"], ["す", "べ", "や"], ["こ", "ぬ", "ぜ"],
    ["ま", "り", "ど"], ["は", "ぐ", "ね"], ["ぼ", "ち", "わ"], ["て", "む", "ざ"],
    ["き", "の", "ぶ"], ["ら", "ぱ", "せ"],
];

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

/// 語を流して末尾モーラ中の出力スパイク数 [段 3 本] を返す。
/// 5 列: [CN84 / M1出力40 / **M1全皮質** / M2出力20 / **M2全皮質**]
///
/// **全皮質列 (2026-08-29・§14.60 読み出しの拡幅)**: §14.58 で「皮質の PR は 78% と高く、
/// 低実効次元は出力層 20ch の側にある」と出た。**開口の上限**を測る —
/// 全集団の読み出しがゲートを通れば、残る差は「表現の欠如」でなく「開口の狭さ」と確定する。
/// **モデルには一切触れない (計器のみ)。**
fn tail_spikes(word: &str, seed: u16, f0: f64,
               co0: &Cochlea, cn0: &CochlearNucleus,
               m10: &ThermoNetwork, m20: &ThermoNetwork) -> [Vec<f64>; 5] {
    let (mut co, mut cn, mut m1, mut m2) = (co0.clone(), cn0.clone(), m10.clone(), m20.clone());
    let mut noise = LfsrNoise::new(seed);
    let (m, sk) = moras_from_kana(word);
    assert_eq!(sk, 0, "未対応: {}", word);
    let wave = synth_utterance(&m, f0, &mut noise);
    let tail = (2 * MORA_STEPS)..(3 * MORA_STEPS);
    let n1_in = m1.input_neurons.len();
    let n2_in = m2.input_neurons.len();
    let mut out = [vec![0f64; N_CN_OUTPUT],
                   vec![0f64; m1.output_neurons.len()],
                   vec![0f64; m1.n_neurons() - n1_in],
                   vec![0f64; m2.output_neurons.len()],
                   vec![0f64; m2.n_neurons() - n2_in]];
    for (step, chunk) in wave.chunks(SAMPLES_PER_STEP).enumerate() {
        if chunk.len() < SAMPLES_PER_STEP { break; }
        let m0 = co.process_step(chunk);
        let cno = cn.process_step(&m0);
        let fired1 = m1.step(&cno);
        let inp2 = m2_input(&m1, &fired1, n2_in);
        let fired2 = m2.step(&inp2);
        if tail.contains(&step) {
            for (i, &x) in cno.iter().enumerate() { if x != 0 { out[0][i] += 1.0; } }
            for &nid in &fired1 {
                if let Some(oi) = m1.output_index_of(nid) { out[1][oi] += 1.0; }
                if nid >= n1_in { out[2][nid - n1_in] += 1.0; }
            }
            for nid in fired2 {
                if let Some(oi) = m2.output_index_of(nid) { out[3][oi] += 1.0; }
                if nid >= n2_in { out[4][nid - n2_in] += 1.0; }
            }
        }
    }
    out
}

fn pearson(a: &[f64], b: &[f64]) -> Option<f64> {
    let n = a.len().min(b.len()) as f64;
    let (ma, mb) = (a.iter().sum::<f64>() / n, b.iter().sum::<f64>() / n);
    let (mut num, mut da, mut db) = (0f64, 0f64, 0f64);
    for i in 0..a.len().min(b.len()) {
        let (x, y) = (a[i] - ma, b[i] - mb);
        num += x * y; da += x * x; db += y * y;
    }
    if da == 0.0 || db == 0.0 { None } else { Some(num / (da * db).sqrt()) }
}

/// [組][変種][段] の署名 Δ
fn signatures(co: &Cochlea, cn: &CochlearNucleus, m1: &ThermoNetwork, m2: &ThermoNetwork)
    -> Vec<Vec<[Vec<f64>; 5]>> {
    let mut sig = Vec::new();
    for (g, t) in TRIPLES.iter().enumerate() {
        let abc = format!("{}{}{}", t[0], t[1], t[2]);
        let bac = format!("{}{}{}", t[1], t[0], t[2]);
        let mut per_var = Vec::new();
        for v in 0..N_VAR {
            let seed = (0x2468u16.wrapping_add(g as u16 * 89).wrapping_add(v as u16 * 7)) | 1;
            let a = tail_spikes(&abc, seed, F0S[v], co, cn, m1, m2);
            let b = tail_spikes(&bac, seed, F0S[v], co, cn, m1, m2);
            let mut d: [Vec<f64>; 5] = [vec![], vec![], vec![], vec![], vec![]];
            for s in 0..N_STAGES {
                d[s] = a[s].iter().zip(b[s].iter()).map(|(x, y)| x - y).collect();
            }
            per_var.push(d);
        }
        sig.push(per_var);
    }
    sig
}

struct ArmStats { within: [f64; N_STAGES], within_n: [usize; N_STAGES], null95: [f64; N_STAGES], null95_x: [f64; N_STAGES], over: [usize; N_STAGES], over_x: [usize; N_STAGES], pairs: [usize; N_STAGES] }

fn analyze(sig: &Vec<Vec<[Vec<f64>; 5]>>) -> ArmStats {
    let ng = sig.len();
    let mut within = [0f64; N_STAGES];
    let mut within_n = [0usize; N_STAGES];
    let mut over = [0usize; N_STAGES];
    let mut over_x = [0usize; N_STAGES];
    let mut pairs = [0usize; N_STAGES];
    let mut null: Vec<Vec<f64>> = vec![Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new()];
    let mut null_x: Vec<Vec<f64>> = vec![Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new()];
    // 群間 (対照)。**2 種類**:
    //   null   = 宣言どおり (全変種組・同一変種の対を含む)
    //   null_x = **条件を揃えた帰無** (v1 != v2 のみ)。本統計は変種間相関なので、
    //            同一変種の群間対を混ぜると「同じ F0 への共通応答」が帰無を膨らませる。
    //            (2026-08-29 の走行で発見した計器の非対称。宣言側も残して両方報告する。)
    for g1 in 0..ng { for g2 in 0..ng { if g1 == g2 { continue; }
        for v1 in 0..N_VAR { for v2 in 0..N_VAR {
            for s in 0..N_STAGES {
                if let Some(r) = pearson(&sig[g1][v1][s], &sig[g2][v2][s]) {
                    null[s].push(r);
                    if v1 != v2 { null_x[s].push(r); }
                }
            }
        }}
    }}
    let mut null95 = [0f64; N_STAGES];
    let mut null95_x = [0f64; N_STAGES];
    for s in 0..N_STAGES {
        null[s].sort_by(|a, b| a.partial_cmp(b).unwrap());
        null95[s] = if null[s].is_empty() { 1.0 }
                    else { null[s][(null[s].len() as f64 * 0.95) as usize] };
        null_x[s].sort_by(|a, b| a.partial_cmp(b).unwrap());
        null95_x[s] = if null_x[s].is_empty() { 1.0 }
                      else { null_x[s][(null_x[s].len() as f64 * 0.95) as usize] };
    }
    // 群内・変種間
    for g in 0..ng {
        for v1 in 0..N_VAR { for v2 in (v1 + 1)..N_VAR {
            for s in 0..N_STAGES {
                if let Some(r) = pearson(&sig[g][v1][s], &sig[g][v2][s]) {
                    within[s] += r;
                    within_n[s] += 1;
                    pairs[s] += 1;
                    if r > null95[s] { over[s] += 1; }
                    if r > null95_x[s] { over_x[s] += 1; }
                }
            }
        }}
    }
    for s in 0..N_STAGES { if within_n[s] > 0 { within[s] /= within_n[s] as f64; } }
    ArmStats { within, within_n, null95, null95_x, over, over_x, pairs }
}

fn print_arm(label: &str, st: &ArmStats) {
    const STAGES: [&str; N_STAGES] = ["M0.5(84)", "M1出力(40)", "**M1全皮質**", "M2出力(20)", "**M2全皮質**"];
    println!("--- {} ---", label);
    println!("  {:<6} | {:>12} | {:>14} | {:>18} | {:>14}", "段", "群内平均相関",
             "帰無95%(宣言)", "**帰無95%(v1!=v2)**", "揃えた帰無超え");
    for s in 0..N_STAGES {
        println!("  {:<6} | {:>12.3} | {:>14.3} | {:>18.3} | {:>10}/{}", STAGES[s], st.within[s],
                 st.null95[s], st.null95_x[s], st.over_x[s], st.pairs[s]);
    }
}

/// **共通モード除去** (2026-08-29 追加): 変種・段ごとに全組の Δ の平均 (= 内容非特異な
/// 共通軌道ずれ) を引く。**群間相関が 0.6 に達する = 分岐が共通モードに支配されている**
/// ことが判明したため。除去後に群内相関が残れば「共通モードの下に順序特異な符号がある」、
/// 消えれば「順序特異な成分に不変性が無い」— 道が分かれる。
const N_STAGES: usize = 5;

fn remove_common_mode(sig: &Vec<Vec<[Vec<f64>; 5]>>) -> Vec<Vec<[Vec<f64>; 5]>> {
    let ng = sig.len();
    let mut out = sig.clone();
    for v in 0..N_VAR {
        for s in 0..N_STAGES {
            let d = sig[0][v][s].len();
            let mut mean = vec![0f64; d];
            for g in 0..ng { for i in 0..d { mean[i] += sig[g][v][s][i]; } }
            for m in mean.iter_mut() { *m /= ng as f64; }
            for g in 0..ng { for i in 0..d { out[g][v][s][i] -= mean[i]; } }
        }
    }
    out
}

fn main() {
    let n_moras: usize = std::env::var("DRPNN_CORPUS_MORAS").ok()
        .and_then(|v| v.parse().ok()).unwrap_or(4000);

    println!("=== 履歴署名の変種間一致度 — 符号か、カオス的分岐か ===");
    println!();
    println!("【何を測るか】署名 Δ = 末尾スパイク数(ABC) − (BAC)。**共通応答は差で消える。**");
    println!("群内・変種間の Pearson 相関 (同じ順序差が変種を跨いで同じ形か) vs 群間 (対照・無関係)。");
    println!();
    println!("【判定・実測前に固定】G104b 本丸: 群内相関が群間の 95% 点を超えるか。");
    println!("  **超える -> 「弱いが符号」= 量・冗長化の道 /**");
    println!("  **超えない -> 「カオス的分岐」= 軌道を安定させる機構 (抑制正規化など) の是正が要る**");
    println!("【予測】§14.54 の語末窓 58.3% (帰無超え) から、**相関は正だが弱いはず。**");
    println!("純カオスなら §14.54 と矛盾するので予測しない。問いは「どれだけ弱いか」「学習で伸びるか」。");
    println!();

    let cfg1 = if N_CN_OUTPUT == 164 { ThermoNetworkConfig::for_m1_cn_80() }
               else { ThermoNetworkConfig::for_m1_cn_40() };
    let mut m1 = ThermoNetwork::new(cfg1);
    let mut m2 = ThermoNetwork::new(ThermoNetworkConfig::for_m2());
    let (mut co, mut cn) = (Cochlea::new(), CochlearNucleus::new());

    let sig0 = signatures(&co, &cn, &m1, &m2);
    print_arm("学習前 (0 モーラ)", &analyze(&sig0));

    let moras = load_moras(n_moras);
    let mut noise = LfsrNoise::new(0xACE1);
    for (i, m) in moras.iter().enumerate() {
        let w = synth_utterance(std::slice::from_ref(m), F0S[i % 4], &mut noise);
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
    let sig1 = signatures(&co, &cn, &m1, &m2);
    let st1 = analyze(&sig1);
    print_arm(&format!("学習後 ({} モーラ)", n_moras), &st1);

    // ---- 共通モード除去後の再解析 ----
    println!();
    let st1c = analyze(&remove_common_mode(&sig1));
    print_arm("学習後・**共通モード除去後**", &st1c);

    println!();
    println!("=== 判定 ===");
    const STAGES: [&str; N_STAGES] = ["M0.5(84)", "M1出力(40)", "**M1全皮質**", "M2出力(20)", "**M2全皮質**"];
    println!("  【共通モード除去後 (本判定)】");
    for s in 0..N_STAGES {
        let verdict = if st1c.within[s] > st1c.null95_x[s] {
            "**共通モードの下に順序特異な符号がある — 冗長化の道**"
        } else if st1c.within[s] > 0.05 {
            "**正だが帰無帯の中 — 特異成分は弱い**"
        } else {
            "**順序特異な成分に不変性が無い — 抑制正規化 (軌道の安定化) の道**"
        };
        println!("  {} : 群内 {:.3} vs 揃えた帰無 {:.3} ({} 対) -> {}",
                 STAGES[s], st1c.within[s], st1c.null95_x[s], st1c.pairs[s], verdict);
    }
    println!();
    println!("  【除去前 (参考)】");
    for s in 0..N_STAGES {
        let verdict = if st1.within[s] > st1.null95_x[s] {
            "**符号 — 弱くても変種を跨いで再現している (量・冗長化の道)**"
        } else if st1.within[s] > 0.0 {
            "**正だが帰無帯の中 — 符号と呼ぶには弱い**"
        } else {
            "**カオス的分岐 — 軌道の安定化 (抑制正規化など) が要る**"
        };
        println!("  {} : 群内 {:.3} vs 宣言帰無 {:.3} / **揃えた帰無 {:.3}** -> {}",
                 STAGES[s], st1.within[s], st1.null95[s], st1.null95_x[s], verdict);
    }
    println!();
    println!("  G104e コーパスの内容 -> **一切出力していない (学習のみに使用・数値のみ)**");
}
