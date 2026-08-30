//! 状態分岐実験 — 履歴はどの状態変数に、どれだけ保持されるのか (2026-08-29)
//!
//! ## なぜ — G100b の壁の正体を二分する
//!
//! Day 2 で確定した: 時間スケールの統合は STDP 窓でなく**再帰層の回路動態**が担うべき。
//! だが「順序の空間化」は全段で未達 (§14.54)。原因は 2 つに分かれ、対処が正反対:
//!
//! - **(i) 状態に履歴が無い** — 動態が速すぎて前史が消える → 時定数・回路の問題
//! - **(ii) 状態に履歴はあるが、スパイク数に出ていない** → 読み出しの問題
//!
//! ## 設計
//!
//! **接頭辞だけが違い、末尾を共有する語対** (ABC vs BAC — 同じモーラ集合・同じ最終モーラ・
//! 最初の 2 モーラの順序だけ違う) を、**同一の網の複製**に流す。
//! **共通の末尾モーラ C を聞いている間**、内部状態の差がどれだけ保持されるかを
//! 段ごと (M0.5 / M1 / M2)・**状態変数ごと (膜電位 = 速い変数 / 局所エントロピー = 遅い変数)**
//! に測る。語末後は無音 1 モーラぶんの残存も見る。
//!
//! **熱力学的描像の直接検証でもある**: DESIGN_PHILOSOPHY §11 は「慣化 = 局所エントロピー
//! 蓄積」と置く。**履歴の担い手はエントロピーのはず** — それが本当かを測る。
//!
//! 距離 = 正規化 L1: d(a,b) = Σ|aᵢ−bᵢ| / (Σ|aᵢ|+Σ|bᵢ|) ∈ [0,1]。10 対の平均。
//!
//! ## ゲート (実測前に固定・以後動かさない)
//!
//! - **G103a 対照**: 同一語対の距離が全変数・全ステップで厳密に 0 (決定論)。
//! - **G103b 記述**: 分岐軌跡 (接頭辞終端 → 末尾モーラ終端 → 無音 +120ms)。
//! - **G103c 本丸**: **末尾モーラ終端での保持率** d(末尾終端)/d(接頭辞終端)。
//!   膜 vs エントロピー・段間・学習前後の比較で「履歴の担い手」を特定する。
//! - **G103d 学習効果**: コーパス 4,000 モーラ後に保持が変わるか。
//! - **G103e スパイク分岐 vs 状態分岐**: 末尾モーラ中の**出力スパイク数**の分岐が
//!   エントロピー分岐より系統的に小さければ、**(ii) 読み出し問題**の直接証拠。
//! - **G103f** コーパスの内容は一切出力しない (学習のみに使用)。
//!
//! ## 予測 (実測前・数値は置かない)
//!
//! 1. 膜電位の分岐は速く消えるはず (leak と発火リセットで洗い流される)。
//! 2. **エントロピーの分岐は保持されるはず** — 設計哲学どおりなら。
//!    特に M0.5 (慣化が強い)。**外れたら熱力学的描像の「記憶」側に穴がある。**
//! 3. **本丸 (M2 と読み出し) は分からない。** (i)/(ii) のどちらに出るかが Day 3 の分岐点。
//!
//! CLI: state_divergence  (DRPNN_CORPUS_MORAS 既定 4000 = 学習後アームの学習量)

use spiking_brain::phase2_f::cochlea::{Cochlea, SAMPLES_PER_STEP};
use spiking_brain::phase2_f::cochlear_nucleus::{CochlearNucleus, N_CN_OUTPUT};
use spiking_brain::phase2_f::kana::{moras_from_kana, synth_utterance, Mora, MORA_MS};
use spiking_brain::phase2_f::phoneme_synth::LfsrNoise;
use spiking_brain::phase2_f::thermo_network::{ThermoNetwork, ThermoNetworkConfig};
use std::io::Read;

const CORPUS: &str = "../data/corpus/roleplay_kana.txt";
const F0S: [f64; 4] = [100.0, 130.0, 160.0, 200.0];
const INPUT_CURRENT_M2: i32 = 60;
const MORA_STEPS: usize = (MORA_MS as usize) * 16 / SAMPLES_PER_STEP;   // 240
const F0_TEST: f64 = 130.0;

/// §14.54 と同じ 10 組。対 = (ABC, BAC): 最初の 2 モーラだけ入れ替え・末尾は共通。
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
    // §14.62: 共有ヘルパへ委譲 (DRPNN_M2_PROJ で legacy / topo を切替)
    spiking_brain::phase2_f::thermo_network::project_m1_m2(m1, fired, n_in)
}

/// 1 本の語を流した軌跡: 各ステップの状態と、末尾モーラ中の出力スパイク数。
struct Trace {
    /// [step][段0..3] 膜電位ベクトル (CN=octopus+bushy+stellate / M1 / M2 全ニューロン)
    mem: Vec<[Vec<i32>; 3]>,
    ent: Vec<[Vec<i32>; 3]>,
    /// 末尾モーラ中の出力スパイク数 (段 0 = CN 出力 84ch / 1 = M1 出力 40 / 2 = M2 出力 20)
    tail_spikes: [Vec<f64>; 3],
}

fn cn_state(cn: &CochlearNucleus) -> (Vec<i32>, Vec<i32>) {
    let all = cn.octopus.iter().chain(cn.bushy.iter()).chain(cn.stellate.iter());
    let mem: Vec<i32> = all.clone().map(|n| n.membrane).collect();
    let ent: Vec<i32> = all.map(|n| n.local_entropy).collect();
    (mem, ent)
}

fn net_state(net: &ThermoNetwork) -> (Vec<i32>, Vec<i32>) {
    (net.neurons.iter().map(|n| n.membrane).collect(),
     net.neurons.iter().map(|n| n.local_entropy).collect())
}

/// 語 + 無音 1 モーラを流して軌跡を取る。網は**呼び出し側が渡した複製**。
fn run_word(word: &str, seed: u16,
            co: &mut Cochlea, cn: &mut CochlearNucleus,
            m1: &mut ThermoNetwork, m2: &mut ThermoNetwork) -> Trace {
    let mut noise = LfsrNoise::new(seed);
    let (m, sk) = moras_from_kana(word);
    assert_eq!(sk, 0, "未対応: {}", word);
    let mut wave = synth_utterance(&m, F0_TEST, &mut noise);
    wave.extend(std::iter::repeat(0).take(MORA_STEPS * SAMPLES_PER_STEP));   // 無音尾
    let n_steps = wave.len() / SAMPLES_PER_STEP;
    let tail_range = (2 * MORA_STEPS)..(3 * MORA_STEPS);   // 末尾モーラ (共通の C)
    let n2_in = m2.input_neurons.len();
    let mut tr = Trace {
        mem: Vec::with_capacity(n_steps),
        ent: Vec::with_capacity(n_steps),
        tail_spikes: [vec![0f64; N_CN_OUTPUT],
                      vec![0f64; m1.output_neurons.len()],
                      vec![0f64; m2.output_neurons.len()]],
    };
    for (step, chunk) in wave.chunks(SAMPLES_PER_STEP).enumerate() {
        if chunk.len() < SAMPLES_PER_STEP { break; }
        let m0 = co.process_step(chunk);
        let cno = cn.process_step(&m0);
        let fired1 = m1.step(&cno);
        let inp2 = m2_input(m1, &fired1, n2_in);
        let fired2 = m2.step(&inp2);
        if tail_range.contains(&step) {
            for (i, &x) in cno.iter().enumerate() { if x != 0 { tr.tail_spikes[0][i] += 1.0; } }
            for &nid in &fired1 {
                if let Some(oi) = m1.output_index_of(nid) { tr.tail_spikes[1][oi] += 1.0; }
            }
            for nid in fired2 {
                if let Some(oi) = m2.output_index_of(nid) { tr.tail_spikes[2][oi] += 1.0; }
            }
        }
        let (c_m, c_e) = cn_state(cn);
        let (a_m, a_e) = net_state(m1);
        let (b_m, b_e) = net_state(m2);
        tr.mem.push([c_m, a_m, b_m]);
        tr.ent.push([c_e, a_e, b_e]);
    }
    tr
}

/// 正規化 L1 距離 ∈ [0,1]
fn nd(a: &[i32], b: &[i32]) -> f64 {
    let num: i64 = a.iter().zip(b.iter()).map(|(&x, &y)| (x as i64 - y as i64).abs()).sum();
    let den: i64 = a.iter().map(|&x| (x as i64).abs()).sum::<i64>()
                 + b.iter().map(|&y| (y as i64).abs()).sum::<i64>();
    if den == 0 { 0.0 } else { num as f64 / den as f64 }
}

fn ndf(a: &[f64], b: &[f64]) -> f64 {
    let num: f64 = a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).sum();
    let den: f64 = a.iter().map(|x| x.abs()).sum::<f64>() + b.iter().map(|y| y.abs()).sum::<f64>();
    if den == 0.0 { 0.0 } else { num / den }
}

struct ArmResult {
    /// [段][時点 (接頭辞終端 / 末尾中央 / 末尾終端 / 無音+120ms)] の平均距離
    mem: [[f64; 4]; 3],
    ent: [[f64; 4]; 3],
    /// [段] 末尾モーラ中の出力スパイク数の分岐
    spk: [f64; 3],
    /// 対照 (同一語対) の最大距離 — 0 であるべき
    ctrl_max: f64,
}

fn measure(co0: &Cochlea, cn0: &CochlearNucleus, m10: &ThermoNetwork, m20: &ThermoNetwork) -> ArmResult {
    let t_pts = [2 * MORA_STEPS - 1, 2 * MORA_STEPS + MORA_STEPS / 2, 3 * MORA_STEPS - 1, 4 * MORA_STEPS - 1];
    let mut mem = [[0f64; 4]; 3];
    let mut ent = [[0f64; 4]; 3];
    let mut spk = [0f64; 3];
    let mut ctrl_max = 0f64;
    for (g, t) in TRIPLES.iter().enumerate() {
        let abc: String = format!("{}{}{}", t[0], t[1], t[2]);
        let bac: String = format!("{}{}{}", t[1], t[0], t[2]);
        let seed = 0x1357u16.wrapping_add(g as u16 * 97) | 1;
        let run = |w: &str| {
            let (mut co, mut cn, mut m1, mut m2) = (co0.clone(), cn0.clone(), m10.clone(), m20.clone());
            run_word(w, seed, &mut co, &mut cn, &mut m1, &mut m2)
        };
        let ta = run(&abc);
        let tb = run(&bac);
        let ta2 = run(&abc);   // 対照: 同一語をもう一度
        for (pi, &tp) in t_pts.iter().enumerate() {
            for s in 0..3 {
                mem[s][pi] += nd(&ta.mem[tp][s], &tb.mem[tp][s]);
                ent[s][pi] += nd(&ta.ent[tp][s], &tb.ent[tp][s]);
                ctrl_max = ctrl_max.max(nd(&ta.mem[tp][s], &ta2.mem[tp][s]))
                                   .max(nd(&ta.ent[tp][s], &ta2.ent[tp][s]));
            }
        }
        for s in 0..3 { spk[s] += ndf(&ta.tail_spikes[s], &tb.tail_spikes[s]); }
    }
    let n = TRIPLES.len() as f64;
    for s in 0..3 { for p in 0..4 { mem[s][p] /= n; ent[s][p] /= n; } spk[s] /= n; }
    ArmResult { mem, ent, spk, ctrl_max }
}

fn print_arm(label: &str, r: &ArmResult) {
    const STAGES: [&str; 3] = ["M0.5", "M1", "M2"];
    println!("--- {} ---", label);
    println!("  {:<6} {:<14} | {:>10} {:>10} {:>10} {:>12} | {:>8}",
             "段", "変数", "接頭辞終端", "末尾中央", "末尾終端", "無音+120ms", "**保持率**");
    for s in 0..3 {
        for (nm, v) in [("膜電位", &r.mem[s]), ("エントロピー", &r.ent[s])] {
            let keep = if v[0] > 1e-12 { v[2] / v[0] } else { 0.0 };
            println!("  {:<6} {:<14} | {:>10.4} {:>10.4} {:>10.4} {:>12.4} | {:>7.2}",
                     STAGES[s], nm, v[0], v[1], v[2], v[3], keep);
        }
        println!("  {:<6} {:<14} | {:>46} 分岐 = {:.4}", STAGES[s], "末尾スパイク数", "", r.spk[s]);
    }
    println!("  対照 (同一語対) の最大距離 = {:.6}  ({})", r.ctrl_max,
             if r.ctrl_max == 0.0 { "**厳密に 0 = G103a PASS**" } else { "**FAIL**" });
}

fn main() {
    let n_moras: usize = std::env::var("DRPNN_CORPUS_MORAS").ok()
        .and_then(|v| v.parse().ok()).unwrap_or(4000);

    println!("=== 状態分岐実験 — 履歴はどの状態変数に、どれだけ保持されるのか ===");
    println!();
    println!("【設計】接頭辞だけが違い末尾を共有する語対 (ABC vs BAC・10 対) を同一の網の複製に流し、");
    println!("**共通の末尾モーラの間、状態の差がどれだけ保持されるか**を測る。");
    println!("距離 = 正規化 L1・保持率 = d(末尾終端) / d(接頭辞終端)。");
    println!();
    println!("【二分】(i) 状態に履歴が無い -> 時定数・回路の問題 / ");
    println!("       (ii) 状態にあるがスパイク数に出ていない -> 読み出しの問題");
    println!();
    println!("【ゲート・実測前に固定】G103a 同一語対 = 厳密 0 / G103b 軌跡 (記述) /");
    println!("**G103c 保持率 (本丸: 履歴の担い手の特定)** / G103d 学習効果 (0 vs {} モーラ) /", n_moras);
    println!("**G103e スパイク分岐 vs 状態分岐** ((ii) の直接証拠) / G103f 内容非出力");
    println!();
    println!("【予測・実測前】①膜電位の分岐は速く消えるはず ②**エントロピーは保持されるはず**");
    println!("(設計哲学「慣化 = エントロピー蓄積」どおりなら。**外れたら熱力学的描像の記憶側に穴**)");
    println!("③本丸 (M2 と読み出し) は分からない — (i)/(ii) のどちらに出るかが Day 3 の分岐点。");
    println!();

    let cfg1 = if N_CN_OUTPUT == 164 { ThermoNetworkConfig::for_m1_cn_80() }
               else { ThermoNetworkConfig::for_m1_cn_40() };
    let mut m1 = ThermoNetwork::new(cfg1);
    let mut m2 = ThermoNetwork::new(ThermoNetworkConfig::for_m2());
    let (mut co, mut cn) = (Cochlea::new(), CochlearNucleus::new());

    // ---- 学習前アーム ----
    let untrained = measure(&co, &cn, &m1, &m2);
    print_arm("学習前 (0 モーラ)", &untrained);

    // ---- コーパスで学習 ----
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
    let trained = measure(&co, &cn, &m1, &m2);
    print_arm(&format!("学習後 ({} モーラ)", n_moras), &trained);

    // ---- 判定 ----
    println!();
    println!("=== 判定 ===");
    const STAGES: [&str; 3] = ["M0.5", "M1", "M2"];
    println!("  **G103c 履歴の担い手** (学習後・末尾終端で分岐が最も残る変数):");
    for s in 0..3 {
        let (m_end, e_end) = (trained.mem[s][2], trained.ent[s][2]);
        println!("     {} -> 膜 {:.4} / エントロピー {:.4} -> **{}**",
                 STAGES[s], m_end, e_end,
                 if e_end > m_end { "エントロピーが担い手 (設計哲学どおり)" }
                 else { "膜が担い手 (設計哲学と不整合 — 要精査)" });
    }
    println!();
    println!("  **G103e (ii) 読み出し問題の判定** (学習後・末尾モーラ):");
    for s in 0..3 {
        let e_mid = (trained.ent[s][1] + trained.ent[s][2]) / 2.0;
        println!("     {} -> 状態分岐 (エントロピー) {:.4} vs スパイク分岐 {:.4} -> {}",
                 STAGES[s], e_mid, trained.spk[s],
                 if e_mid > trained.spk[s] * 2.0 {
                     "**状態にあるがスパイクに出ていない = 読み出し問題 (ii)**"
                 } else if trained.spk[s] > 1e-9 {
                     "スパイクにも同程度出ている"
                 } else { "どちらにも無い (i)" });
    }
    println!();
    println!("  G103f コーパスの内容 -> **一切出力していない (学習のみに使用・数値のみ)**");
}
