//! E/I 診断 — 抑制系は生きているのか、死んでいるのか (2026-08-29)
//!
//! ## なぜ
//!
//! §14.57 で確定: 順序署名の符号は存在するが (帰無超えの濃縮 5〜10 倍)、
//! **対ごとの信頼性が壊れている** = 軌道が不安定。力学系の定石は **E/I バランス崩壊**。
//!
//! そして盲点がある: **E+F+D の是正はすべて興奮系の解析に基づいて行った。
//! 同じ力学 (STDP + 受動減衰) に晒された抑制シナプスがどうなったかは、一度も測っていない。**
//! これまでの census の「皮質内など」は E→E・I→E・E→I を**混ぜて**いた。ここで分ける。
//!
//! ## 何を測るか
//!
//! 1. **シナプス census を pre→post 種別で分割** (入力→E / 入力→I / E→E / E→I / I→E / I→I)
//! 2. **配送電流の E/I 収支** (`stat_exc/inh_delivered` — 計測専用カウンタを追加した)
//! 3. **ニューロン種別の発火率** (入力 / 皮質E / 皮質I・音あり vs 無音)
//! 4. **活動の参加率** = 実効ニューロン数 PR = (Σr)²/Σr² (実効次元の低さの直接原因の候補)
//!
//! ## ゲート (実測前に固定・以後動かさない)
//!
//! - **G105a 本丸: I→E シナプスは生きているか** (伝達可・平均 G)。
//! - **G105b E/I 電流収支** (配送量の比・区間ごと)。
//! - **G105c 参加率** (皮質で実際に応答に参加しているニューロン数)。
//! - **G105d 抑制ニューロンの発火率** (音あり / 無音)。
//! - **G105e 決定論性 / G105f 内容非出力。**
//!
//! ## 予測 (実測前)
//!
//! **I→E は死んでいるはず** (会議で置いた予測)。ただし正直な留保:
//! これまでの census で「皮質内など」(I→E を含む混合) は健全だった (伝達可 65%) ので、
//! **分割したら「生きていた」と出る可能性も十分ある。** その場合、犯人は抑制の死でなく
//! 別のもの (抑制の「形」= 減算 vs シャント、あるいは配線) に移る。
//!
//! CLI: ei_census  (DRPNN_CORPUS_MORAS 既定 4000)

use spiking_brain::phase2_f::cochlea::{Cochlea, SAMPLES_PER_STEP};
use spiking_brain::phase2_f::cochlear_nucleus::{CochlearNucleus, N_CN_OUTPUT};
use spiking_brain::phase2_f::kana::{moras_from_kana, synth_utterance, Mora, MORA_MS};
use spiking_brain::phase2_f::phoneme_synth::LfsrNoise;
use spiking_brain::phase2_f::thermo_network::{ThermoNetwork, ThermoNetworkConfig, SIGNAL_SCALE_DIVISOR};
use std::io::Read;

const CORPUS: &str = "../data/corpus/roleplay_kana.txt";
const F0S: [f64; 4] = [100.0, 130.0, 160.0, 200.0];
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
    // §14.62: 共有ヘルパへ委譲 (DRPNN_M2_PROJ で legacy / topo を切替)
    spiking_brain::phase2_f::thermo_network::project_m1_m2(m1, fired, n_in)
}

/// pre→post の種別 (0=入力→E, 1=入力→I, 2=E→E, 3=E→I, 4=**I→E**, 5=I→I)
fn class_of(net: &ThermoNetwork, s: &spiking_brain::phase2_f::thermo_synapse::ThermoSynapse,
            is_input: &[bool]) -> usize {
    let pre_inh = net.neurons[s.pre].is_inhibitory;
    let post_inh = net.neurons[s.post].is_inhibitory;
    if is_input[s.pre] { if post_inh { 1 } else { 0 } }
    else if pre_inh { if post_inh { 5 } else { 4 } }
    else if post_inh { 3 } else { 2 }
}

const CLASS_NAMES: [&str; 6] = ["入力→E", "入力→I", "E→E", "E→I", "**I→E**", "I→I"];

fn census_by_class(net: &ThermoNetwork, is_input: &[bool]) {
    println!("  {:<10} {:>7} {:>10} {:>9} {:>12} {:>12}",
             "種別", "本数", "伝達可", "平均G", "LTP事象", "LTD事象");
    for c in 0..6 {
        let sel: Vec<_> = net.synapses.iter()
            .filter(|s| class_of(net, s, is_input) == c).collect();
        if sel.is_empty() { continue; }
        let live = sel.iter().filter(|s| s.alive && s.conductance >= SIGNAL_SCALE_DIVISOR).count();
        let mg = sel.iter().map(|s| s.conductance.max(0) as f64).sum::<f64>() / sel.len() as f64;
        let ltp: u64 = sel.iter().map(|s| s.n_ltp as u64).sum();
        let ltd: u64 = sel.iter().map(|s| s.n_ltd as u64).sum();
        println!("  {:<10} {:>7} {:>7} ({:>3.0}%) {:>9.2} {:>12} {:>12}",
                 CLASS_NAMES[c], sel.len(), live,
                 live as f64 / sel.len() as f64 * 100.0, mg, ltp, ltd);
    }
}

/// テスト語 10 本を複製に流し、(発火率 [入力/皮質E/皮質I], 参加率, PR) を返す。
fn probe_activity(net0: &ThermoNetwork, co0: &Cochlea, cn0: &CochlearNucleus,
                  m2_side: Option<&ThermoNetwork>, silent: bool)
    -> ([f64; 3], f64, f64) {
    let is_input: Vec<bool> = (0..net0.n_neurons()).map(|i| net0.input_neurons.contains(&i)).collect();
    let mut fires = vec![0f64; net0.n_neurons()];
    let mut steps = 0u64;
    for (g, t) in TRIPLES.iter().enumerate() {
        let word = format!("{}{}{}", t[0], t[1], t[2]);
        let (mut co, mut cn, mut m1) = (co0.clone(), cn0.clone(), net0.clone());
        let mut m2c = m2_side.map(|m| m.clone());
        let mut noise = LfsrNoise::new((0x77u16.wrapping_add(g as u16 * 13)) | 1);
        let (m, sk) = moras_from_kana(&word);
        assert_eq!(sk, 0);
        let wave = if silent {
            vec![0i32; 3 * MORA_STEPS * SAMPLES_PER_STEP]
        } else {
            synth_utterance(&m, F0S[g % 4], &mut noise)
        };
        for chunk in wave.chunks(SAMPLES_PER_STEP) {
            if chunk.len() < SAMPLES_PER_STEP { break; }
            let m0 = co.process_step(chunk);
            let cno = cn.process_step(&m0);
            let fired1 = m1.step(&cno);
            for &nid in &fired1 { fires[nid] += 1.0; }
            if let Some(m2c) = m2c.as_mut() {
                let inp2 = m2_input(&m1, &fired1, m2c.input_neurons.len());
                let _ = m2c.step(&inp2);
            }
            steps += 1;
        }
    }
    let mut rate = [0f64; 3];
    let mut cnt = [0f64; 3];
    let (mut sum, mut sq, mut active, mut n_cortex) = (0f64, 0f64, 0usize, 0usize);
    for i in 0..fires.len() {
        let k = if is_input[i] { 0 } else if net0.neurons[i].is_inhibitory { 2 } else { 1 };
        rate[k] += fires[i];
        cnt[k] += 1.0;
        if k != 0 {
            n_cortex += 1;
            sum += fires[i];
            sq += fires[i] * fires[i];
            if fires[i] > 0.0 { active += 1; }
        }
    }
    for k in 0..3 { rate[k] /= cnt[k].max(1.0) * steps as f64; }
    let pr = if sq > 0.0 { sum * sum / sq } else { 0.0 };
    (rate, active as f64 / n_cortex as f64 * 100.0, pr)
}

fn main() {
    let n_moras: usize = std::env::var("DRPNN_CORPUS_MORAS").ok()
        .and_then(|v| v.parse().ok()).unwrap_or(4000);

    println!("=== E/I 診断 — 抑制系は生きているのか、死んでいるのか ===");
    println!();
    println!("【なぜ】§14.57: 符号は存在するが対ごとの信頼性が壊れている = 軌道不安定。");
    println!("定石は E/I バランス崩壊。**E+F+D の是正は興奮系の解析だけに基づいており、");
    println!("同じ力学に晒された抑制シナプスは一度も分けて測っていない** (「皮質内など」は混合)。");
    println!();
    println!("【ゲート・実測前に固定】**G105a 本丸: I→E は生きているか** / G105b E/I 電流収支 /");
    println!("G105c 参加率 (実効ニューロン数 PR) / G105d 抑制ニューロンの発火率 (音/無音) /");
    println!("G105e 決定論性 / G105f 内容非出力");
    println!();
    println!("【予測・実測前】**I→E は死んでいるはず** (会議で置いた予測)。**正直な留保**:");
    println!("混合 census では皮質内は健全 (伝達可 65%) だったので、**生きていたと出る可能性も");
    println!("十分ある**。その場合、犯人は抑制の死でなく別のもの (減算 vs シャント・配線) に移る。");

    let cfg1 = if N_CN_OUTPUT == 164 { ThermoNetworkConfig::for_m1_cn_80() }
               else { ThermoNetworkConfig::for_m1_cn_40() };
    let mut m1 = ThermoNetwork::new(cfg1);
    let mut m2 = ThermoNetwork::new(ThermoNetworkConfig::for_m2());
    let is_in1: Vec<bool> = (0..m1.n_neurons()).map(|i| m1.input_neurons.contains(&i)).collect();
    let is_in2: Vec<bool> = (0..m2.n_neurons()).map(|i| m2.input_neurons.contains(&i)).collect();
    let (mut co, mut cn) = (Cochlea::new(), CochlearNucleus::new());
    let mut noise = LfsrNoise::new(0xACE1);

    let moras = load_moras(n_moras);
    let cps = [0usize, 1500, n_moras];
    let mut next = 0usize;
    let (mut last_e, mut last_i) = (0u64, 0u64);
    for i in 0..=moras.len() {
        if next < cps.len() && cps[next] == i {
            println!();
            println!("======== {} モーラ聞いた時点 ========", i);
            println!("--- M1 のシナプス (pre→post 種別) ---");
            census_by_class(&m1, &is_in1);
            println!("--- M2 のシナプス ---");
            census_by_class(&m2, &is_in2);
            let de = m1.stat_exc_delivered - last_e;
            let di = m1.stat_inh_delivered - last_i;
            println!("--- G105b M1 配送電流の E/I 収支 (この区間) ---");
            println!("  E {} / I {} -> **I/E 比 = {:.4}**", de, di,
                     di as f64 / de.max(1) as f64);
            last_e = m1.stat_exc_delivered;
            last_i = m1.stat_inh_delivered;
            let (r_snd, act_snd, pr_snd) = probe_activity(&m1, &co, &cn, Some(&m2), false);
            let (r_sil, _, _) = probe_activity(&m1, &co, &cn, Some(&m2), true);
            println!("--- G105c/d 発火率と参加率 (テスト語 10 本・複製側) ---");
            println!("  発火率 [入力 {:.4} / 皮質E {:.4} / **皮質I {:.4}**] (無音: I {:.4})",
                     r_snd[0], r_snd[1], r_snd[2], r_sil[2]);
            println!("  皮質の参加率 {:.1}% / **実効ニューロン数 PR = {:.1}** (皮質 {} 個中)",
                     act_snd, pr_snd, m1.n_neurons() - m1.input_neurons.len());
            next += 1;
        }
        if i == moras.len() { break; }
        let w = synth_utterance(std::slice::from_ref(&moras[i]), F0S[i % 4], &mut noise);
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
    println!("  G105f コーパスの内容 -> **一切出力していない (学習のみに使用・数値のみ)**");
}
