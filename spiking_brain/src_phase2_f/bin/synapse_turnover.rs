//! ① 頭打ちの正体 — 平衡後、シナプスは回り続けているのか凍っているのか (2026-08-28)
//!
//! ## なぜ
//!
//! §14.49 で「読み出しの頭打ち (~2,000 モーラ)」と「平均 conductance の安定 (~1,500)」が
//! ほぼ同時と分かった。仮説は 3 つ:
//!
//! - (a) 表現容量の限界
//! - (b) コーパス多様性の限界
//! - **(c) 平衡に達したから学習が止まった** — 動的平衡は「統計が変わらなくなる」ことなので、
//!   **平衡到達と学習停止は同じ事象の裏表**かもしれない。欠陥ではなく**この学習則の性質**。
//!
//! ## 何を測るか
//!
//! **平均は §14.49 で測った (不動)。ここでは個々のシナプスを見る。**
//! 平均が不動でも、個々が入れ替わり続けている (= 探索は続くが正味が動かない) なら (c)。
//! 個々も凍っているなら「平衡」ではなく「固化」であり、別の話になる。
//!
//! チェックポイントごとに全シナプスの conductance を写し取り、隣接区間で比べる:
//! - **平均 |ΔG|** と **変化したシナプスの率** (|ΔG| ≥ 5 = LTP 1 量子)
//! - **伝達可能集合の Jaccard** (集合として入れ替わっているか)
//! - 軸索成長 (新設) と死亡の数
//!
//! (b) は `corpus_word_learning` の **ループ腕** (`DRPNN_CORPUS_LOOP` = 最初の N モーラを
//! 繰り返す・新素材なし) で切り分ける。fresh と loop で読み出しが違えば (b)。
//!
//! ## 本線との同一性
//!
//! この走行は `corpus_word_learning` の本線と**同じ moras・同じ F0 巡回・同じ雑音種・
//! 同じ網**なので、決定論により**状態軌道はバイト同一**である。
//! したがってここで測る構造動態は、あちらの読み出し曲線と**同じ軌道上の量**である。
//!
//! ## ゲート (実測前に固定・以後動かさない)
//!
//! - **G99a** 区間ごとの平均 |ΔG| と変化率 (記述)
//! - **G99b** 伝達可能集合の Jaccard (記述)
//! - **G99c 判定**: 平衡後 (2,000 モーラ以降) も **1,000 モーラあたり 1% 以上**のシナプスが
//!   LTP 1 量子以上動くか。動く → **回り続けている = 動的平衡 = 仮説 (c) を支持**。
//!   動かない → **凍結 = (c) は棄却され、(a)/(b) が残る**。
//! - **G99d** 決定論性 / **G99e** コーパスの内容は一切出力しない
//!
//! ## 予測 (実測前)
//!
//! **回り続けているはず (c)。** LTP 4,700 万事象が平衡中も起き続けていた (§14.49.6) ので、
//! 個々が凍っているとは考えにくい。**ループ腕は差が出ないはず** (かなの種類は 1,500 モーラで
//! ほぼ飽和しているので、新素材の追加は統計をほとんど変えない = (b) は棄却される方向)。
//!
//! CLI: synapse_turnover  (DRPNN_CORPUS_MORAS 既定 8000)

use spiking_brain::phase2_f::cochlea::{Cochlea, SAMPLES_PER_STEP};
use spiking_brain::phase2_f::cochlear_nucleus::{CochlearNucleus, N_CN_OUTPUT};
use spiking_brain::phase2_f::kana::{moras_from_kana, synth_utterance, Mora};
use spiking_brain::phase2_f::phoneme_synth::LfsrNoise;
use spiking_brain::phase2_f::thermo_network::{ThermoNetwork, ThermoNetworkConfig, SIGNAL_SCALE_DIVISOR};
use std::collections::BTreeSet;
use std::io::Read;

const CORPUS: &str = "../data/corpus/roleplay_kana.txt";
const F0S: [f64; 4] = [100.0, 130.0, 160.0, 200.0];

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
    for c in text.chars() {
        if out.len() >= n { break; }
        if c == '\n' || c == ' ' { continue; }
        let (m, _) = moras_from_kana(&c.to_string());
        out.extend(m);
    }
    out.truncate(n);
    out
}

struct Snap { mora: usize, g: Vec<i32>, live: BTreeSet<usize>, n_syn: usize }

fn snap(net: &ThermoNetwork, mora: usize) -> Snap {
    Snap {
        mora,
        g: net.synapses.iter().map(|s| if s.alive { s.conductance } else { -1 }).collect(),
        live: net.synapses.iter().enumerate()
            .filter(|(_, s)| s.alive && s.conductance >= SIGNAL_SCALE_DIVISOR)
            .map(|(i, _)| i).collect(),
        n_syn: net.synapses.len(),
    }
}

fn main() {
    let n_moras: usize = std::env::var("DRPNN_CORPUS_MORAS").ok()
        .and_then(|v| v.parse().ok()).unwrap_or(8000);
    let cps = [0usize, 1000, 1500, 2000, 3000, 4000, 6000, 8000];

    println!("=== ① 頭打ちの正体 — 平衡後、シナプスは回り続けているのか凍っているのか ===");
    println!();
    println!("【仮説】(a) 容量 / (b) コーパス多様性 / **(c) 平衡到達 = 学習停止は同じ事象の裏表**");
    println!("【何を測るか】平均は不動と分かっている (§14.49.6)。**ここでは個々を見る。**");
    println!("個々が入れ替わり続けている (探索は続くが正味が動かない) なら (c)。");
    println!("個々も凍っているなら「平衡」でなく「固化」。");
    println!();
    println!("【ゲート・実測前に固定】");
    println!("  G99a 区間ごとの平均|ΔG|と変化率 / G99b 伝達可能集合の Jaccard");
    println!("  **G99c 判定: 平衡後 (2,000以降) も 1,000モーラあたり 1% 以上が LTP 1量子(5)以上動くか**");
    println!("     動く -> **回り続けている = 動的平衡 = (c) を支持** / 動かない -> **凍結 = (c) 棄却**");
    println!("  G99d 決定論性 / G99e 内容非出力");
    println!();
    println!("【予測・実測前】**回り続けているはず (c)。**LTP 4,700万事象が平衡中も起きていた。");
    println!("(b) のループ腕は corpus_word_learning 側で切り分ける (差は出ないはず)。");

    let moras = load_moras(n_moras);
    let cfg = if N_CN_OUTPUT == 164 { ThermoNetworkConfig::for_m1_cn_80() }
              else { ThermoNetworkConfig::for_m1_cn_40() };
    assert_eq!(cfg.n_input, N_CN_OUTPUT);
    let mut net = ThermoNetwork::new(cfg);
    let is_input: Vec<bool> = (0..net.n_neurons()).map(|i| net.input_neurons.contains(&i)).collect();
    let (mut co, mut cn) = (Cochlea::new(), CochlearNucleus::new());
    let mut noise = LfsrNoise::new(0xACE1);

    let mut snaps: Vec<Snap> = Vec::new();
    let mut next = 0usize;
    for i in 0..=moras.len() {
        if next < cps.len() && cps[next] == i {
            snaps.push(snap(&net, i));
            next += 1;
        }
        if i == moras.len() { break; }
        let w = synth_utterance(std::slice::from_ref(&moras[i]), F0S[i % 4], &mut noise);
        for chunk in w.chunks(SAMPLES_PER_STEP) {
            if chunk.len() < SAMPLES_PER_STEP { break; }
            let m0 = co.process_step(chunk);
            let cno = cn.process_step(&m0);
            let _ = net.step(&cno);
        }
    }

    println!();
    println!("--- 隣接チェックポイント間の構造動態 (入力→皮質 / 皮質内 の順) ---");
    println!("  {:>13} | {:>9} {:>12} | {:>9} {:>12} | {:>8} {:>7} {:>7}",
             "区間", "平均|ΔG|", "変化率(≥5)", "平均|ΔG|", "変化率(≥5)", "Jaccard", "新設", "死亡");
    let mut eq_rates: Vec<f64> = Vec::new();
    for w in snaps.windows(2) {
        let (a, b) = (&w[0], &w[1]);
        let common = a.g.len().min(b.g.len());
        let span_k = (b.mora - a.mora) as f64 / 1000.0;
        let mut stats = [[0f64; 2]; 2];   // [入力/皮質][sum|Δ|, changed]
        let mut counts = [0f64; 2];
        for i in 0..common {
            if a.g[i] < 0 || b.g[i] < 0 { continue; }
            let pre = net.synapses[i].pre;
            let k = if is_input[pre] { 0 } else { 1 };
            let d = (b.g[i] - a.g[i]).abs();
            stats[k][0] += d as f64;
            if d >= 5 { stats[k][1] += 1.0; }
            counts[k] += 1.0;
        }
        let inter = a.live.intersection(&b.live).count() as f64;
        let uni = a.live.union(&b.live).count() as f64;
        let born = b.n_syn - a.n_syn;
        let died = (0..common).filter(|&i| a.g[i] >= 0 && b.g[i] < 0).count();
        let rate_in = stats[0][1] / counts[0].max(1.0) * 100.0 / span_k;
        let rate_cx = stats[1][1] / counts[1].max(1.0) * 100.0 / span_k;
        println!("  {:>5}-{:>5}k | {:>9.2} {:>10.2}%/k | {:>9.2} {:>10.2}%/k | {:>8.3} {:>7} {:>7}",
                 a.mora, b.mora, stats[0][0] / counts[0].max(1.0), rate_in,
                 stats[1][0] / counts[1].max(1.0), rate_cx,
                 if uni > 0.0 { inter / uni } else { 1.0 }, born, died);
        if a.mora >= 2000 { eq_rates.push(rate_in.max(rate_cx)); }
    }

    println!();
    let sustained = !eq_rates.is_empty() && eq_rates.iter().all(|&r| r >= 1.0);
    let min_rate = eq_rates.iter().cloned().fold(f64::INFINITY, f64::min);
    println!("  **G99c 判定** (平衡後 = 2,000 モーラ以降の全区間・最小の変化率 {:.2}%/1000モーラ):", min_rate);
    println!("  -> {}", if sustained {
        "**回り続けている — 平均が不動のまま個々は入れ替わっている = 動的平衡 = 仮説 (c) を支持**"
    } else {
        "**凍結している — (c) は棄却され、(a)/(b) が残る**"
    });
    println!();
    println!("  G99e コーパスの内容 -> **一切出力していない (数値のみ)**");
    println!("  【この測定が答えないこと】(c) の支持は「読み出しがなぜその水準か」(a) を排除しない。");
    println!("  (b) はループ腕 (corpus_word_learning · DRPNN_CORPUS_LOOP) の結果と併せて判定する。");
}
