//! LTP は本当に起きているのか — 診断①と②を分ける (2026-08-27)
//!
//! ## なぜ
//!
//! §14.43 で「**M1 は単語について何も表現していない。入力から切れていた**」が出た。
//! **1,000 モーラで伝達可能シナプスが 22,740 → 519 (97.7% 減)、
//! M1 発火/語 は 250 モーラ以降ずっと 70.5 で一定** (= 出力が入力に依存していない)。
//!
//! **診断が 2 つに分かれ、直し方が正反対である:**
//!
//! - **診断① 減衰が速すぎる** — 強化は起きているが、追いつかない。
//!   → 時定数の問題。
//! - **診断② そもそも強化が一度も起きていない** — 入力が M1 の皮質ニューロンを
//!   発火させられず、因果ペア (LTP の条件) が成立していない。
//!   → 入力の駆動力の問題。**時定数をいじっても何も変わらない。**
//!
//! **いまのデータでは区別できない。ここで確定させる。**
//!
//! ## 何を数えるか
//!
//! `ThermoSynapse` に **計測専用のカウンタ** `n_ltp` / `n_ltd` を足した
//! (**振る舞いには一切使わない**)。コーパスを流しながら、
//! **入力→皮質のシナプス**と**それ以外**に分けて数える。
//! **入力→皮質こそが本丸である** (聴覚入力が M1 に届いているかを決めるのはここ)。
//!
//! ## ゲート (実測前に固定・以後動かさない)
//!
//! - **G97a LTP は起きているか**: 総 LTP 事象数と、一度でも増強されたシナプスの本数。
//! - **G97b 入力→皮質に限ると LTP は起きているか**: ***これが本丸。***
//! - **G97c 正味で増強されたシナプスはあるか**: `conductance` が初期値 (80) を超えている本数。
//!   *減衰は単調に下げるだけなので、80 超は「LTP が減衰に勝った」ことの証拠になる。*
//! - **G97d LTP と LTD の比**。
//! - **G97e 決定論性 / G97f コーパスの内容は一切出力しない。**
//!
//! ## 予測 (実測前・数値は置かない)
//!
//! 1. **LTP は起きているはず** — 皮質内の再帰結線では同時発火が起きるはず。
//! 2. **入力→皮質では、LTP はほとんど起きていないはず。**
//!    §14.43 で M1 の発火が入力に依存していなかったので、
//!    入力が皮質を発火させていないなら因果ペアが成立しない。
//! 3. **正味で増強されたシナプス (conductance > 80) はほぼゼロのはず。**
//!
//! **2 が当たれば診断②、外れれば診断①である。**
//!
//! CLI: plasticity_census  (DRPNN_CORPUS_MORAS でモーラ数・既定 4000)

use spiking_brain::phase2_f::cochlea::{Cochlea, SAMPLES_PER_STEP};
use spiking_brain::phase2_f::cochlear_nucleus::{CochlearNucleus, N_CN_OUTPUT};
use spiking_brain::phase2_f::kana::{moras_from_kana, synth_utterance, Mora};
use spiking_brain::phase2_f::phoneme_synth::LfsrNoise;
use spiking_brain::phase2_f::thermo_network::{ThermoNetwork, ThermoNetworkConfig, SIGNAL_SCALE_DIVISOR};
use spiking_brain::phase2_f::thermo_synapse::{LTD_AMOUNT, LTP_AMOUNT, OPEN_THRESHOLD};
use std::io::Read;

const CORPUS: &str = "../data/corpus/roleplay_kana.txt";
const F0S: [f64; 4] = [100.0, 130.0, 160.0, 200.0];
/// 生成時の初期 conductance (`ThermoNetwork::new` が `OPEN_THRESHOLD + 50` で作る)
const INITIAL_CONDUCTANCE: i32 = OPEN_THRESHOLD + 50;

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

/// (本数, LTP事象, LTD事象, 一度でもLTPされた本数, conductance>初期値 の本数, 最大conductance, 伝達可)
fn census(net: &ThermoNetwork, input_only: bool) -> (usize, u64, u64, usize, usize, i32, usize) {
    let is_in = |i: usize| net.input_neurons.contains(&i);
    let sel: Vec<&spiking_brain::phase2_f::thermo_synapse::ThermoSynapse> =
        net.synapses.iter().filter(|s| is_in(s.pre) == input_only).collect();
    let ltp: u64 = sel.iter().map(|s| s.n_ltp as u64).sum();
    let ltd: u64 = sel.iter().map(|s| s.n_ltd as u64).sum();
    let touched = sel.iter().filter(|s| s.n_ltp > 0).count();
    let above = sel.iter().filter(|s| s.conductance > INITIAL_CONDUCTANCE).count();
    let maxc = sel.iter().map(|s| s.conductance).max().unwrap_or(0);
    let live = sel.iter().filter(|s| s.alive && s.conductance >= SIGNAL_SCALE_DIVISOR).count();
    (sel.len(), ltp, ltd, touched, above, maxc, live)
}

fn row(name: &str, c: (usize, u64, u64, usize, usize, i32, usize)) {
    println!("  {:<14} {:>8} {:>12} {:>12} {:>12} {:>10} {:>8} {:>10}",
             name, c.0, c.1, c.2, c.3, c.4, c.5, c.6);
}

fn main() {
    let n_moras: usize = std::env::var("DRPNN_CORPUS_MORAS").ok()
        .and_then(|v| v.parse().ok()).unwrap_or(4000);

    println!("=== LTP は本当に起きているのか — 診断①と②を分ける ===");
    println!();
    println!("【なぜ】§14.43 で「**M1 は単語について何も表現していない。入力から切れていた**」。");
    println!("**診断が2つに分かれ、直し方が正反対:**");
    println!("  **① 減衰が速すぎる** (強化は起きているが追いつかない) -> 時定数の問題");
    println!("  **② そもそも強化が一度も起きていない** (入力が皮質を発火させられない)");
    println!("     -> 入力の駆動力の問題。**時定数をいじっても何も変わらない。**");
    println!("**いまのデータでは区別できない。ここで確定させる。**");
    println!();
    println!("【何を数えるか】ThermoSynapse に **計測専用のカウンタ** n_ltp / n_ltd を足した");
    println!("(**振る舞いには一切使わない**)。**入力→皮質**と**それ以外**に分けて数える。");
    println!("**入力→皮質こそが本丸**(聴覚入力が M1 に届くかを決めるのはここ)。");
    println!();
    println!("【ゲート・実測前に固定】");
    println!("  G97a LTP は起きているか / **G97b 入力→皮質に限ると起きているか (本丸)**");
    println!("  G97c **正味で増強された本数** (conductance > 初期値 {}。減衰は下げるだけなので", INITIAL_CONDUCTANCE);
    println!("       **超えていれば LTP が減衰に勝った証拠**) / G97d LTP と LTD の比");
    println!("  G97e 決定論性 / G97f 内容非出力");
    println!();
    println!("【予測・実測前】①LTP は起きているはず(皮質内の再帰結線) ");
    println!("  ②**入力→皮質では LTP はほとんど起きていないはず** ③正味で増強された本数はほぼゼロ");
    println!("  **②が当たれば診断②、外れれば診断①。**");
    println!();
    println!("  定数: LTP_AMOUNT={} / LTD_AMOUNT={} / 初期 conductance={}",
             LTP_AMOUNT, LTD_AMOUNT, INITIAL_CONDUCTANCE);

    let (moras, kinds) = load_moras(n_moras);
    println!("  コーパス {} モーラ / **{} 種類のかなが鳴った**。**内容は出力しない。**", moras.len(), kinds);

    let cfg = if N_CN_OUTPUT == 164 { ThermoNetworkConfig::for_m1_cn_80() }
              else { ThermoNetworkConfig::for_m1_cn_40() };
    assert_eq!(cfg.n_input, N_CN_OUTPUT);
    let mut net = ThermoNetwork::new(cfg);
    let (mut co, mut cn) = (Cochlea::new(), CochlearNucleus::new());
    let mut noise = LfsrNoise::new(0xACE1);

    let cps = [0usize, 100, 300, 1000, n_moras];
    let mut next = 0usize;
    let (mut fire_in, mut fire_cx, mut n_steps) = (0u64, 0u64, 0u64);
    println!();
    println!("  {:<14} {:>8} {:>12} {:>12} {:>12} {:>10} {:>8} {:>10}",
             "", "本数", "LTP事象", "LTD事象", "一度でもLTP", "**80超**", "最大G", "伝達可");
    for i in 0..=moras.len() {
        if next < cps.len() && cps[next] == i {
            println!("  --- {} モーラ聞いた時点 ---", i);
            row("**入力→皮質**", census(&net, true));
            row("皮質内など", census(&net, false));
            next += 1;
        }
        if i == moras.len() { break; }
        let w = synth_utterance(std::slice::from_ref(&moras[i]), F0S[i % 4], &mut noise);
        for chunk in w.chunks(SAMPLES_PER_STEP) {
            if chunk.len() < SAMPLES_PER_STEP { break; }
            let m0 = co.process_step(chunk);
            let cno = cn.process_step(&m0);
            // **発火率を数える** (§14.44.4 の推論を確かめる。振る舞いは変えない)
            for nid in net.step(&cno) {
                if net.input_neurons.contains(&nid) { fire_in += 1u64; } else { fire_cx += 1u64; }
            }
            n_steps += 1u64;
        }
    }

    let inp = census(&net, true);
    let oth = census(&net, false);
    println!();
    println!("  **G97a LTP は起きているか** -> 全体で {} 事象 / 一度でも増強された {} 本 -> {}",
             inp.1 + oth.1, inp.3 + oth.3,
             if inp.1 + oth.1 > 0 { "**起きている**" } else { "**一度も起きていない**" });
    println!("  **G97b 入力→皮質に限ると (本丸)** -> {} 事象 / {} 本 ({} 本中) -> {}",
             inp.1, inp.3, inp.0,
             if inp.3 == 0 { "**一度も起きていない = 診断②**" }
             else if (inp.3 as f64) < (inp.0 as f64) * 0.01 { "**ほぼ起きていない = 診断②寄り**" }
             else { "**起きている = 診断①寄り**" });
    println!("  **G97c 正味で増強された本数 (conductance > {})** -> 入力→皮質 {} 本 / それ以外 {} 本 -> {}",
             INITIAL_CONDUCTANCE, inp.4, oth.4,
             if inp.4 + oth.4 == 0 { "**ゼロ。LTP は一度も減衰に勝っていない**" } else { "**存在する**" });
    println!("  G97d LTP と LTD の比 -> 入力→皮質 {}:{} / それ以外 {}:{}", inp.1, inp.2, oth.1, oth.2);
    let n_in = net.input_neurons.len() as f64;
    let n_cx = (net.n_neurons() - net.input_neurons.len()) as f64;
    let st = n_steps.max(1) as f64;
    println!();
    println!("  **G97e 発火率 (§14.44.4 の推論の検証)** — 1 ニューロンあたり 1 step の発火確率");
    println!("     **入力ニューロン ({:.0} 個): {:.4}**", n_in, fire_in as f64 / st / n_in);
    println!("     皮質ニューロン ({:.0} 個): {:.4}", n_cx, fire_cx as f64 / st / n_cx);
    println!("     -> **比 {:.2} 倍**  {}", (fire_in as f64 / n_in) / (fire_cx as f64 / n_cx).max(1e-9),
             if fire_in as f64 / n_in > fire_cx as f64 / n_cx { "**入力側が高頻度 = 推論どおり**" } else { "**推論が外れた**" });
    println!("  G97f コーパスの内容 -> **一切出力していない (数値のみ)**");
    println!();
    println!("  【この測定が答えないこと】**なぜ因果ペアが成立しないのか**までは見ていない");
    println!("  (入力ニューロンが発火していないのか、皮質ニューロンが発火していないのか、");
    println!("   両方発火しても窓に入らないのか)。**次に切り分ける。**");
}
