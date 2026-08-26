//! 実コーパスを音として聞かせる (2026-08-27)
//!
//! ## なぜ
//!
//! ユーザーの指示: 「触れる音の種類を多くしよう、多様性を増やしてかつ単語や熟語の音も
//! 聞かせたい。実際にチャットコーパスの文章を音として聞かせたらどうなるのか」
//! 「刈込や新造が実物と同じスケールで起こるようにして、刺激も同じスケールに
//! そろえる必要があるんじゃない?」
//!
//! §14.20 でコーパスをかな (発音) 化済み: 1億かな・75 種・逐次構造 0.613 bit。
//! これを M0 → M0.5 → M1 に**連続で**流す。**リセットしない = 実際の聴取と同じ。**
//!
//! ## 刈込の算術 (**予測はここから出る。当てるためではなく私のコード理解の検算**)
//!
//! ```
//! VITALITY_INITIAL       = 100
//! VITALITY_DECAY_INTERVAL = 10,000 step (この step 数ごとに vitality -= 1)
//! 1 モーラ = MORA_MS 120ms × 16kHz ÷ SAMPLES_PER_STEP 8 = 240 step
//!
//! 一度も信号を通さないシナプスが死ぬまで:
//!   100 × 10,000 ÷ 240 = **4,166.7 モーラ**
//! ```
//!
//! ## ゲート (実測前に固定・以後動かさない)
//!
//! **正解の出どころ**: 刈込開始時刻は**上の算術**が決めている (実験者の側にある)。
//!
//! - **G89a 刈込は算術どおりに始まるか**: alive が最初に減るモーラ数が **4,100〜4,300**。
//!   *外れたら私のコード理解が間違っている。*
//! - **G89b 平衡に達するか**: alive と open の変化がブロックごとに小さくなるか。*記述。*
//! - **G89c コーパスは実際に多様か**: 実際に鳴ったかなの種類数。*記述。*
//! - **G89d 決定論性**: 先頭 500 モーラを 2 回走らせて一致。
//! - **G89e コーパスの内容は一切出力しない**: **数値のみ。**
//!
//! ## この測定の既知の欠陥 (**実測前に書いておく**)
//!
//! **M1 の入力数が現在の M0.5 と噛み合っていない。**
//! `for_m1_cn_80` は入力 164 だが `CochlearNucleus` の出力は 84 (Octopus 4 + Bushy 40 +
//! Stellate 40)。**80 本の入力ニューロンが一度も駆動されない。**
//! (44 = 4+20+20 は 20 帯域時代、164 = 4+80+80 は 80 帯域時代の設定で、
//!  40 帯域の 84 に合う設定が存在しない。**パラメータを発明しない**ので噛み合わせない。)
//!
//! したがって **刈込の「量」は私の配線ミスマッチに汚染される。**
//! ただし **G89a が見るのは「いつ始まるか」であって「いくつ死ぬか」ではない**ので、
//! ミスマッチに対して頑健である。**量の方は判定に使わない。**
//!
//! CLI: corpus_run   (DRPNN_CORPUS_MORAS でモーラ数・既定 12000)

use spiking_brain::phase2_f::cochlea::{Cochlea, SAMPLES_PER_STEP};
use spiking_brain::phase2_f::cochlear_nucleus::CochlearNucleus;
use spiking_brain::phase2_f::kana::{moras_from_kana, synth_utterance, Mora, MORA_MS};
use spiking_brain::phase2_f::phoneme_synth::LfsrNoise;
use spiking_brain::phase2_f::thermo_network::{ThermoNetwork, ThermoNetworkConfig};
use spiking_brain::phase2_f::thermo_synapse::OPEN_THRESHOLD;
use std::collections::BTreeSet;
use std::io::Read;

const CORPUS: &str = "../data/corpus/roleplay_kana.txt";
const F0: f64 = 150.0;
const SEED: u16 = 0xACE1;
const STEPS_PER_MORA: usize = (MORA_MS as usize) * 16 / SAMPLES_PER_STEP;

/// コーパスの先頭から `n_moras` 分のモーラを取る。**内容は返さない・出力もしない。**
fn load_moras(n_moras: usize) -> (Vec<Mora>, usize, usize) {
    // 1 文字 ≒ 1 モーラなので、余裕をみて 3 倍の文字を読む (UTF-8 で最大 3 バイト/字)
    let want_bytes = n_moras * 3 * 3 + 4096;
    let mut f = std::fs::File::open(CORPUS)
        .unwrap_or_else(|e| panic!("コーパスが開けない ({}): {}", CORPUS, e));
    let mut buf = vec![0u8; want_bytes];
    let mut filled = 0usize;
    while filled < want_bytes {
        match f.read(&mut buf[filled..]) {
            Ok(0) => break,
            Ok(n) => filled += n,
            Err(e) => panic!("読み込み失敗: {}", e),
        }
    }
    buf.truncate(filled);
    // 末尾の壊れた UTF-8 を落とす
    let text = loop {
        match std::str::from_utf8(&buf) {
            Ok(s) => break s.to_string(),
            Err(e) => { buf.truncate(e.valid_up_to()); }
        }
    };
    let mut kinds: BTreeSet<char> = BTreeSet::new();
    let mut out: Vec<Mora> = Vec::new();
    let mut skipped = 0usize;
    for c in text.chars() {
        if out.len() >= n_moras { break; }
        if c == '\n' || c == ' ' { continue; }
        let (m, sk) = moras_from_kana(&c.to_string());
        skipped += sk;
        if !m.is_empty() { kinds.insert(c); }
        out.extend(m);
    }
    out.truncate(n_moras);
    (out, kinds.len(), skipped)
}

struct Snapshot { mora: usize, alive: usize, open: usize, fired: usize }

/// 連続で流す。**一度もリセットしない。**
fn run(moras: &[Mora], checkpoint_every: usize) -> Vec<Snapshot> {
    let mut net = ThermoNetwork::new(ThermoNetworkConfig::for_m1_cn_80());
    let mut co = Cochlea::new();
    let mut cn = CochlearNucleus::new();
    let mut noise = LfsrNoise::new(SEED);
    let mut snaps = Vec::new();
    let mut fired_total = 0usize;
    let mut prev_alive = net.n_open_synapses();
    snaps.push(Snapshot { mora: 0, alive: prev_alive, open: open_by_conductance(&net), fired: 0 });
    for (i, m) in moras.iter().enumerate() {
        let w = synth_utterance(std::slice::from_ref(m), F0, &mut noise);
        for chunk in w.chunks(SAMPLES_PER_STEP) {
            if chunk.len() < SAMPLES_PER_STEP { break; }
            let m0 = co.process_step(chunk);
            let cn_out = cn.process_step(&m0);
            fired_total += net.step(&cn_out).len();
        }
        let alive = net.n_open_synapses();
        // **刈込の開始**: alive が初めて減ったモーラを 1 回だけ記録する
        if alive < prev_alive && !snaps.iter().any(|s| s.mora == usize::MAX) {
            snaps.push(Snapshot { mora: usize::MAX, alive: i + 1, open: prev_alive - alive, fired: 0 });
        }
        prev_alive = alive;
        if (i + 1) % checkpoint_every == 0 {
            snaps.push(Snapshot {
                mora: i + 1, alive,
                open: open_by_conductance(&net), fired: fired_total,
            });
        }
    }
    snaps
}

fn open_by_conductance(net: &ThermoNetwork) -> usize {
    net.synapses.iter().filter(|s| s.alive && s.conductance >= OPEN_THRESHOLD).count()
}

fn main() {
    let n_moras: usize = std::env::var("DRPNN_CORPUS_MORAS").ok()
        .and_then(|v| v.parse().ok()).unwrap_or(12000);

    println!("=== 実コーパスを音として聞かせる ===");
    println!();
    println!("【なぜ】ユーザー指示「多様性を増やして単語や熟語の音も聞かせたい」");
    println!("「刈込や新造が実物と同じスケールで起こるように、刺激も同じスケールに」。");
    println!("§14.20 のかな化コーパス (1億かな・75種・逐次構造 0.613bit) を");
    println!("**連続で・一度もリセットせず** M0 -> M0.5 -> M1 に流す。");
    println!();
    println!("【刈込の算術】**予測はここから出る。当てるためでなく私のコード理解の検算。**");
    println!("  VITALITY_INITIAL 100 × VITALITY_DECAY_INTERVAL 10,000 step");
    println!("  ÷ 1モーラ {} step  =  **4,166.7 モーラ**", STEPS_PER_MORA);
    println!();
    println!("【ゲート・実測前に固定】正解の出どころ = 刈込開始時刻は上の算術が決めている");
    println!("  **G89a 刈込は算術どおり 4,100〜4,300 モーラで始まるか**");
    println!("  G89b 平衡に達するか (記述) / G89c 多様性 (記述)");
    println!("  G89d 決定論性 / **G89e コーパスの内容は一切出力しない (数値のみ)**");
    println!();
    println!("【この測定の既知の欠陥・実測前に書いた】");
    println!("  **M1 の入力数が現在の M0.5 と噛み合っていない。**");
    println!("  for_m1_cn_80 は入力 164 だが CochlearNucleus の出力は 84 (4+40+40)。");
    println!("  **80 本の入力ニューロンが一度も駆動されない。**");
    println!("  (44=4+20+20 は 20帯域時代・164=4+80+80 は 80帯域時代。40帯域の 84 に合う");
    println!("   設定が存在しない。**パラメータを発明しないので噛み合わせない。**)");
    println!("  したがって **刈込の「量」は配線ミスマッチに汚染される。**");
    println!("  **G89a が見るのは「いつ」であって「いくつ」ではないので頑健。量は判定に使わない。**");

    let t0 = std::time::Instant::now();
    let (moras, kinds, skipped) = load_moras(n_moras);
    println!();
    println!("  読み込み: {} モーラ / **{} 種類のかなが鳴った** / 未対応で捨てた {} 文字",
             moras.len(), kinds, skipped);
    println!("  (§14.20 の全体は 75 種。**内容は出力しない。**)");

    let ckpt = (n_moras / 12).max(1);
    let snaps = run(&moras, ckpt);
    let elapsed = t0.elapsed().as_secs_f64();

    // --- G89a ---
    println!();
    let onset = snaps.iter().find(|s| s.mora == usize::MAX);
    match onset {
        Some(s) => {
            let ok = (4100..=4300).contains(&s.alive);
            println!("  **G89a 刈込の開始 -> {} モーラ目 ({} 本が同時に死んだ)**", s.alive, s.open);
            println!("  算術の予測 4,166.7 (許容 4,100〜4,300) -> {}",
                     if ok { "**PASS — コード理解は正しい**" } else { "**FAIL — 私のコード理解が間違っている**" });
        }
        None => println!("  **G89a 刈込は {} モーラまでに一度も起きなかった -> FAIL**", moras.len()),
    }

    // --- G89b / 軌跡 ---
    println!();
    println!("  {:>10} {:>12} {:>12} {:>14} {:>12}", "モーラ", "alive", "Δalive", "open(伝導度)", "M1発火累計");
    let mut prev: Option<&Snapshot> = None;
    for s in snaps.iter().filter(|s| s.mora != usize::MAX) {
        let d = prev.map(|p| s.alive as i64 - p.alive as i64).unwrap_or(0);
        println!("  {:>10} {:>12} {:>+12} {:>14} {:>12}", s.mora, s.alive, d, s.open, s.fired);
        prev = Some(s);
    }
    let ds: Vec<i64> = snaps.iter().filter(|s| s.mora != usize::MAX).collect::<Vec<_>>()
        .windows(2).map(|w| (w[1].alive as i64 - w[0].alive as i64).abs()).collect();
    if ds.len() >= 4 {
        let h = ds.len() / 2;
        let (a, b): (i64, i64) = (ds[..h].iter().sum(), ds[h..].iter().sum());
        println!();
        println!("  G89b |Δalive| の合計: 前半 {} / 後半 {} -> {}", a, b,
                 if b < a { "**後半の方が小さい = 平衡に向かっている**" }
                 else { "**後半の方が大きい = まだ平衡でない**" });
    }

    // --- G89d ---
    let small: Vec<Mora> = moras.iter().take(500).cloned().collect();
    let r1 = run(&small, 500);
    let r2 = run(&small, 500);
    println!();
    println!("  G89d 決定論性 (先頭500モーラ×2回) -> {}",
             if r1.last().map(|s| (s.alive, s.open, s.fired)) == r2.last().map(|s| (s.alive, s.open, s.fired))
             { "PASS" } else { "**FAIL**" });
    println!("  G89e コーパスの内容 -> **一切出力していない (数値のみ)**");
    println!();
    println!("  所要 {:.1} 秒 ({:.1} モーラ/秒)。1億モーラ全体なら {:.0} 日。",
             elapsed, moras.len() as f64 / elapsed,
             100_461_249.0 / (moras.len() as f64 / elapsed) / 86400.0);
}
