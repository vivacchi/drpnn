//! シナプスは動的平衡に至るのか、端に張り付いて飽和するのか (2026-08-26)
//!
//! ## なぜこれを測るのか
//!
//! このプロジェクトの学習の定義は **「シナプスが動的平衡に至ること」** である
//! (6 原理の 6 番目「平衡としての学習」)。
//! **平衡は学習の終わりを知る目印ではなく、学習そのものの定義である。**
//!
//! だとすれば観測すべきは正答率ではなく **シナプスの状態** である。
//!
//! ## 疑い — 平衡ではなく飽和ではないか
//!
//! `thermo_synapse.rs` の定数を読むと:
//!
//! - `conductance` は 0..=100、LTP は +5、LTD は −6、**両端でクランプされる**
//!   (:116 と :127)。つまり **20 回の LTP で上端、17 回の LTD で下端**に届く。
//! - 自然減衰は 1000 step (500ms) ごとに −1。1 モーラは 240 step (120ms) なので、
//!   **提示 1 回あたり 0.24 しか戻らない**。
//! - `vitality` は 0..=200、通過ごと +1、10000 step (5s) ごとに −1。やはり両端クランプ。
//!
//! **端に張り付いた状態は動的平衡ではない。** 流れが釣り合っているのではなく、
//! クランプが押さえつけているだけである。散逸構造は形成されていない。
//!
//! ## 正解の出どころ
//!
//! 何回提示したかは**実験者が決めた**。端点 (0 と `CONDUCTANCE_MAX`) は
//! **実験者が定めた定数**である。あとは値を読んで数えるだけで、判断機構は入らない。
//!
//! ## ゲート (実測前に固定・以後動かさない)
//!
//! - **G71a 飽和か平衡か**: 最終チェックポイントで、`alive` なシナプスのうち
//!   `conductance` が端点 (0 または 100) にいる割合が **50% を超えたら飽和**と呼ぶ。
//!   *帰無 = 動的平衡なら流れは内点で釣り合うので、端点占有率は低いはず。*
//! - **G71b 定常に達するか**: 連続チェックポイント間のヒストグラムの全変動距離が、
//!   **最後の 2 点で最初の 2 点の 1/10 未満**なら定常に達したと言う。
//!   (絶対値の閾値を置くと後から動かしたくなるので、相対で置く)
//! - **G71c 決定論性**: 2 回実行して最終ヒストグラムのハッシュが一致する。
//! - **G71d 端点の内訳**: 上端 (100) と下端 (0) のどちらに寄るか。
//!   *これは判定ではなく記述である。ゲートを置かない。*
//!
//! ## 予測 (結果を見る前に固定)
//!
//! **飽和すると予測する。** 根拠は上の定数計算 (20 イベントで端・戻りは 0.24/提示)。
//!
//! **数値の予測は置かない。** §14.6.4 と §14.7 で数値予測を 2 回連続で外し
//! (20-40% に対し 5.4%、30-60% に対し 4.9%)、どちらも「母音 × 子音の掛け算」という
//! 同じ型の推論だった。**その型を使うのをやめると決めたので、構造の予測だけ置く。**
//!
//! ## この測定が答えないこと
//!
//! - **入力の統計を変えたら平衡が変わるか** (= B) は測らない。
//!   飽和していたら B の前提が崩れるので、A が先。
//! - **平衡が「良い」かどうか**も測らない。良し悪しは外部の物差しであり、
//!   ここで測るのは系がどういう状態に落ち着くかだけ。
//!
//! CLI: synapse_equilibrium

use spiking_brain::phase2_f::cochlea::{Cochlea, SAMPLES_PER_STEP};
use spiking_brain::phase2_f::cochlear_nucleus::{CochlearNucleus, N_CN_OUTPUT};
use spiking_brain::phase2_f::kana::{moras_from_kana, synth_utterance};
use spiking_brain::phase2_f::phoneme_synth::LfsrNoise;
use spiking_brain::phase2_f::thermo_network::{ThermoNetwork, ThermoNetworkConfig};
use spiking_brain::phase2_f::thermo_synapse::{CONDUCTANCE_MAX, OPEN_THRESHOLD, VITALITY_MAX};

const KANA: &[&str] = &[
    "あ","い","う","え","お","か","き","く","け","こ","さ","し","す","せ","そ",
    "た","ち","つ","て","と","な","に","ぬ","ね","の","は","ひ","ふ","へ","ほ",
    "ま","み","む","め","も","や","ゆ","よ","ら","り","る","れ","ろ","わ","を","ん",
];

const F0: f64 = 150.0;
const SEED: u16 = 0xACE1;
const ORDER_SEED: u64 = 0xA5A5_1234_5678_9ABC;
const CHECKPOINTS: [usize; 10] = [0, 1, 2, 5, 10, 20, 50, 100, 200, 500];
/// conductance のビン: [0] / 1-11 / 12-22 / ... / 89-99 / [100] = 11 ビン
const N_BINS: usize = 11;

fn wave_of(text: &str) -> Vec<i32> {
    let mut noise = LfsrNoise::new(SEED);
    let (moras, skipped) = moras_from_kana(text);
    assert_eq!(skipped, 0, "未対応のかな: {}", text);
    synth_utterance(&moras, F0, &mut noise)
}

/// 1 提示: M0 → M0.5 → M1
fn present(net: &mut ThermoNetwork, co: &mut Cochlea, cn: &mut CochlearNucleus, wave: &[i32]) {
    net.reset_trial_state();
    co.reset();
    cn.reset();
    for chunk in wave.chunks(SAMPLES_PER_STEP) {
        if chunk.len() < SAMPLES_PER_STEP { break; }
        let m0 = co.process_step(chunk);
        let cn_out = cn.process_step(&m0);
        let _ = net.step(&cn_out);
    }
}

/// conductance を 11 ビンに落とす。ビン 0 = ちょうど 0、ビン 10 = ちょうど MAX。
fn bin_conductance(c: i32) -> usize {
    if c <= 0 { 0 }
    else if c >= CONDUCTANCE_MAX { N_BINS - 1 }
    else { 1 + ((c - 1) as usize * (N_BINS - 2) / (CONDUCTANCE_MAX as usize - 1)).min(N_BINS - 3) }
}

struct Snapshot {
    trial: usize,
    n_alive: usize,
    n_open: usize,
    hist: [f64; N_BINS],
    at_zero: usize,
    at_max: usize,
    vit_at_zero: usize,
    vit_at_max: usize,
    mean_cond: f64,
}

fn snapshot(net: &ThermoNetwork, trial: usize) -> Snapshot {
    let alive: Vec<&spiking_brain::phase2_f::thermo_synapse::ThermoSynapse> =
        net.synapses.iter().filter(|s| s.alive).collect();
    let n = alive.len().max(1);
    let mut counts = [0usize; N_BINS];
    let mut sum = 0f64;
    let (mut z, mut m) = (0usize, 0usize);
    let (mut vz, mut vm) = (0usize, 0usize);
    for s in alive.iter() {
        counts[bin_conductance(s.conductance)] += 1;
        sum += s.conductance as f64;
        if s.conductance <= 0 { z += 1; }
        if s.conductance >= CONDUCTANCE_MAX { m += 1; }
        if s.vitality <= 0 { vz += 1; }
        if s.vitality >= VITALITY_MAX { vm += 1; }
    }
    let mut hist = [0f64; N_BINS];
    for i in 0..N_BINS { hist[i] = counts[i] as f64 / n as f64; }
    Snapshot {
        trial,
        n_alive: alive.len(),
        n_open: alive.iter().filter(|s| s.conductance >= OPEN_THRESHOLD).count(),
        hist,
        at_zero: z,
        at_max: m,
        vit_at_zero: vz,
        vit_at_max: vm,
        mean_cond: sum / n as f64,
    }
}

fn tv(a: &[f64; N_BINS], b: &[f64; N_BINS]) -> f64 {
    0.5 * (0..N_BINS).map(|i| (a[i] - b[i]).abs()).sum::<f64>()
}

fn fnv(snaps: &[Snapshot]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for s in snaps {
        for v in [s.n_alive as u64, s.n_open as u64, s.at_zero as u64, s.at_max as u64,
                  s.vit_at_zero as u64, s.vit_at_max as u64].iter() {
            for b in v.to_le_bytes().iter() { h ^= *b as u64; h = h.wrapping_mul(0x100000001b3); }
        }
    }
    h
}

fn run(label: &str) -> Vec<Snapshot> {
    let waves: Vec<Vec<i32>> = KANA.iter().map(|k| wave_of(k)).collect();
    let cfg = if N_CN_OUTPUT == 164 {
        ThermoNetworkConfig::for_m1_cn_80()
    } else {
        ThermoNetworkConfig::for_m1_cn_40()
    };
    let mut net = ThermoNetwork::new(cfg);
    let mut co = Cochlea::new();
    let mut cn = CochlearNucleus::new();

    println!("  [{}] M1: ニューロン {} ・ シナプス {} ・ 入力 {} ・ 出力 {}",
             label, net.n_neurons(), net.n_synapses(),
             net.input_neurons.len(), net.output_neurons.len());

    let mut snaps = vec![snapshot(&net, 0)];
    let mut order = ORDER_SEED;
    let last = *CHECKPOINTS.last().unwrap();
    for trial in 1..=last {
        order = order.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let si = ((order >> 33) as usize) % KANA.len();
        present(&mut net, &mut co, &mut cn, &waves[si]);
        if CHECKPOINTS.contains(&trial) { snaps.push(snapshot(&net, trial)); }
    }
    println!("  [{}] 軸索 成長 {} 刈り取り {}", label, net.axons_grown, net.axons_pruned);
    snaps
}

fn main() {
    println!("=== シナプスは動的平衡に至るのか、飽和するのか ===");
    println!();
    println!("学習の定義 = シナプスが動的平衡に至ること (6原理の6番目)。");
    println!("よって観測するのは正答率ではなく **シナプスの状態** である。");
    println!();
    println!("定数から予想される挙動: conductance 0..={} ・ LTP +5 ・ LTD -6 ・両端クランプ",
             CONDUCTANCE_MAX);
    println!("  -> 20 回の LTP で上端 / 17 回の LTD で下端。自然減衰は提示1回あたり 0.24。");
    println!("  **予測: 飽和する。数値の予測は置かない (数値予測を2回連続で外したため)。**");
    println!();
    println!("入力: {} かな ・ F0 {:.0}Hz ・ 決定論的な順序 ・ 最大 {} 提示",
             KANA.len(), F0, CHECKPOINTS.last().unwrap());

    println!();
    let snaps = run("1回目");

    println!();
    println!("--- conductance の分布 (alive なシナプスに対する割合) ---");
    println!("  提示     alive   open  平均   [==0]  内点の分布 (低→高)                      [==100]  端点占有");
    for s in snaps.iter() {
        let inner: String = (1..N_BINS - 1).map(|i| {
            let v = (s.hist[i] * 100.0).round() as i32;
            format!("{:>4}", v)
        }).collect::<Vec<_>>().join("");
        let edge = (s.at_zero + s.at_max) as f64 / s.n_alive.max(1) as f64 * 100.0;
        println!("  {:>5} {:>7} {:>6} {:>5.1}  {:>5.1}% {} {:>6.1}%  {:>6.1}%",
                 s.trial, s.n_alive, s.n_open, s.mean_cond,
                 s.hist[0] * 100.0, inner, s.hist[N_BINS - 1] * 100.0, edge);
    }

    println!();
    println!("--- vitality の端点占有 ---");
    println!("  提示     [==0]     [=={}]", VITALITY_MAX);
    for s in snaps.iter() {
        println!("  {:>5} {:>7.1}% {:>9.1}%",
                 s.trial,
                 s.vit_at_zero as f64 / s.n_alive.max(1) as f64 * 100.0,
                 s.vit_at_max as f64 / s.n_alive.max(1) as f64 * 100.0);
    }

    println!();
    println!("--- 定常性: 連続チェックポイント間の全変動距離 ---");
    let mut tvs: Vec<(usize, usize, f64)> = Vec::new();
    for i in 1..snaps.len() {
        let d = tv(&snaps[i - 1].hist, &snaps[i].hist);
        tvs.push((snaps[i - 1].trial, snaps[i].trial, d));
        println!("  {:>4} -> {:>4} : {:.4}", snaps[i - 1].trial, snaps[i].trial, d);
    }

    println!();
    println!("--- G71c 決定論性 ---");
    let snaps2 = run("2回目");
    let (h1, h2) = (fnv(&snaps), fnv(&snaps2));
    println!("  ハッシュ 1回目 {:016x} / 2回目 {:016x} -> {}",
             h1, h2, if h1 == h2 { "一致 PASS" } else { "**不一致 FAIL**" });

    // --- 判定 ---
    let f = snaps.last().unwrap();
    let edge = (f.at_zero + f.at_max) as f64 / f.n_alive.max(1) as f64 * 100.0;
    let first_tv = tvs.first().map(|t| t.2).unwrap_or(f64::NAN);
    let last_tv = tvs.last().map(|t| t.2).unwrap_or(f64::NAN);
    println!();
    println!("=== 判定 (ゲートは実測前に固定・動かさない) ===");
    println!("  G71a 端点占有率 {:.1}% (上端 {:.1}% / 下端 {:.1}%) -> **{}**",
             edge,
             f.at_max as f64 / f.n_alive.max(1) as f64 * 100.0,
             f.at_zero as f64 / f.n_alive.max(1) as f64 * 100.0,
             if edge > 50.0 { "飽和" } else { "飽和ではない" });
    println!("  G71b 定常性: 最初の TV {:.4} → 最後の TV {:.4} (比 {:.3}) -> **{}**",
             first_tv, last_tv, last_tv / first_tv,
             if last_tv < first_tv / 10.0 { "定常に達した" } else { "定常に達していない" });
    println!("  G71c 決定論性 -> {}", if h1 == h2 { "PASS" } else { "**FAIL**" });
    println!();
    println!("  G71d 端点の内訳は上の表のとおり (記述であって判定ではない)。");
    println!();
    println!("  【この測定が答えないこと】入力の統計を変えたら平衡が変わるか (=B) は測っていない。");
    println!("  平衡が『良い』かどうかも測っていない。良し悪しは外部の物差しである。");
}
