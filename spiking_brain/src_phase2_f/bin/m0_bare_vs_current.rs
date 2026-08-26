//! 素の M0 と現行 M0 で、シナプスの刈り取りと動的平衡がどう変わるか (2026-08-26)
//!
//! ## 経緯
//!
//! §14.8 (A) は 500 提示で「軸索成長 1402・**刈り取り 0**」を出した。
//! ユーザーの読み: **素の M0 なら刈り取りが起きて動的平衡に到達するはず。**
//!
//! ## その前に — A の窓が短すぎた (訂正)
//!
//! `VITALITY_INITIAL = 100`、`VITALITY_DECAY_INTERVAL = 10000` step。
//! 1 提示は 240 step なので、**500 提示では vitality は最大 12 しか減らない**。
//! 0 に達するには **1,000,000 step = 4,166 提示**要る。
//!
//! **A で刈り取りが 0 だったのは M0 の密度のせいではない。単に窓が短すぎた。**
//! 「全シナプスが伝送し続けて +1 を受け取っていたから」という説明は**誤り**で、
//! 伝送がゼロでも 500 提示では 100 → 88 にしかならない。
//!
//! よってここでは **6000 提示**まで伸ばす (刈り取りの開始点 4166 を跨ぐ)。
//! **ゲートの規則は §14.8 のまま動かさない。窓だけを機構に合わせて伸ばす。**
//!
//! ## 素の M0 とは何か
//!
//! git で確定: このセッション開始前の最後の cochlea.rs は **`7c9be36` (2026-05-31)**。
//! それ以降の変更はすべて 2026-08-25 以降 = 今回のセッション。
//!
//! | | 素 (`7c9be36`) | 現行 |
//! |---|---|---|
//! | Q | `erb_q_factor(fc)` (×1.0) | ×0.5 (`Q_SHARPENING`) |
//! | 発火閾値 | **200** | 120 |
//! | 出力段 | 閾値+不応期のみ (最大 400Hz) | 漏れ積分発火 `spike_cost=480` (最大 1853Hz) |
//! | biquad の丸め | `acc >> 15` (自己発振あり) | ゼロ方向切り捨て (自己発振 0) |
//! | biquad の状態 | Q15 | **Q8 高精度 (`STATE_SHIFT=8`)** |
//!
//! **`STATE_SHIFT` は compile-time なので戻せない。** これだけは現行のまま。
//! 高精度側は低域の量子化損失が無いぶん**素より密**になるので、
//! 刈り取りの有無について**保守側 (起きにくい側) に倒れる**。
//!
//! ## ゲート (実測前に固定・§14.8 の規則をそのまま使う)
//!
//! - **G73a 刈り取りの発生**: 素の M0 で `axons_pruned > 0`。
//!   *これはユーザーの予測である。当たったか外れたかを記録する。*
//! - **G73b 平衡到達**: 連続チェックポイント間の全変動距離が、
//!   **最後の 2 点で最初の 2 点の 1/10 未満** (§14.8 の G71b と同一)。
//! - **G73c 端点占有率**: `conductance` が端点にいる割合が **50% 超で飽和**
//!   (§14.8 の G71a と同一)。
//! - **G73d 決定論性**: 短い窓 (500 提示) で 2 回実行してハッシュ一致。
//!
//! ## 予測
//!
//! **数値の予測は置かない** (§14.6.4 / §14.7 / §14.9.7 で 3 回連続で外したため)。
//! 構造の予測だけ置く: **素の M0 のほうが M0 出力のスパイクが疎いので、
//! 伝送しないシナプスが増え、刈り取りは素のほうが多いはず。**
//!
//! ## A の計器欠陥を直した点
//!
//! §14.8.7 で自認した「`==MAX` は減衰 1 tick で外れる脆い統計」を受けて、
//! **vitality は端点だけでなく分布全体を出す。**
//!
//! CLI: m0_bare_vs_current

use spiking_brain::phase2_f::cochlea::{
    erb_q_factor, erb_spaced_freqs, BandpassBiquad, Cochlea, EnvelopeDetector, FireGenerator,
    ENV_LEAK_SHIFT, F_MAX_HZ, F_MIN_HZ, FIRE_REFRACTORY_STEPS, N_BANDS, SAMPLES_PER_STEP,
    SAMPLE_RATE_HZ,
};
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

/// 刈り取りの開始点は 4166 提示 (VITALITY_INITIAL=100 × 10000step ÷ 240step)。
/// それを跨ぐところまで伸ばす。
const CHECKPOINTS: [usize; 14] =
    [0, 1, 10, 50, 100, 500, 1000, 2000, 3000, 4000, 4200, 4500, 5000, 6000];
const DETERMINISM_TRIALS: usize = 500;

const N_BINS: usize = 11;

fn wave_of(text: &str) -> Vec<i32> {
    let mut noise = LfsrNoise::new(SEED);
    let (moras, skipped) = moras_from_kana(text);
    assert_eq!(skipped, 0, "未対応のかな: {}", text);
    synth_utterance(&moras, F0, &mut noise)
}

/// 素の M0 (`7c9be36` 相当)。`STATE_SHIFT` だけは compile-time なので現行のまま。
fn cochlea_bare() -> Cochlea {
    let center_freqs = erb_spaced_freqs(F_MIN_HZ, F_MAX_HZ, N_BANDS);
    let bands: Vec<BandpassBiquad> = center_freqs
        .iter()
        .map(|&fc| {
            let mut b = BandpassBiquad::new(fc, erb_q_factor(fc), SAMPLE_RATE_HZ);
            b.magnitude_truncation = false; // 素は acc >> 15 (自己発振あり)
            b
        })
        .collect();
    let envelopes = (0..N_BANDS).map(|_| EnvelopeDetector::new(ENV_LEAK_SHIFT)).collect();
    let fire_gens = (0..N_BANDS)
        .map(|_| {
            let mut f = FireGenerator::new(200, FIRE_REFRACTORY_STEPS); // 素の閾値 200
            f.spike_cost = 0; // 素は閾値+不応期のみ
            f
        })
        .collect();
    Cochlea { bands, envelopes, fire_gens, center_freqs, ..Cochlea::new() }
}

struct Snap {
    trial: usize,
    n_alive: usize,
    n_open: usize,
    grown: usize,
    pruned: usize,
    hist: [f64; N_BINS],
    at_zero: usize,
    at_max: usize,
    mean_cond: f64,
    vit_hist: [f64; N_BINS],
    vit_min: i32,
    vit_mean: f64,
}

fn bin(v: i32, max: i32) -> usize {
    if v <= 0 { 0 }
    else if v >= max { N_BINS - 1 }
    else { 1 + ((v - 1) as usize * (N_BINS - 2) / (max as usize - 1)).min(N_BINS - 3) }
}

fn snapshot(net: &ThermoNetwork, trial: usize) -> Snap {
    let alive: Vec<_> = net.synapses.iter().filter(|s| s.alive).collect();
    let n = alive.len().max(1);
    let mut c_counts = [0usize; N_BINS];
    let mut v_counts = [0usize; N_BINS];
    let (mut csum, mut vsum) = (0f64, 0f64);
    let (mut z, mut m) = (0usize, 0usize);
    let mut vmin = i32::MAX;
    for s in alive.iter() {
        c_counts[bin(s.conductance, CONDUCTANCE_MAX)] += 1;
        v_counts[bin(s.vitality, VITALITY_MAX)] += 1;
        csum += s.conductance as f64;
        vsum += s.vitality as f64;
        if s.conductance <= 0 { z += 1; }
        if s.conductance >= CONDUCTANCE_MAX { m += 1; }
        vmin = vmin.min(s.vitality);
    }
    let mut hist = [0f64; N_BINS];
    let mut vit_hist = [0f64; N_BINS];
    for i in 0..N_BINS {
        hist[i] = c_counts[i] as f64 / n as f64;
        vit_hist[i] = v_counts[i] as f64 / n as f64;
    }
    Snap {
        trial,
        n_alive: alive.len(),
        n_open: alive.iter().filter(|s| s.conductance >= OPEN_THRESHOLD).count(),
        grown: net.axons_grown as usize,
        pruned: net.axons_pruned as usize,
        hist,
        at_zero: z,
        at_max: m,
        mean_cond: csum / n as f64,
        vit_hist,
        vit_min: if vmin == i32::MAX { 0 } else { vmin },
        vit_mean: vsum / n as f64,
    }
}

fn tv(a: &[f64; N_BINS], b: &[f64; N_BINS]) -> f64 {
    0.5 * (0..N_BINS).map(|i| (a[i] - b[i]).abs()).sum::<f64>()
}

fn fnv(s: &[Snap]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for x in s {
        for v in [x.n_alive as u64, x.n_open as u64, x.at_zero as u64,
                  x.at_max as u64, x.grown as u64, x.pruned as u64].iter() {
            for b in v.to_le_bytes().iter() { h ^= *b as u64; h = h.wrapping_mul(0x100000001b3); }
        }
    }
    h
}

/// 1 アーム分の走行。`bare=true` で素の M0。
/// 返り値: (スナップショット列, M0 スパイク/提示, M0.5 スパイク/提示)
fn run(bare: bool, n_trials: usize, label: &str) -> (Vec<Snap>, f64, f64) {
    let waves: Vec<Vec<i32>> = KANA.iter().map(|k| wave_of(k)).collect();
    let cfg = if N_CN_OUTPUT == 164 {
        ThermoNetworkConfig::for_m1_cn_80()
    } else {
        ThermoNetworkConfig::for_m1_cn_40()
    };
    let mut net = ThermoNetwork::new(cfg);
    let mut co = if bare { cochlea_bare() } else { Cochlea::new() };
    let mut cn = CochlearNucleus::new();

    // --- 入力の密度を先に測る (機構を見るため) ---
    let (mut m0_spikes, mut m05_spikes) = (0u64, 0u64);
    for w in waves.iter() {
        let mut c2 = if bare { cochlea_bare() } else { Cochlea::new() };
        let mut n2 = CochlearNucleus::new();
        for chunk in w.chunks(SAMPLES_PER_STEP) {
            if chunk.len() < SAMPLES_PER_STEP { break; }
            let m0 = c2.process_step(chunk);
            m0_spikes += m0.iter().filter(|&&v| v != 0).count() as u64;
            m05_spikes += n2.process_step(&m0).iter().filter(|&&v| v != 0).count() as u64;
        }
    }
    let d0 = m0_spikes as f64 / KANA.len() as f64;
    let d05 = m05_spikes as f64 / KANA.len() as f64;
    println!("  [{}] M0 スパイク {:.0}/提示 ・ M0.5 スパイク {:.0}/提示 ・ シナプス {}",
             label, d0, d05, net.n_synapses());

    let mut snaps = vec![snapshot(&net, 0)];
    let mut order = ORDER_SEED;
    for trial in 1..=n_trials {
        order = order.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let si = ((order >> 33) as usize) % KANA.len();
        net.reset_trial_state();
        co.reset();
        cn.reset();
        for chunk in waves[si].chunks(SAMPLES_PER_STEP) {
            if chunk.len() < SAMPLES_PER_STEP { break; }
            let m0 = co.process_step(chunk);
            let cn_out = cn.process_step(&m0);
            let _ = net.step(&cn_out);
        }
        if CHECKPOINTS.contains(&trial) { snaps.push(snapshot(&net, trial)); }
    }
    (snaps, d0, d05)
}

fn report(label: &str, snaps: &[Snap]) -> (f64, f64, bool) {
    println!();
    println!("--- [{}] conductance の分布 (alive に対する割合) ---", label);
    println!("  提示     alive    open   刈取  平均  [==0]  内点 (低→高)                    [==MAX]  端点占有");
    for s in snaps.iter() {
        let inner: String = (1..N_BINS - 1)
            .map(|i| format!("{:>4}", (s.hist[i] * 100.0).round() as i32))
            .collect::<Vec<_>>().join("");
        let edge = (s.at_zero + s.at_max) as f64 / s.n_alive.max(1) as f64 * 100.0;
        println!("  {:>5} {:>7} {:>7} {:>6} {:>5.1} {:>5.1}% {} {:>6.1}% {:>7.1}%",
                 s.trial, s.n_alive, s.n_open, s.pruned, s.mean_cond,
                 s.hist[0] * 100.0, inner, s.hist[N_BINS - 1] * 100.0, edge);
    }

    println!();
    println!("--- [{}] vitality の分布 (端点だけでなく全体・A の脆い統計を修正) ---", label);
    println!("  提示   最小  平均   [==0]  内点 (低→高)                    [=={}]", VITALITY_MAX);
    for s in snaps.iter() {
        let inner: String = (1..N_BINS - 1)
            .map(|i| format!("{:>4}", (s.vit_hist[i] * 100.0).round() as i32))
            .collect::<Vec<_>>().join("");
        println!("  {:>5} {:>5} {:>5.1} {:>6.1}% {} {:>6.1}%",
                 s.trial, s.vit_min, s.vit_mean, s.vit_hist[0] * 100.0, inner,
                 s.vit_hist[N_BINS - 1] * 100.0);
    }

    println!();
    println!("--- [{}] 定常性: 連続チェックポイント間の全変動距離 ---", label);
    let mut tvs = Vec::new();
    for i in 1..snaps.len() {
        let d = tv(&snaps[i - 1].hist, &snaps[i].hist);
        tvs.push(d);
        println!("  {:>5} -> {:>5} : {:.4}", snaps[i - 1].trial, snaps[i].trial, d);
    }
    let first = *tvs.first().unwrap_or(&f64::NAN);
    let last = *tvs.last().unwrap_or(&f64::NAN);
    let f = snaps.last().unwrap();
    let edge = (f.at_zero + f.at_max) as f64 / f.n_alive.max(1) as f64 * 100.0;
    (edge, last / first, f.pruned > 0)
}

fn main() {
    println!("=== 素の M0 と現行 M0 — 刈り取りと動的平衡 ===");
    println!();
    println!("【A の訂正】§14.8 で刈り取り 0 だったのは M0 の密度のせいではない。");
    println!("VITALITY_INITIAL=100 ・ 減衰 -1/10000step ・ 1提示=240step なので、");
    println!("**500 提示では vitality は最大 12 しか減らない**。0 に達するには 4,166 提示要る。");
    println!("『全シナプスが伝送し続けて +1 を受け取っていたから』という説明は誤りだった。");
    println!("よって窓を {} 提示まで伸ばす。**ゲートの規則は §14.8 のまま動かさない。**",
             CHECKPOINTS.last().unwrap());
    println!();
    println!("【素の M0 = git 7c9be36 (2026-05-31)】Q×1.0 / 閾値200 / 閾値+不応期のみ /");
    println!("biquad は acc>>15 (自己発振あり)。**STATE_SHIFT だけ compile-time で戻せない**");
    println!("(高精度側は素より密になるので、刈り取りについて保守側に倒れる)。");
    println!();
    println!("【予測】数値は置かない (3回連続で外したため)。構造だけ:");
    println!("  素の M0 のほうが M0 出力が疎いので、刈り取りは素のほうが多いはず。");

    println!();
    println!("--- 走行 ---");
    let n = *CHECKPOINTS.last().unwrap();
    let (cur, cd0, cd05) = run(false, n, "現行");
    let (bare, bd0, bd05) = run(true, n, "素");

    println!();
    println!("=== 入力の密度 ===");
    println!("  現行: M0 {:.0} / M0.5 {:.0}  (スパイク/提示)", cd0, cd05);
    println!("  素  : M0 {:.0} / M0.5 {:.0}  → 素は現行の {:.2} 倍 (M0)", bd0, bd05, bd0 / cd0);

    let (cur_edge, cur_ratio, cur_pruned) = report("現行", &cur);
    let (bare_edge, bare_ratio, bare_pruned) = report("素", &bare);

    println!();
    println!("--- G73d 決定論性 (短い窓 {} 提示で 2 回) ---", DETERMINISM_TRIALS);
    let (a, _, _) = run(true, DETERMINISM_TRIALS, "素-det1");
    let (b, _, _) = run(true, DETERMINISM_TRIALS, "素-det2");
    let (ha, hb) = (fnv(&a), fnv(&b));
    println!("  ハッシュ {:016x} / {:016x} -> {}",
             ha, hb, if ha == hb { "一致 PASS" } else { "**不一致 FAIL**" });

    println!();
    println!("=== 判定 (規則は §14.8 のまま・実測前に固定) ===");
    println!("  {:<6} {:>10} {:>12} {:>14} {:>12}", "アーム", "刈り取り", "端点占有率", "TV比(要<0.1)", "判定");
    for (name, edge, ratio, pruned, snaps) in
        [("現行", cur_edge, cur_ratio, cur_pruned, &cur), ("素", bare_edge, bare_ratio, bare_pruned, &bare)].iter()
    {
        let last = snaps.last().unwrap();
        println!("  {:<6} {:>10} {:>11.1}% {:>14.3} {:>12}",
                 name, last.pruned, edge, ratio,
                 if *ratio < 0.1 { "平衡に到達" } else { "未到達" });
        let _ = pruned;
    }
    println!();
    println!("  G73a 刈り取りの発生 (素で pruned > 0 ・**ユーザーの予測**) -> {}",
             if bare_pruned { "**発生した = 予測は当たり**" } else { "**発生しなかった = 予測は外れ**" });
    println!("  G73b 平衡到達 (TV比 < 0.1) -> 現行 {} / 素 {}",
             if cur_ratio < 0.1 { "到達" } else { "未到達" },
             if bare_ratio < 0.1 { "到達" } else { "未到達" });
    println!("  G73c 端点占有率 (>50% で飽和) -> 現行 {:.1}% {} / 素 {:.1}% {}",
             cur_edge, if cur_edge > 50.0 { "飽和" } else { "飽和でない" },
             bare_edge, if bare_edge > 50.0 { "飽和" } else { "飽和でない" });
    println!("  G73d 決定論性 -> {}", if ha == hb { "PASS" } else { "**FAIL**" });
}
