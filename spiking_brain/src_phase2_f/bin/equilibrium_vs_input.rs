//! B: 入力の統計を変えると、シナプスの平衡は変わるか (2026-08-26)
//!
//! ## なぜこれを測るのか
//!
//! 学習の定義は「シナプスが動的平衡に至ること」。§14.10 (G73b) で
//! **平衡に到達することは確認できた** (TV比 現行 0.033 / 素 0.015)。
//! では **その平衡は入力の統計で変わるのか。** 変わるなら、それが「学習の内容」である。
//!
//! ## 記録の捜索結果 — 測られていなかった
//!
//! 5 通りの探し方で捜索した結果:
//!
//! - **PAPER_DRAFT の 3 箇所 (§5.9.6・§5.14.7・§5.11.7) に
//!   「動的平衡点は入力統計で決まる」と書かれている。**
//! - **根拠として引かれている実測はすべて出力側** (selectivity / active / entropy)。
//!   `conductance` / `vitality` を入力統計間で比べた実測は**一度もない**。
//! - 近いものは 2 種類あり、どちらも半分だけ:
//!   §5.5 は入力統計を 3 種振るが測るのは selectivity。
//!   §5.9.5-6 はシナプス側を比べるが振ったのは vitality 減衰周期。
//! - `PHASE2_INSTRUCTION.md:449` は「シナプスの conductance 分布」を観測項目として
//!   指示しているが、**分布としては一度も実行されていない** (既存 CSV は mean/max/std のみ)。
//!
//! ## 自発発火はいまどこにも無い (2026-08-26 に判明)
//!
//! | | 素 (`7c9be36`) | 現行 |
//! |---|---|---|
//! | M0 蝸牛 | 無し (「自発発火は M0 内では生成しない」) | `spontaneous_amplitude: 0` 既定 OFF |
//! | M1 非入力ニューロン | `idx % 4` | `idx % 4` (同じ) |
//! | **M1 入力ニューロン** | **`idx % 4`** | **0** |
//!
//! 入力層の自発発火は `d918cad` (2026-05-24)
//! 「**仮想 M0 等価性の発見: M1 input spont=2 + 固定パターンで POST=0.795 (過去最高)**」
//! で意図的に入ったものだった。ユーザーの「これは仮想 M0 を実装したことに相当するのでは」
//! という洞察から生まれ、**過去最高の結果を出している。**
//!
//! **私は 2026-08-26 の ③ でこれを「死んだガード」として消した** (`22467fb`)。
//! G67b は FAIL (1/3) だったが、原理 (入力ニューロンは受信専用) を根拠に残した。
//! **その判断は、この由来を知らずに下したものである。**
//!
//! コードのコメントは「自発発火は M0 蝸牛が担当する設計に」と言うが、
//! **その M0 側は既定 OFF なので、いま担当がどちらにも無い。**
//!
//! よってここでは **素の M1 (入力層の自発発火あり)** を構成の一つとして測る。
//!
//! ## ゲート (実測前に固定・以後動かさない)
//!
//! **正解の出どころ: どの入力を与えたかは実験者が決めた。**
//!
//! - **G74a 平衡到達**: 各アームで TV比 (最後の TV / 最初の TV) < 0.1。
//!   §14.10 の G73b と**同一規則**。
//! - **G74b 入力依存**: 同一構成内で、**アーム間の conductance 分布の TV 距離**が、
//!   **同一アーム内の後期の揺らぎ (最後 3 区間の TV の最大値) より大きい**。
//!   *帰無 = 平衡は入力統計に依らない → アーム間 TV ≦ アーム内の揺らぎ。*
//!   **閾値を後から置かないために、対照は同じ量 (TV 距離) の中で取る。**
//! - **G74c 決定論性**: 短い窓で 2 回実行してハッシュ一致。
//!
//! ## 予測
//!
//! **数値は置かない** (§14.6.4 / §14.7 / §14.9.7 で 3 回連続、
//! §14.10.4 で構造予測も外した)。構造だけ:
//! **入力依存はあるはず。** 無音アームは自発発火だけで駆動されるので、
//! かな・雑音とは違う平衡になるはず。
//!
//! CLI: equilibrium_vs_input

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
/// 刈り取りの開始点 4166 提示を跨ぐ
const CHECKPOINTS: [usize; 9] = [0, 100, 500, 1000, 2000, 3000, 4000, 4500, 5000];
const DETERMINISM_TRIALS: usize = 300;
const N_BINS: usize = 11;

#[derive(Clone, Copy, PartialEq)]
enum Input { Kana, WhiteNoise, Silence }

impl Input {
    fn name(&self) -> &'static str {
        match self { Input::Kana => "かな46音", Input::WhiteNoise => "白色雑音", Input::Silence => "無音" }
    }
}

fn kana_waves() -> Vec<Vec<i32>> {
    KANA.iter().map(|k| {
        let mut noise = LfsrNoise::new(SEED);
        let (moras, skipped) = moras_from_kana(k);
        assert_eq!(skipped, 0);
        synth_utterance(&moras, F0, &mut noise)
    }).collect()
}

fn rms(w: &[i32]) -> f64 {
    if w.is_empty() { return 0.0; }
    (w.iter().map(|&s| (s as f64) * (s as f64)).sum::<f64>() / w.len() as f64).sqrt()
}

/// 入力を作る。長さと本数は かな に揃える (提示回数・順序を同一にするため)。
/// 白色雑音は かな の平均 RMS に合わせる。
fn make_input(kind: Input) -> Vec<Vec<i32>> {
    let kana = kana_waves();
    match kind {
        Input::Kana => kana,
        Input::Silence => kana.iter().map(|w| vec![0i32; w.len()]).collect(),
        Input::WhiteNoise => {
            let target: f64 = kana.iter().map(|w| rms(w)).sum::<f64>() / kana.len() as f64;
            kana.iter().enumerate().map(|(i, w)| {
                let mut n = LfsrNoise::new((0x1000u32 + (i as u32) * 7919) as u16 | 1);
                let raw: Vec<i32> = (0..w.len()).map(|_| n.next_sample()).collect();
                let r = rms(&raw).max(1.0);
                let g = target / r;
                raw.iter().map(|&s| ((s as f64) * g).round() as i32).collect()
            }).collect()
        }
    }
}

/// 素の M0 (`7c9be36` 相当)。`STATE_SHIFT` だけ compile-time なので現行のまま。
fn cochlea_bare() -> Cochlea {
    let center_freqs = erb_spaced_freqs(F_MIN_HZ, F_MAX_HZ, N_BANDS);
    let bands: Vec<BandpassBiquad> = center_freqs.iter().map(|&fc| {
        let mut b = BandpassBiquad::new(fc, erb_q_factor(fc), SAMPLE_RATE_HZ);
        b.magnitude_truncation = false;
        b
    }).collect();
    let envelopes = (0..N_BANDS).map(|_| EnvelopeDetector::new(ENV_LEAK_SHIFT)).collect();
    let fire_gens = (0..N_BANDS).map(|_| {
        let mut f = FireGenerator::new(200, FIRE_REFRACTORY_STEPS);
        f.spike_cost = 0;
        f
    }).collect();
    Cochlea { bands, envelopes, fire_gens, center_freqs, ..Cochlea::new() }
}

struct Snap {
    trial: usize,
    n_alive: usize,
    n_open: usize,
    pruned: usize,
    hist: [f64; N_BINS],
    at_zero: usize,
    at_max: usize,
    mean_cond: f64,
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
    let mut counts = [0usize; N_BINS];
    let (mut csum, mut vsum) = (0f64, 0f64);
    let (mut z, mut m) = (0usize, 0usize);
    let mut vmin = i32::MAX;
    for s in alive.iter() {
        counts[bin(s.conductance, CONDUCTANCE_MAX)] += 1;
        csum += s.conductance as f64;
        vsum += s.vitality as f64;
        if s.conductance <= 0 { z += 1; }
        if s.conductance >= CONDUCTANCE_MAX { m += 1; }
        vmin = vmin.min(s.vitality);
    }
    let mut hist = [0f64; N_BINS];
    for i in 0..N_BINS { hist[i] = counts[i] as f64 / n as f64; }
    Snap {
        trial, n_alive: alive.len(),
        n_open: alive.iter().filter(|s| s.conductance >= OPEN_THRESHOLD).count(),
        pruned: net.axons_pruned as usize,
        hist, at_zero: z, at_max: m,
        mean_cond: csum / n as f64,
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
                  x.at_max as u64, x.pruned as u64].iter() {
            for b in v.to_le_bytes().iter() { h ^= *b as u64; h = h.wrapping_mul(0x100000001b3); }
        }
    }
    h
}

struct Arm { snaps: Vec<Snap>, m0_density: f64, m05_density: f64 }

fn run(bare: bool, kind: Input, n_trials: usize) -> Arm {
    let waves = make_input(kind);
    let cfg = if N_CN_OUTPUT == 164 {
        ThermoNetworkConfig::for_m1_cn_80()
    } else {
        ThermoNetworkConfig::for_m1_cn_40()
    };
    let mut net = ThermoNetwork::new(cfg);
    // 素の M1 = 入力層にも自発発火 (idx % 4)。d918cad の状態。
    if bare { net.reproduce_broken_input_guard(); }
    let mut co = if bare { cochlea_bare() } else { Cochlea::new() };
    let mut cn = CochlearNucleus::new();

    let (mut s0, mut s05) = (0u64, 0u64);
    for w in waves.iter() {
        let mut c2 = if bare { cochlea_bare() } else { Cochlea::new() };
        let mut n2 = CochlearNucleus::new();
        for chunk in w.chunks(SAMPLES_PER_STEP) {
            if chunk.len() < SAMPLES_PER_STEP { break; }
            let m0 = c2.process_step(chunk);
            s0 += m0.iter().filter(|&&v| v != 0).count() as u64;
            s05 += n2.process_step(&m0).iter().filter(|&&v| v != 0).count() as u64;
        }
    }

    let mut snaps = vec![snapshot(&net, 0)];
    let mut order = ORDER_SEED;
    for trial in 1..=n_trials {
        order = order.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let si = ((order >> 33) as usize) % waves.len();
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
    Arm { snaps, m0_density: s0 as f64 / waves.len() as f64, m05_density: s05 as f64 / waves.len() as f64 }
}

/// 同一アーム内の後期の揺らぎ = 最後 3 区間の TV の最大値
fn late_fluctuation(s: &[Snap]) -> f64 {
    let mut tvs = Vec::new();
    for i in 1..s.len() { tvs.push(tv(&s[i - 1].hist, &s[i].hist)); }
    let k = tvs.len().saturating_sub(3);
    tvs[k..].iter().cloned().fold(0f64, f64::max)
}

fn tv_ratio(s: &[Snap]) -> f64 {
    let mut tvs = Vec::new();
    for i in 1..s.len() { tvs.push(tv(&s[i - 1].hist, &s[i].hist)); }
    *tvs.last().unwrap() / *tvs.first().unwrap()
}

fn main() {
    println!("=== B: 入力の統計を変えると、シナプスの平衡は変わるか ===");
    println!();
    println!("【記録の捜索結果】この測定は**一度も行われていない**。");
    println!("PAPER_DRAFT の 3 箇所に『動的平衡点は入力統計で決まる』と書かれているが、");
    println!("根拠の実測はすべて出力側 (selectivity/active/entropy)。");
    println!("conductance/vitality を入力統計間で比べた実測は無い。");
    println!();
    println!("【自発発火の所在 (2026-08-26 判明)】");
    println!("  M0 蝸牛: 素=無し / 現行=既定 OFF");
    println!("  M1 非入力: 素・現行とも idx%4 (同じ)");
    println!("  M1 入力層: 素=idx%4 / 現行=0  ← 私が③で消した");
    println!("  入力層の自発発火は d918cad『仮想 M0 等価性の発見・POST=0.795 過去最高』で");
    println!("  **意図的に入ったもの**だった。バグではなかった。");
    println!();
    println!("【ゲート・実測前に固定】正解の出どころ = どの入力を与えたかは実験者が決めた");
    println!("  G74a 平衡到達: 各アームで TV比 < 0.1 (§14.10 の G73b と同一規則)");
    println!("  G74b 入力依存: アーム間 TV > 同一アーム内の後期の揺らぎ (最後3区間の最大)");
    println!("       帰無 = 平衡は入力統計に依らない -> アーム間 TV <= アーム内の揺らぎ");
    println!("  G74c 決定論性");
    println!();
    println!("【予測】数値は置かない (3回連続で外し、構造予測も1回外したため)。");
    println!("  構造のみ: 入力依存はあるはず。無音は自発発火だけで駆動されるので違う平衡になるはず。");

    let n = *CHECKPOINTS.last().unwrap();
    let kinds = [Input::Kana, Input::WhiteNoise, Input::Silence];

    for &(bare, cfg_name) in [(true, "素 (M0素 + M1入力層に自発発火あり)"),
                              (false, "現行 (M0現行 + M1入力層の自発発火なし)")].iter() {
        println!();
        println!("################ 構成: {} ################", cfg_name);
        let arms: Vec<Arm> = kinds.iter().map(|&k| run(bare, k, n)).collect();

        println!();
        println!("  入力            M0/提示  M0.5/提示   alive    open   刈取  平均cond  [==0]  vit最小  TV比");
        for (i, &k) in kinds.iter().enumerate() {
            let f = arms[i].snaps.last().unwrap();
            println!("  {:<12} {:>8.0} {:>10.0} {:>7} {:>7} {:>6} {:>9.1} {:>6.1}% {:>7} {:>6.3}",
                     k.name(), arms[i].m0_density, arms[i].m05_density,
                     f.n_alive, f.n_open, f.pruned, f.mean_cond,
                     f.hist[0] * 100.0, f.vit_min, tv_ratio(&arms[i].snaps));
        }

        println!();
        println!("  --- 最終分布 (5000提示) ---");
        println!("  入力          [==0]  内点 (低→高)                    [==MAX]");
        for (i, &k) in kinds.iter().enumerate() {
            let f = arms[i].snaps.last().unwrap();
            let inner: String = (1..N_BINS - 1)
                .map(|j| format!("{:>4}", (f.hist[j] * 100.0).round() as i32))
                .collect::<Vec<_>>().join("");
            println!("  {:<12} {:>5.1}% {} {:>6.1}%", k.name(), f.hist[0] * 100.0, inner, f.hist[N_BINS - 1] * 100.0);
        }

        // --- G74b: アーム間 TV vs アーム内の揺らぎ ---
        println!();
        println!("  --- G74b 入力依存 ---");
        let within: Vec<f64> = arms.iter().map(|a| late_fluctuation(&a.snaps)).collect();
        let within_max = within.iter().cloned().fold(0f64, f64::max);
        for (i, &k) in kinds.iter().enumerate() {
            println!("  アーム内の後期の揺らぎ [{}] = {:.4}", k.name(), within[i]);
        }
        println!("  -> アーム内の揺らぎの最大 = {:.4}", within_max);
        println!();
        let mut all_bigger = true;
        for i in 0..kinds.len() {
            for j in (i + 1)..kinds.len() {
                let d = tv(&arms[i].snaps.last().unwrap().hist, &arms[j].snaps.last().unwrap().hist);
                let ok = d > within_max;
                if !ok { all_bigger = false; }
                println!("  アーム間 TV [{} vs {}] = {:.4}  {}",
                         kinds[i].name(), kinds[j].name(), d,
                         if ok { "> 揺らぎ" } else { "**<= 揺らぎ**" });
            }
        }
        println!();
        println!("  G74a 平衡到達 -> {}",
                 arms.iter().enumerate()
                     .map(|(i, a)| format!("{} {}", kinds[i].name(),
                          if tv_ratio(&a.snaps) < 0.1 { "到達" } else { "未到達" }))
                     .collect::<Vec<_>>().join(" / "));
        println!("  G74b 入力依存 -> {}",
                 if all_bigger { "**あり (全対でアーム間 > アーム内)**" }
                 else { "**全対では示せない (下を見ること)**" });
    }

    println!();
    println!("--- G74c 決定論性 (素・かな・{} 提示で 2 回) ---", DETERMINISM_TRIALS);
    let a = run(true, Input::Kana, DETERMINISM_TRIALS);
    let b = run(true, Input::Kana, DETERMINISM_TRIALS);
    let (ha, hb) = (fnv(&a.snaps), fnv(&b.snaps));
    println!("  ハッシュ {:016x} / {:016x} -> {}",
             ha, hb, if ha == hb { "一致 PASS" } else { "**不一致 FAIL**" });
}
