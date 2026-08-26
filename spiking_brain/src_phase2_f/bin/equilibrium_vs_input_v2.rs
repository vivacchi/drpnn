//! B 修正版: 入力の統計を変えると平衡は変わるか — 個体数を主たる比較量にする (2026-08-26)
//!
//! ## なぜ測り直すのか
//!
//! §14.11 (B 第 1 回) で宣言したゲート G74b は**発火しなかった**。
//! だが原因は「差が無い」ことではなく、**私が選んだ統計量が効果を割り算で消していた**こと。
//!
//! `conductance` のヒストグラムは **alive なシナプスに対する割合**である。
//! 入力 A が 23571 本を生かし入力 B が 5895 本しか生かさなくても、
//! **残った集団の形が似ていればヒストグラムは似てしまう。**
//! 生の個体数では、現行構成で無音がかなの **24.8 倍**刈り取っていた (685 vs 17020)。
//!
//! **スケールこそが答えである問いに、スケール不変な統計量を選んだ。**
//! よって主たる比較量を **個体数 (alive / open / pruned)** に置き直す。
//!
//! ## 構成の変更 (ユーザー決定・2026-08-26)
//!
//! **M0 の自発発火を既定 ON にした** (振幅 8)。
//! コードのコメントは以前から「自発発火は M0 蝸牛が担当する設計に」と言っていたが、
//! M0 側が既定 OFF・M1 入力層も 0 (§14.4 の③で私が消した) で、
//! **担当がどちらにも無い状態**になっていた。M0 に担当させる。
//!
//! 振幅 8 は `spontaneous_probe` が**実測前に宣言した選定規則**
//! 「中央値レートが設計範囲の中央 75 Hz に最も近い振幅」による (中央値 90.5 Hz)。
//! 50-100 Hz は `M0_COCHLEA_DESIGN.md` §3.6 が指定した**設計側の正解**。
//!
//! **既知の未解決 (G13 FAIL)**: 振幅 8 では 40 帯域中 10 本が無音のまま。
//! 個体差倍率 `idx%4` (1..4) が 4 倍の振幅差を作り、レート応答が超線形なので
//! 0〜251 Hz に広がる。設計の窓 50-100 Hz は 2 倍しかない。
//! **両立する振幅は存在しない** (16 なら全帯域鳴るが中央値 335.8 Hz = 設計の 3.4 倍)。
//! 個体差機構は 2026-08-25 の `ae51754` で私が足したもので設計書側の指定ではない。
//!
//! ## ゲート (実測前に固定・以後動かさない)
//!
//! **正解の出どころ: どの入力を与えたかは実験者が決めた。**
//!
//! - **G75a 平衡到達**: 各アームで TV比 < 0.1。§14.10 の G73b と**同一規則**。
//! - **G75b 入力依存 (個体数)**: 同一構成内で、**アーム間の open シナプス数の差**が、
//!   **同一アーム内の後期の変動 (最後 3 区間の |Δopen| の最大値) より大きい**。
//!   *帰無 = 平衡は入力統計に依らない → アーム間の差はアーム内の変動に埋もれる。*
//!   **閾値を後から置かないために、対照は同じ量の中で取る。**
//! - **G75c 入力依存 (分布の形)**: §14.11 の G74b と**同一規則**。
//!   発火しなかった記録を保つために残す。**判定はこちらでは行わない。**
//! - **G75d 決定論性**: 短い窓で 2 回実行してハッシュ一致。
//!
//! ## 予測
//!
//! **数値は置かない** (§14.6.4 / §14.7 / §14.9.7 で 3 連続、§14.10.4 で構造予測も外した)。
//!
//! 構造: **個体数では入力依存が出るはず。**
//! ただし**これは独立な予言ではない。** §14.11 の生データで既に 24.8 倍の差を見ており、
//! **事後の観察に基づく予測である。**当たっても予測能力の証拠にはならない。
//!
//! CLI: equilibrium_vs_input_v2

use spiking_brain::phase2_f::cochlea::{Cochlea, SAMPLES_PER_STEP};
use spiking_brain::phase2_f::cochlear_nucleus::{CochlearNucleus, N_CN_OUTPUT};
use spiking_brain::phase2_f::kana::{moras_from_kana, synth_utterance};
use spiking_brain::phase2_f::phoneme_synth::LfsrNoise;
use spiking_brain::phase2_f::thermo_network::{ThermoNetwork, ThermoNetworkConfig};
use spiking_brain::phase2_f::thermo_synapse::{CONDUCTANCE_MAX, OPEN_THRESHOLD};

const KANA: &[&str] = &[
    "あ","い","う","え","お","か","き","く","け","こ","さ","し","す","せ","そ",
    "た","ち","つ","て","と","な","に","ぬ","ね","の","は","ひ","ふ","へ","ほ",
    "ま","み","む","め","も","や","ゆ","よ","ら","り","る","れ","ろ","わ","を","ん",
];

const F0: f64 = 150.0;
const SEED: u16 = 0xACE1;
const ORDER_SEED: u64 = 0xA5A5_1234_5678_9ABC;
const CHECKPOINTS: [usize; 9] = [0, 100, 500, 1000, 2000, 3000, 4000, 4500, 5000];
const DETERMINISM_TRIALS: usize = 300;
const N_BINS: usize = 11;

#[derive(Clone, Copy)]
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
                let g = target / rms(&raw).max(1.0);
                raw.iter().map(|&s| ((s as f64) * g).round() as i32).collect()
            }).collect()
        }
    }
}

struct Snap {
    trial: usize,
    n_alive: usize,
    n_open: usize,
    pruned: usize,
    hist: [f64; N_BINS],
    mean_cond: f64,
    at_zero: usize,
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
    let mut csum = 0f64;
    let mut z = 0usize;
    for s in alive.iter() {
        counts[bin(s.conductance, CONDUCTANCE_MAX)] += 1;
        csum += s.conductance as f64;
        if s.conductance <= 0 { z += 1; }
    }
    let mut hist = [0f64; N_BINS];
    for i in 0..N_BINS { hist[i] = counts[i] as f64 / n as f64; }
    Snap {
        trial, n_alive: alive.len(),
        n_open: alive.iter().filter(|s| s.conductance >= OPEN_THRESHOLD).count(),
        pruned: net.axons_pruned as usize,
        hist, mean_cond: csum / n as f64, at_zero: z,
    }
}

fn tv(a: &[f64; N_BINS], b: &[f64; N_BINS]) -> f64 {
    0.5 * (0..N_BINS).map(|i| (a[i] - b[i]).abs()).sum::<f64>()
}

fn fnv(s: &[Snap]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for x in s {
        for v in [x.n_alive as u64, x.n_open as u64, x.pruned as u64, x.at_zero as u64].iter() {
            for b in v.to_le_bytes().iter() { h ^= *b as u64; h = h.wrapping_mul(0x100000001b3); }
        }
    }
    h
}

struct Arm { snaps: Vec<Snap>, m0_density: f64, m05_density: f64 }

fn run(spont: i32, kind: Input, n_trials: usize) -> Arm {
    let waves = make_input(kind);
    let cfg = if N_CN_OUTPUT == 164 {
        ThermoNetworkConfig::for_m1_cn_80()
    } else {
        ThermoNetworkConfig::for_m1_cn_40()
    };
    let mut net = ThermoNetwork::new(cfg);
    let mut co = Cochlea::new();
    co.spontaneous_amplitude = spont;
    let mut cn = CochlearNucleus::new();

    let (mut s0, mut s05) = (0u64, 0u64);
    for w in waves.iter() {
        let mut c2 = Cochlea::new();
        c2.spontaneous_amplitude = spont;
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

fn tv_ratio(s: &[Snap]) -> f64 {
    let mut t = Vec::new();
    for i in 1..s.len() { t.push(tv(&s[i - 1].hist, &s[i].hist)); }
    *t.last().unwrap() / *t.first().unwrap()
}

fn late_tv(s: &[Snap]) -> f64 {
    let mut t = Vec::new();
    for i in 1..s.len() { t.push(tv(&s[i - 1].hist, &s[i].hist)); }
    let k = t.len().saturating_sub(3);
    t[k..].iter().cloned().fold(0f64, f64::max)
}

/// 同一アーム内の後期の変動 = 最後 3 区間の |Δopen| の最大値
fn late_open_var(s: &[Snap]) -> f64 {
    let mut d = Vec::new();
    for i in 1..s.len() {
        d.push((s[i].n_open as f64 - s[i - 1].n_open as f64).abs());
    }
    let k = d.len().saturating_sub(3);
    d[k..].iter().cloned().fold(0f64, f64::max)
}

fn main() {
    println!("=== B 修正版: 入力の統計で平衡は変わるか (個体数を主たる比較量に) ===");
    println!();
    println!("【なぜ測り直すか】§14.11 の G74b が発火しなかったのは差が無いからではなく、");
    println!("**私の統計量が効果を割り算で消していた**から。conductance ヒストグラムは");
    println!("alive に対する割合なので、23571本 vs 5895本 でも形が似れば似てしまう。");
    println!("生の個体数では無音がかなの 24.8 倍刈り取っていた。");
    println!("**スケールこそが答えである問いにスケール不変な統計量を選んだ。**");
    println!();
    println!("【構成変更・ユーザー決定】M0 の自発発火を既定 ON (振幅 8) にした。");
    println!("値は spontaneous_probe が実測前に宣言した規則 (中央値が 75Hz に最も近い) による。");
    println!("50-100Hz は M0_COCHLEA_DESIGN.md §3.6 の設計側の正解。");
    println!("**既知の未解決 (G13 FAIL)**: 振幅 8 では 40 帯域中 10 本が無音のまま。");
    println!("個体差の広がり (4倍) が設計の窓 (2倍) より広く、両立する振幅は存在しない。");
    println!();
    println!("【ゲート・実測前に固定】正解の出どころ = どの入力を与えたかは実験者が決めた");
    println!("  G75a 平衡到達: TV比 < 0.1 (§14.10 G73b と同一規則)");
    println!("  G75b 入力依存(個体数): アーム間 |Δopen| > アーム内の後期変動 (最後3区間の最大)");
    println!("  G75c 入力依存(分布の形): §14.11 G74b と同一規則。記録のため残すが判定しない");
    println!("  G75d 決定論性");
    println!();
    println!("【予測】数値は置かない。構造: 個体数では入力依存が出るはず。");
    println!("**ただし独立な予言ではない。** §14.11 の生データで既に差を見ている。");
    println!("**事後の観察に基づく予測なので、当たっても予測能力の証拠にはならない。**");

    let n = *CHECKPOINTS.last().unwrap();
    let kinds = [Input::Kana, Input::WhiteNoise, Input::Silence];

    for &(spont, cname) in [(8, "M0自発発火 ON (新既定・振幅8)"),
                            (0, "M0自発発火 OFF (旧既定・対照)")].iter() {
        println!();
        println!("################ 構成: {} ################", cname);
        let arms: Vec<Arm> = kinds.iter().map(|&k| run(spont, k, n)).collect();

        println!();
        println!("  入力          M0/提示  M0.5/提示    alive     open     刈取  平均cond  [==0]   TV比");
        for (i, &k) in kinds.iter().enumerate() {
            let f = arms[i].snaps.last().unwrap();
            println!("  {:<12} {:>8.0} {:>10.0} {:>8} {:>8} {:>8} {:>9.1} {:>6.1}% {:>6.3}",
                     k.name(), arms[i].m0_density, arms[i].m05_density,
                     f.n_alive, f.n_open, f.pruned, f.mean_cond,
                     f.hist[0] * 100.0, tv_ratio(&arms[i].snaps));
        }

        // --- G75b: 個体数での入力依存 ---
        println!();
        println!("  --- G75b 入力依存 (個体数・**主たる判定**) ---");
        let within: Vec<f64> = arms.iter().map(|a| late_open_var(&a.snaps)).collect();
        let wmax = within.iter().cloned().fold(0f64, f64::max);
        for (i, &k) in kinds.iter().enumerate() {
            println!("  アーム内の後期変動 |Δopen| [{}] = {:.0}", k.name(), within[i]);
        }
        println!("  -> アーム内の変動の最大 = {:.0}", wmax);
        println!();
        let mut all_b = true;
        for i in 0..kinds.len() {
            for j in (i + 1)..kinds.len() {
                let d = (arms[i].snaps.last().unwrap().n_open as f64
                       - arms[j].snaps.last().unwrap().n_open as f64).abs();
                let ok = d > wmax;
                if !ok { all_b = false; }
                println!("  アーム間 |Δopen| [{} vs {}] = {:>6.0}  {}",
                         kinds[i].name(), kinds[j].name(), d,
                         if ok { "> 変動" } else { "**<= 変動**" });
            }
        }

        // --- G75c: 分布の形 (記録用・判定しない) ---
        println!();
        println!("  --- G75c 入力依存 (分布の形・記録用・判定しない) ---");
        let w2: Vec<f64> = arms.iter().map(|a| late_tv(&a.snaps)).collect();
        let w2max = w2.iter().cloned().fold(0f64, f64::max);
        let mut all_c = true;
        for i in 0..kinds.len() {
            for j in (i + 1)..kinds.len() {
                let d = tv(&arms[i].snaps.last().unwrap().hist, &arms[j].snaps.last().unwrap().hist);
                if d <= w2max { all_c = false; }
                println!("  アーム間 TV [{} vs {}] = {:.4} (アーム内 {:.4}) {}",
                         kinds[i].name(), kinds[j].name(), d, w2max,
                         if d > w2max { "> 変動" } else { "**<= 変動**" });
            }
        }

        println!();
        println!("  G75a 平衡到達 -> {}",
                 arms.iter().enumerate().map(|(i, a)| format!("{} {}", kinds[i].name(),
                      if tv_ratio(&a.snaps) < 0.1 { "到達" } else { "**未到達**" }))
                     .collect::<Vec<_>>().join(" / "));
        println!("  **G75b 入力依存 (個体数) -> {}**",
                 if all_b { "**あり (全対でアーム間 > アーム内)**" } else { "**全対では示せない**" });
        println!("  (参考) G75c 分布の形 -> {}",
                 if all_c { "あり" } else { "全対では示せない" });
    }

    println!();
    println!("--- G75d 決定論性 (自発ON・かな・{} 提示で 2 回) ---", DETERMINISM_TRIALS);
    let a = run(8, Input::Kana, DETERMINISM_TRIALS);
    let b = run(8, Input::Kana, DETERMINISM_TRIALS);
    let (ha, hb) = (fnv(&a.snaps), fnv(&b.snaps));
    println!("  ハッシュ {:016x} / {:016x} -> {}",
             ha, hb, if ha == hb { "一致 PASS" } else { "**不一致 FAIL**" });
}
