//! M1 はカオス的か — 1 step・1 ニューロン・+1 の摂動感度 (S11・2026-08-25)
//!
//! ## なぜ測るか
//!
//! ユーザーの狙いは「人間の脳内の活動ノイズ／JEPA のノイズ追加と同じ」。
//! `spontaneous_jitter` として実装し振幅 0-8 を掃引したところ、
//! **最小振幅 1 でも再現性 (within) が 0.966 → 0.373 に崩壊**した。
//!
//! 最初の仮説「ノイズが信号を埋めた」は**実測で外れた** (`jitter_probe`):
//!   jitter=1 で 30 step に積もるノイズは 4.0 (閾値 30 の 13%)、
//!   無音時の出力自発発火率も 3.2 → 3.3 Hz とほぼ動かない。ノイズは小さい。
//!
//! 残る説明: **M1 の再現性 0.966 は「頑健さ」ではなく「摂動がゼロであること」から来ている**。
//! つまりカオス的で、わずかな差が指紋レベルまで増幅される。
//!
//! ## ゲート (実測前に固定)
//!
//!   G21 摂動感度: 同一刺激・同一初期状態の 2 本を走らせ、片方にだけ
//!       **1 step だけ 1 ニューロンに +1** の膜電位摂動を入れて指紋のコサイン類似度を測る。
//!       正解の出どころ: 同じ刺激を与えたのも、摂動を 1 step だけ入れたのも実験者。
//!       **1 step の +1 で大きく落ちるなら、このネットワークはカオス的であり
//!       「穏やかなノイズ」は原理的に存在しない** = JEPA 型のノイズ追加は
//!       このアーキではそのままでは成立しない。
//!
//! 対照として「摂動なし 2 本」も測る (完全一致 = 1.000 になるはず。
//! ならなければ測定系自体が非決定論的ということなので、その時点で判定は無効)。
//!
//! CLI: perturbation_probe

use spiking_brain::phase2_f::cochlea::{Cochlea, SAMPLES_PER_STEP};
use spiking_brain::phase2_f::cochlear_nucleus::CochlearNucleus;
use spiking_brain::phase2_f::phoneme_synth::{standard_syllables, synth_syllable_scaled, LfsrNoise};
use spiking_brain::phase2_f::thermo_network::{ThermoNetwork, ThermoNetworkConfig};
use spiking_brain::trace::{cosine_similarity, OutputTrace};

const TRIAL_DURATION_MS: f64 = 300.0;
const DT_MS: f64 = 0.5;
const TRIAL_STEPS: usize = (TRIAL_DURATION_MS / DT_MS) as usize;
const FINGERPRINT_BIN_WIDTH_MS: f64 = 10.0;
/// 摂動を入れる step (試行の序盤・中盤・終盤で比べる)
const PERTURB_STEPS: [usize; 4] = [10, 100, 300, 500];

fn fingerprint(log: &[(usize, f64)], n_out: usize) -> Vec<f64> {
    let mut tr = OutputTrace::new(n_out, 50.0);
    for &(oi, t) in log {
        tr.record_spike(oi, t);
    }
    tr.time_binned_fingerprint(TRIAL_DURATION_MS, FINGERPRINT_BIN_WIDTH_MS)
}

/// 1 trial 提示。`perturb` = Some((step, neuron_id, delta)) でその step に膜電位を足す。
fn present(
    net: &mut ThermoNetwork,
    cochlea: &mut Cochlea,
    cn: &mut CochlearNucleus,
    waveform: &[i32],
    perturb: Option<(usize, usize, i32)>,
) -> Vec<(usize, f64)> {
    net.reset_trial_state();
    cochlea.reset();
    cn.reset();
    let t0 = net.current_time;
    let mut out_log = Vec::new();
    for step in 0..TRIAL_STEPS {
        if let Some((ps, nid, delta)) = perturb {
            if step == ps {
                net.neurons[nid].membrane = net.neurons[nid].membrane.saturating_add(delta);
            }
        }
        let s0 = step * SAMPLES_PER_STEP;
        let mut samples = [0i32; SAMPLES_PER_STEP];
        for i in 0..SAMPLES_PER_STEP {
            let idx = s0 + i;
            if idx < waveform.len() {
                samples[i] = waveform[idx];
            }
        }
        let coch = cochlea.process_step(&samples);
        let cn_out = cn.process_step(&coch);
        for nid in net.step(&cn_out) {
            if let Some(oi) = net.output_index_of(nid) {
                out_log.push((oi, (net.current_time - t0) as f64 * DT_MS));
            }
        }
    }
    out_log
}

fn fresh() -> (ThermoNetwork, Cochlea, CochlearNucleus) {
    (
        ThermoNetwork::new(ThermoNetworkConfig::for_m1_cn_40()),
        Cochlea::new(),
        CochlearNucleus::new(),
    )
}

fn main() {
    let syl = standard_syllables()[0]; // pa
    let mut noise = LfsrNoise::new(0xACE1);
    let wave = synth_syllable_scaled(&syl, &mut noise, 1.0);

    println!("=== M1 の摂動感度 (音節 {} ・ trial {} step) ===", syl.label, TRIAL_STEPS);
    println!("正解の出どころ: 同じ刺激を与えたのも、摂動を 1 step 入れたのも実験者");
    println!();

    // --- 対照: 摂動なし 2 本 ---
    let (mut n1, mut c1, mut k1) = fresh();
    let a = present(&mut n1, &mut c1, &mut k1, &wave, None);
    let n_out = n1.output_neurons.len();
    let (mut n2, mut c2, mut k2) = fresh();
    let b = present(&mut n2, &mut c2, &mut k2, &wave, None);
    let control = cosine_similarity(&fingerprint(&a, n_out), &fingerprint(&b, n_out));
    println!("対照 (摂動なし 2 本): コサイン類似度 {:.6}  スパイク {} vs {}",
             control, a.len(), b.len());
    if (control - 1.0).abs() > 1e-9 {
        println!("  ⚠️ 完全一致しない = 測定系が非決定論的。この時点で G21 の判定は無効。");
        return;
    }

    // --- 摂動あり ---
    println!();
    println!("摂動step  対象ニューロン  Δ   コサイン類似度  スパイク数 (基準 {})", a.len());
    let targets = [0usize, 100, 200, 439.min(n1.neurons.len() - 1)];
    let mut worst = 1.0f64;
    for &ps in PERTURB_STEPS.iter() {
        for &nid in targets.iter() {
            if nid >= n1.neurons.len() {
                continue;
            }
            let (mut np, mut cp, mut kp) = fresh();
            let p = present(&mut np, &mut cp, &mut kp, &wave, Some((ps, nid, 1)));
            let sim = cosine_similarity(&fingerprint(&a, n_out), &fingerprint(&p, n_out));
            worst = worst.min(sim);
            println!("{:>8}  {:>14}  +1  {:>14.6}  {:>8}", ps, nid, sim, p.len());
        }
    }

    println!();
    println!("--- G21 判定 ---");
    println!("最悪のコサイン類似度: {:.6}", worst);
    if worst < 0.99 {
        println!("**カオス的**: 1 step・1 ニューロン・+1 の摂動で指紋が {:.1}% 崩れる。", (1.0 - worst) * 100.0);
        println!("整数の膜電位では ±1/step が最小の刻みなので、");
        println!("**これより穏やかなノイズはこの表現のままでは作れない**。");
        println!("→ JEPA 型のノイズ追加は、このアーキにそのままでは載らない。");
    } else {
        println!("**頑健**: 1 step の摂動では指紋がほとんど動かない。");
        println!("→ jitter の崩壊は摂動感度でなく別の理由 (累積量) による。");
    }
}
