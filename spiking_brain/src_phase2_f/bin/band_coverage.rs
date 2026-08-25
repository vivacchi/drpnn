//! 帯域数と選択性の幾何 — Q を上げるなら帯域は何本要るか (2026-08-25)
//!
//! ## 問題
//!
//! `formant_probe` で判明: **Q×3 にすると周波数軸の 20-35% が動作レベルで聞こえなくなる**
//! (3 レベルで掃引した最悪値で 40-69 / 200 点)。
//! 帯域を 3 倍狭くしても中心は 40 個のままなので、軸を覆いきれないため。
//!
//! 5 母音の 15 本のフォルマントが拾えている (被覆 15/15) のは、
//! **たまたま帯域中心の近くに落ちているから**であって、
//! 別の母音・別の話者のフォルマントが穴に落ちれば拾えない。
//!
//! ## この診断が問うこと
//!
//! **Q を上げるなら帯域は何本要るか。** 蝸牛の配列サイズを変えずに測れるよう、
//! フィルタの幾何だけで計算する (Vec で任意本数の filterbank を組み、
//! 純音を通して床を超える帯域が 1 本でもあるかを見る)。
//!
//! 正解の出どころ: どの周波数・どのレベルを入力したかは実験者が決めた。
//! 「聞こえるべき」は実験者が音を入れた事実から来る。
//!
//! CLI: band_coverage

use spiking_brain::phase2_f::cochlea::{
    compress_sqrt, erb_q_factor, erb_spaced_freqs, BandpassBiquad, EnvelopeDetector,
    FireGenerator, ENV_LEAK_SHIFT, FIRE_REFRACTORY_STEPS, FIRE_THRESHOLD, F_MAX_HZ, F_MIN_HZ,
    N_BANDS, Q_SHARPENING, SAMPLES_PER_STEP,
};
use spiking_brain::phase2_f::phoneme_synth::{
    freq_to_phase_step, sin_lookup, vowels, SAMPLE_RATE_HZ,
};

const N_PROBE: usize = 200;
const PROBE_MS: f64 = 170.0;
const BAND_COUNTS: [usize; 6] = [40, 60, 80, 120, 160, 240];
const Q_MULS: [f64; 4] = [1.0, 2.0, 3.0, 4.0];

/// 任意本数の filterbank に純音を通し、1 本でも発火する帯域があるか。
fn any_band_fires(f_hz: f64, amp: i32, n_bands: usize, q_mul: f64, threshold: i32) -> bool {
    let freqs = erb_spaced_freqs(F_MIN_HZ, F_MAX_HZ, n_bands);
    let mut bands: Vec<BandpassBiquad> = freqs
        .iter()
        .map(|&fc| BandpassBiquad::new(fc, erb_q_factor(fc) * q_mul, SAMPLE_RATE_HZ))
        .collect();
    let mut envs: Vec<EnvelopeDetector> =
        (0..n_bands).map(|_| EnvelopeDetector::new(ENV_LEAK_SHIFT)).collect();
    let mut fires: Vec<FireGenerator> = (0..n_bands)
        .map(|_| FireGenerator::new(threshold, FIRE_REFRACTORY_STEPS))
        .collect();

    let n_samples = (PROBE_MS * SAMPLE_RATE_HZ / 1000.0) as usize;
    let step = freq_to_phase_step(f_hz);
    let mut phase = 0u32;
    let mut n_steps = 0usize;
    let mut buf = [0i32; SAMPLES_PER_STEP];
    let mut i = 0usize;
    while i + SAMPLES_PER_STEP <= n_samples {
        for k in 0..SAMPLES_PER_STEP {
            buf[k] = (sin_lookup(phase) * amp) >> 14;
            phase = phase.wrapping_add(step);
        }
        for &x in buf.iter() {
            for ch in 0..n_bands {
                let y = bands[ch].process(x);
                envs[ch].process(y);
            }
        }
        for ch in 0..n_bands {
            if fires[ch].process(compress_sqrt(envs[ch].env)) {
                return true;
            }
        }
        n_steps += 1;
        i += SAMPLES_PER_STEP;
        let _ = n_steps;
    }
    false
}

/// 3 レベル (最弱フォルマント / その 2 倍 / 最強フォルマント) の最悪値で穴を数える。
fn holes(n_bands: usize, q_mul: f64, threshold: i32) -> usize {
    let vs = vowels();
    let weakest = vs.iter().flat_map(|v| v.amplitudes.iter()).cloned().min().unwrap_or(800);
    let strongest = vs.iter().flat_map(|v| v.amplitudes.iter()).cloned().max().unwrap_or(4000);
    let levels = [weakest, weakest * 2, strongest];
    let mut worst = 0usize;
    for &amp in levels.iter() {
        let mut dead = 0usize;
        for k in 0..N_PROBE {
            let t = k as f64 / (N_PROBE - 1) as f64;
            let f = F_MIN_HZ * (F_MAX_HZ / F_MIN_HZ).powf(t);
            if !any_band_fires(f, amp, n_bands, q_mul, threshold) {
                dead += 1;
            }
        }
        worst = worst.max(dead);
    }
    worst
}

fn main() {
    let vs = vowels();
    let weakest = vs.iter().flat_map(|v| v.amplitudes.iter()).cloned().min().unwrap();
    let strongest = vs.iter().flat_map(|v| v.amplitudes.iter()).cloned().max().unwrap();
    println!("=== 帯域数と選択性の幾何 ===");
    println!("出荷構成: N_BANDS={} ・ Q_SHARPENING=×{:.0} ・ FIRE_THRESHOLD={}",
             N_BANDS, Q_SHARPENING, FIRE_THRESHOLD);
    println!("検査音のレベル: 最弱フォルマント {} / その2倍 {} / 最強フォルマント {}",
             weakest, weakest * 2, strongest);
    println!("{} 点を {:.0}-{:.0}Hz に対数間隔で掃引し、3 レベルの**最悪値**で穴を数える",
             N_PROBE, F_MIN_HZ, F_MAX_HZ);
    println!();
    println!("穴の数 / {}  (0 が目標)", N_PROBE);
    print!("帯域数 ");
    for &q in Q_MULS.iter() {
        print!("   Q×{:.0}", q);
    }
    println!();
    for &nb in BAND_COUNTS.iter() {
        print!("{:>5}  ", nb);
        for &q in Q_MULS.iter() {
            print!("{:>6}", holes(nb, q, FIRE_THRESHOLD));
        }
        println!();
    }

    // --- 残差の正体: 穴はどこにあるか ---
    println!();
    println!("--- 穴の位置 (帯域数 240・Q×1・最弱レベル {}) ---", weakest);
    let mut dead_lo = Vec::new();
    for k in 0..N_PROBE {
        let t = k as f64 / (N_PROBE - 1) as f64;
        let f = F_MIN_HZ * (F_MAX_HZ / F_MIN_HZ).powf(t);
        if !any_band_fires(f, weakest, 240, 1.0, FIRE_THRESHOLD) {
            dead_lo.push(f);
        }
    }
    let shown: Vec<String> = dead_lo.iter().map(|f| format!("{:.0}", f)).collect();
    println!("  {} 個: {}", dead_lo.len(), shown.join(", "));

    // 同じ周波数を最強レベルで鳴らしたら聞こえるか (レベル依存かの切り分け)
    let still: Vec<String> = dead_lo
        .iter()
        .filter(|&&f| !any_band_fires(f, strongest, 240, 1.0, FIRE_THRESHOLD))
        .map(|f| format!("{:.0}", f))
        .collect();
    println!("  うち最強レベル {} でも聞こえないもの: {} 個 {}",
             strongest, still.len(),
             if still.is_empty() { "(= 全部レベル依存の欠落)".to_string() } else { still.join(", ") });

    println!();
    println!("--- 読み方 ---");
    println!("Q を上げると帯域が狭くなるが中心の数は変わらないので、軸に穴が空く。");
    println!("穴を 0 に保ったまま Q を上げるには、**帯域数も一緒に増やす**必要がある。");
    println!("帯域数は M1 の入力数に直結する (N_BANDS → M0.5 で 84ch → M1 入力) ので、");
    println!("ここを動かすと M1 側の再設計が要る。**M0 単体の幾何としての答えがこの表。**");
}
