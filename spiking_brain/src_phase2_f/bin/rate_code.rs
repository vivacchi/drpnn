//! 出力段に rate code を持たせる — 漏れ積分発火の掃引 (2026-08-25)
//!
//! ## 問題（実測済み）
//!
//! 旧 `FireGenerator` は「閾値を超えていたら 1/(1+不応期) step ごとに発火」で、
//! **発火が env にも膜電位にも戻らない**。そのため 1ch の rate-level 関数は
//! 閾値を跨いだ瞬間に上限 400Hz へ飛び、閾値→飽和は fc により **0.50-3.25 dB** しかない。
//! 中間レートが出る振幅は掃引の 0.6-3.6%。
//! 設計書 §1.4 の「動的レンジ 30-130 dB SPL」に対し出力段は実質 2 状態だった。
//!
//! ## 直し方（6原理と整合）
//!
//! `ThermoNeuron` が既に持っている物理（溜める・漏れる・閾値で**消費**する）と同じ形を
//! `FireGenerator` にも入れる。判断機構ではない。
//! 漏れは現行 `FIRE_THRESHOLD` がそのまま担うので**無音床は保存される**。
//!
//! ## ゲート（実測前に固定）
//!
//!   G43 レート符号の動的レンジ: 1ch の発火率が閾値を跨いでから飽和するまでの入力範囲 [dB]。
//!       正解の出どころ = 入力振幅を置いたのは実験者。
//!   G44 レベル順序の復元: 同一音素を複数レベルで提示し、
//!       集団スパイク数からレベル順序が**単調に**復元できるか。
//!       正解の出どころ = レベルを決めたのは実験者。
//!   G45 無音床の保存: 無音で 1 発も出ない。
//!   G46 0dB の場所符号を壊さない: 被覆 15/15・場所符号 10/10 を維持。
//!
//!   リップル対照（監査の提案を採用）: 改善を主張する前に `ENV_LEAK_SHIFT` を伸ばして
//!   幅が残るか必ず確認する。伸ばして消えるならレベル符号ではなくリップル。
//!
//! **禁止する直し方**: `FIRE_THRESHOLD` を下げる / `FIRE_REFRACTORY_STEPS` を増やす /
//! `ENV_LEAK_SHIFT` をいじって窓を広く見せる。いずれもパラメータ調整でゲートを通す類。
//!
//! CLI: rate_code

use spiking_brain::phase2_f::cochlea::{
    Cochlea, ENV_LEAK_SHIFT, FIRE_THRESHOLD, N_BANDS, SAMPLES_PER_STEP,
};
use spiking_brain::phase2_f::phoneme_synth::{
    freq_to_phase_step, sin_lookup, synth_vowel, vowels, SAMPLE_RATE_HZ,
};

const STEPS_PER_SEC: f64 = 2000.0;
const PROBE_MS: f64 = 170.0;
const VOWEL_MS: f64 = 170.0;
const SPIKE_COSTS: [i32; 4] = [0, 240, 480, 960];
/// rate-level を測る中心周波数 [Hz]
const RL_FC: f64 = 1000.0;
const LEVELS_DB: [f64; 9] = [0.0, -3.0, -6.0, -9.0, -12.0, -15.0, -18.0, -21.0, -24.0];

fn cochlea_with(spike_cost: i32, env_leak_shift: i32) -> Cochlea {
    let mut c = Cochlea::new();
    for f in c.fire_gens.iter_mut() {
        f.spike_cost = spike_cost;
    }
    if env_leak_shift != ENV_LEAK_SHIFT {
        for e in c.envelopes.iter_mut() {
            e.leak_shift = env_leak_shift;
        }
    }
    c
}

fn gain_apply(x: i32, num: i32, den: i32) -> i32 {
    ((x as i64 * num as i64) / den as i64) as i32
}

fn db_gain(db: f64) -> (i32, i32) {
    let den = 4096i32;
    ((10f64.powf(db / 20.0) * den as f64).round() as i32, den)
}

/// 純音を通し、指定帯域の発火率 [Hz] を返す。
fn tone_rate(fc: f64, amp: i32, spike_cost: i32, env_leak_shift: i32, ch: usize) -> f64 {
    let mut c = cochlea_with(spike_cost, env_leak_shift);
    let n_samples = (PROBE_MS * SAMPLE_RATE_HZ / 1000.0) as usize;
    let step = freq_to_phase_step(fc);
    let mut phase = 0u32;
    let mut spikes = 0usize;
    let mut steps = 0usize;
    let mut i = 0usize;
    while i + SAMPLES_PER_STEP <= n_samples {
        let mut buf = [0i32; SAMPLES_PER_STEP];
        for b in buf.iter_mut() {
            *b = (sin_lookup(phase) * amp) >> 14;
            phase = phase.wrapping_add(step);
        }
        if c.process_step(&buf)[ch] != 0 {
            spikes += 1;
        }
        steps += 1;
        i += SAMPLES_PER_STEP;
    }
    spikes as f64 * STEPS_PER_SEC / steps as f64
}

/// 監査の critical 指摘の検証:
/// 「M1 が見るチャネルにはレベルが一切残らない。M0.5 bushy は全域で 24Hz 固定」。
/// 事実なら M0 の出力段を直しても M1 には届かない。
fn cn_passthrough() {
    use spiking_brain::phase2_f::cochlear_nucleus::{CochlearNucleus, N_CN_OUTPUT};
    let vs = vowels();
    println!();
    println!("--- M0.5 は M0 のレートを通すか (監査 critical の検証) ---");
    println!("音素 /a/ ・ M0 総スパイク と M0.5 総スパイク をレベル別に");
    println!("消費量  レベル  M0総スパイク  M0.5総スパイク  M0.5発火ch数");
    for &sc in [0i32, 480].iter() {
        for &db in [0.0f64, -6.0, -12.0, -18.0].iter() {
            let (num, den) = db_gain(db);
            let mut co = cochlea_with(sc, ENV_LEAK_SHIFT);
            let mut cn = CochlearNucleus::new();
            let wave = synth_vowel(&vs[0], VOWEL_MS);
            let mut m0 = 0u32;
            let mut m05 = 0u32;
            let mut active = vec![false; N_CN_OUTPUT];
            for chunk in wave.chunks(SAMPLES_PER_STEP) {
                if chunk.len() < SAMPLES_PER_STEP {
                    break;
                }
                let amp: Vec<i32> = chunk.iter().map(|&x| gain_apply(x, num, den)).collect();
                let out = co.process_step(&amp);
                m0 += out.iter().filter(|&&v| v != 0).count() as u32;
                let cn_out = cn.process_step(&out);
                for (i, &v) in cn_out.iter().enumerate() {
                    if v != 0 {
                        m05 += 1;
                        active[i] = true;
                    }
                }
            }
            println!(
                "{:>6}  {:>5.0}  {:>12}  {:>14}  {:>12}",
                if sc == 0 { "旧".to_string() } else { sc.to_string() },
                db, m0, m05,
                active.iter().filter(|&&b| b).count()
            );
        }
    }
}

fn main() {
    cn_passthrough();
    let c0 = Cochlea::new();
    let ch = c0
        .center_freqs
        .iter()
        .enumerate()
        .min_by(|a, b| (a.1 - RL_FC).abs().partial_cmp(&(b.1 - RL_FC).abs()).unwrap())
        .unwrap()
        .0;
    let fc = c0.center_freqs[ch];
    println!("=== 出力段の rate code (漏れ積分発火) ===");
    println!("rate-level は 帯域{} fc={:.1}Hz で測る ・ 無音床 = FIRE_THRESHOLD {}", ch, fc, FIRE_THRESHOLD);
    println!();

    // --- G43: rate-level の動的レンジ ---
    println!("--- G43 レート符号の動的レンジ ---");
    println!("消費量  無音時  初発火 [dBFS]  飽和 [dBFS]  動的レンジ[dB]  最大レート[Hz]  中間段数");
    for &sc in SPIKE_COSTS.iter() {
        // 無音
        let silent = tone_rate(fc, 0, sc, ENV_LEAK_SHIFT, ch);
        // 振幅を 0.5dB 刻みで掃引
        let mut first: Option<f64> = None;
        let mut sat: Option<f64> = None;
        let mut rates: Vec<f64> = Vec::new();
        let mut max_rate = 0.0f64;
        let mut k = -80.0f64;
        while k <= 0.0 {
            let amp = (32000.0 * 10f64.powf(k / 20.0)) as i32;
            let r = tone_rate(fc, amp, sc, ENV_LEAK_SHIFT, ch);
            rates.push(r);
            if r > 0.0 && first.is_none() {
                first = Some(k);
            }
            if r > max_rate {
                max_rate = r;
            }
            k += 0.5;
        }
        // 飽和 = 最大レートの 99% に初めて到達した点
        let mut kk = -80.0f64;
        for &r in rates.iter() {
            if r >= max_rate * 0.99 && sat.is_none() {
                sat = Some(kk);
            }
            kk += 0.5;
        }
        let n_mid = rates
            .iter()
            .filter(|&&r| r > 0.0 && r < max_rate * 0.99)
            .count();
        let range = match (first, sat) {
            (Some(f), Some(s)) => s - f,
            _ => f64::NAN,
        };
        println!(
            "{:>6}  {:>6.0}  {:>13}  {:>11}  {:>14.2}  {:>14.0}  {:>8}",
            if sc == 0 { "旧".to_string() } else { sc.to_string() },
            silent,
            first.map(|v| format!("{:.1}", v)).unwrap_or("-".into()),
            sat.map(|v| format!("{:.1}", v)).unwrap_or("-".into()),
            range,
            max_rate,
            n_mid
        );
    }

    // --- リップル対照 (監査の提案) ---
    println!();
    println!("--- リップル対照: ENV_LEAK_SHIFT を 4 → 7 に伸ばしても幅が残るか ---");
    println!("消費量  shift=4 の動的レンジ  shift=7 の動的レンジ");
    for &sc in SPIKE_COSTS.iter() {
        let mut widths = Vec::new();
        for &shift in [ENV_LEAK_SHIFT, 7].iter() {
            let mut first = None;
            let mut max_rate = 0.0f64;
            let mut rates = Vec::new();
            let mut k = -80.0f64;
            while k <= 0.0 {
                let amp = (32000.0 * 10f64.powf(k / 20.0)) as i32;
                let r = tone_rate(fc, amp, sc, shift, ch);
                rates.push((k, r));
                if r > 0.0 && first.is_none() {
                    first = Some(k);
                }
                if r > max_rate {
                    max_rate = r;
                }
                k += 0.5;
            }
            let sat = rates.iter().find(|(_, r)| *r >= max_rate * 0.99).map(|(k, _)| *k);
            widths.push(match (first, sat) {
                (Some(f), Some(s)) => s - f,
                _ => f64::NAN,
            });
        }
        println!("{:>6}  {:>20.2}  {:>20.2}",
                 if sc == 0 { "旧".to_string() } else { sc.to_string() },
                 widths[0], widths[1]);
    }

    // --- G44/G45/G46 ---
    println!();
    println!("--- G44 レベル順序の復元 / G45 無音床 / G46 0dB の場所符号 ---");
    println!("消費量  G45無音  G46被覆/15 場所符号/10  G44 レベル順序が単調な音素数/5");
    let vs = vowels();
    for &sc in SPIKE_COSTS.iter() {
        // G45
        let mut c = cochlea_with(sc, ENV_LEAK_SHIFT);
        let mut silent_spikes = 0usize;
        for _ in 0..4000 {
            if c.process_step(&[0i32; SAMPLES_PER_STEP]).iter().any(|&v| v != 0) {
                silent_spikes += 1;
            }
        }

        // レベルごとの総スパイク数 (G44) と 0dB のプロファイル (G46)
        let mut monotone = 0usize;
        let mut profiles0: Vec<[u32; N_BANDS]> = Vec::new();
        for (k, v) in vs.iter().enumerate() {
            let wave = synth_vowel(v, VOWEL_MS);
            let mut totals = Vec::new();
            for &db in LEVELS_DB.iter() {
                let (num, den) = db_gain(db);
                let mut co = cochlea_with(sc, ENV_LEAK_SHIFT);
                let mut counts = [0u32; N_BANDS];
                let mut total = 0u32;
                for chunk in wave.chunks(SAMPLES_PER_STEP) {
                    if chunk.len() < SAMPLES_PER_STEP {
                        break;
                    }
                    let amp: Vec<i32> =
                        chunk.iter().map(|&x| gain_apply(x, num, den)).collect();
                    let out = co.process_step(&amp);
                    for b in 0..N_BANDS {
                        if out[b] != 0 {
                            counts[b] += 1;
                            total += 1;
                        }
                    }
                }
                if db == 0.0 {
                    profiles0.push(counts);
                }
                totals.push(total);
            }
            // レベルが下がるほど総スパイクが減る (単調非増加) か
            if totals.windows(2).all(|w| w[0] >= w[1]) {
                monotone += 1;
            }
            let _ = k;
        }

        // G46
        let freqs = &c0.center_freqs;
        let mut recall = 0usize;
        for (k, v) in vs.iter().enumerate() {
            for f in 0..3 {
                let bi = freqs
                    .iter()
                    .enumerate()
                    .min_by(|a, b| {
                        (a.1 - v.formants_hz[f])
                            .abs()
                            .partial_cmp(&(b.1 - v.formants_hz[f]).abs())
                            .unwrap()
                    })
                    .unwrap()
                    .0;
                if profiles0[k][bi] > 0 {
                    recall += 1;
                }
            }
        }
        let sets: Vec<Vec<usize>> = profiles0
            .iter()
            .map(|p| (0..N_BANDS).filter(|&i| p[i] > 0).collect())
            .collect();
        let mut distinct = 0usize;
        for i in 0..sets.len() {
            for j in (i + 1)..sets.len() {
                if sets[i] != sets[j] {
                    distinct += 1;
                }
            }
        }

        println!(
            "{:>6}  {:>7}  {:>10} {:>11}  {:>28}",
            if sc == 0 { "旧".to_string() } else { sc.to_string() },
            if silent_spikes == 0 { "PASS" } else { "**FAIL**" },
            format!("{}/15", recall),
            format!("{}/10", distinct),
            format!("{}/5", monotone)
        );
    }
}
