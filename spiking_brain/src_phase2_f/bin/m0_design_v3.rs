//! M0 の設計点を **M0.5 出力**で取り直す (v3・2026-08-26)
//!
//! ## なぜ v3 か
//!
//! v2 は**蝸牛 (M0) の出力**で測っていた。そこでは母音の識別率が 35% で頭打ちになり、
//! 「パラメータでは解けない」と結論した。
//!
//! しかし **M1 が実際に見るのは M0.5 (蝸牛神経核) の出力**であり、
//! そこで測ると **55%** だった (`lateral_inhibition` bin で判明)。
//! M0.5 の Stellate (±1 帯域プール) と時間適応が既に 35% → 55% に上げている。
//!
//! **測る段を間違えていた。** v2 の設計点 (Q×0.5 / 閾120) は
//! 35% の測定点で選んだものなので、55% の測定点では別の最適解かもしれない。
//! 「間違った段・間違った刺激で最適化した」を繰り返さないために取り直す。
//!
//! ## 制約 (先に明記する)
//!
//! `CochlearNucleus` は `N_BUSHY = N_BANDS` でコンパイル時に固定なので、
//! **N_BANDS は掃引できない**。M0.5 を Vec 版に作り直せば可能だが実物と乖離する危険がある。
//! 蝸牛出力での測定では N_BANDS 40 / 80 / 120 が 30-35% とほぼ同じだったので、
//! **40 に固定して Q × 閾値を掃引**する。
//!
//! ## ゲート (実測前に固定)
//!
//!   G61 母音の識別率: 20 条件 (5 母音 × 4 F0) の leave-one-out 1-NN。
//!       **M0.5 出力**で測る。チャンス 15.8%。**飽和しない。**
//!       正解の出どころ = どれが同じ母音かは実験者が決めた。
//!   G62 F0 不変性  : 同一母音の最小 > 別母音の最大
//!   G63 沈黙なし   : どの条件も無音でない ・ 無音入力で発火しない
//!   G64 穴なし     : 3 レベル (最弱フォルマント基準) でスペクトルの穴 0
//!
//! **採用規則 (先に宣言)**: G63・G64 を満たすうち **G61 最大**。
//!
//! CLI: m0_design_v3

use spiking_brain::phase2_f::cochlea::{
    compress_sqrt, erb_q_factor, erb_spaced_freqs, BandpassBiquad, Cochlea, EnvelopeDetector,
    FireGenerator, ENV_LEAK_SHIFT, FIRE_REFRACTORY_STEPS, FIRE_SPIKE_COST, F_MAX_HZ, F_MIN_HZ,
    N_BANDS, Q_SHARPENING, SAMPLES_PER_STEP,
};
use spiking_brain::phase2_f::cochlear_nucleus::{CochlearNucleus, N_CN_OUTPUT};
use spiking_brain::phase2_f::phoneme_synth::{
    freq_to_phase_step, sin_lookup, synth_vowel_f0, vowels, SAMPLE_RATE_HZ,
};

const VOWEL_MS: f64 = 170.0;
const PROBE_MS: f64 = 170.0;
const N_PROBE: usize = 100;
const F0S: [f64; 4] = [100.0, 150.0, 200.0, 250.0];

const Q_MULS: [f64; 7] = [0.15, 0.25, 0.35, 0.5, 0.75, 1.0, 1.5];
const THRESHOLDS: [i32; 5] = [60, 80, 120, 160, 200];

fn cochlea_of(q_mul: f64, threshold: i32) -> Cochlea {
    let center_freqs = erb_spaced_freqs(F_MIN_HZ, F_MAX_HZ, N_BANDS);
    let bands = center_freqs
        .iter()
        .map(|&fc| BandpassBiquad::new(fc, erb_q_factor(fc) * q_mul, SAMPLE_RATE_HZ))
        .collect();
    let envelopes = (0..N_BANDS).map(|_| EnvelopeDetector::new(ENV_LEAK_SHIFT)).collect();
    let fire_gens = (0..N_BANDS)
        .map(|_| {
            let mut f = FireGenerator::new(threshold, FIRE_REFRACTORY_STEPS);
            f.spike_cost = FIRE_SPIKE_COST;
            f
        })
        .collect();
    Cochlea { bands, envelopes, fire_gens, center_freqs, ..Cochlea::new() }
}

/// **M0.5 の出力**チャネルごとのスパイク数 (M1 が実際に見るもの)
fn cn_counts(wave: &[i32], q_mul: f64, threshold: i32) -> Vec<u32> {
    let mut co = cochlea_of(q_mul, threshold);
    let mut cn = CochlearNucleus::new();
    let mut counts = vec![0u32; N_CN_OUTPUT];
    for chunk in wave.chunks(SAMPLES_PER_STEP) {
        if chunk.len() < SAMPLES_PER_STEP {
            break;
        }
        let out = co.process_step(chunk);
        for (i, &v) in cn.process_step(&out).iter().enumerate() {
            if v != 0 {
                counts[i] += 1;
            }
        }
    }
    counts
}

fn cosine(a: &[u32], b: &[u32]) -> f64 {
    let dot: f64 = a.iter().zip(b.iter()).map(|(&x, &y)| x as f64 * y as f64).sum();
    let na: f64 = a.iter().map(|&x| (x as f64).powi(2)).sum::<f64>().sqrt();
    let nb: f64 = b.iter().map(|&x| (x as f64).powi(2)).sum::<f64>().sqrt();
    if na == 0.0 || nb == 0.0 {
        0.0
    } else {
        dot / (na * nb)
    }
}

fn nearest_band(freqs: &[f64], f_hz: f64) -> usize {
    freqs
        .iter()
        .enumerate()
        .min_by(|a, b| (a.1 - f_hz).abs().partial_cmp(&(b.1 - f_hz).abs()).unwrap())
        .unwrap()
        .0
}

/// G64: 純音を掃引して応答しない周波数があるか (蝸牛段で測る)。
fn holes(q_mul: f64, threshold: i32) -> usize {
    let freqs = erb_spaced_freqs(F_MIN_HZ, F_MAX_HZ, N_BANDS);
    let vs = vowels();
    let weakest = vs.iter().flat_map(|v| v.amplitudes.iter()).cloned().min().unwrap();
    let strongest = vs.iter().flat_map(|v| v.amplitudes.iter()).cloned().max().unwrap();
    let mut worst = 0usize;
    for &amp in [weakest, weakest * 2, strongest].iter() {
        let mut dead = 0usize;
        for k in 0..N_PROBE {
            let t = k as f64 / (N_PROBE - 1) as f64;
            let f = F_MIN_HZ * (F_MAX_HZ / F_MIN_HZ).powf(t);
            // 最寄り帯域まわりだけ回す (遠い帯域が近い帯域より強く応答することはない)
            let c = nearest_band(&freqs, f);
            let lo = c.saturating_sub(3);
            let hi = (c + 4).min(N_BANDS);
            let sub: Vec<f64> = freqs[lo..hi].to_vec();
            let n = sub.len();
            let mut bands: Vec<BandpassBiquad> = sub
                .iter()
                .map(|&fc| BandpassBiquad::new(fc, erb_q_factor(fc) * q_mul, SAMPLE_RATE_HZ))
                .collect();
            let mut envs: Vec<EnvelopeDetector> =
                (0..n).map(|_| EnvelopeDetector::new(ENV_LEAK_SHIFT)).collect();
            let mut fires: Vec<FireGenerator> = (0..n)
                .map(|_| {
                    let mut fg = FireGenerator::new(threshold, FIRE_REFRACTORY_STEPS);
                    fg.spike_cost = FIRE_SPIKE_COST;
                    fg
                })
                .collect();
            let n_samples = (PROBE_MS * SAMPLE_RATE_HZ / 1000.0) as usize;
            let step = freq_to_phase_step(f);
            let mut phase = 0u32;
            let mut heard = false;
            let mut i = 0usize;
            while i + SAMPLES_PER_STEP <= n_samples && !heard {
                for _ in 0..SAMPLES_PER_STEP {
                    let x = (sin_lookup(phase) * amp) >> 14;
                    phase = phase.wrapping_add(step);
                    for ch in 0..n {
                        let y = bands[ch].process(x);
                        envs[ch].process(y);
                    }
                }
                for ch in 0..n {
                    if fires[ch].process(compress_sqrt(envs[ch].env)) {
                        heard = true;
                    }
                }
                i += SAMPLES_PER_STEP;
            }
            if !heard {
                dead += 1;
            }
        }
        worst = worst.max(dead);
    }
    worst
}

fn main() {
    let vs = vowels();
    println!("=== M0 の設計点を M0.5 出力で取り直す (v3) ===");
    println!("測定点: **M0.5 出力** ({} ch) = M1 が実際に見るもの", N_CN_OUTPUT);
    println!("N_BANDS = {} に固定 (CochlearNucleus がコンパイル時固定のため掃引不可)", N_BANDS);
    println!("主指標 G61 = leave-one-out 1-NN の母音識別率 (チャンス 15.8%)");
    println!("採用規則: G63(沈黙なし) かつ G64(穴0) のうち G61 最大");
    println!();
    println!("  Q    閾値  G61識別率  G62(同最小/異最大)  無音入力  沈黙  穴/{}", N_PROBE);

    struct R {
        q: f64,
        thr: i32,
        ident: f64,
        min_same: f64,
        max_diff: f64,
        silent: usize,
        silence_fire: u32,
        holes: usize,
    }
    let mut rows: Vec<R> = Vec::new();
    for &q in Q_MULS.iter() {
        for &thr in THRESHOLDS.iter() {
            // 無音入力での発火
            let mut co = cochlea_of(q, thr);
            let mut cn = CochlearNucleus::new();
            let mut silence_fire = 0u32;
            for _ in 0..4000 {
                let out = co.process_step(&[0i32; SAMPLES_PER_STEP]);
                silence_fire += cn.process_step(&out).iter().filter(|&&v| v != 0).count() as u32;
            }

            let mut conds: Vec<(usize, Vec<u32>)> = Vec::new();
            let mut silent = 0usize;
            for (k, v) in vs.iter().enumerate() {
                for &f0 in F0S.iter() {
                    let c = cn_counts(&synth_vowel_f0(v, f0, VOWEL_MS), q, thr);
                    if c.iter().all(|&x| x == 0) {
                        silent += 1;
                    }
                    conds.push((k, c));
                }
            }
            let mut hit = 0usize;
            for i in 0..conds.len() {
                let mut best = (-2.0f64, usize::MAX);
                for j in 0..conds.len() {
                    if i == j {
                        continue;
                    }
                    let c = cosine(&conds[i].1, &conds[j].1);
                    if c > best.0 {
                        best = (c, conds[j].0);
                    }
                }
                if best.1 == conds[i].0 {
                    hit += 1;
                }
            }
            let mut min_same = f64::INFINITY;
            let mut max_diff = f64::NEG_INFINITY;
            for i in 0..conds.len() {
                for j in (i + 1)..conds.len() {
                    let c = cosine(&conds[i].1, &conds[j].1);
                    if conds[i].0 == conds[j].0 {
                        min_same = min_same.min(c);
                    } else {
                        max_diff = max_diff.max(c);
                    }
                }
            }
            let h = if silent == 0 && silence_fire == 0 {
                holes(q, thr)
            } else {
                usize::MAX
            };
            rows.push(R {
                q,
                thr,
                ident: hit as f64 / conds.len() as f64,
                min_same,
                max_diff,
                silent,
                silence_fire,
                holes: h,
            });
        }
    }

    let mut pass: Vec<&R> = Vec::new();
    for r in rows.iter() {
        let ok = r.silent == 0 && r.silence_fire == 0 && r.holes == 0;
        println!(
            "{:>4.2}x {:>5}  {:>8.1}%  {:>7.3} / {:<7.3}  {:>8}  {:>4}  {:>5}  {}",
            r.q,
            r.thr,
            r.ident * 100.0,
            r.min_same,
            r.max_diff,
            r.silence_fire,
            r.silent,
            if r.holes == usize::MAX { "-".to_string() } else { r.holes.to_string() },
            if ok { "候補" } else { "-" }
        );
        if ok {
            pass.push(r);
        }
    }

    println!();
    match pass.iter().max_by(|a, b| a.ident.partial_cmp(&b.ident).unwrap()) {
        Some(b) => {
            println!("採用規則の結果: **Q ×{:.2} ・ FIRE_THRESHOLD = {}**", b.q, b.thr);
            println!("  G61 母音の識別率 {:.1}% (チャンス 15.8%)", b.ident * 100.0);
            println!(
                "  G62 同一母音の最小 {:.3} vs 別母音の最大 {:.3} → {}",
                b.min_same,
                b.max_diff,
                if b.min_same > b.max_diff { "PASS" } else { "**FAIL**" }
            );
            println!();
            println!("  現行の出荷値は Q ×{:.2} / 閾120 (蝸牛出力で決めたもの)", Q_SHARPENING);
        }
        None => println!("**該当なし — 不成立。**"),
    }
}
