//! M0 の設計点を**倍音つき刺激で**取り直す (v2・2026-08-26)
//!
//! ## なぜ取り直すか
//!
//! §12.7 の設計点 (N_BANDS=80 / Q×6 / 閾120) は **純音 3 本の刺激**で最適化した。
//! F0 を実装して倍音つき刺激で測ったところ (§13.7):
//!   Q×1 で母音の識別率 30.0% / Q×6 で 5.0% (チャンス 15.8%)
//! **Q を上げるほど単調に悪化する。** 高い Q は倍音を 1 本ずつ分解するので、
//! 場所符号が「倍音の位置」(同じ F0 なら全母音で同じ) を追ってしまうため。
//!
//! **純音 3 本という刺激そのものが測定対象を歪めていた**
//! = 「計器が答えを決める」の刺激版。実音声に近い刺激で取り直す。
//!
//! ## ゲート (実測前に固定)
//!
//!   G55 母音の識別率 (主指標): 5 母音 × 4 F0 = 20 条件の leave-one-out 1-NN。
//!       **飽和しない**・正解は完全に実験者側 (どれが同じ母音か)。
//!       チャンスレベル 3/19 = 15.8%。
//!   G56 F0 不変性 (直接形): 「同一母音・F0違い」の最小 > 「別母音・同F0」の最大
//!   G57 沈黙なし: どの条件も無音でない
//!   G58 穴なし  : 3 レベルでスペクトルの穴 0 (既存の穴テストを継承)
//!
//! **採用規則 (先に宣言)**: G57・G58 を満たすうち **G55 識別率が最大**。
//! 同点なら **N_BANDS 最小** (M1 への波及を抑える)。
//!
//! CLI: m0_design_v2

use spiking_brain::phase2_f::cochlea::{
    compress_sqrt, erb_q_factor, erb_spaced_freqs, BandpassBiquad, EnvelopeDetector,
    FireGenerator, ENV_LEAK_SHIFT, FIRE_REFRACTORY_STEPS, FIRE_SPIKE_COST, F_MAX_HZ, F_MIN_HZ,
    SAMPLES_PER_STEP,
};
use spiking_brain::phase2_f::phoneme_synth::{
    freq_to_phase_step, sin_lookup, synth_vowel_f0, vowels, SAMPLE_RATE_HZ,
};

const VOWEL_MS: f64 = 170.0;
const PROBE_MS: f64 = 170.0;
const N_PROBE: usize = 100;
const NEAREST_FOR_HOLES: usize = 7;
const F0S: [f64; 4] = [100.0, 150.0, 200.0, 250.0];

const BAND_COUNTS: [usize; 2] = [40, 80];
const Q_MULS: [f64; 6] = [0.1, 0.15, 0.25, 0.35, 0.5, 1.0];
const THRESHOLDS: [i32; 3] = [80, 120, 160];

struct Bank {
    bands: Vec<BandpassBiquad>,
    envs: Vec<EnvelopeDetector>,
    fires: Vec<FireGenerator>,
}

fn make_bank(freqs: &[f64], q_mul: f64, threshold: i32) -> Bank {
    Bank {
        bands: freqs
            .iter()
            .map(|&fc| BandpassBiquad::new(fc, erb_q_factor(fc) * q_mul, SAMPLE_RATE_HZ))
            .collect(),
        envs: (0..freqs.len()).map(|_| EnvelopeDetector::new(ENV_LEAK_SHIFT)).collect(),
        fires: (0..freqs.len())
            .map(|_| {
                let mut f = FireGenerator::new(threshold, FIRE_REFRACTORY_STEPS);
                f.spike_cost = FIRE_SPIKE_COST;
                f
            })
            .collect(),
    }
}

fn counts_of(wave: &[i32], freqs: &[f64], q_mul: f64, threshold: i32) -> Vec<u32> {
    let n = freqs.len();
    let mut bank = make_bank(freqs, q_mul, threshold);
    let mut out = vec![0u32; n];
    for chunk in wave.chunks(SAMPLES_PER_STEP) {
        if chunk.len() < SAMPLES_PER_STEP {
            break;
        }
        for &x in chunk {
            for ch in 0..n {
                let y = bank.bands[ch].process(x);
                bank.envs[ch].process(y);
            }
        }
        for ch in 0..n {
            if bank.fires[ch].process(compress_sqrt(bank.envs[ch].env)) {
                out[ch] += 1;
            }
        }
    }
    out
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

/// 純音を最寄り数本だけに通し、1 本でも発火するか (穴の検査)。
fn tone_heard(f_hz: f64, amp: i32, freqs: &[f64], q_mul: f64, threshold: i32) -> bool {
    let c = nearest_band(freqs, f_hz);
    let lo = c.saturating_sub(NEAREST_FOR_HOLES / 2);
    let hi = (c + NEAREST_FOR_HOLES / 2 + 1).min(freqs.len());
    let sub: Vec<f64> = freqs[lo..hi].to_vec();
    let mut bank = make_bank(&sub, q_mul, threshold);
    let n = sub.len();
    let n_samples = (PROBE_MS * SAMPLE_RATE_HZ / 1000.0) as usize;
    let step = freq_to_phase_step(f_hz);
    let mut phase = 0u32;
    let mut i = 0usize;
    while i + SAMPLES_PER_STEP <= n_samples {
        for _ in 0..SAMPLES_PER_STEP {
            let x = (sin_lookup(phase) * amp) >> 14;
            phase = phase.wrapping_add(step);
            for ch in 0..n {
                let y = bank.bands[ch].process(x);
                bank.envs[ch].process(y);
            }
        }
        for ch in 0..n {
            if bank.fires[ch].process(compress_sqrt(bank.envs[ch].env)) {
                return true;
            }
        }
        i += SAMPLES_PER_STEP;
    }
    false
}

fn holes(freqs: &[f64], q_mul: f64, threshold: i32) -> usize {
    let vs = vowels();
    let weakest = vs.iter().flat_map(|v| v.amplitudes.iter()).cloned().min().unwrap();
    let strongest = vs.iter().flat_map(|v| v.amplitudes.iter()).cloned().max().unwrap();
    let mut worst = 0usize;
    for &amp in [weakest, weakest * 2, strongest].iter() {
        let mut dead = 0usize;
        for k in 0..N_PROBE {
            let t = k as f64 / (N_PROBE - 1) as f64;
            let f = F_MIN_HZ * (F_MAX_HZ / F_MIN_HZ).powf(t);
            if !tone_heard(f, amp, freqs, q_mul, threshold) {
                dead += 1;
            }
        }
        worst = worst.max(dead);
    }
    worst
}

struct R {
    n_bands: usize,
    q: f64,
    thr: i32,
    ident: f64,
    min_same: f64,
    max_diff: f64,
    silent: usize,
    holes: usize,
}

fn measure(n_bands: usize, q: f64, thr: i32) -> R {
    let freqs = erb_spaced_freqs(F_MIN_HZ, F_MAX_HZ, n_bands);
    let vs = vowels();
    let mut conds: Vec<(usize, Vec<u32>)> = Vec::new();
    let mut silent = 0usize;
    for (k, v) in vs.iter().enumerate() {
        for &f0 in F0S.iter() {
            let c = counts_of(&synth_vowel_f0(v, f0, VOWEL_MS), &freqs, q, thr);
            if c.iter().all(|&x| x == 0) {
                silent += 1;
            }
            conds.push((k, c));
        }
    }
    // G55: leave-one-out 1-NN
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
    // G56
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
    R {
        n_bands,
        q,
        thr,
        ident: hit as f64 / conds.len() as f64,
        min_same,
        max_diff,
        silent,
        holes: usize::MAX,
    }
}

fn main() {
    println!("=== M0 の設計点を倍音つき刺激で取り直す (v2) ===");
    println!("刺激: synth_vowel_f0 (声帯源 -> 全極共鳴器 -> 唇からの放射)");
    println!("F0 = {:?} Hz ・ 5 母音 × 4 F0 = 20 条件", F0S);
    println!("主指標 G55 = leave-one-out 1-NN の母音識別率 (チャンス 15.8%)");
    println!("採用規則: G57(沈黙なし) かつ G58(穴0) のうち G55 最大、同点なら N_BANDS 最小");
    println!();

    let mut rows: Vec<R> = Vec::new();
    for &nb in BAND_COUNTS.iter() {
        for &q in Q_MULS.iter() {
            for &thr in THRESHOLDS.iter() {
                rows.push(measure(nb, q, thr));
            }
        }
    }
    // 沈黙なしのものだけ穴を測る (重い)
    for r in rows.iter_mut() {
        if r.silent == 0 {
            let freqs = erb_spaced_freqs(F_MIN_HZ, F_MAX_HZ, r.n_bands);
            r.holes = holes(&freqs, r.q, r.thr);
        }
    }

    println!("帯域  Q    閾値  G55識別率  G56(同最小/異最大)  沈黙  穴/{}", N_PROBE);
    let mut pass: Vec<&R> = Vec::new();
    for r in rows.iter() {
        if r.silent > 0 {
            continue;
        }
        let ok = r.holes == 0;
        println!(
            "{:>4} {:>4.1}x {:>5}  {:>8.1}%  {:>7.3} / {:<7.3}  {:>4}  {:>5}  {}",
            r.n_bands,
            r.q,
            r.thr,
            r.ident * 100.0,
            r.min_same,
            r.max_diff,
            r.silent,
            if r.holes == usize::MAX {
                "-".to_string()
            } else {
                r.holes.to_string()
            },
            if ok { "候補" } else { "-" }
        );
        if ok {
            pass.push(r);
        }
    }

    println!();
    match pass.iter().max_by(|a, b| {
        a.ident
            .partial_cmp(&b.ident)
            .unwrap()
            .then(b.n_bands.cmp(&a.n_bands))
    }) {
        Some(b) => {
            println!(
                "採用規則の結果: **N_BANDS = {} ・ Q ×{:.1} ・ FIRE_THRESHOLD = {}**",
                b.n_bands, b.q, b.thr
            );
            println!("  G55 母音の識別率 {:.1}% (チャンス 15.8%)", b.ident * 100.0);
            println!(
                "  G56 同一母音の最小 {:.3} vs 別母音の最大 {:.3} → {}",
                b.min_same,
                b.max_diff,
                if b.min_same > b.max_diff { "PASS" } else { "**FAIL**" }
            );
            println!("  沈黙 0 ・ 穴 0");
            println!();
            println!("  現行の出荷値は N_BANDS=80 / Q×6.0 / 閾120 (純音刺激で決めたもの)");
        }
        None => println!("**該当なし — 不成立。**"),
    }
}
