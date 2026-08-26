//! 子音は蝸牛から見て区別されているか (S1 検証・2026-08-25)
//!
//! 背景: `synth_consonant` は `Plosive { burst_freq_low: _, burst_freq_high: _ }` と
//! 帯域指定を両方捨てており、pa/ki/tu は構造的に同一波形だった。摩擦音も同様。
//! `synth_consonant_banded` が帯域を実際に効かせる。ここでは
//! **M1 に届く手前 (蝸牛出力) で違いが見えるか**を測る。
//!
//! ゲート (実測前に固定):
//!   G1: 旧版で pa/ki/tu の相互コサイン類似度 = 1.000 (欠陥の再現)
//!   G2: 帯域版で pa/ki/tu の相互類似度がすべて < 1.000
//!   G3: 帯域版のスパイク重心帯域が帯域指定の順 pa < tu < ki < se
//!
//! 正解の出どころ: 帯域を指定したのは実験者。「その順に並ぶべき」は既知。
//! 「音素として正しいか」は測らない (正解を持たないので計量できない)。
//!
//! CLI: consonant_probe

use spiking_brain::phase2_f::cochlea::{Cochlea, N_BANDS, SAMPLES_PER_STEP};
use spiking_brain::phase2_f::phoneme_synth::{F0_DEFAULT_HZ, 
    standard_syllables, synth_consonant, synth_consonant_banded, LfsrNoise,
};

const CONSONANT_MS: f64 = 30.0;
const SEED: u16 = 0xACE1;

/// 波形を蝸牛に通し、帯域ごとのスパイク数を返す。
fn band_spikes(wave: &[i32]) -> [u32; N_BANDS] {
    let mut c = Cochlea::new();
    let mut counts = [0u32; N_BANDS];
    for chunk in wave.chunks(SAMPLES_PER_STEP) {
        if chunk.len() < SAMPLES_PER_STEP {
            break;
        }
        let out = c.process_step(chunk);
        for ch in 0..N_BANDS {
            if out[ch] != 0 {
                counts[ch] += 1;
            }
        }
    }
    counts
}

fn cosine(a: &[u32; N_BANDS], b: &[u32; N_BANDS]) -> f64 {
    let dot: f64 = (0..N_BANDS).map(|i| a[i] as f64 * b[i] as f64).sum();
    let na: f64 = (0..N_BANDS).map(|i| (a[i] as f64).powi(2)).sum::<f64>().sqrt();
    let nb: f64 = (0..N_BANDS).map(|i| (b[i] as f64).powi(2)).sum::<f64>().sqrt();
    if na == 0.0 || nb == 0.0 { 0.0 } else { dot / (na * nb) }
}

/// スパイク重心の中心周波数 (Hz)。全帯域無音なら None。
fn centroid_hz(counts: &[u32; N_BANDS], freqs: &[f64]) -> Option<f64> {
    let mass: f64 = counts.iter().map(|&v| v as f64).sum();
    if mass == 0.0 {
        return None;
    }
    Some((0..N_BANDS).map(|i| counts[i] as f64 * freqs[i]).sum::<f64>() / mass)
}

fn rms(wave: &[i32]) -> f64 {
    if wave.is_empty() { return 0.0; }
    let sq: f64 = wave.iter().map(|&v| (v as f64) * (v as f64)).sum();
    (sq / wave.len() as f64).sqrt()
}

fn main() {
    // --- 診断: 各音源の振幅指標 (ピーク vs RMS) ---
    {
        use spiking_brain::phase2_f::phoneme_synth::{synth_vowel, vowels};
        println!("=== 振幅診断 (子音 30ms / 母音 170ms) ===");
        println!("音源                       ピーク      RMS   crest");
        let a = synth_vowel(&vowels()[0], 170.0);
        println!("{:<24} {:>8} {:>8.0} {:>7.2}", "母音 /a/ (基準)",
                 a.iter().map(|v| v.abs()).max().unwrap(), rms(&a),
                 a.iter().map(|v| v.abs()).max().unwrap() as f64 / rms(&a));
        for s in standard_syllables().iter() {
            for (tag, banded) in [("旧", false), ("帯域", true)] {
                let mut n = LfsrNoise::new(SEED);
                let w = if banded {
                    synth_consonant_banded(s.consonant, CONSONANT_MS, F0_DEFAULT_HZ, &mut n)
                } else {
                    synth_consonant(s.consonant, CONSONANT_MS, &mut n)
                };
                if w.is_empty() { continue; }
                let pk = w.iter().map(|v| v.abs()).max().unwrap();
                println!("{:<24} {:>8} {:>8.0} {:>7.2}",
                         format!("{} 子音 {}", tag, s.label), pk, rms(&w),
                         pk as f64 / rms(&w).max(1.0));
            }
        }
    }

    let syls = standard_syllables();
    let freqs = Cochlea::new().center_freqs.clone();
    let labels: Vec<&str> = syls.iter().map(|s| s.label).collect();

    for (mode, banded) in [("旧 synth_consonant", false), ("帯域版 banded", true)] {
        println!("\n=== {} ===", mode);
        let mut profiles: Vec<[u32; N_BANDS]> = Vec::new();
        for s in syls.iter() {
            let mut noise = LfsrNoise::new(SEED);
            let wave = if banded {
                synth_consonant_banded(s.consonant, CONSONANT_MS, F0_DEFAULT_HZ, &mut noise)
            } else {
                synth_consonant(s.consonant, CONSONANT_MS, &mut noise)
            };
            profiles.push(band_spikes(&wave));
        }

        println!("音節  総スパイク  重心Hz   発火帯域数");
        for (i, label) in labels.iter().enumerate() {
            let total: u32 = profiles[i].iter().sum();
            let active = profiles[i].iter().filter(|&&v| v > 0).count();
            let cent = centroid_hz(&profiles[i], &freqs)
                .map(|v| format!("{:7.1}", v))
                .unwrap_or_else(|| "   ----".to_string());
            println!("{:>4}  {:>10}  {}  {:>10}", label, total, cent, active);
        }

        println!("コサイン類似度:");
        print!("      ");
        for l in labels.iter() {
            print!("{:>8}", l);
        }
        println!();
        for i in 0..labels.len() {
            print!("{:>4}  ", labels[i]);
            for j in 0..labels.len() {
                print!("{:>8.3}", cosine(&profiles[i], &profiles[j]));
            }
            println!();
        }

        // --- ゲート判定 ---
        let plosive_idx = [0usize, 1, 2]; // pa, ki, tu
        let mut sims = Vec::new();
        for a in 0..plosive_idx.len() {
            for b in (a + 1)..plosive_idx.len() {
                sims.push(cosine(&profiles[plosive_idx[a]], &profiles[plosive_idx[b]]));
            }
        }
        let max_sim = sims.iter().cloned().fold(0.0f64, f64::max);
        if !banded {
            let g1 = sims.iter().all(|&v| (v - 1.0).abs() < 1e-9);
            println!("G1 (旧版で pa/ki/tu が完全同一): {} (最大類似度 {:.6})",
                     if g1 { "PASS" } else { "FAIL" }, max_sim);
        } else {
            let g2 = sims.iter().all(|&v| v < 1.0);
            println!("G2 (帯域版で pa/ki/tu が非同一): {} (最大類似度 {:.6})",
                     if g2 { "PASS" } else { "FAIL" }, max_sim);

            // G3: pa(0) < tu(2) < ki(1) < se(3)
            let order = [0usize, 2, 1, 3];
            let cents: Vec<Option<f64>> = order.iter()
                .map(|&i| centroid_hz(&profiles[i], &freqs)).collect();
            let names: Vec<&str> = order.iter().map(|&i| labels[i]).collect();
            let g3 = cents.windows(2).all(|w| match (w[0], w[1]) {
                (Some(a), Some(b)) => a < b,
                _ => false,
            });
            let shown: Vec<String> = cents.iter().zip(names.iter())
                .map(|(c, n)| match c {
                    Some(v) => format!("{} {:.0}Hz", n, v),
                    None => format!("{} 無音", n),
                }).collect();
            println!("G3 (重心が帯域順 pa<tu<ki<se): {} [{}]",
                     if g3 { "PASS" } else { "FAIL" }, shown.join(" < "));
        }
    }
}
