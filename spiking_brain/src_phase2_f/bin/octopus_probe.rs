//! Octopus 細胞が音素ごとに異なるオンセットパターンを出すか検証する診断
//!
//! M1 統合の前段検証: Octopus 出力が音素間で分化していなければ、
//! M1 に繋いでも意味がない。 先にここで確認する。
//!
//! 出力:
//!   - 各音素の Octopus 発火回数 (細胞別)
//!   - 各音素の Octopus 発火時刻 (オンセットのタイミング)
//!   - 音素間 Octopus fingerprint の cosine (between)

use spiking_brain::phase2_f::cochlea::{Cochlea, SAMPLES_PER_STEP};
use spiking_brain::phase2_f::cochlear_nucleus::{CochlearNucleus, N_OCTOPUS};
use spiking_brain::phase2_f::phoneme_synth::{standard_syllables, synth_syllable, LfsrNoise};
use spiking_brain::trace::cosine_similarity;

const TRIAL_DURATION_MS: f64 = 300.0;
const DT_MS: f64 = 0.5;
const TRIAL_STEPS: usize = (TRIAL_DURATION_MS / DT_MS) as usize;  // 600
const N_BINS: usize = 30;  // 10ms bin
const STEPS_PER_BIN: usize = TRIAL_STEPS / N_BINS;

/// 音素を cochlea + octopus に通し、 (発火回数[N_OCTOPUS], fingerprint[N_OCTOPUS×N_BINS], 発火時刻リスト) を返す
fn run_phoneme(
    cochlea: &mut Cochlea,
    cn: &mut CochlearNucleus,
    waveform: &[i32],
) -> (Vec<u32>, Vec<f64>, Vec<(usize, f64)>) {
    cochlea.reset();
    cn.reset();
    let mut counts = vec![0u32; N_OCTOPUS];
    let mut fp = vec![0.0f64; N_OCTOPUS * N_BINS];
    let mut spike_times: Vec<(usize, f64)> = Vec::new();

    for step in 0..TRIAL_STEPS {
        let s0 = step * SAMPLES_PER_STEP;
        let mut samples = [0i32; SAMPLES_PER_STEP];
        for i in 0..SAMPLES_PER_STEP {
            let idx = s0 + i;
            if idx < waveform.len() { samples[i] = waveform[idx]; }
        }
        let coch = cochlea.process_step(&samples);
        let oct = cn.process_step(&coch);
        let bin = (step / STEPS_PER_BIN).min(N_BINS - 1);
        for k in 0..N_OCTOPUS {
            if oct[k] > 0 {
                counts[k] += 1;
                fp[k * N_BINS + bin] += 1.0;
                spike_times.push((k, step as f64 * DT_MS));
            }
        }
    }
    (counts, fp, spike_times)
}

fn main() {
    println!("== Octopus 細胞 音素別オンセット応答 診断 ==");
    println!("  Octopus {} 細胞 (同時発火閾値 3/5/8/12 帯域)", N_OCTOPUS);

    let syllables = standard_syllables();
    let mut noise = LfsrNoise::new(0xACE1);
    let waveforms: Vec<Vec<i32>> = syllables.iter()
        .map(|s| synth_syllable(s, &mut noise))
        .collect();

    let mut cochlea = Cochlea::new();
    let mut cn = CochlearNucleus::new();

    let mut fps: Vec<Vec<f64>> = Vec::new();

    println!("\n  ── 音素別 Octopus 発火 ──");
    println!("  音素 | oct0(th3) oct1(th5) oct2(th8) oct3(th12) | 発火時刻(ms)");
    for (si, syl) in syllables.iter().enumerate() {
        let (counts, fp, times) = run_phoneme(&mut cochlea, &mut cn, &waveforms[si]);
        // 最初の数発火時刻
        let first_times: Vec<String> = times.iter().take(6)
            .map(|(k, t)| format!("c{}@{:.0}", k, t))
            .collect();
        println!("  {:>3}  |   {:>4}     {:>4}     {:>4}      {:>4}    | {}",
            syl.label, counts[0], counts[1], counts[2], counts[3],
            first_times.join(" "));
        fps.push(fp);
    }

    // 音素間 between
    println!("\n  ── Octopus fingerprint 音素間 cosine (between) ──");
    let mut sum = 0.0;
    let mut n = 0;
    for i in 0..syllables.len() {
        for j in (i+1)..syllables.len() {
            // 両方とも全ゼロなら cosine 未定義 → スキップ表示
            let zero_i = fps[i].iter().all(|&v| v == 0.0);
            let zero_j = fps[j].iter().all(|&v| v == 0.0);
            let sim = if zero_i || zero_j { 0.0 } else {
                cosine_similarity(&fps[i], &fps[j])
            };
            let note = if zero_i || zero_j { " (一方無応答)" } else { "" };
            println!("    {} vs {}: {:.4}{}",
                syllables[i].label, syllables[j].label, sim, note);
            sum += sim;
            n += 1;
        }
    }
    println!("    平均 between = {:.4}", sum / n as f64);

    println!("\n  ── 判定 ──");
    println!("  Octopus 発火回数・時刻が音素ごとに異なれば、 M1 を別アトラクタへ蹴り分ける材料になる");
    println!("  between が低い (< 0.6) ほど音素分化に有効");
}
