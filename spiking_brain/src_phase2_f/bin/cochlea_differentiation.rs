//! M0 蝸牛出力の音素間分化度を直接測定する診断ツール
//!
//! 目的: M0+M1+音素 タスクで between が高い (分化しない) 問題が、
//!       上流 (M0 cochlea が音素を区別できていない) か
//!       下流 (M1 が cochlea の分化を潰す) かを切り分ける。
//!
//! 手法:
//!   - 5 音素 (pa, ki, tu, se, mo) を cochlea に通し、 20ch × 30bin fingerprint を作る
//!   - cochlea は決定論的なので within=1.0 自明 → between (音素間 cosine) のみ測定
//!   - between が低い (例 0.2) → cochlea は音素をよく分化している → 問題は M1 (下流)
//!   - between が高い (例 0.9) → cochlea が音素を区別できていない → 問題は M0/入力 (上流)
//!
//! 比較対象: M1 出力の between (M0+M1 5K で 0.600)

use spiking_brain::phase2_f::cochlea::{Cochlea, SAMPLES_PER_STEP};
use spiking_brain::phase2_f::phoneme_synth::{standard_syllables, synth_syllable, LfsrNoise};
use spiking_brain::trace::cosine_similarity;

const TRIAL_DURATION_MS: f64 = 300.0;
const DT_MS: f64 = 0.5;
const TRIAL_STEPS: usize = (TRIAL_DURATION_MS / DT_MS) as usize;  // 600
const N_CH: usize = 20;
const N_BINS: usize = 30;  // 10ms bin
const STEPS_PER_BIN: usize = TRIAL_STEPS / N_BINS;  // 20

/// cochlea 出力を 20ch × 30bin fingerprint に集約 (各 bin で電流を合計)
fn cochlea_fingerprint(cochlea: &mut Cochlea, waveform: &[i32]) -> Vec<f64> {
    cochlea.reset();
    let mut fp = vec![0.0f64; N_CH * N_BINS];
    let mut ch_total = vec![0i64; N_CH];
    for step in 0..TRIAL_STEPS {
        let s0 = step * SAMPLES_PER_STEP;
        let mut samples = [0i32; SAMPLES_PER_STEP];
        for i in 0..SAMPLES_PER_STEP {
            let idx = s0 + i;
            if idx < waveform.len() { samples[i] = waveform[idx]; }
        }
        let out = cochlea.process_step(&samples);
        let bin = (step / STEPS_PER_BIN).min(N_BINS - 1);
        for ch in 0..N_CH.min(out.len()) {
            fp[ch * N_BINS + bin] += out[ch] as f64;
            ch_total[ch] += out[ch] as i64;
        }
    }
    // チャネル別総活動を stderr 的に保持したいが、 ここでは fp だけ返す
    let _ = ch_total;
    fp
}

/// チャネル別総電流 (音素ごとの「どの周波数帯が活性か」 を見る)
fn channel_totals(cochlea: &mut Cochlea, waveform: &[i32]) -> Vec<i64> {
    cochlea.reset();
    let mut ch_total = vec![0i64; N_CH];
    for step in 0..TRIAL_STEPS {
        let s0 = step * SAMPLES_PER_STEP;
        let mut samples = [0i32; SAMPLES_PER_STEP];
        for i in 0..SAMPLES_PER_STEP {
            let idx = s0 + i;
            if idx < waveform.len() { samples[i] = waveform[idx]; }
        }
        let out = cochlea.process_step(&samples);
        for ch in 0..N_CH.min(out.len()) {
            ch_total[ch] += out[ch] as i64;
        }
    }
    ch_total
}

fn main() {
    println!("== M0 蝸牛出力 音素間分化度診断 ==");
    println!("  5 音素を cochlea に通し、 20ch×30bin fingerprint の音素間 cosine を測定");
    println!("  比較: M1 出力 between = 0.600 (M0+M1 5K)");

    let syllables = standard_syllables();
    let mut noise = LfsrNoise::new(0xACE1);
    let waveforms: Vec<Vec<i32>> = syllables.iter()
        .map(|s| synth_syllable(s, &mut noise))
        .collect();

    let mut cochlea = Cochlea::new();

    // fingerprint 構築
    let fps: Vec<Vec<f64>> = waveforms.iter()
        .map(|w| cochlea_fingerprint(&mut cochlea, w))
        .collect();

    // ── 音素間 between (cosine) ──
    println!("\n  ── 音素間 cosine 類似度 (between) ──");
    let mut sum = 0.0;
    let mut n = 0;
    for i in 0..syllables.len() {
        for j in (i+1)..syllables.len() {
            let sim = cosine_similarity(&fps[i], &fps[j]);
            println!("    {} vs {}: {:.4}", syllables[i].label, syllables[j].label, sim);
            sum += sim;
            n += 1;
        }
    }
    let mean_between = sum / n as f64;
    println!("    平均 between = {:.4}", mean_between);

    // ── チャネル別活動プロファイル (どの周波数帯が音素を分けるか) ──
    println!("\n  ── チャネル別総電流 (周波数帯プロファイル) ──");
    println!("  ch | {}", syllables.iter().map(|s| format!("{:>7}", s.label)).collect::<String>());
    let totals: Vec<Vec<i64>> = waveforms.iter()
        .map(|w| channel_totals(&mut cochlea, w))
        .collect();
    for ch in 0..N_CH {
        print!("  {:>2} |", ch);
        for si in 0..syllables.len() {
            print!(" {:>6}", totals[si][ch]);
        }
        println!();
    }

    // ── 判定 ──
    println!("\n  ── 判定 ──");
    if mean_between < 0.5 {
        println!("  cochlea between = {:.3} < 0.5 → M0 は音素をよく分化している", mean_between);
        println!("  → 問題は下流 (M1 が cochlea の分化を潰している)");
    } else if mean_between < 0.8 {
        println!("  cochlea between = {:.3} (中間) → M0 の分化は部分的", mean_between);
        println!("  → 上流・下流 両方に改善余地");
    } else {
        println!("  cochlea between = {:.3} > 0.8 → M0 が音素を区別できていない", mean_between);
        println!("  → 問題は上流 (M0 cochlea or 音素生成、 garbage in)");
    }
}
