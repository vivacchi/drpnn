//! 音素生成器 — 整数フォルマント合成 + LFSR ノイズ
//!
//! 設計: M0_COCHLEA_DESIGN.md §4
//!
//! 目的: M0 蝸牛 + M1 評価用の音響信号 (16 kHz / i16) を生成する.
//! 評価入力 A-E パターン (時間オフセット) の代わりに、生物的に妥当な音素を使う.
//!
//! 内容:
//!   - 整数 sin テーブル (256 entry) で母音 5 種を加算合成
//!   - LFSR (線形帰還シフトレジスタ) で決定論的ノイズ
//!   - 子音 (破裂、摩擦、鼻音) を簡易合成
//!   - 音節 (CV 構造) と sequence
//!
//! 設計原則:
//!   - 整数演算のみ (sin テーブルも i32 で持つ)
//!   - 決定論的 (確率なし、初期化時の seed のみ)
//!   - 16 kHz / i16 出力

// ──────────────────────────────────────────────────────────────
// 整数 sin テーブル
// ──────────────────────────────────────────────────────────────

pub const SIN_TABLE_SIZE: usize = 256;
pub const SIN_AMPLITUDE: i32 = 16384;  // Q15 内の単位振幅 (1.0 を 16384 で表現)
pub const SAMPLE_RATE_HZ: f64 = 16000.0;

/// 256 エントリの整数 sin テーブル (振幅 SIN_AMPLITUDE).
/// 初期化時に std::sync::OnceLock で 1 回計算 (浮動小数は初期化のみ許容).
pub fn sin_table() -> &'static [i32; SIN_TABLE_SIZE] {
    use std::sync::OnceLock;
    static TABLE: OnceLock<[i32; SIN_TABLE_SIZE]> = OnceLock::new();
    TABLE.get_or_init(|| {
        let mut t = [0i32; SIN_TABLE_SIZE];
        for i in 0..SIN_TABLE_SIZE {
            let phase = 2.0 * std::f64::consts::PI * (i as f64) / (SIN_TABLE_SIZE as f64);
            t[i] = (SIN_AMPLITUDE as f64 * phase.sin()).round() as i32;
        }
        t
    })
}

/// 位相 (Q24 = 24bit 固定小数点、上 8bit がテーブルインデックス) から sin 値を取得.
/// 線形補間あり.
#[inline]
pub fn sin_lookup(phase_q24: u32) -> i32 {
    let table = sin_table();
    // 上位 8bit: テーブルインデックス
    let idx = ((phase_q24 >> 16) & 0xFF) as usize;
    let idx_next = (idx + 1) & 0xFF;
    // 下位 16bit: 補間係数 (0..65535)
    let frac = (phase_q24 & 0xFFFF) as i32;

    let s0 = table[idx];
    let s1 = table[idx_next];
    // s0 + (s1 - s0) * frac / 65536
    s0 + (((s1 - s0) * frac) >> 16)
}

/// 周波数 f [Hz] から 1 サンプルあたりの位相増分 (Q24) を計算.
#[inline]
pub fn freq_to_phase_step(f_hz: f64) -> u32 {
    let step = f_hz * (1u64 << 24) as f64 / SAMPLE_RATE_HZ;
    step.round() as u32
}

// ──────────────────────────────────────────────────────────────
// 母音フォルマント合成
// ──────────────────────────────────────────────────────────────

/// 母音 1 つを定義する 3 フォルマント. 周波数 [Hz] と振幅比 (× 1024 で整数化).
#[derive(Clone, Copy, Debug)]
pub struct Vowel {
    pub label: char,
    pub formants_hz: [f64; 3],  // F1, F2, F3
    pub amplitudes: [i32; 3],   // 振幅 (生成時 i32 範囲)
}

/// 標準的な日本語 5 母音.
/// 値は Klatt 1980 と日本語音声学の文献に基づく中央値.
pub fn vowels() -> [Vowel; 5] {
    [
        Vowel {
            label: 'a',
            formants_hz: [800.0, 1300.0, 2700.0],
            amplitudes: [4000, 2800, 1200],
        },
        Vowel {
            label: 'i',
            formants_hz: [300.0, 2300.0, 3000.0],
            amplitudes: [4000, 2000, 1600],
        },
        Vowel {
            label: 'u',
            formants_hz: [350.0, 850.0, 2400.0],
            amplitudes: [4000, 3200, 800],
        },
        Vowel {
            label: 'e',
            formants_hz: [500.0, 2000.0, 2700.0],
            amplitudes: [4000, 2400, 1200],
        },
        Vowel {
            label: 'o',
            formants_hz: [500.0, 900.0, 2400.0],
            amplitudes: [4000, 3200, 1200],
        },
    ]
}

/// 母音波形を生成.
/// duration_ms: 持続時間 [ms]、戻り値: i32 波形 (i16 範囲を想定).
pub fn synth_vowel(vowel: &Vowel, duration_ms: f64) -> Vec<i32> {
    let n_samples = (duration_ms * SAMPLE_RATE_HZ / 1000.0) as usize;
    let phase_steps: [u32; 3] = [
        freq_to_phase_step(vowel.formants_hz[0]),
        freq_to_phase_step(vowel.formants_hz[1]),
        freq_to_phase_step(vowel.formants_hz[2]),
    ];
    let mut phases: [u32; 3] = [0; 3];
    let mut out = Vec::with_capacity(n_samples);

    // attack/release エンベロープ (前後 10ms はランプ)
    let ramp_samples = ((10.0 * SAMPLE_RATE_HZ / 1000.0) as usize).min(n_samples / 4);

    for i in 0..n_samples {
        let mut sample: i32 = 0;
        for f in 0..3 {
            let s = sin_lookup(phases[f]);
            // s は -SIN_AMPLITUDE..+SIN_AMPLITUDE 範囲 (= ±16384)
            // amplitude をかけてシフト ( >> 14 で正規化)
            sample = sample.saturating_add((s * vowel.amplitudes[f]) >> 14);
            phases[f] = phases[f].wrapping_add(phase_steps[f]);
        }
        // attack/release エンベロープ
        let env = if i < ramp_samples {
            (i * 1024 / ramp_samples) as i32
        } else if i >= n_samples - ramp_samples {
            (((n_samples - i) * 1024) / ramp_samples) as i32
        } else {
            1024
        };
        sample = (sample * env) >> 10;
        out.push(sample);
    }
    out
}

// ──────────────────────────────────────────────────────────────
// LFSR ノイズ (決定論的)
// ──────────────────────────────────────────────────────────────

/// 16-bit LFSR ノイズ生成器. 周期 65535.
#[derive(Clone, Debug)]
pub struct LfsrNoise {
    pub state: u16,
}

impl LfsrNoise {
    pub fn new(seed: u16) -> Self {
        // 0 を避ける (LFSR は 0 から脱出できない)
        let s = if seed == 0 { 0xACE1 } else { seed };
        Self { state: s }
    }

    /// 1 サンプルのノイズを生成. 戻り値: -SIN_AMPLITUDE..+SIN_AMPLITUDE 程度
    #[inline]
    pub fn next_sample(&mut self) -> i32 {
        let bit = ((self.state >> 0)
            ^ (self.state >> 2)
            ^ (self.state >> 3)
            ^ (self.state >> 5)) & 1;
        self.state = (self.state >> 1) | (bit << 15);
        // i16 範囲のノイズに変換
        (self.state as i16) as i32
    }
}

// ──────────────────────────────────────────────────────────────
// 子音合成
// ──────────────────────────────────────────────────────────────

/// 子音の種類.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Consonant {
    /// 破裂音 (例 /p/, /t/, /k/): 短い無音 + broadband burst
    Plosive { burst_freq_low: f64, burst_freq_high: f64 },
    /// 摩擦音 (例 /s/, /sh/): 持続的な高周波ノイズ
    Fricative { freq_low: f64, freq_high: f64 },
    /// 鼻音 (例 /n/, /m/): 低周波フォルマント
    Nasal { f1: f64, f2: f64 },
}

/// 子音波形を生成. duration_ms: 持続時間.
pub fn synth_consonant(c: Consonant, duration_ms: f64, noise: &mut LfsrNoise) -> Vec<i32> {
    let n_samples = (duration_ms * SAMPLE_RATE_HZ / 1000.0) as usize;
    let mut out = Vec::with_capacity(n_samples);
    match c {
        Consonant::Plosive { burst_freq_low: _, burst_freq_high: _ } => {
            // 10ms 無音 + 20ms broadband burst (LFSR ノイズの振幅変調)
            let silent = ((10.0 * SAMPLE_RATE_HZ / 1000.0) as usize).min(n_samples / 2);
            for _ in 0..silent {
                out.push(0);
            }
            // burst エンベロープ (急速減衰)
            for i in silent..n_samples {
                let decay = ((n_samples - i) * 1024 / (n_samples - silent)) as i32;
                let n = noise.next_sample();
                out.push((n * decay) >> 10);
            }
        }
        Consonant::Fricative { .. } => {
            // 持続的 broadband ノイズ (前後 attack/release エンベロープ付き)
            let ramp = ((5.0 * SAMPLE_RATE_HZ / 1000.0) as usize).min(n_samples / 4);
            for i in 0..n_samples {
                let n = noise.next_sample();
                let env = if i < ramp {
                    (i * 1024 / ramp) as i32
                } else if i >= n_samples - ramp {
                    (((n_samples - i) * 1024) / ramp) as i32
                } else {
                    1024
                };
                out.push((n * env) >> 10);
            }
        }
        Consonant::Nasal { f1, f2 } => {
            // 低周波 2 フォルマント
            let ps = [freq_to_phase_step(f1), freq_to_phase_step(f2)];
            let amps = [3000i32, 1500];
            let mut phases = [0u32; 2];
            let ramp = ((5.0 * SAMPLE_RATE_HZ / 1000.0) as usize).min(n_samples / 4);
            for i in 0..n_samples {
                let mut s = 0i32;
                for f in 0..2 {
                    s = s.saturating_add((sin_lookup(phases[f]) * amps[f]) >> 14);
                    phases[f] = phases[f].wrapping_add(ps[f]);
                }
                let env = if i < ramp {
                    (i * 1024 / ramp) as i32
                } else if i >= n_samples - ramp {
                    (((n_samples - i) * 1024) / ramp) as i32
                } else {
                    1024
                };
                out.push((s * env) >> 10);
            }
        }
    }
    out
}

// ──────────────────────────────────────────────────────────────
// 音節 (CV) と sequence
// ──────────────────────────────────────────────────────────────

/// 1 音節 = 子音 + 母音.
#[derive(Clone, Copy, Debug)]
pub struct Syllable {
    pub label: &'static str,
    pub consonant: Consonant,
    pub vowel: Vowel,
}

/// 標準的な 5 音節 (pa, ki, tu, se, mo) を生成.
/// A-E パターンの置き換え.
pub fn standard_syllables() -> [Syllable; 5] {
    let v = vowels();
    [
        Syllable {
            label: "pa",
            consonant: Consonant::Plosive { burst_freq_low: 500.0, burst_freq_high: 2000.0 },
            vowel: v[0], // a
        },
        Syllable {
            label: "ki",
            consonant: Consonant::Plosive { burst_freq_low: 2000.0, burst_freq_high: 4000.0 },
            vowel: v[1], // i
        },
        Syllable {
            label: "tu",
            consonant: Consonant::Plosive { burst_freq_low: 1500.0, burst_freq_high: 3500.0 },
            vowel: v[2], // u
        },
        Syllable {
            label: "se",
            consonant: Consonant::Fricative { freq_low: 3000.0, freq_high: 8000.0 },
            vowel: v[3], // e
        },
        Syllable {
            label: "mo",
            consonant: Consonant::Nasal { f1: 250.0, f2: 1500.0 },
            vowel: v[4], // o
        },
    ]
}

/// 音節を合成. 子音 (30ms) + 母音 (170ms) = 200ms.
/// 戻り値: 16kHz / i32 (i16 範囲) の波形.
pub fn synth_syllable(syl: &Syllable, noise: &mut LfsrNoise) -> Vec<i32> {
    synth_syllable_scaled(syl, noise, 1.0)
}

/// 時間圧縮版: duration を speed 倍だけ短縮 (formant 周波数は保持)。
/// speed=1.0 で標準 (200ms)、 speed=3.0 で 67ms (STDP 因果窓 80ms 内に収まる)。
///
/// 「入力を速める」 仮説の検証用。 ピッチ (resampling) ではなく duration 短縮なので
/// formant 周波数は変わらず、 cochlea 帯域から外れない。 音素を STDP 窓に収め、
/// 「音素全体を 1 つの因果イベントとして M1 が学習できるか」 を試す。
pub fn synth_syllable_scaled(syl: &Syllable, noise: &mut LfsrNoise, speed: f64) -> Vec<i32> {
    let consonant_ms = 30.0 / speed;
    let vowel_ms = 170.0 / speed;
    let mut wave = synth_consonant(syl.consonant, consonant_ms, noise);
    let vowel_wave = synth_vowel(&syl.vowel, vowel_ms);
    wave.extend(vowel_wave);
    wave
}

// ──────────────────────────────────────────────────────────────
// テスト
// ──────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sin_table_initialized() {
        let t = sin_table();
        // sin(0) = 0
        assert_eq!(t[0], 0);
        // sin(π/2) ≈ 1.0 = SIN_AMPLITUDE
        assert!((t[64] - SIN_AMPLITUDE).abs() < 2);
        // sin(π) ≈ 0
        assert!(t[128].abs() < 2);
        // sin(3π/2) ≈ -1.0
        assert!((t[192] + SIN_AMPLITUDE).abs() < 2);
    }

    #[test]
    fn sin_lookup_smooth() {
        // 隣接インデックス間で線形補間
        let v_at_0 = sin_lookup(0);
        let v_mid = sin_lookup(1 << 15);  // インデックス 0 と 1 の中間
        let v_at_1 = sin_lookup(1 << 16);
        // 中間値は両端の中間付近
        let avg = (v_at_0 + v_at_1) / 2;
        assert!((v_mid - avg).abs() < 100);
    }

    #[test]
    fn vowels_have_distinct_formants() {
        let vs = vowels();
        // a と i の F1, F2 は十分違う
        assert!((vs[0].formants_hz[0] - vs[1].formants_hz[0]).abs() > 400.0,
            "a vs i F1");
        assert!((vs[0].formants_hz[1] - vs[1].formants_hz[1]).abs() > 800.0,
            "a vs i F2");
    }

    #[test]
    fn vowel_synthesis_correct_length() {
        let v = vowels()[0];  // /a/
        let wave = synth_vowel(&v, 100.0);
        assert_eq!(wave.len(), 1600); // 100ms × 16 kHz
    }

    #[test]
    fn vowel_synthesis_has_signal() {
        let v = vowels()[0];
        let wave = synth_vowel(&v, 100.0);
        // エネルギー (二乗和) が十分大きい
        let energy: i64 = wave.iter().map(|&x| (x as i64) * (x as i64)).sum();
        assert!(energy > 100_000_000, "vowel energy too low: {}", energy);
        // 中央 200 サンプル窓の RMS が信号として認識される強度
        let mid_start = wave.len() / 2 - 100;
        let mid_sq: i64 = wave[mid_start..mid_start+200]
            .iter()
            .map(|&x| (x as i64) * (x as i64))
            .sum();
        let mid_rms = ((mid_sq / 200) as f64).sqrt();
        assert!(mid_rms > 500.0, "vowel mid RMS too low: {}", mid_rms);
    }

    #[test]
    fn lfsr_deterministic() {
        let mut n1 = LfsrNoise::new(0xACE1);
        let mut n2 = LfsrNoise::new(0xACE1);
        for _ in 0..1000 {
            assert_eq!(n1.next_sample(), n2.next_sample());
        }
    }

    #[test]
    fn lfsr_period_long() {
        let mut n = LfsrNoise::new(0xACE1);
        let initial = n.state;
        let mut periods = 0;
        for _ in 0..70000 {
            n.next_sample();
            if n.state == initial {
                periods += 1;
            }
        }
        // 16-bit LFSR は周期 65535
        assert!(periods >= 1, "LFSR should cycle within 70000 samples");
    }

    #[test]
    fn consonant_plosive_has_silence_then_burst() {
        let mut n = LfsrNoise::new(0xACE1);
        let wave = synth_consonant(
            Consonant::Plosive { burst_freq_low: 500.0, burst_freq_high: 2000.0 },
            30.0,
            &mut n,
        );
        assert_eq!(wave.len(), 480);  // 30ms × 16 kHz
        // 最初の 10ms は無音
        for i in 0..160 {
            assert_eq!(wave[i], 0, "plosive should be silent at sample {}", i);
        }
        // burst エネルギーがある
        let burst_energy: i64 = wave[160..].iter().map(|&x| (x as i64).abs()).sum();
        assert!(burst_energy > 10_000, "plosive burst too weak: {}", burst_energy);
    }

    #[test]
    fn syllable_pa_has_consonant_then_vowel() {
        let syls = standard_syllables();
        let mut n = LfsrNoise::new(0xACE1);
        let wave = synth_syllable(&syls[0], &mut n); // "pa"
        // 30ms + 170ms = 200ms = 3200 sample
        assert_eq!(wave.len(), 3200);
        // 子音部分の最初の 10ms (160 sample) は無音
        for i in 0..160 {
            assert_eq!(wave[i], 0);
        }
        // 母音部分 (480..3200) の RMS が十分大きい
        let vowel_sq: i64 = wave[480..3200].iter()
            .map(|&x| (x as i64) * (x as i64))
            .sum();
        let vowel_rms = ((vowel_sq / (3200 - 480) as i64) as f64).sqrt();
        assert!(vowel_rms > 500.0, "vowel RMS too low: {}", vowel_rms);
    }

    #[test]
    fn all_syllables_synthesizable() {
        let syls = standard_syllables();
        let mut n = LfsrNoise::new(0xACE1);
        for syl in &syls {
            let wave = synth_syllable(syl, &mut n);
            assert_eq!(wave.len(), 3200, "{} duration", syl.label);
            // 有意なエネルギー
            let energy: i64 = wave.iter().map(|&x| (x as i64).abs()).sum();
            assert!(energy > 100_000, "{} energy too low: {}", syl.label, energy);
        }
    }
}
