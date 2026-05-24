//! M0 蝸牛 (Cochlea) — 音波 → 構造化スパイク列変換
//!
//! 設計: M0_COCHLEA_DESIGN.md
//!
//! Step 1 (本ファイル): 整数 biquad IIR バンドパスフィルタ
//!   - Q1.15 固定小数点演算
//!   - 中心周波数 fc、Q ファクタを指定して帯域通過特性
//!   - 浮動小数禁止 (DESIGN_PHILOSOPHY.md 原理 4)
//!
//! 後続 Step:
//!   - Step 2: 包絡線検出 + 圧縮 + 閾値発火 (1 チャンネル)
//!   - Step 3: 20 帯域に拡張
//!   - Step 4: 音素生成器との接続
//!
//! 「学習なし固定」のため、ThermoSynapse のような可塑性は不要。
//! ニューロン構造体も持たず、純粋アルゴリズム的処理。

// ──────────────────────────────────────────────────────────────
// Q1.15 固定小数点演算ユーティリティ
// ──────────────────────────────────────────────────────────────

/// Q1.15 固定小数点のスケール (= 2^15 = 32768)。
/// 係数 c は `(c * Q15_SCALE).round() as i32` として保存。
/// 演算後は `>> 15` でスケールを戻す。
pub const Q15_SCALE: i32 = 1 << 15;

/// 整数 sin テーブル (256 entry × Q15、フィルタ係数の事前計算で参照する想定)。
/// Step 2 以降で活用、現状は将来用に予約。
pub const SIN_TABLE_SIZE: usize = 256;

// ──────────────────────────────────────────────────────────────
// 2 次 IIR バンドパスフィルタ (biquad)
// ──────────────────────────────────────────────────────────────

/// 整数 biquad IIR バンドパスフィルタ。
///
/// 差分方程式 (Direct Form I):
///   y[n] = b0*x[n] + b1*x[n-1] + b2*x[n-2] - a1*y[n-1] - a2*y[n-2]
///
/// 全係数は Q1.15 固定小数点 (i32 で保持、実効範囲 -1.0..+1.0 を ±32767 で表現)。
/// 状態 x1, x2, y1, y2 は通常スケール (Q1.15 ではない、波形そのもの)。
#[derive(Clone, Debug)]
pub struct BandpassBiquad {
    /// 分子係数 b0, b1, b2 (Q1.15)
    pub b0: i32,
    pub b1: i32,
    pub b2: i32,
    /// 分母係数 a1, a2 (Q1.15、a0=1.0 で正規化済み)
    pub a1: i32,
    pub a2: i32,
    /// 過去の入力 x[n-1], x[n-2]
    pub x1: i32,
    pub x2: i32,
    /// 過去の出力 y[n-1], y[n-2]
    pub y1: i32,
    pub y2: i32,
}

impl BandpassBiquad {
    /// 中心周波数 fc [Hz]、Q ファクタ、サンプリングレート sr [Hz] からフィルタ係数を計算。
    ///
    /// バンドパス (Constant Skirt Gain) の設計式 (RBJ Audio EQ Cookbook):
    ///   ω₀ = 2π fc / sr
    ///   α = sin(ω₀) / (2Q)
    ///
    ///   b0 =  α
    ///   b1 =  0
    ///   b2 = -α
    ///   a0 =  1 + α
    ///   a1 = -2 cos(ω₀)
    ///   a2 =  1 - α
    ///
    /// 正規化: 全係数を a0 で割る。
    ///
    /// 注: 設計時の浮動小数点計算は **初期化時のみ許容** (DESIGN_PHILOSOPHY.md 原理 3
    /// 「初期化時を除く」)。ランタイム処理は完全整数。
    pub fn new(fc_hz: f64, q: f64, sample_rate: f64) -> Self {
        let omega0 = 2.0 * std::f64::consts::PI * fc_hz / sample_rate;
        let sin_w = omega0.sin();
        let cos_w = omega0.cos();
        let alpha = sin_w / (2.0 * q);

        let a0 = 1.0 + alpha;
        let b0 = alpha / a0;
        let b1 = 0.0;
        let b2 = -alpha / a0;
        let a1 = -2.0 * cos_w / a0;
        let a2 = (1.0 - alpha) / a0;

        // Q1.15 固定小数点に変換 (round() で四捨五入)
        Self {
            b0: (b0 * Q15_SCALE as f64).round() as i32,
            b1: (b1 * Q15_SCALE as f64).round() as i32,
            b2: (b2 * Q15_SCALE as f64).round() as i32,
            a1: (a1 * Q15_SCALE as f64).round() as i32,
            a2: (a2 * Q15_SCALE as f64).round() as i32,
            x1: 0, x2: 0, y1: 0, y2: 0,
        }
    }

    /// 1 サンプル処理。x0 は入力サンプル (例: i16 範囲)、戻り値はフィルタ出力。
    ///
    /// 内部は i64 で計算してオーバーフローを防止、最後に Q15 シフト。
    #[inline]
    pub fn process(&mut self, x0: i32) -> i32 {
        // y0 = b0*x0 + b1*x1 + b2*x2 - a1*y1 - a2*y2 (全て Q15 係数 × 通常スケール状態)
        let acc: i64 = (self.b0 as i64) * (x0 as i64)
            + (self.b1 as i64) * (self.x1 as i64)
            + (self.b2 as i64) * (self.x2 as i64)
            - (self.a1 as i64) * (self.y1 as i64)
            - (self.a2 as i64) * (self.y2 as i64);
        let y0 = (acc >> 15) as i32;

        // 状態更新
        self.x2 = self.x1;
        self.x1 = x0;
        self.y2 = self.y1;
        self.y1 = y0;
        y0
    }

    /// 状態をゼロクリア (新しいトライアル開始時など)
    pub fn reset(&mut self) {
        self.x1 = 0; self.x2 = 0;
        self.y1 = 0; self.y2 = 0;
    }

    /// 複数サンプルを一括処理。
    pub fn process_block(&mut self, input: &[i32], output: &mut [i32]) {
        assert_eq!(input.len(), output.len(), "input/output length mismatch");
        for (i, &x) in input.iter().enumerate() {
            output[i] = self.process(x);
        }
    }
}

// ──────────────────────────────────────────────────────────────
// ERB スケール周波数配置
// ──────────────────────────────────────────────────────────────

/// Cambridge ERB scale (Glasberg & Moore 1990):
///   ERBs(f) = 21.4 × log10(1 + 0.00437 × f)
///   ERB(f)  = 24.7 × (4.37 × f / 1000 + 1)  [Hz]
///
/// 中心周波数 f_min..f_max の範囲を ERB 単位で等間隔に n_bands 分割。
/// 戻り値: 各帯域の中心周波数 [Hz] (Vec、長さ n_bands)。
pub fn erb_spaced_freqs(f_min: f64, f_max: f64, n_bands: usize) -> Vec<f64> {
    let erbs_min = 21.4 * (1.0 + 0.00437 * f_min).log10();
    let erbs_max = 21.4 * (1.0 + 0.00437 * f_max).log10();
    let step = (erbs_max - erbs_min) / ((n_bands - 1) as f64).max(1.0);
    (0..n_bands).map(|i| {
        let erbs = erbs_min + (i as f64) * step;
        // 逆変換: f = (10^(erbs/21.4) - 1) / 0.00437
        (10f64.powf(erbs / 21.4) - 1.0) / 0.00437
    }).collect()
}

/// 各帯域の ERB 帯域幅 [Hz]
pub fn erb_bandwidth(f_hz: f64) -> f64 {
    24.7 * (4.37 * f_hz / 1000.0 + 1.0)
}

/// 帯域フィルタの Q ファクタ (Q = fc / ERB(fc))
pub fn erb_q_factor(f_hz: f64) -> f64 {
    f_hz / erb_bandwidth(f_hz)
}

// ──────────────────────────────────────────────────────────────
// テスト
// ──────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// 純音を入れて、その周波数で大きく、別周波数で小さく出力されること
    fn sine_wave(fc: f64, sample_rate: f64, n_samples: usize, amplitude: i32) -> Vec<i32> {
        (0..n_samples).map(|i| {
            let phase = 2.0 * std::f64::consts::PI * fc * (i as f64) / sample_rate;
            ((amplitude as f64) * phase.sin()) as i32
        }).collect()
    }

    /// 信号の RMS (整数 sqrt は不要、二乗和の平方根を f64 で近似)
    fn rms(samples: &[i32]) -> f64 {
        let sum_sq: f64 = samples.iter()
            .map(|&x| (x as f64).powi(2))
            .sum();
        (sum_sq / samples.len() as f64).sqrt()
    }

    #[test]
    fn biquad_passes_center_frequency() {
        // 1000 Hz バンドパス、Q=4、サンプリング 16 kHz
        let mut filter = BandpassBiquad::new(1000.0, 4.0, 16000.0);

        // 1000 Hz の純音 (中心周波数) を入力
        let input = sine_wave(1000.0, 16000.0, 16000, 10000);
        let mut output = vec![0i32; input.len()];
        filter.process_block(&input, &mut output);

        // 過渡応答を避けて後半 8000 サンプルで RMS 比較
        let in_rms = rms(&input[8000..]);
        let out_rms = rms(&output[8000..]);
        // 中心周波数では出力 ≒ 入力 (constant skirt gain では peak gain ≈ Q)
        // ただし正規化されているので 0.7 倍 ~ 1.0 倍程度の通過
        let ratio = out_rms / in_rms;
        assert!(ratio > 0.3 && ratio < 2.0,
            "1000 Hz at fc=1000: out/in = {}, expected ~0.5-1.5", ratio);
    }

    #[test]
    fn biquad_attenuates_off_center_frequency() {
        // 1000 Hz バンドパス、Q=4
        let mut filter = BandpassBiquad::new(1000.0, 4.0, 16000.0);

        // 4000 Hz (中心から 2 oct 上) を入力
        let input = sine_wave(4000.0, 16000.0, 16000, 10000);
        let mut output = vec![0i32; input.len()];
        filter.process_block(&input, &mut output);

        let in_rms = rms(&input[8000..]);
        let out_rms = rms(&output[8000..]);
        let ratio = out_rms / in_rms;
        // 中心から離れた周波数は大幅に減衰
        assert!(ratio < 0.3,
            "4000 Hz at fc=1000: out/in = {}, expected < 0.3", ratio);
    }

    #[test]
    fn biquad_attenuates_low_frequency() {
        // 1000 Hz バンドパス
        let mut filter = BandpassBiquad::new(1000.0, 4.0, 16000.0);
        // 100 Hz (中心から 3.3 oct 下) を入力
        let input = sine_wave(100.0, 16000.0, 16000, 10000);
        let mut output = vec![0i32; input.len()];
        filter.process_block(&input, &mut output);

        let in_rms = rms(&input[8000..]);
        let out_rms = rms(&output[8000..]);
        let ratio = out_rms / in_rms;
        assert!(ratio < 0.3,
            "100 Hz at fc=1000: out/in = {}, expected < 0.3", ratio);
    }

    #[test]
    fn erb_freqs_monotonic_and_logarithmic() {
        let freqs = erb_spaced_freqs(50.0, 4000.0, 20);
        assert_eq!(freqs.len(), 20);
        // 単調増加
        for i in 1..freqs.len() {
            assert!(freqs[i] > freqs[i-1],
                "erb_spaced_freqs not monotonic at i={}", i);
        }
        // 端点が指定どおり
        assert!((freqs[0] - 50.0).abs() < 0.5);
        assert!((freqs[19] - 4000.0).abs() < 5.0);
        // 対数的: 後半の方が間隔が広い
        let first_step = freqs[1] - freqs[0];
        let last_step = freqs[19] - freqs[18];
        assert!(last_step > first_step * 5.0,
            "ERB scale should be logarithmic: first_step={}, last_step={}",
            first_step, last_step);
    }

    #[test]
    fn erb_bandwidth_reasonable() {
        // Glasberg & Moore 1990 の値と整合
        assert!((erb_bandwidth(100.0) - 35.5).abs() < 1.0);   // ~35.5 Hz
        assert!((erb_bandwidth(1000.0) - 132.6).abs() < 1.0); // ~132 Hz
        assert!((erb_bandwidth(4000.0) - 456.8).abs() < 2.0); // ~457 Hz
    }

    #[test]
    fn reset_clears_state() {
        let mut filter = BandpassBiquad::new(1000.0, 4.0, 16000.0);
        // しばらく処理
        for _ in 0..100 {
            filter.process(10000);
        }
        assert_ne!(filter.x1, 0);
        // reset で状態クリア
        filter.reset();
        assert_eq!(filter.x1, 0);
        assert_eq!(filter.x2, 0);
        assert_eq!(filter.y1, 0);
        assert_eq!(filter.y2, 0);
    }
}
