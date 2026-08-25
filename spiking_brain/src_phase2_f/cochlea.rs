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
    /// 過去の出力 y[n-1], y[n-2]。**Q{STATE_SHIFT} の高精度表現** (2026-08-25)。
    ///
    /// 旧実装は毎サンプル整数に丸めて状態に戻していた。低周波・高 Q では
    /// `b0 = alpha/a0` が極端に小さく (50Hz・ERB Q で約 192/32768 = 0.0059)、
    /// 1 サンプルあたりの寄与が数十しかないので、丸め損失が相対的に巨大だった。
    /// 実測: 最弱フォルマント振幅 3200 の純音が **50-65Hz で全く発火しない**
    /// (帯域数を 240 本に増やしても同じ = 幾何でなく量子化が原因)。
    /// しかも Q を上げると alpha は 1/Q に比例して小さくなるので、
    /// **選択性を上げようとするほど低域が聞こえなくなる**という悪循環だった。
    ///
    /// 状態に小数ビットを持たせるのは固定小数点 IIR の定石。
    /// **整数のまま桁を増やすだけ**なので原理 4 (整数演算) に一切触れない。
    pub y1: i64,
    pub y2: i64,
    /// Q15 の戻し方を「ゼロ方向切り捨て」にするか (**既定 true**・2026-08-25 ユーザー判断 案ア)。
    ///
    /// `false` にすると従来の算術シフト (floor) に戻る = ロールバック経路。
    ///
    /// 2026-08-25 実測: 既定の `acc >> 15` (算術シフト = floor) では
    /// **40 帯域中 24 本のインパルス応答が減衰しきらず自己発振する**
    /// (帯域0 fc=50Hz は前半 max|y|=1876 に対し n>=3900 の末尾 max|y|=2491 で増大)。
    /// 極半径 r=0.994108 の理論減衰は 1275 サンプルなので、フィルタが自分の
    /// 線形ダイナミクスに従っていない = 量子化由来のリミットサイクル。
    ///
    /// 丸めモード比較 (F_MAX=4000・40帯域・4000サンプル):
    ///   `acc >> 15`        (floor)          残存 24 帯域・最悪 2491
    ///   `(acc+1<<14) >> 15` (最近接)         残存 40 帯域・最悪 1231 (悪化)
    ///   `acc / 32768`      (ゼロ方向)       **残存 0 帯域・最悪 0**
    ///
    /// ゼロ方向切り捨ては常に |y| を減らすので受動的な損失として働き、
    /// リミットサイクルを構造的に不可能にする (固定小数点 IIR の定石)。
    /// 正解はフィルタ理論側にある — 安定な線形フィルタの
    /// インパルス応答はゼロに収束する。調整パラメータではない。
    pub magnitude_truncation: bool,
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
            magnitude_truncation: true,
        }
    }

    /// 1 サンプル処理。x0 は入力サンプル (例: i16 範囲)、戻り値はフィルタ出力。
    ///
    /// 内部は i64 で計算してオーバーフローを防止、最後に Q15 シフト。
    #[inline]
    pub fn process(&mut self, x0: i32) -> i32 {
        // y0 = b0*x0 + b1*x1 + b2*x2 - a1*y1 - a2*y2
        // 状態 y1/y2 は 2^STATE_SHIFT 倍で保持しているので、分子側も同じだけ持ち上げる。
        let num: i64 = (self.b0 as i64) * (x0 as i64)
            + (self.b1 as i64) * (self.x1 as i64)
            + (self.b2 as i64) * (self.x2 as i64);
        let acc: i64 = (num << STATE_SHIFT)
            - (self.a1 as i64) * self.y1
            - (self.a2 as i64) * self.y2;

        // Q15 を戻す。ゼロ方向切り捨て (受動的損失) でリミットサイクルを構造的に潰す。
        let y0_hp: i64 = if self.magnitude_truncation {
            let bias: i64 = if acc < 0 { 32767 } else { 0 };
            (acc + bias) >> 15
        } else {
            acc >> 15 // 旧挙動 (算術シフト = floor)
        };

        self.x2 = self.x1;
        self.x1 = x0;
        self.y2 = self.y1;
        self.y1 = y0_hp;

        // 返り値は従来と同じ整数スケール (下流は無変更)
        let out_bias: i64 = if y0_hp < 0 && self.magnitude_truncation {
            (1 << STATE_SHIFT) - 1
        } else {
            0
        };
        ((y0_hp + out_bias) >> STATE_SHIFT) as i32
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
// Step 2: 包絡線検出 + 圧縮 + 閾値発火 (1 チャンネル)
// ──────────────────────────────────────────────────────────────

/// 包絡線検出器: 半波整流 + 整数 leaky integrator.
///
/// 生物対応: IHC の機械→電気変換における AC→DC 変換 (DC 成分 = rate code 源)。
/// 出力は |bandpass_out| を low-pass で平滑した「強度」の指標。
///
/// 漏れ係数 LEAK_SHIFT で時定数を制御:
///   env[n+1] = env[n] - (env[n] >> LEAK_SHIFT) + |x|
///   LEAK_SHIFT=4 で 2^4=16 サンプル時定数 ≒ 1ms @ 16kHz
#[derive(Clone, Debug)]
pub struct EnvelopeDetector {
    pub env: i32,
    pub leak_shift: i32,
}

impl EnvelopeDetector {
    pub fn new(leak_shift: i32) -> Self {
        Self { env: 0, leak_shift }
    }

    /// 1 サンプル処理。入力は biquad の出力 (符号あり)、戻り値は包絡線値 (非負)。
    #[inline]
    pub fn process(&mut self, x: i32) -> i32 {
        let rectified = x.abs();
        // leaky integrator: env -= env >> shift; env += rectified
        self.env -= self.env >> self.leak_shift;
        self.env = self.env.saturating_add(rectified);
        self.env
    }

    pub fn reset(&mut self) { self.env = 0; }
}

/// 整数平方根 (Newton-Raphson、~5 反復で 32bit 入力に対し収束)
pub fn isqrt(n: i32) -> i32 {
    if n <= 0 { return 0; }
    let mut x = n;
    let mut y = (x + 1) / 2;
    while y < x {
        x = y;
        y = (x + n / x) / 2;
    }
    x
}

/// 圧縮器: 整数平方根による動的レンジ圧縮.
///
/// 生物対応: 外有毛細胞 (OHC) の能動増幅による対数圧縮の簡易版.
/// 130 dB の物理ダイナミックレンジを 50 dB 程度の神経出力に圧縮.
///
/// sqrt(x) は対数より穏やかな圧縮だが整数演算で実装容易.
/// log2 を使うなら別実装 (将来検討).
#[inline]
pub fn compress_sqrt(env: i32) -> i32 {
    isqrt(env)
}

/// 閾値発火生成器: 1 チャンネル分.
///
/// 入力: 圧縮済みの包絡線値 (非負).
/// 出力: 発火イベント (この step で発火したか) のフラグ.
///
/// 内部に蓄積カウンタを持ち、入力が閾値を超え続けると周期的に発火.
/// パルス幅 (連続発火継続 step 数) と不応期を持つ.
#[derive(Clone, Debug)]
pub struct FireGenerator {
    /// 発火させる包絡線下限 (これ未満は発火しない)
    pub threshold: i32,
    /// 不応期残り step (>0 なら発火しない)
    pub refractory_remaining: i32,
    /// 不応期長 step
    pub refractory_period: i32,
}

impl FireGenerator {
    pub fn new(threshold: i32, refractory_period: i32) -> Self {
        Self { threshold, refractory_remaining: 0, refractory_period }
    }

    /// 1 step 処理. 戻り値: 発火したか.
    #[inline]
    pub fn process(&mut self, compressed_env: i32) -> bool {
        if self.refractory_remaining > 0 {
            self.refractory_remaining -= 1;
            return false;
        }
        if compressed_env >= self.threshold {
            self.refractory_remaining = self.refractory_period;
            true
        } else {
            false
        }
    }

    pub fn reset(&mut self) { self.refractory_remaining = 0; }
}

// ──────────────────────────────────────────────────────────────
// Step 3: 20 帯域に拡張 (Cochlea 構造体)
// ──────────────────────────────────────────────────────────────

/// 蝸牛 1 つ分の構造 (20 帯域).
///
/// 入力: 1 サンプル (i32, i16 範囲)
/// 出力: 1 step (8 サンプル) ごとに 20 input neuron 用電流ベクトル
///
/// パイプライン:
///   各サンプルごと:
///     [BandpassBiquad × 20] → [EnvelopeDetector × 20]
///   8 サンプル (1 step) ごと:
///     [圧縮 + 閾値発火] → input current[20] を生成
pub const N_BANDS: usize = 40;  // 20→40: 解像度倍化 (ki/se 分化改善実験 Step 1 で確認済み)
pub const SAMPLE_RATE_HZ: f64 = 16000.0;
pub const SAMPLES_PER_STEP: usize = 8;  // DT_MS=0.5ms × 16kHz = 8
pub const F_MIN_HZ: f64 = 50.0;
pub const F_MAX_HZ: f64 = 4000.0;

/// 発火閾値 (圧縮済み包絡線の閾値).
/// 純音 amplitude 8000 で中心帯域は sqrt(env)≈230、非中心帯域は ~150 程度になる.
/// 200 にすると中心帯域のみ発火 (周波数選択性確保).
pub const FIRE_THRESHOLD: i32 = 160;

/// biquad の内部状態が持つ小数ビット数 (2026-08-25 追加)。
///
/// 係数は Q1.15 のまま。**状態 y[n-1], y[n-2] だけ**を 2^STATE_SHIFT 倍して保持し、
/// 再帰の途中で整数に丸めないようにする。量子化雑音が 1/2^STATE_SHIFT になる。
/// 返り値は従来どおり整数スケールなので、下流 (包絡線検出器) は無変更。
pub const STATE_SHIFT: i32 = 8;

/// 周波数選択性の鋭さ倍率 (ERB の Q に掛ける・2026-08-25 追加)。
///
/// 設計書 §1.5: 外有毛細胞 (OHC) は「周波数選択性を **1/3 oct → 1/10 oct** まで
/// 鋭くする。これがないと『補聴器をつけても言葉が聞き取れない』
/// (sensorineural hearing loss の典型症状)」。**未実装だった。**
///
/// 実測 (`formant_probe`・母音の指定フォルマントを正解として):
///   Q倍率  被覆15/15を保った最良精度  スペクトルの穴/200
///     ×1              47.6%                 0
///     ×2              66.0%                 1
///     **×3              84.2%                 0**
///     ×4              84.2%                 1
///     ×6             100.0%                18  ← 穴だらけ (精度は「聞こえないから」上がる)
///
/// ×6 の 100% は帯域が狭くなって鳴らなくなったための見せかけ。
/// **×3 が「穴ゼロ・非減衰ゼロで最良」**であり、設計書の生物学的数値
/// (1/3 oct → 1/10 oct ≈ 3.3 倍) ともほぼ一致する。
pub const Q_SHARPENING: f64 = 3.0;
/// パルス幅 (M1 input への electric current 値). M1 の INPUT_CURRENT=60 と整合.
pub const FIRE_CURRENT: i32 = 60;
/// 不応期 (step 単位). 連続音でも発火が頭打ちになる.
pub const FIRE_REFRACTORY_STEPS: i32 = 4;
/// 包絡線検出器の leak_shift (4 = 約 1ms 時定数)
pub const ENV_LEAK_SHIFT: i32 = 4;

/// 聴神経 spontaneous rate 用の 16-bit LFSR (決定論的擬似ノイズ)。
///
/// 原理 3「確率や乱数を使わない (初期化時を除く)」を満たすため、
/// 乱数ではなく LFSR を使う。同じ種から必ず同じ列が出る。
/// `phoneme_synth::LfsrNoise` と同じタップだが、**M0 は自己完結させる**
/// (刺激生成器に依存させない = DRP 実装との整合)。
#[derive(Clone, Debug)]
pub struct SpontaneousLfsr {
    pub state: u16,
}

impl SpontaneousLfsr {
    pub fn new(seed: u16) -> Self {
        Self { state: if seed == 0 { 0xACE1 } else { seed } }
    }

    /// -128..127 程度の小振幅ノイズ。
    #[inline]
    pub fn next(&mut self) -> i32 {
        let bit = ((self.state >> 0) ^ (self.state >> 2) ^ (self.state >> 3) ^ (self.state >> 5)) & 1;
        self.state = (self.state >> 1) | (bit << 15);
        ((self.state & 0xFF) as i32) - 128
    }
}

/// 自発発火の帯域間個体差の段数 (M1 の `idx % 4` と同じ idiom)。
///
/// 生物対応: 聴神経線維の spontaneous rate は個体差が大きい
/// (high-SR / medium-SR / low-SR fiber)。帯域ごとに決定論的に割り当てる。
pub const SPONTANEOUS_INDIVIDUALITY: usize = 4;

/// 設計が指定する自発発火率の範囲 [Hz] (M0_COCHLEA_DESIGN.md §3.6)。
/// **この値は設計側にある正解であって、こちらで決めた閾値ではない。**
pub const SPONTANEOUS_RATE_TARGET_HZ: (f64, f64) = (50.0, 100.0);

#[derive(Clone, Debug)]
pub struct Cochlea {
    pub bands: Vec<BandpassBiquad>,
    pub envelopes: Vec<EnvelopeDetector>,
    pub fire_gens: Vec<FireGenerator>,
    /// 各帯域の中心周波数 (デバッグ・可視化用)
    pub center_freqs: Vec<f64>,
    /// 自発発火の駆動振幅 (**0 = 無効**)。
    ///
    /// `M0_COCHLEA_DESIGN.md` §3.6 で「M0 蝸牛が聴神経 spontaneous rate を含む
    /// スパイク列を生成する」と 2026-05-24 に確定しながら、
    /// `// cochlea.rs (Step 3 で実装)` のまま未実装だったもの (2026-08-25 実装)。
    ///
    /// 注入点は**包絡線検出器の入力** (biquad の後)。生物で spontaneous rate が
    /// 生じるのは内有毛細胞→聴神経シナプスであり、**機械的フィルタリングの下流**だから。
    /// 帯域ごとに独立な LFSR を使う (相関ノイズだと広帯域オンセットに見え、
    /// M0.5 の Octopus 細胞が偽発火する)。
    pub spontaneous_amplitude: i32,
    /// 帯域ごとの独立な決定論的ノイズ源
    pub spontaneous: Vec<SpontaneousLfsr>,
}

impl Cochlea {
    /// 20 帯域、ERB スケール、サンプリングレート 16 kHz で構築.
    pub fn new() -> Self {
        let center_freqs = erb_spaced_freqs(F_MIN_HZ, F_MAX_HZ, N_BANDS);
        let bands: Vec<BandpassBiquad> = center_freqs.iter()
            .map(|&fc| BandpassBiquad::new(fc, erb_q_factor(fc) * Q_SHARPENING, SAMPLE_RATE_HZ))
            .collect();
        let envelopes: Vec<EnvelopeDetector> = (0..N_BANDS)
            .map(|_| EnvelopeDetector::new(ENV_LEAK_SHIFT))
            .collect();
        let fire_gens: Vec<FireGenerator> = (0..N_BANDS)
            .map(|_| FireGenerator::new(FIRE_THRESHOLD, FIRE_REFRACTORY_STEPS))
            .collect();
        // 帯域ごとに異なる種 (決定論的・index 由来)
        let spontaneous = (0..N_BANDS)
            .map(|ch| SpontaneousLfsr::new((0xACE1u16).wrapping_add((ch as u16).wrapping_mul(2654))))
            .collect();
        Self {
            bands,
            envelopes,
            fire_gens,
            center_freqs,
            spontaneous_amplitude: 0, // 既定 OFF
            spontaneous,
        }
    }

    /// 1 step (= SAMPLES_PER_STEP サンプル) 処理.
    /// 戻り値: 各 input neuron への電流 [N_BANDS]. 発火したら FIRE_CURRENT、それ以外 0.
    pub fn process_step(&mut self, samples: &[i32]) -> [i32; N_BANDS] {
        assert!(samples.len() == SAMPLES_PER_STEP,
            "step samples must be {}", SAMPLES_PER_STEP);

        // (a) 各サンプルを 20 帯域並列で処理し、包絡線を更新
        for &x in samples {
            for ch in 0..N_BANDS {
                let bp_out = self.bands[ch].process(x);
                // 聴神経 spontaneous rate: 帯域ごとの独立な決定論的ノイズを
                // 包絡線検出器の入力に足す (0 のとき従来とバイト同一)。
                let drive = if self.spontaneous_amplitude > 0 {
                    let indiv = 1 + (ch % SPONTANEOUS_INDIVIDUALITY) as i32;
                    bp_out + self.spontaneous[ch].next() * self.spontaneous_amplitude * indiv
                } else {
                    bp_out
                };
                let _env = self.envelopes[ch].process(drive);
            }
        }
        // (b) step の最後で各帯域の包絡線を圧縮して閾値判定
        let mut output = [0i32; N_BANDS];
        for ch in 0..N_BANDS {
            let env = self.envelopes[ch].env;
            let compressed = compress_sqrt(env);
            if self.fire_gens[ch].process(compressed) {
                output[ch] = FIRE_CURRENT;
            }
        }
        output
    }

    /// 状態リセット (新セッション開始時)
    pub fn reset(&mut self) {
        for b in &mut self.bands { b.reset(); }
        for e in &mut self.envelopes { e.reset(); }
        for f in &mut self.fire_gens { f.reset(); }
        // 自発ノイズ源も同じ種に戻す (決定論性: 同じ条件で必ず同じ列)
        for (ch, s) in self.spontaneous.iter_mut().enumerate() {
            *s = SpontaneousLfsr::new((0xACE1u16).wrapping_add((ch as u16).wrapping_mul(2654)));
        }
    }
}

impl Default for Cochlea {
    fn default() -> Self { Self::new() }
}

// ──────────────────────────────────────────────────────────────
// テスト
// ──────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// 40 帯域のうち、インパルス応答が末尾までゼロに落ちない帯域数を数える。
    fn limit_cycling_bands(magtrunc: bool) -> usize {
        let freqs = erb_spaced_freqs(F_MIN_HZ, F_MAX_HZ, N_BANDS);
        let mut bad = 0;
        for &fc in freqs.iter() {
            // 出荷フィルタを反映させる (Q_SHARPENING 込み)
            let mut bp = BandpassBiquad::new(fc, erb_q_factor(fc) * Q_SHARPENING, 16000.0);
            bp.magnitude_truncation = magtrunc;
            let mut tail = 0i32;
            for n in 0..4000 {
                let y = bp.process(if n == 0 { 10000 } else { 0 }).abs();
                if n >= 3900 {
                    tail = tail.max(y);
                }
            }
            if tail != 0 {
                bad += 1;
            }
        }
        bad
    }

    /// 算術シフト (旧既定) では自己発振が起きることを固定する (バグの記録・2026-08-25)。
    /// 仕様ではない。`magnitude_truncation` が直す対象。
    ///
    /// 本数は出荷フィルタに依存する (履歴):
    ///   Q_SHARPENING 導入前 (ERB Q・Q15 状態)      24 本
    ///   Q ×3 導入後 (Q15 状態)                     20 本
    ///   状態を Q{STATE_SHIFT} に高精度化した後      25 本
    /// 高精度化で増えるのは、細かい振幅の振動が丸めで消えなくなるため。
    /// **どの構成でもゼロではない**ことがこのテストの主張。
    #[test]
    fn default_shift_produces_limit_cycles() {
        let n = limit_cycling_bands(false);
        assert!(n > 0, "旧既定の acc>>15 で自己発振が 1 本も起きない — バグの記録が失効した");
        assert_eq!(n, 25,
            "自己発振する帯域数が変わった (Q_SHARPENING={} / STATE_SHIFT={} 前提)",
            Q_SHARPENING, STATE_SHIFT);
    }

    /// ゼロ方向切り捨てなら 1 本も自己発振しない。
    /// 正解の出どころ: 安定な線形フィルタのインパルス応答はゼロに収束する (フィルタ理論)。
    #[test]
    fn magnitude_truncation_kills_limit_cycles() {
        assert_eq!(limit_cycling_bands(true), 0,
            "ゼロ方向切り捨てでも自己発振が残る");
    }

    /// ゼロ方向切り捨ての実装 (シフト + 条件加算) が整数除算と厳密同値であること。
    /// DRP には除算器が無いので、この同値性が「実機に載る」ことの根拠になる。
    #[test]
    fn magnitude_truncation_equals_division_without_divider() {
        for acc in [
            0i64, 1, -1, 32767, -32767, 32768, -32768, 32769, -32769,
            1_000_000, -1_000_000, 123_456_789, -123_456_789,
            i32::MAX as i64, i32::MIN as i64,
        ] {
            let bias: i64 = if acc < 0 { 32767 } else { 0 };
            let shifted = (acc + bias) >> 15;
            assert_eq!(shifted, acc / 32768,
                "acc={} でシフト実装 {} が除算 {} と不一致", acc, shifted, acc / 32768);
        }
    }

    /// 既定は ON (2026-08-25 ユーザー判断 案ア: 数学的に正しい方を基準にする)。
    /// `false` に落とせば従来の算術シフトへロールバックできる。
    #[test]
    fn magnitude_truncation_defaults_on() {
        let bp = BandpassBiquad::new(1000.0, 1.0, 16000.0);
        assert!(bp.magnitude_truncation, "既定は ゼロ方向切り捨て であるべき");
        let mut legacy = BandpassBiquad::new(1000.0, 1.0, 16000.0);
        legacy.magnitude_truncation = false;
        // ロールバック経路が実際に別の波形を出すこと (経路が死んでいないことの確認)
        let (mut a, mut b) = (Vec::new(), Vec::new());
        let mut d = BandpassBiquad::new(1000.0, 1.0, 16000.0);
        for n in 0..2000 {
            let x = if n == 0 { 10000 } else { 0 };
            a.push(d.process(x));
            b.push(legacy.process(x));
        }
        assert_ne!(a, b, "ロールバック経路が既定と同じ出力になっている");
    }

    /// 自発発火は既定 OFF (2026-08-25 ユーザー判断 案イ)。
    ///
    /// 実装はしたが、`FireGenerator` が閾値+不応期の装置なのでレートが
    /// 0Hz か 400Hz にしかならず、設計書 §3.6 の「50-100Hz」を満たせない (S9)。
    /// 有効化するときは `spontaneous_amplitude` を明示的に設定すること。
    #[test]
    fn cochlea_spontaneous_defaults_off() {
        let c = Cochlea::new();
        assert_eq!(c.spontaneous_amplitude, 0, "蝸牛の自発発火は既定 OFF");
        assert_eq!(c.spontaneous.len(), N_BANDS, "ノイズ源が帯域数だけ無い");
        // 種が帯域ごとに違うこと (同期しないことの保証)
        let seeds: std::collections::HashSet<u16> =
            c.spontaneous.iter().map(|s| s.state).collect();
        assert!(seeds.len() > N_BANDS / 2, "ノイズ源の種が重複しすぎている: {}", seeds.len());
    }

    /// 低域の弱い純音が聞こえること (2026-08-25 に直した欠陥の回帰テスト)。
    ///
    /// 状態を高精度化する前は、最弱フォルマント振幅 3200 の純音が
    /// **50-65Hz で全く発火しなかった** (帯域を 240 本に増やしても同じ)。
    /// 正解の出どころ: その周波数・その振幅の音を入れたのは実験者。
    #[test]
    fn weak_low_frequency_tone_is_heard() {
        use super::super::phoneme_synth::{freq_to_phase_step, sin_lookup};
        for &f_hz in [50.0f64, 55.0, 60.0, 65.0].iter() {
            let mut c = Cochlea::new();
            let step = freq_to_phase_step(f_hz);
            let mut phase = 0u32;
            let mut fired = false;
            // 170ms
            for _ in 0..(170 * 16 / SAMPLES_PER_STEP as i32) {
                let mut buf = [0i32; SAMPLES_PER_STEP];
                for b in buf.iter_mut() {
                    *b = (sin_lookup(phase) * 3200) >> 14;
                    phase = phase.wrapping_add(step);
                }
                if c.process_step(&buf).iter().any(|&v| v != 0) {
                    fired = true;
                    break;
                }
            }
            assert!(fired, "{:.0}Hz・振幅3200 の純音が 1 帯域も発火させない", f_hz);
        }
    }

    /// 既定の蝸牛は自己発振しない (出荷状態の不変条件)。
    #[test]
    fn default_cochlea_has_no_limit_cycles() {
        let mut c = Cochlea::new();
        for b in c.bands.iter_mut() {
            let mut tail = 0i32;
            for n in 0..4000 {
                let y = b.process(if n == 0 { 10000 } else { 0 }).abs();
                if n >= 3900 {
                    tail = tail.max(y);
                }
            }
            assert_eq!(tail, 0, "既定の蝸牛に自己発振する帯域がある");
        }
    }

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

    // ─── Step 2 テスト ───

    #[test]
    fn isqrt_basic() {
        assert_eq!(isqrt(0), 0);
        assert_eq!(isqrt(1), 1);
        assert_eq!(isqrt(4), 2);
        assert_eq!(isqrt(9), 3);
        assert_eq!(isqrt(100), 10);
        assert_eq!(isqrt(10000), 100);
        // 非完全平方数: 切り捨て
        let s = isqrt(50);
        assert!(s == 7 || s == 8, "isqrt(50) ≈ 7.07, got {}", s);
    }

    #[test]
    fn envelope_rises_on_signal() {
        let mut env = EnvelopeDetector::new(4);
        // 振幅 1000 の DC 入力 (簡易テスト)
        for _ in 0..200 {
            env.process(1000);
        }
        // 漏れと積分の平衡: env ≈ 1000 × 2^4 = 16000
        assert!(env.env > 10000 && env.env < 20000,
            "envelope on DC=1000: {}", env.env);
    }

    #[test]
    fn envelope_decays_after_signal() {
        let mut env = EnvelopeDetector::new(4);
        // 信号を入れる
        for _ in 0..100 {
            env.process(1000);
        }
        let peak = env.env;
        // 信号停止
        for _ in 0..200 {
            env.process(0);
        }
        assert!(env.env < peak / 10, "envelope should decay: peak={}, after={}",
            peak, env.env);
    }

    #[test]
    fn fire_generator_respects_threshold() {
        let mut fg = FireGenerator::new(100, 4);
        // 閾値未満
        assert!(!fg.process(50));
        assert!(!fg.process(99));
        // 閾値以上で発火
        assert!(fg.process(100));
        // 不応期中は発火しない
        assert!(!fg.process(1000));
        assert!(!fg.process(1000));
        assert!(!fg.process(1000));
        assert!(!fg.process(1000));
        // 不応期明けで再発火可能
        assert!(fg.process(1000));
    }

    // ─── Step 3 テスト ───

    #[test]
    fn cochlea_constructs_with_n_bands() {
        // 2026-08-25 修正: 20 をハードコードしていたため N_BANDS 20->40 拡張以降
        // ずっと FAILED のままだった。`Cochlea::process_step` は定数 N_BANDS で
        // ループするので、**この長さ不変条件を守るテストはこれ 1 本しかない**
        // (独立レビュー指摘)。定数に追従させて本来の役目を果たさせる。
        let c = Cochlea::new();
        assert_eq!(c.bands.len(), N_BANDS);
        assert_eq!(c.envelopes.len(), N_BANDS);
        assert_eq!(c.fire_gens.len(), N_BANDS);
        assert_eq!(c.center_freqs.len(), N_BANDS);
        // 周波数範囲は定数どおり (端点は erb_spaced_freqs が厳密に取る)
        assert!((c.center_freqs[0] - F_MIN_HZ).abs() < 1.0);
        assert!((c.center_freqs[N_BANDS - 1] - F_MAX_HZ).abs() < 5.0);
    }

    /// 純音入力でその周波数帯域のみが発火する (周波数選択性)
    #[test]
    fn cochlea_frequency_selectivity() {
        let mut c = Cochlea::new();
        // 1000 Hz 純音、振幅 8000、1 秒分 (16000 sample) = 2000 step
        let n_step = 2000;
        let mut fire_counts = [0u32; N_BANDS];
        // 過渡応答を避けて後半で集計
        for step in 0..n_step {
            let samples: Vec<i32> = (0..SAMPLES_PER_STEP).map(|s| {
                let t = (step * SAMPLES_PER_STEP + s) as f64;
                let phase = 2.0 * std::f64::consts::PI * 1000.0 * t / 16000.0;
                (8000.0 * phase.sin()) as i32
            }).collect();
            let out = c.process_step(&samples);
            if step >= n_step / 2 {
                for (ch, &v) in out.iter().enumerate() {
                    if v > 0 { fire_counts[ch] += 1; }
                }
            }
        }
        // 1000 Hz に最も近い帯域を見つける
        let best_ch = c.center_freqs.iter()
            .enumerate()
            .min_by_key(|&(_, &f)| ((f - 1000.0).abs() * 1000.0) as i64)
            .map(|(i, _)| i)
            .unwrap();
        let best_count = fire_counts[best_ch];

        // 隣接以外の遠い帯域の発火は best より大幅に少ない
        let far_ch_low = 0;   // 50 Hz
        let far_ch_high = 19; // 4000 Hz
        assert!(best_count > fire_counts[far_ch_low] * 3,
            "near 1000Hz ch{} ({:.0}Hz) fires {}, far low ch{} ({:.0}Hz) fires {}",
            best_ch, c.center_freqs[best_ch], best_count,
            far_ch_low, c.center_freqs[far_ch_low], fire_counts[far_ch_low]);
        assert!(best_count > fire_counts[far_ch_high] * 2,
            "near 1000Hz ch{} fires {}, far high ch{} fires {}",
            best_ch, best_count, far_ch_high, fire_counts[far_ch_high]);
    }

    #[test]
    fn cochlea_silence_no_firing() {
        let mut c = Cochlea::new();
        // 無音 (zero) を入れて 100 step 動かす
        let zero = vec![0i32; SAMPLES_PER_STEP];
        let mut total_fires = 0u32;
        for _ in 0..100 {
            let out = c.process_step(&zero);
            total_fires += out.iter().filter(|&&v| v > 0).count() as u32;
        }
        // 無音では発火しない (自発発火は M0 内では生成しない)
        assert_eq!(total_fires, 0,
            "silence should produce no firing, got {}", total_fires);
    }
}
