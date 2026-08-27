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
/// 子音合成で F0 を明示しないときの既定値 (2026-08-27)。
/// voice bar の周波数に使う。無声子音では使われない。
pub const F0_DEFAULT_HZ: f64 = 150.0;

/// 破裂音の閉鎖区間が子音長に占める割合 [%] (2026-08-27)。
///
/// 生体の破裂音は**閉鎖が破裂より長い** (閉鎖 50ms / 破裂 10-20ms = 閉鎖が 70-80%)。
/// 旧実装は固定 10ms で、子音長 30ms のうち 33% しかなく**比率が逆**だった。
///
/// **有声のときこの区間は voice bar だけが鳴る = 前有声の長さそのもの。**
/// 日本語の前有声は生体で 20-70ms。
pub const CLOSURE_FRACTION_PERCENT: usize = 67;

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
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Vowel {
    pub label: char,
    pub formants_hz: [f64; 3],  // F1, F2, F3
    pub amplitudes: [i32; 3],   // 振幅 (生成時 i32 範囲)
    /// フォルマント帯域幅 [Hz] (Klatt 1980 の標準値・2026-08-25 追加)。
    /// `synth_vowel_f0` の全極共鳴器で使う。`synth_vowel` (純音3本) は使わない。
    pub bandwidths_hz: [f64; 3],
}

/// 標準的な日本語 5 母音.
/// 周波数は Klatt 1980 と日本語音声学の文献に基づく中央値.
///
/// **振幅の絶対スケールは 2026-08-25 に ×4 した**（相対比は不変）。
///
/// 旧スケールは「**波形のピーク**が純音基準 8000 に合う」ように選ばれていたが、
/// `FIRE_THRESHOLD` は**帯域ごとに効く**。1 本の帯域に届くのは最強フォルマント
/// F1 の振幅だけなので、旧 F1=4000 は床 (旧: A>=3927) のわずか 1.9% 上、
/// F2=2000-3200 と F3=800-1600 は**構造的に床の下**で一度も鳴らなかった。
/// 結果、どの母音も発火帯域は 1 本 (F1 のみ)、/e/ と /o/ は F1 が同じ 500Hz で衝突。
///
/// 正しい規則は「**実験者が置いた最弱のフォルマントが床を超える**」こと。
/// 相対比は任意ではないが**絶対スケールは元から任意**だったので、これを直した。
/// (S1 の「ピークで校正したがノイズは RMS で効く」と同じ型の取り違え。3 度目。)
pub fn vowels() -> [Vowel; 5] {
    [
        Vowel {
            label: 'a',
            formants_hz: [800.0, 1300.0, 2700.0],
            amplitudes: [16000, 11200, 4800],
            bandwidths_hz: [60.0, 90.0, 150.0],
        },
        Vowel {
            label: 'i',
            formants_hz: [300.0, 2300.0, 3000.0],
            amplitudes: [16000, 8000, 6400],
            bandwidths_hz: [60.0, 90.0, 150.0],
        },
        Vowel {
            label: 'u',
            formants_hz: [350.0, 850.0, 2400.0],
            amplitudes: [16000, 12800, 3200],
            bandwidths_hz: [60.0, 90.0, 150.0],
        },
        Vowel {
            label: 'e',
            formants_hz: [500.0, 2000.0, 2700.0],
            amplitudes: [16000, 9600, 4800],
            bandwidths_hz: [60.0, 90.0, 150.0],
        },
        Vowel {
            label: 'o',
            formants_hz: [500.0, 900.0, 2400.0],
            amplitudes: [16000, 12800, 4800],
            bandwidths_hz: [60.0, 90.0, 150.0],
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
// F0 (音程) つき母音合成 — 声帯パルス列 + 全極フォルマント共鳴器
// ──────────────────────────────────────────────────────────────

/// **全極 2 次共鳴器** (Klatt 型のフォルマント共鳴器・2026-08-25 追加)。
///
/// `cochlea::BandpassBiquad` は共鳴器として使えない (独立監査の指摘):
///   - RBJ の bandpass は **0 dB peak gain 正規化**なので**一切増幅しない**
///   - 2 極 2 零なので中心の下側スカートが -6dB/oct で落ちる (全極なら平坦)
///   - `erb_q_factor` は Q 上限 9.264 で高域フォルマント帯域幅を表現できない
///
/// 伝達関数: `H(z) = G / (1 - a1 z^-1 - a2 z^-2)`、極は `r·e^(±jθ)`
///   `r = exp(-π·BW/fs)` 、 `θ = 2π·F/fs`
///   `a1 = 2r·cosθ` 、 `a2 = -r²`
/// `G` は共鳴点のピーク応答が指定振幅になるよう初期化時に決める
/// (設計時の f64 は初期化のみ — 原理 3 の明文化された例外)。
///
/// 状態は `cochlea::BandpassBiquad` と同じく **i64 の Q8** で保持する。
/// 低周波・高 Q ほど 1 サンプルの寄与が小さく、整数丸めで削られるため
/// (2026-08-25 に蝸牛側で実証した欠陥と同型)。
#[derive(Clone, Debug)]
pub struct FormantResonator {
    /// Q1.15 係数
    pub a1: i32,
    pub a2: i32,
    /// 入力利得 (Q1.15)
    pub g: i32,
    /// Q8 の高精度状態
    pub y1: i64,
    pub y2: i64,
}

/// 共鳴器の状態が持つ小数ビット数
pub const RESONATOR_STATE_SHIFT: i32 = 8;

impl FormantResonator {
    /// `f_hz` に共鳴し、帯域幅 `bw_hz`、共鳴点のピーク利得が `peak_gain` になる共鳴器。
    pub fn new(f_hz: f64, bw_hz: f64, peak_gain: f64, sample_rate: f64) -> Self {
        let r = (-std::f64::consts::PI * bw_hz / sample_rate).exp();
        let theta = 2.0 * std::f64::consts::PI * f_hz / sample_rate;
        let a1 = 2.0 * r * theta.cos();
        let a2 = -(r * r);
        // 共鳴点 z = e^{jθ} での |H| を求めて G を逆算する
        let (cw, sw) = (theta.cos(), theta.sin());
        let (c2w, s2w) = ((2.0 * theta).cos(), (2.0 * theta).sin());
        let re = 1.0 - a1 * cw - a2 * c2w;
        let im = a1 * sw + a2 * s2w;
        let mag = (re * re + im * im).sqrt();
        let g = peak_gain * mag;
        Self {
            a1: (a1 * 32768.0).round() as i32,
            a2: (a2 * 32768.0).round() as i32,
            g: (g * 32768.0).round() as i32,
            y1: 0,
            y2: 0,
        }
    }

    /// **状態 (y1,y2) を保ったまま中心周波数だけ差し替える** (2026-08-27)。
    ///
    /// フォルマント遷移のために必要。`new` を呼び直すと状態が 0 に戻り、
    /// 共鳴が毎サンプル切れて**別の音**になってしまう。
    /// 係数の算出式は `new` と同一 (重複を避けるため `new` から係数だけ借りる)。
    pub fn retune(&mut self, f_hz: f64, bw_hz: f64, peak_gain: f64, sample_rate: f64) {
        let fresh = Self::new(f_hz, bw_hz, peak_gain, sample_rate);
        self.a1 = fresh.a1;
        self.a2 = fresh.a2;
        self.g = fresh.g;
    }

    #[inline]
    pub fn process(&mut self, x: i32) -> i32 {
        let num: i64 = (self.g as i64) * (x as i64);
        let acc: i64 = (num << RESONATOR_STATE_SHIFT)
            + (self.a1 as i64) * self.y1
            + (self.a2 as i64) * self.y2;
        // ゼロ方向切り捨て (受動的損失でリミットサイクルを構造的に潰す)
        let bias: i64 = if acc < 0 { 32767 } else { 0 };
        let y0 = (acc + bias) >> 15;
        self.y2 = self.y1;
        self.y1 = y0;
        let ob: i64 = if y0 < 0 { (1 << RESONATOR_STATE_SHIFT) - 1 } else { 0 };
        ((y0 + ob) >> RESONATOR_STATE_SHIFT) as i32
    }
}

/// **声帯パルスの速度商** (rise ÷ fall) [%]。(2026-08-27)
///
/// Rosenberg (1971) の声帯流モデルでは、**開く相は閉じる相より長い** (速度商 2〜3)。
/// 開放商 (`GLOTTAL_OPEN_QUOTIENT_PERCENT`) は既に Rosenberg を引いていたのに、
/// **速度商だけが 1 (左右対称) のままだった。**
///
/// ## なぜ直すか
///
/// **左右対称の三角波の DTFT は Dirichlet 核の 2 乗であり、厳密な零点の櫛を持つ。**
/// 零点は `k · SAMPLE_RATE / rise` に立ち、`rise` は小さな整数なので
/// 500 / 667 / 800 / 1000 Hz といった「切りのいい」周波数に固定される。
/// これが `vowels()` のフォルマント表 (300/350/500/800/850/900/1300/2000/2300/2400/2700/3000)
/// と衝突し、**F0 ごとに違うフォルマントが消える。**
///
/// 出荷コードでの実測 (2026-08-27・§14.32): /a/ の M0 argmax が
/// F0=100 で 820Hz (F1) → F0=130/150/160 で **293Hz** → F0=200 で 820Hz と飛ぶ。
/// **5 母音のうち /a/ の 2 点を除き、場所符号のピークはフォルマントではなかった。**
///
/// **非対称にすると厳密零点は消える** (Dirichlet 核の 2 乗でなくなる)。
///
/// **100 = 左右対称 = 2026-08-27 以前の挙動。** A/B のために残してある。
pub const GLOTTAL_SPEED_QUOTIENT_PERCENT: usize = 200;

static GLOTTAL_SQ: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(usize::MAX);

/// 速度商 [%]。初回だけ環境変数 `DRPNN_GLOTTAL_SQ` を読む。**乱数は使わない。**
pub fn glottal_speed_quotient_percent() -> usize {
    use std::sync::atomic::Ordering;
    let v = GLOTTAL_SQ.load(Ordering::Relaxed);
    if v == usize::MAX {
        let q = std::env::var("DRPNN_GLOTTAL_SQ").ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(GLOTTAL_SPEED_QUOTIENT_PERCENT)
            .max(1);
        GLOTTAL_SQ.store(q, Ordering::Relaxed);
        return q;
    }
    v
}

/// 速度商を実行時に切り替える (対照実験用)。**100 で従来と同一波形。**
pub fn set_glottal_speed_quotient(percent: usize) {
    GLOTTAL_SQ.store(percent.max(1), std::sync::atomic::Ordering::Relaxed);
}

/// 声帯パルス列の開放商 (パルス幅 / 周期)。Rosenberg モデルの標準値付近。
pub const GLOTTAL_OPEN_QUOTIENT_PERCENT: usize = 40;

/// 声帯パルス列を作る (決定論的・整数)。
///
/// 単純なインパルス列は全倍音が等振幅になるが、実際の声帯波は高域が落ちる。
/// **パルスに幅を持たせることで傾斜を作る** (後から傾斜を掛けるのではない・監査の指摘)。
/// 三角波パルス (立ち上がり→立ち下がり) は約 -12 dB/oct を与える。
///
/// F0 [Hz] から周期 = SAMPLE_RATE / F0 サンプル。整数周期なので折り返さない。
pub fn glottal_pulse_train(f0_hz: f64, n_samples: usize, amplitude: i32) -> Vec<i32> {
    let period = (SAMPLE_RATE_HZ / f0_hz).round() as usize;
    let width = (period * GLOTTAL_OPEN_QUOTIENT_PERCENT / 100).max(2);
    // **速度商つきの非対称パルス** (2026-08-27)。sq=100 で `width / 2` = 従来と厳密に同一。
    let sq = glottal_speed_quotient_percent();
    let rise = (width * sq / (sq + 100)).clamp(1, width.saturating_sub(1).max(1));
    let mut out = Vec::with_capacity(n_samples);
    for i in 0..n_samples {
        let phase = i % period;
        let v = if phase < rise {
            // 立ち上がり
            (amplitude as i64 * phase as i64 / rise.max(1) as i64) as i32
        } else if phase < width {
            // 立ち下がり
            (amplitude as i64 * (width - phase) as i64 / (width - rise).max(1) as i64) as i32
        } else {
            0
        };
        out.push(v);
    }
    out
}

/// **F0 (音程) つきの母音合成**。
///
/// `synth_vowel` (純音 3 本) との違い:
///   - 純音 3 本は「フォルマントの**包絡**」を直接鳴らしていた。倍音が無いので
///     F0 という概念自体が存在せず、「低いあ」と「高いあ」を作れなかった。
///   - こちらは**声帯パルス列を 3 つの全極共鳴器に通す**。
///     倍音が F0 間隔で並び、共鳴器がその振幅を形づくる = 本物の音声と同じ構造。
///     **音程を変えても包絡は動かない**ので、同じ母音として読めるはず。
///
/// `synth_vowel` は**変更しない** (過去ログとの比較の連続性)。
/// フォルマント遷移の長さ。子音の解放から母音の定常状態に達するまで。
///
/// 実音声では 40-60ms 程度 (Delattre, Liberman & Cooper 1955)。
pub const TRANSITION_MS: f64 = 40.0;

/// `synth_vowel_f0_glide` に `None` を渡すのと同じ。**既存の呼び出しは一切変わらない。**
pub fn synth_vowel_f0(vowel: &Vowel, f0_hz: f64, duration_ms: f64) -> Vec<i32> {
    synth_vowel_f0_glide(vowel, f0_hz, duration_ms, None)
}

/// **フォルマント遷移つきの母音合成** (2026-08-27)。
///
/// `locus_hz` = 直前の子音の調音位置が決める遷移の**始点** [F1, F2, F3]。
/// `None` なら遷移なし = 従来と**完全に同一の波形**。
///
/// ## なぜ必要か
///
/// §14.26 で「連続にすると有声性が崩壊する (76.8% -> 3.3%)」が出た。
/// 本実装の子音の手がかりは**子音区間の中にしか無い**ので、
/// 直前の母音に埋もれると何も残らない。
///
/// 実音声では、子音の調音位置は**後続母音のフォルマントの動き**に転写される
/// (locus theory)。**遷移は文脈でしか存在しない手がかり**であり、
/// 本実装に無かったものである (旧コメント「遷移は未実装」がそれを認めていた)。
///
/// ## 何が動くか
///
/// 3 つの共鳴器の中心周波数を locus から母音の目標値へ**線形に**動かす
/// (`TRANSITION_MS` かけて)。帯域幅・利得は動かさない。
/// **RMS 正規化は `synth_vowel` を基準にしており locus に依存しない**ので、
/// 遷移の有無で音量は変わらない = 音量では当てられない。
/// `synth_vowel_f0_full` に気音なしを渡すのと同じ。**既存の呼び出しは一切変わらない。**
pub fn synth_vowel_f0_glide(
    vowel: &Vowel,
    f0_hz: f64,
    duration_ms: f64,
    locus_hz: Option<[f64; 3]>,
) -> Vec<i32> {
    let mut unused = LfsrNoise::new(1);
    synth_vowel_f0_full(vowel, f0_hz, duration_ms, locus_hz, 0.0, &mut unused)
}

/// **VOT (気音) つきの母音合成** (2026-08-27)。
///
/// `aspiration_ms` = 破裂の解放から声帯振動が始まるまでの時間 (Voice Onset Time)。
/// 0 なら気音なし = `synth_vowel_f0_glide` と**完全に同一の波形**。
///
/// ## なぜ必要か
///
/// §14.26 で「連続にすると有声性の伝達情報量が 76.8% → 3.3% に崩壊する」が出た。
/// 原因は**本実装の有声性の手がかりが voice bar (閉鎖中の声帯振動) 1 本しか無く、
/// それが最も文脈に弱い手がかりだった**こと。連続では直前の母音の低域に埋もれる。
///
/// 実音声で有声/無声を分ける最も強い手がかりは **VOT** である。
/// 無声破裂音は解放のあと声帯振動が始まるまで**気音** (声門での乱流雑音が
/// 声道を通ったもの) が入る。有声破裂音には入らない (むしろ解放前から声帯が鳴っている)。
///
/// **VOT は解放「後」の手がかりなので、フォルマント遷移と同じく連続発話でも生き残る。**
/// これが voice bar との決定的な違いである。
///
/// ## 何が動くか
///
/// 最初の `aspiration_ms` の間、共鳴器を**声帯パルス列でなく雑音で駆動する**。
/// 共鳴器・遷移・放射・正規化は一切変えない。
/// **物理的にこれが気音そのものである** (声門の乱流が同じ声道を通る)。
///
/// 雑音の RMS は**声帯パルス列の RMS に合わせる**。
/// 合わせないと「気音の区間だけ音量が違う」ことになり、
/// **有声性を音量で当てられてしまう** (§14.27 の G87c で学んだ交絡)。
/// 手がかりを**周期性の有無**だけに絞るための措置である。
pub fn synth_vowel_f0_full(
    vowel: &Vowel,
    f0_hz: f64,
    duration_ms: f64,
    locus_hz: Option<[f64; 3]>,
    aspiration_ms: f64,
    noise: &mut LfsrNoise,
) -> Vec<i32> {
    let n_samples = (duration_ms * SAMPLE_RATE_HZ / 1000.0) as usize;
    // 声帯源。共鳴器のピーク利得で振幅を作るので、源は控えめにする。
    let mut source = glottal_pulse_train(f0_hz, n_samples, 4096);
    // **気音 (VOT)**: 最初の aspiration_ms を雑音駆動に差し替える。
    // 声帯パルス列と同じ RMS に揃えるので、**手がかりは周期性の有無だけ**になる。
    let n_asp = ((aspiration_ms * SAMPLE_RATE_HZ / 1000.0) as usize).min(n_samples);
    if n_asp > 0 {
        let src_rms = rms_i64(&source);
        let mut asp: Vec<i32> = (0..n_asp).map(|_| noise.next_sample()).collect();
        let a_rms = rms_i64(&asp);
        if a_rms > 0 && src_rms > 0 {
            for v in asp.iter_mut() { *v = ((*v as i64) * src_rms / a_rms) as i32; }
        }
        source[..n_asp].copy_from_slice(&asp);
    }
    let mut resonators: Vec<FormantResonator> = (0..3)
        .map(|k| {
            FormantResonator::new(
                match locus_hz { Some(l) => l[k], None => vowel.formants_hz[k] },
                vowel.bandwidths_hz[k],
                vowel.amplitudes[k] as f64 / 4096.0,
                SAMPLE_RATE_HZ,
            )
        })
        .collect();
    let mut voiced = Vec::with_capacity(n_samples);
    // 遷移の長さ (サンプル数)。母音が短ければそれに収める。
    let n_trans = ((TRANSITION_MS * SAMPLE_RATE_HZ / 1000.0) as usize).min(n_samples);
    for (i, &x) in source.iter().enumerate() {
        if let Some(locus) = locus_hz {
            if i < n_trans && n_trans > 0 {
                // locus -> 目標値 への線形移動。i = n_trans で目標値に一致する。
                for (k, r) in resonators.iter_mut().enumerate() {
                    let target = vowel.formants_hz[k];
                    let f = locus[k] + (target - locus[k]) * (i as f64) / (n_trans as f64);
                    r.retune(
                        f,
                        vowel.bandwidths_hz[k],
                        vowel.amplitudes[k] as f64 / 4096.0,
                        SAMPLE_RATE_HZ,
                    );
                }
            }
        }
        let mut sample = 0i32;
        for r in resonators.iter_mut() {
            sample = sample.saturating_add(r.process(x));
        }
        voiced.push(sample);
    }

    // **唇からの放射特性** (2026-08-26 追加)。
    //
    // 本物の音声生成は 3 段:
    //   声帯源 (-12 dB/oct) → 声道の共鳴 (フォルマント) → **唇からの放射 (+6 dB/oct)**
    // 3 段目を落とすと全体が -12 dB/oct のままになり、**F0 と低次倍音が支配的**になる。
    // 実測 (2026-08-26): 放射なしだと /a/ も /i/ も上位が 91-207Hz (F0 の低次倍音) で
    // 埋まり、母音の識別率が**全 Q で 0.0%** (チャンス 15.8%) になった。
    // 共鳴器自体は効いていた (/a/ は 802Hz、/i/ は 307Hz が立っていた) が、
    // 源の低域に埋もれていた。
    //
    // 放射は一次差分 (微分) = +6 dB/oct。整数・決定論的。
    let mut raw = Vec::with_capacity(n_samples);
    let mut prev = 0i32;
    for &v in voiced.iter() {
        raw.push(v.saturating_sub(prev));
        prev = v;
    }

    // **RMS を `synth_vowel` (純音3本版) に揃える** (2026-08-25)。
    //
    // 声帯パルス列の倍音は F0 が高いほど本数が減り 1 本あたりが強くなるので、
    // 源の振幅を固定すると**音量が F0 に依存する**。それでは音程不変性を測るとき
    // 「音程の違い」でなく「音量の違い」を見てしまう。
    // 既存の基準 (`synth_vowel`) と同じ RMS に揃えることで音量を交絡から外す。
    // (蝸牛を駆動するのは包絡線 ≒ RMS であってピークではない — S1 で実証済み。)
    let reference = synth_vowel(vowel, duration_ms);
    let target_rms = rms_i64(&reference);
    let cur_rms = rms_i64(&raw);
    if cur_rms > 0 && target_rms > 0 {
        for v in raw.iter_mut() {
            *v = ((*v as i64) * target_rms / cur_rms) as i32;
        }
    }

    // attack/release (既存 synth_vowel と同じ形)
    let ramp = ((10.0 * SAMPLE_RATE_HZ / 1000.0) as usize).min(n_samples / 4).max(1);
    let mut out = Vec::with_capacity(n_samples);
    for (i, &sample) in raw.iter().enumerate() {
        let env = if i < ramp {
            (i * 1024 / ramp) as i32
        } else if i + ramp >= n_samples {
            (((n_samples - i) * 1024) / ramp) as i32
        } else {
            1024
        };
        out.push(((sample as i64 * env as i64) >> 10) as i32);
    }
    out
}

/// 整数 RMS (決定論的)
fn rms_i64(w: &[i32]) -> i64 {
    if w.is_empty() {
        return 0;
    }
    let sq: i64 = w.iter().map(|&v| (v as i64) * (v as i64)).sum();
    isqrt_i64(sq / w.len() as i64)
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
    /// 破裂音 (例 /p/, /t/, /k/, /b/, /d/, /g/): 短い無音 + broadband burst
    ///
    /// `voiced` (2026-08-27 追加): **前有声 (prevoicing)**。
    /// 生体では有声破裂音の閉鎖中も声帯が振動し、声道が閉じているので
    /// **低周波だけが組織を通って漏れる** (スペクトログラムの voice bar)。
    /// **日本語の /b d g/ と /p t k/ を分ける主要な手がかりはこれである。**
    ///
    /// これを入れる前は `'k'|'g'` `'t'|'d'` `'p'|'b'` が同じ子音にマップされており、
    /// **濁音と清音が完全に同一の波形**だった。実コーパスでは
    /// **モーラの 41.7% がこの縮退の影響を受けていた** (§14.21)。
    Plosive { burst_freq_low: f64, burst_freq_high: f64, voiced: bool },
    /// 摩擦音 (例 /s/, /sh/, /z/, /ʑ/): 持続的な高周波ノイズ
    ///
    /// `voiced` (2026-08-27 追加): 有声摩擦音では摩擦と**同時に**声帯が振動する。
    Fricative { freq_low: f64, freq_high: f64, voiced: bool },
    /// 鼻音 (例 /n/, /m/): 低周波フォルマント
    Nasal { f1: f64, f2: f64 },
    /// 子音なし (「あいうえお」など母音のみのかな)
    None,
    /// **接近音** /j/ /w/ (2026-08-26 追加)。
    ///
    /// ヤ行・ワ行。母音的だが F1/F2 が母音と違う位置から始まる。
    /// 遷移 (formant transition) は未実装なので、**その位置の短い有声区間**で近似する。
    /// /j/ ≈ F1 300 / F2 2200 (硬口蓋)、 /w/ ≈ F1 300 / F2 700 (両唇軟口蓋)。
    ///
    /// これを入れる前は `Consonant::None` で近似しており、
    /// **や=あ / ゆ=う / よ=お / わ=あ / を=お が完全に同一の応答**になっていた (実測)。
    Approximant { f1: f64, f2: f64 },
    /// **破擦音** /ts/ /tɕ/ (2026-08-26 追加)。
    ///
    /// タ行の「つ」「ち」。破裂 (閉鎖の開放) の直後に摩擦が続く複合音。
    /// 破裂区間と摩擦区間を連結して作る。
    ///
    /// これを入れる前は摩擦音のみで近似しており、
    /// **す=つ / し=ち が完全に同一の応答**になっていた (実測)。
    Affricate {
        burst_freq_low: f64,
        burst_freq_high: f64,
        fric_freq_low: f64,
        fric_freq_high: f64,
    },
}

/// 子音波形を生成. duration_ms: 持続時間.
pub fn synth_consonant(c: Consonant, duration_ms: f64, noise: &mut LfsrNoise) -> Vec<i32> {
    let n_samples = (duration_ms * SAMPLE_RATE_HZ / 1000.0) as usize;
    let mut out = Vec::with_capacity(n_samples);
    match c {
        Consonant::Plosive { burst_freq_low: _, burst_freq_high: _, .. } => {
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
            //
            // 振幅校正 (2026-05-30 修正):
            //   旧 [3000, 1500] (peak 4500) → 蝸牛 firing threshold 200 (純音 amp 8000 想定) 未満で
            //   全帯域取りこぼし発生。 他音素は Plosive/Fricative peak ~32768、 Vowel peak ~8000。
            //   新 [6000, 3000] (peak 9000) で 母音 と同等の cochlea 反応を確保。
            let ps = [freq_to_phase_step(f1), freq_to_phase_step(f2)];
            let amps = [6000i32, 3000];
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
        Consonant::None => {}
        // 2026-08-26 追加。旧実装は帯域指定を無視する設計なので、
        // 接近音は鼻音相当・破擦音は破裂音相当で近似する (旧経路の連続性のため)。
        Consonant::Approximant { f1, f2 } => {
            return synth_consonant(Consonant::Nasal { f1, f2 }, duration_ms, noise);
        }
        Consonant::Affricate { burst_freq_low, burst_freq_high, .. } => {
            return synth_consonant(
                Consonant::Plosive { burst_freq_low, burst_freq_high, voiced: false },
                duration_ms,
                noise,
            );
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
            consonant: Consonant::Plosive { burst_freq_low: 500.0, burst_freq_high: 2000.0, voiced: false },
            vowel: v[0], // a
        },
        Syllable {
            label: "ki",
            consonant: Consonant::Plosive { burst_freq_low: 2000.0, burst_freq_high: 4000.0, voiced: false },
            vowel: v[1], // i
        },
        Syllable {
            label: "tu",
            consonant: Consonant::Plosive { burst_freq_low: 1500.0, burst_freq_high: 3500.0, voiced: false },
            vowel: v[2], // u
        },
        Syllable {
            label: "se",
            consonant: Consonant::Fricative { freq_low: 3000.0, freq_high: 8000.0, voiced: false },
            vowel: v[3], // e
        },
        Syllable {
            label: "mo",
            consonant: Consonant::Nasal { f1: 250.0, f2: 1500.0 },
            vowel: v[4], // o
        },
    ]
}

/// 帯域を効かせた子音波形 (2026-08-25 追加).
///
/// **既存の `synth_consonant` は変更しない**（None 腕の追加のみ）——
/// 過去の測定ログ (sel 0.508 / per-pair 0.765 / ki-tu 0.883) との比較の連続性を保つため。
///
/// 動機: `standard_syllables()` は pa/ki/tu に別々の burst 周波数
/// (500-2000 / 2000-4000 / 1500-3500 Hz) を指定しているが、`synth_consonant` は
/// `Plosive { burst_freq_low: _, burst_freq_high: _ }` と**両方を捨てている**。
/// 摩擦音も `Fricative { .. }` で全部無視。したがって pa/ki/tu は構造的に同一波形
/// (LFSR 状態差のみ) であり、5 音素を分けていたのは母音 5 種と鼻音だけだった。
/// かな 100 種を流しても 6 通りにしか写像されない。この関数は指定帯域を実際に効かせる。
///
/// 手段: `cochlea::BandpassBiquad`（整数 Q1.15・`process` は完全整数）に
/// LFSR ノイズを通すだけ。f64 は初期化時のみ（原理 3 の明文化された例外）。
///
/// 振幅: 帯域通過で減衰するため正規化する。**揃えるのはピークではなく RMS**。
///
/// 初版はピーク 9000 に揃えたが、蝸牛は完全に無音になった (2026-08-25 実測:
/// pa/tu/se とも総スパイク 0)。原因は基準の取り違え——蝸牛を駆動するのは
/// 包絡線 ≒ RMS であり、ピークではない。実測 crest は母音 2.28 に対し
/// 帯域ノイズ 4.07-4.53 なので、ピークを揃えると RMS が 1.8 倍不足する。
///
/// 目標値は**コードが既に宣言している定数から導く**: `cochlea::FIRE_THRESHOLD` 200 は
/// 「純音 amp 8000 想定」で置かれている (cochlea.rs)。純音 amp 8000 の RMS は
/// 8000/√2 ≈ 5657。これを目標にする。ゲートを通すために選んだ値ではない。
///
/// **全子音を同一 RMS に揃えるので、以後に測る分離は音量差でなくスペクトル差に帰する。**
/// これは刺激生成側の校正であり、ネットワーク側の判断機構ではない。
pub fn synth_consonant_banded(c: Consonant, duration_ms: f64, f0_hz: f64, noise: &mut LfsrNoise) -> Vec<i32> {
    /// 帯域通過後の目標 RMS。
    ///
    /// 2026-08-25 に 5657 → 11314 (×2)。旧値は「純音 amp 8000 の RMS」だったが、
    /// 子音は広帯域なので 1 帯域に届く量が総 RMS よりずっと小さい。
    /// 実測 (`consonant_gate`・Q×3/閾160) で指定帯域の被覆が
    /// ×1 で 62.8%、×2 で 90.7% (精度 82.4% → 72.3%)。
    /// 母音とは必要な絶対スケールが違うことが実測で出たので、別々に校正する。
    const TARGET_RMS: i32 = 11314;

    let n_samples = (duration_ms * SAMPLE_RATE_HZ / 1000.0) as usize;

    match c {
        Consonant::Plosive { burst_freq_low, burst_freq_high, voiced } => {
            let mut bp = band_filter(burst_freq_low, burst_freq_high);
            let mut out = Vec::with_capacity(n_samples);
            // 2026-08-27: **閉鎖長を生体の比率に合わせた (軌道修正)。**
            //
            // 旧: 固定 10ms。子音長 30ms のうち **33%** が閉鎖。
            // 生体: 破裂音は**閉鎖が破裂より長い** (閉鎖 50ms / 破裂 10-20ms = 70-80%)。
            //       **比率が逆になっていた。**
            // 新: 子音長の CLOSURE_FRACTION_PERCENT (= 67%)。
            //     **CONSONANT_MS 自体は変えない** — 変えると母音との配分が動いて
            //     これまでの測定が比較不能になるため。
            //
            // **有声のときこの区間は voice bar だけが鳴る = 前有声の長さそのもの。**
            // §14.22.4 で「閉鎖 10ms はモーラ 120ms の 8% で、マージンが
            // 小数第4位になる」と記録した。それへの手当て。
            //
            // **旧経路 (synth_consonant・456 行) は変えない。**
            // あちらは有声性を実装していないので閉鎖長が手がかりにならず、
            // 変えると G1 (旧版で pa/ki/tu が完全同一) の対照が動くだけである。
            let silent = (n_samples * CLOSURE_FRACTION_PERCENT / 100).min(n_samples * 4 / 5);
            for _ in 0..silent {
                out.push(0);
            }
            for i in silent..n_samples {
                let decay = ((n_samples - i) * 1024 / (n_samples - silent).max(1)) as i32;
                let n = bp.process(noise.next_sample());
                out.push((n * decay) >> 10);
            }
            // 2026-08-27: 有声なら voice bar (前有声) を重ねる。
            // 閉鎖区間 (先頭 10ms の無音) にも乗るので、それが前有声そのものになる。
            let out = if voiced { mix_voice_bar(out, f0_hz) } else { out };
            normalize_rms(out, TARGET_RMS)
        }
        Consonant::Fricative { freq_low, freq_high, voiced } => {
            let mut bp = band_filter(freq_low, freq_high);
            let mut out = Vec::with_capacity(n_samples);
            let ramp = ((5.0 * SAMPLE_RATE_HZ / 1000.0) as usize).min(n_samples / 4).max(1);
            for i in 0..n_samples {
                let n = bp.process(noise.next_sample());
                let env = if i < ramp {
                    (i * 1024 / ramp) as i32
                } else if i + ramp >= n_samples {
                    (((n_samples - i) * 1024) / ramp) as i32
                } else {
                    1024
                };
                out.push((n * env) >> 10);
            }
            // 2026-08-27: 有声摩擦音は摩擦と同時に声帯が振動する
            let out = if voiced { mix_voice_bar(out, f0_hz) } else { out };
            normalize_rms(out, TARGET_RMS)
        }
        // 接近音: その位置の短い有声区間 (鼻音と同じ 2 フォルマント合成) で近似
        Consonant::Approximant { f1, f2 } => {
            normalize_rms(
                synth_consonant(Consonant::Nasal { f1, f2 }, duration_ms, noise),
                TARGET_RMS,
            )
        }
        // 破擦音: 破裂 (前半) + 摩擦 (後半) を連結
        Consonant::Affricate {
            burst_freq_low,
            burst_freq_high,
            fric_freq_low,
            fric_freq_high,
        } => {
            let burst_ms = duration_ms * 0.4;
            let fric_ms = duration_ms - burst_ms;
            let mut w = synth_consonant_banded(
                Consonant::Plosive { burst_freq_low, burst_freq_high, voiced: false },
                burst_ms,
                f0_hz,
                noise,
            );
            w.extend(synth_consonant_banded(
                Consonant::Fricative { freq_low: fric_freq_low, freq_high: fric_freq_high, voiced: false },
                fric_ms,
                f0_hz,
                noise,
            ));
            normalize_rms(w, TARGET_RMS)
        }
        // 鼻音は既に f1/f2 を使っているので波形生成は既存実装に委ねるが、
        // **RMS 正規化は掛ける** (2026-08-25 修正)。
        // 初版は素通ししており、mo だけ RMS 4189 と他子音 5657 より 26% 小さかった。
        // これは「全子音を同一 RMS に揃える」という本関数自身の宣言に反する実装バグ
        // (独立レビューで発覚)。既存 `synth_consonant` 側の 2026-05-30 校正は変えない。
        Consonant::Nasal { .. } => normalize_rms(synth_consonant(c, duration_ms, noise), TARGET_RMS),
        Consonant::None => Vec::new(),
    }
}

/// 帯域 [lo, hi] → 中心は幾何平均、Q = 中心 / 帯域幅（音響の慣例）。
fn band_filter(lo: f64, hi: f64) -> super::cochlea::BandpassBiquad {
    let fc = (lo * hi).sqrt();
    let bw = (hi - lo).max(1.0);
    super::cochlea::BandpassBiquad::new(fc, fc / bw, SAMPLE_RATE_HZ)
}

/// RMS を target に合わせる（整数演算のみ・平方根は整数ニュートン法）。
/// 有声子音の **voice bar (前有声)** を作る。
///
/// ## 物理
///
/// 有声破裂音では**閉鎖中も声帯が振動する**。だが声道が閉じているので
/// 高周波は放射されず、**低周波だけが組織を通って漏れる**。
/// スペクトログラムでは低域の帯 (voice bar) に見える。
/// **日本語の /b d g/ と /p t k/ を分ける主要な手がかりはこれ (前有声)。**
/// 有声摩擦音では摩擦と同時に声帯が振動する。
///
/// ## 実装 (実測前に宣言・以後動かさない)
///
/// F0 の**基本波 + 第 2 倍音** (振幅比 2:1)。閉鎖で高次倍音が落ちることの近似。
/// **振幅は子音本体の RMS の 1/4 (−12 dB)** とする。
///
/// ## 重要 — 音量では区別できないようにする
///
/// `normalize_rms` は**最後に**掛かるので、**有声も無声も総 RMS は同じ**になる。
/// **差は周波数構造だけであり、音量で当てられることはない。**
/// (音量で当てられると、測定が「有声性を聞き分けた」ことの証拠にならない)
fn voice_bar(base: &[i32], f0_hz: f64) -> Vec<i32> {
    let n = base.len();
    if n == 0 { return Vec::new(); }
    let rms = (base.iter().map(|&s| (s as f64) * (s as f64)).sum::<f64>() / n as f64).sqrt();
    let target = rms / 4.0; // 宣言: 1/4 (−12 dB)
    let (p1, p2) = (freq_to_phase_step(f0_hz), freq_to_phase_step(f0_hz * 2.0));
    let (mut ph1, mut ph2) = (0u32, 0u32);
    let raw: Vec<i32> = (0..n).map(|_| {
        let v = (sin_lookup(ph1) * 1000 + sin_lookup(ph2) * 500) >> 14;
        ph1 = ph1.wrapping_add(p1);
        ph2 = ph2.wrapping_add(p2);
        v
    }).collect();
    let r = (raw.iter().map(|&s| (s as f64) * (s as f64)).sum::<f64>() / n as f64).sqrt().max(1.0);
    let g = target / r;
    raw.iter().map(|&s| ((s as f64) * g).round() as i32).collect()
}

/// voice bar を重ねる (有声のときだけ呼ぶ)
fn mix_voice_bar(base: Vec<i32>, f0_hz: f64) -> Vec<i32> {
    let vb = voice_bar(&base, f0_hz);
    base.iter().zip(vb.iter()).map(|(&a, &b)| a.saturating_add(b)).collect()
}

fn normalize_rms(mut wave: Vec<i32>, target: i32) -> Vec<i32> {
    if wave.is_empty() {
        return wave;
    }
    let sum_sq: i64 = wave.iter().map(|&v| (v as i64) * (v as i64)).sum();
    let rms = isqrt_i64(sum_sq / wave.len() as i64);
    if rms > 0 {
        for v in wave.iter_mut() {
            *v = ((*v as i64) * (target as i64) / rms) as i32;
        }
    }
    wave
}

/// 整数平方根（ニュートン法・決定論的）。
fn isqrt_i64(n: i64) -> i64 {
    if n <= 0 {
        return 0;
    }
    let mut x = n;
    let mut y = (x + 1) / 2;
    while y < x {
        x = y;
        y = (x + n / x) / 2;
    }
    x
}

/// `synth_syllable_scaled` の帯域版。子音だけ `synth_consonant_banded` に差し替える。
pub fn synth_syllable_banded(syl: &Syllable, noise: &mut LfsrNoise, speed: f64) -> Vec<i32> {
    let consonant_ms = 30.0 / speed;
    let vowel_ms = 170.0 / speed;
    let mut wave = synth_consonant_banded(syl.consonant, consonant_ms, F0_DEFAULT_HZ, noise);
    wave.extend(synth_vowel(&syl.vowel, vowel_ms));
    wave
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

    // ──────────────────────────────────────────────────────────
    // 帯域つき子音 (2026-08-25)
    // ──────────────────────────────────────────────────────────

    /// ゼロ交差数 = 支配周波数の代理指標（整数のみ）。
    fn zero_crossings(wave: &[i32]) -> usize {
        wave.windows(2)
            .filter(|w| (w[0] < 0) != (w[1] < 0))
            .count()
    }

    fn plosive(lo: f64, hi: f64) -> Consonant {
        Consonant::Plosive { burst_freq_low: lo, burst_freq_high: hi, voiced: false }
    }

    /// 既存 `synth_consonant` の**欠陥を固定する**テスト。
    /// 破裂音の帯域指定は無視されるので、別の周波数を与えても波形が同一になる。
    /// これは仕様ではなくバグの記録——`synth_consonant_banded` が直す対象。
    #[test]
    fn legacy_consonant_ignores_burst_frequency() {
        let a = synth_consonant(plosive(500.0, 2000.0), 30.0, &mut LfsrNoise::new(0xACE1));
        let b = synth_consonant(plosive(2000.0, 4000.0), 30.0, &mut LfsrNoise::new(0xACE1));
        assert_eq!(a, b, "既存実装は帯域を捨てている（この一致が壊れたら既存挙動が変わった）");
    }

    /// 帯域版は同じ LFSR 種でも帯域が違えば波形が違う。
    #[test]
    fn banded_consonant_uses_burst_frequency() {
        let a = synth_consonant_banded(plosive(500.0, 2000.0), 30.0, F0_DEFAULT_HZ, &mut LfsrNoise::new(0xACE1));
        let b = synth_consonant_banded(plosive(2000.0, 4000.0), 30.0, F0_DEFAULT_HZ, &mut LfsrNoise::new(0xACE1));
        assert_ne!(a, b, "帯域を変えたのに波形が同一");
        assert_eq!(a.len(), b.len());
    }

    /// 支配周波数が帯域の順に並ぶ: pa(fc≈1000) < tu(fc≈2291) < ki(fc≈2828) < se(fc≈4899).
    /// 閾値は置かず**順序だけ**を見る（順序の正解は帯域指定＝実験者側にある）。
    #[test]
    fn banded_consonants_ordered_by_band() {
        let pa = synth_consonant_banded(plosive(500.0, 2000.0), 100.0, F0_DEFAULT_HZ, &mut LfsrNoise::new(0xACE1));
        let tu = synth_consonant_banded(plosive(1500.0, 3500.0), 100.0, F0_DEFAULT_HZ, &mut LfsrNoise::new(0xACE1));
        let ki = synth_consonant_banded(plosive(2000.0, 4000.0), 100.0, F0_DEFAULT_HZ, &mut LfsrNoise::new(0xACE1));
        let se = synth_consonant_banded(
            Consonant::Fricative { freq_low: 3000.0, freq_high: 8000.0, voiced: false },
            F0_DEFAULT_HZ,
            100.0, &mut LfsrNoise::new(0xACE1));
        let (z_pa, z_tu, z_ki, z_se) = (
            zero_crossings(&pa), zero_crossings(&tu), zero_crossings(&ki), zero_crossings(&se));
        assert!(z_pa < z_tu, "pa {} < tu {}", z_pa, z_tu);
        assert!(z_tu < z_ki, "tu {} < ki {}", z_tu, z_ki);
        assert!(z_ki < z_se, "ki {} < se {}", z_ki, z_se);
    }

    /// 実運用長 (30ms) でも 3 破裂音が区別できる。
    #[test]
    fn banded_consonants_differ_at_production_length() {
        let waves: Vec<Vec<i32>> = [(500.0, 2000.0), (1500.0, 3500.0), (2000.0, 4000.0)]
            .iter()
            .map(|&(lo, hi)| synth_consonant_banded(plosive(lo, hi), 30.0, F0_DEFAULT_HZ,
                                                    &mut LfsrNoise::new(0xACE1)))
            .collect();
        let z: Vec<usize> = waves.iter().map(|w| zero_crossings(w)).collect();
        assert!(z[0] < z[1] && z[1] < z[2], "30ms でのゼロ交差 {:?} が帯域順でない", z);
    }

    fn wave_rms(w: &[i32]) -> f64 {
        (w.iter().map(|&v| (v as f64) * (v as f64)).sum::<f64>() / w.len() as f64).sqrt()
    }

    /// 鼻音も含めて全帯域子音の RMS が揃うこと (2026-08-25 のレビュー指摘の回帰テスト)。
    /// 初版は Nasal だけ normalize_rms を素通りしており mo が 26% 小さかった。
    #[test]
    fn banded_nasal_shares_rms_with_others() {
        let nasal = synth_consonant_banded(
            Consonant::Nasal { f1: 250.0, f2: 1500.0 }, 30.0, F0_DEFAULT_HZ, &mut LfsrNoise::new(0xACE1));
        let plos = synth_consonant_banded(plosive(500.0, 2000.0), 30.0, F0_DEFAULT_HZ, &mut LfsrNoise::new(0xACE1));
        let (rn, rp) = (wave_rms(&nasal), wave_rms(&plos));
        assert!((rn - rp).abs() / rp < 0.20,
                "鼻音 RMS {:.0} が破裂音 {:.0} と揃っていない", rn, rp);
    }

    /// 振幅校正: 全帯域子音の RMS が TARGET_RMS (11314) に揃う。
    /// **ピークでなく RMS** を揃える——蝸牛を駆動するのは包絡線であり、
    /// ノイズの crest (実測 4.1-4.5) は母音 (2.28) と違うのでピークは比較不能。
    /// 分離が音量差に化けないための校正でもある。
    #[test]
    fn banded_consonants_share_rms() {
        let mut all = vec![
            Consonant::Fricative { freq_low: 3000.0, freq_high: 8000.0, voiced: false },
        ];
        for &(lo, hi) in &[(500.0, 2000.0), (1500.0, 3500.0), (2000.0, 4000.0)] {
            all.push(plosive(lo, hi));
        }
        for c in all {
            let w = synth_consonant_banded(c, 30.0, F0_DEFAULT_HZ, &mut LfsrNoise::new(0xACE1));
            let r = wave_rms(&w);
            // 破裂音は先頭 10ms が無音なので、その分だけ全体 RMS は目標を下回る。
            // 整数化と無音区間を許容して ±15% で判定。
            assert!((r - 11314.0).abs() / 11314.0 < 0.15,
                    "{:?} の RMS {:.0} が目標 11314 から外れる", c, r);
        }
    }

    #[test]
    fn isqrt_matches_float() {
        for n in [0i64, 1, 2, 3, 4, 99, 100, 12345, 1_000_000, 8_000_000_000] {
            let want = (n as f64).sqrt() as i64;
            let got = isqrt_i64(n);
            assert!((got - want).abs() <= 1, "isqrt({}) = {} (期待 {})", n, got, want);
        }
    }

    #[test]
    fn banded_consonant_deterministic() {
        let a = synth_consonant_banded(plosive(500.0, 2000.0), 30.0, F0_DEFAULT_HZ, &mut LfsrNoise::new(0xACE1));
        let b = synth_consonant_banded(plosive(500.0, 2000.0), 30.0, F0_DEFAULT_HZ, &mut LfsrNoise::new(0xACE1));
        assert_eq!(a, b);
    }

    /// 子音なし（母音のみのかな）は空の子音区間を返す。
    #[test]
    fn consonant_none_is_empty() {
        assert!(synth_consonant(Consonant::None, 30.0, &mut LfsrNoise::new(0xACE1)).is_empty());
        assert!(synth_consonant_banded(Consonant::None, 30.0, F0_DEFAULT_HZ, &mut LfsrNoise::new(0xACE1))
            .is_empty());
    }

    /// 標準 5 音節が帯域版では全ペア相異なる（従来は pa/ki/tu が同一子音だった）。
    #[test]
    fn banded_standard_syllables_pairwise_distinct() {
        let syls = standard_syllables();
        let waves: Vec<Vec<i32>> = syls.iter()
            .map(|s| synth_syllable_banded(s, &mut LfsrNoise::new(0xACE1), 1.0))
            .collect();
        for i in 0..waves.len() {
            for j in (i + 1)..waves.len() {
                assert_ne!(waves[i], waves[j], "{} と {} が同一波形",
                           syls[i].label, syls[j].label);
            }
        }
    }

    /// 帯域版の音節も長さは従来版と同じ（下流のパイプラインを壊さない）。
    #[test]
    fn banded_syllable_same_length_as_scaled() {
        for s in standard_syllables().iter() {
            let a = synth_syllable_scaled(s, &mut LfsrNoise::new(0xACE1), 1.0);
            let b = synth_syllable_banded(s, &mut LfsrNoise::new(0xACE1), 1.0);
            assert_eq!(a.len(), b.len(), "{} の長さが違う", s.label);
        }
    }


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
            Consonant::Plosive { burst_freq_low: 500.0, burst_freq_high: 2000.0, voiced: false },
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
