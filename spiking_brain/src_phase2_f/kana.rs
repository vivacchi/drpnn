//! かな → モーラ → 波形 (2026-08-26)
//!
//! 元の目的「コーパスを流してストリーミングで学ぶ」への復帰。
//! text → かな → モーラ列 → 音素合成 → 蝸牛 → M0.5 → M1 の、かな→モーラ の段。
//!
//! ## 方針
//!
//! - **静的表**。形態素解析も辞書も使わない (原理 3 決定論性・外部依存を持たない)。
//! - 子音のパラメータは Klatt 1980 と日本語音声学の標準値に基づいて置く。
//!   **どこから採ったかをコードに明記する** (後から検証できるように)。
//! - 漢字は扱わない。**ひらがな・カタカナ・長音符・促音・撥音**まで。
//!   漢字→かな変換には辞書が要る (未着手・§14 に記録)。
//!
//! ## モーラの型
//!
//! 日本語の音韻単位はモーラ。CV (子音+母音) のほか、
//! **長音「ー」・促音「っ」・撥音「ん」**が独立した 1 モーラを成す。
//! これらは CV でないので別の型にする。

use super::phoneme_synth::{
    band_filter, glottal_pulse_train, normalize_rms, synth_consonant_banded,
    synth_vowel_f0, synth_vowel_f0_full, vowels, FormantResonator,
    ANTIFORMANT_BW_HZ, CLOSURE_FRACTION_PERCENT, TRANSITION_MS, UTTERANCE_TARGET_RMS, Consonant, LfsrNoise, Vowel, SAMPLE_RATE_HZ,
};

/// 日本語のモーラ。
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Mora {
    /// 子音 + 母音 (子音なしの「あいうえお」は `Consonant::None`)
    Cv { consonant: Consonant, vowel: Vowel, locus: Option<[f64; 3]>, vot_ms: f64 },
    /// 長音「ー」: 直前のモーラの母音を 1 モーラぶん伸ばす
    Long,
    /// 促音「っ」: 無音 1 モーラ (次の子音の閉鎖に相当)
    Sokuon,
    /// 撥音「ん」: 鼻音 1 モーラ
    Moraic,
}

/// 子音のパラメータ表。
///
/// **出典と根拠**:
/// - 破裂音の burst 帯域: 調音位置で決まる。
///   両唇 /p,b/ は低め (500-2000Hz)、歯茎 /t,d/ は中 (1500-3500Hz)、
///   軟口蓋 /k,g/ は高め (2000-4000Hz)。既存の `standard_syllables()` の
///   pa/ki/tu と同じ値を踏襲する (過去の測定との連続性)。
/// - 摩擦音: /s/ は 3000-8000Hz、/h/ は広帯域 (500-4000Hz)、
///   /sh/(し) は /s/ より低め (2000-6000Hz)。
/// - 鼻音: /m/ は F1=250 F2=1500 (既存 mo と同じ)、/n/ は F1=250 F2=1700。
/// - 有声破裂音 /b,d,g/ は無声 /p,t,k/ と同じ帯域を使う。
///   **有声/無声の区別 (VOT・声帯振動の有無) は未実装**。§14 に記録。
///
/// **注意**: 蝸牛の `F_MAX_HZ = 4000` なので、/s/ の 3000-8000Hz は
/// 上の 5/8 が可測域の外にある (既知の制約・§8.1)。
/// フォルマント遷移の ON/OFF。**同じビルドの中で対照を取るために要る。**
///
/// 既定 ON。`DRPNN_FORMANT_TRANSITION=0` で OFF。`set_formant_transition` で実行時にも切れる。
/// (§14.21 の教訓: **同じビルドの OFF 対照が無いと A/B が成立しない**。
///  自発発火のとき `DRPNN_M0_SPONTANEOUS` を足したのと同じ理由。)
static TRANSITION: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(2);

/// 遷移が有効か。初回だけ環境変数を読む。**乱数は使わない。**
pub fn formant_transition_enabled() -> bool {
    use std::sync::atomic::Ordering;
    let v = TRANSITION.load(Ordering::Relaxed);
    if v == 2 {
        let on = std::env::var("DRPNN_FORMANT_TRANSITION").map(|s| s != "0").unwrap_or(true);
        TRANSITION.store(on as u8, Ordering::Relaxed);
        return on;
    }
    v == 1
}

/// 遷移を実行時に切り替える (対照実験用)。
pub fn set_formant_transition(on: bool) {
    TRANSITION.store(on as u8, std::sync::atomic::Ordering::Relaxed);
}

/// VOT (気音) の ON/OFF。**同じビルドの中で対照を取るために要る。**
/// 既定 ON。`DRPNN_VOT=0` で OFF。`set_vot` で実行時にも切れる。
static VOT_ON: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(2);

/// VOT が有効か。初回だけ環境変数を読む。**乱数は使わない。**
pub fn vot_enabled() -> bool {
    use std::sync::atomic::Ordering;
    let v = VOT_ON.load(Ordering::Relaxed);
    if v == 2 {
        let on = std::env::var("DRPNN_VOT").map(|s| s != "0").unwrap_or(true);
        VOT_ON.store(on as u8, Ordering::Relaxed);
        return on;
    }
    v == 1
}

/// VOT を実行時に切り替える (対照実験用)。
pub fn set_vot(on: bool) {
    VOT_ON.store(on as u8, std::sync::atomic::Ordering::Relaxed);
}

/// 無声摩擦音の解放後気音 (10ms)。**既定 ON** (2026-08-28・§14.51 案2)。
/// `DRPNN_FRIC_VOT=0` で従来 (気音なし = さ の接合部に −180dB の谷)。
static FRIC_VOT: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(2);

/// 有効か。初回だけ環境変数を読む。**乱数は使わない。**
pub fn fric_vot_enabled() -> bool {
    use std::sync::atomic::Ordering;
    let v = FRIC_VOT.load(Ordering::Relaxed);
    if v == 2 {
        let on = std::env::var("DRPNN_FRIC_VOT").map(|s| s != "0").unwrap_or(true);
        FRIC_VOT.store(on as u8, Ordering::Relaxed);
        return on;
    }
    v == 1
}

/// 実行時に切り替える (対照実験用)。
pub fn set_fric_vot(on: bool) {
    FRIC_VOT.store(on as u8, std::sync::atomic::Ordering::Relaxed);
}

/// /h/ を「雑音を後続母音の声道に通したもの」として合成する。**既定 ON** (2026-08-28・§14.51 /h/物理)。
///
/// **声門摩擦音の物理的真実**: /h/ は声門での乱流が**声道全体**を通ったもの =
/// 無声化した母音である。入り口監査の所見 M6 (「/h/ が後続母音と無関係な固定帯域ノイズ
/// なのは誤り」) への是正。**発明ではない。**
/// 歯擦音 (さ行・し) には適用しない — 歯擦音は前腔だけを励振するので、
/// 全声道に通すのは物理的に誤り (Klatt が並列合成を選んだ理由)。
/// `DRPNN_GLOTTAL_H=0` で従来 (固定帯域ノイズ 500-4000)。
static GLOTTAL_H: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(2);

/// 有効か。初回だけ環境変数を読む。**乱数は使わない。**
pub fn glottal_h_enabled() -> bool {
    use std::sync::atomic::Ordering;
    let v = GLOTTAL_H.load(Ordering::Relaxed);
    if v == 2 {
        let on = std::env::var("DRPNN_GLOTTAL_H").map(|s| s != "0").unwrap_or(true);
        GLOTTAL_H.store(on as u8, Ordering::Relaxed);
        return on;
    }
    v == 1
}

/// 実行時に切り替える (対照実験用)。
pub fn set_glottal_h(on: bool) {
    GLOTTAL_H.store(on as u8, std::sync::atomic::Ordering::Relaxed);
}

/// **VOT (Voice Onset Time)** = 破裂の解放から声帯振動が始まるまでの時間 [ms]。(2026-08-27)
///
/// 日本語の語頭無声破裂音の代表値 (Shimizu 1996 / Riney et al. 2007)。
/// **軟口蓋が最も長いのは言語普遍**である (閉鎖の解放が遅く、声門下圧が抜けにくい)。
///
/// **有声破裂音は 0**。有声側の手がかりは解放「前」の voice bar (前有声) であり、
/// 解放「後」に気音は入らない。**この非対称そのものが有声性の手がかりである。**
///
/// 摩擦音・鼻音・接近音・弾き音は 0。破裂という事象が無いので VOT が定義できない。
///
/// **これらの値は文献の代表値であり、結果を見てから動かさない。**
fn vot_ms_of(row: char) -> f64 {
    if !vot_enabled() {
        return 0.0;
    }
    match row {
        'p' => 25.0,          // 両唇 無声破裂
        't' => 25.0,          // 歯茎 無声破裂
        'k' => 45.0,          // 軟口蓋 無声破裂 (**最も長い**)
        'c' | 'C' => 10.0,    // 無声破擦 (つ [ts] ・ち [tɕ]) — 摩擦の後の気音は短い
        // §14.51 案2 (2026-08-28): **無声摩擦音にも解放後の気音 10ms。**
        // 値は破擦音の 10ms の転用 (**発明ではない**)。これが無いと連続合成で
        // 「摩擦音の間、声道への入力ゼロ → 共鳴器が死ぬ → 有声化の瞬間にパルス閉相なら
        //  厳密にゼロ」= さ の接合部の −180dB の谷 (§14.41.3) になる。
        // 気音は毎サンプル雑音で声道を叩くので、ゼロ状態から即座に立ち上がる。
        's' | 'S' | 'h' if fric_vot_enabled() => 10.0,
        _ => 0.0,             // 有声破裂 (前有声で符号化) / 摩擦 / 鼻音 / 接近 / 弾き
    }
}

/// 調音位置ごとの**フォルマント遷移の始点** (locus)。[F1, F2, F3] Hz。(2026-08-27)
///
/// locus theory (Delattre, Liberman & Cooper 1955) の代表値。
/// 子音の調音位置は、**後続母音のフォルマントがどこから動き出すか**に転写される。
///
/// **F1 が全ての阻害音で低いのは閉鎖そのものの帰結**であり (狭めが強いほど F1 は下がる)、
/// 位置によらない。したがって **F1 の遷移は調音方法を、F2 の遷移は調音位置を運ぶ。**
///
/// `None` = 遷移なし。/h/ は声門摩擦音で**後続母音の声道形状をそのまま使う**
/// (無声化した母音に等しい) ので locus を持たない。**これは音声学的に正しい。**
/// (ひ [çi]・ふ [ɸɯ] が実際には硬口蓋・両唇であることは、は行を 'h' 一つに畳んでいる
///  既存の単純化の側の問題であって、ここでは触らない。)
fn locus_of(row: char) -> Option<[f64; 3]> {
    if !formant_transition_enabled() {
        return None;
    }
    match row {
        'p' | 'b' | 'm' => Some([250.0, 750.0, 2100.0]),                 // 両唇
        'w' => Some([250.0, 600.0, 2100.0]),                             // 両唇軟口蓋 (F2 最低)
        't' | 'd' | 'n' | 's' | 'z' | 'c' | 'r' => Some([250.0, 1750.0, 2700.0]), // 歯茎
        'k' | 'g' => Some([250.0, 2200.0, 2400.0]),                      // 軟口蓋 (velar pinch)
        'S' | 'Z' | 'C' | 'y' => Some([250.0, 2600.0, 3000.0]),          // 硬口蓋
        _ => None,                                                        // 声門 /h/・母音単独
    }
}

fn consonant_of(row: char) -> Consonant {
    match row {
        // 2026-08-27: **有声/無声を分離した (軌道修正)**。
        //
        // それまで 'k'|'g' / 't'|'d' / 'p'|'b' / 's'|'z' を**同じ子音にマップ**しており、
        // **濁音と清音が完全に同一の波形**だった。実コーパスでは
        // **モーラの 41.7% がこの縮退の影響を受けていた** (§14.21)。
        // 46 項目のコーパスには濁音が無かったので、この穴は一度も表に出ていなかった。
        //
        // 帯域 (調音位置) は同じで、**違うのは声帯振動の有無だけ**。
        // これは生体でも同じで、/k/ と /g/ は同じ軟口蓋閉鎖である。
        'k' => Consonant::Plosive { burst_freq_low: 2000.0, burst_freq_high: 4000.0, voiced: false },
        'g' => Consonant::Plosive { burst_freq_low: 2000.0, burst_freq_high: 4000.0, voiced: true },
        't' => Consonant::Plosive { burst_freq_low: 1500.0, burst_freq_high: 3500.0, voiced: false },
        'd' => Consonant::Plosive { burst_freq_low: 1500.0, burst_freq_high: 3500.0, voiced: true },
        'p' => Consonant::Plosive { burst_freq_low: 500.0, burst_freq_high: 2000.0, voiced: false },
        'b' => Consonant::Plosive { burst_freq_low: 500.0, burst_freq_high: 2000.0, voiced: true },
        's' => Consonant::Fricative { freq_low: 3000.0, freq_high: 8000.0, voiced: false },
        'z' => Consonant::Fricative { freq_low: 3000.0, freq_high: 8000.0, voiced: true },
        'S' => Consonant::Fricative { freq_low: 2000.0, freq_high: 6000.0, voiced: false }, // し・しゃ行
        // 2026-08-27 新設: じ・ぢ (/ʑ/)。現代日本語で じ=ぢ は同音なので同じ記号でよい。
        'Z' => Consonant::Fricative { freq_low: 2000.0, freq_high: 6000.0, voiced: true },
        // 破擦音 (2026-08-26): 摩擦のみの近似では す=つ / し=ち が
        // 完全に同一の応答になっていた (実測)。破裂 + 摩擦の複合として作る。
        'c' => Consonant::Affricate {
            burst_freq_low: 1500.0,
            burst_freq_high: 3500.0,   // /t/ と同じ歯茎の破裂
            fric_freq_low: 3000.0,
            fric_freq_high: 8000.0,    // /s/ と同じ摩擦
        },
        'C' => Consonant::Affricate {
            burst_freq_low: 1500.0,
            burst_freq_high: 3500.0,
            fric_freq_low: 2000.0,
            fric_freq_high: 6000.0,    // /sh/ と同じ摩擦 (ち)
        },
        'h' => Consonant::Fricative { freq_low: 500.0, freq_high: 4000.0, voiced: false }, // は行は無声
        'm' => Consonant::Nasal { f1: 250.0, f2: 1500.0, zero_hz: 1000.0 },  // 両唇
        'n' => Consonant::Nasal { f1: 250.0, f2: 1700.0, zero_hz: 1800.0 },  // 歯茎
        // ラ行 (弾き音) は破裂音で近似する。**正確でない**ことを明記 (§14)。
        // 2026-08-27: ら行 /ɾ/ は**有声**の弾き音。破裂音で近似しているが声帯は振動する。
        // これで た(無声) と ら(有声) に手がかりが増える
        // (§14.6.2 で た-ら が最も紛らわしい対 0.9995 だった)。
        'r' => Consonant::Plosive { burst_freq_low: 1200.0, burst_freq_high: 2800.0, voiced: true },
        // 接近音 (2026-08-26): 母音のみの近似では や=あ / ゆ=う / よ=お / わ=あ / を=お が
        // 完全に同一の応答になっていた (実測) ので、専用の型を与える。
        'y' => Consonant::Approximant { f1: 300.0, f2: 2200.0 }, // 硬口蓋
        'w' => Consonant::Approximant { f1: 300.0, f2: 700.0 },  // 両唇軟口蓋
        _ => Consonant::None,
    }
}

/// 母音のインデックス (0=a, 1=i, 2=u, 3=e, 4=o)
fn vowel_of(idx: usize) -> Vowel {
    vowels()[idx]
}

/// かな 1 文字 → (子音の行, 母音のインデックス)。
///
/// 静的表。カタカナはひらがなに正規化してから引く。
fn kana_to_cv(c: char) -> Option<(char, usize)> {
    let table: &[(char, char, usize)] = &[
        ('あ', '-', 0), ('い', '-', 1), ('う', '-', 2), ('え', '-', 3), ('お', '-', 4),
        ('か', 'k', 0), ('き', 'k', 1), ('く', 'k', 2), ('け', 'k', 3), ('こ', 'k', 4),
        ('が', 'g', 0), ('ぎ', 'g', 1), ('ぐ', 'g', 2), ('げ', 'g', 3), ('ご', 'g', 4),
        ('さ', 's', 0), ('し', 'S', 1), ('す', 's', 2), ('せ', 's', 3), ('そ', 's', 4),
        ('ざ', 'z', 0), ('じ', 'Z', 1), ('ず', 'z', 2), ('ぜ', 'z', 3), ('ぞ', 'z', 4),
        ('た', 't', 0), ('ち', 'C', 1), ('つ', 'c', 2), ('て', 't', 3), ('と', 't', 4),
        ('だ', 'd', 0), ('ぢ', 'Z', 1), ('づ', 'z', 2), ('で', 'd', 3), ('ど', 'd', 4),
        ('な', 'n', 0), ('に', 'n', 1), ('ぬ', 'n', 2), ('ね', 'n', 3), ('の', 'n', 4),
        ('は', 'h', 0), ('ひ', 'h', 1), ('ふ', 'h', 2), ('へ', 'h', 3), ('ほ', 'h', 4),
        ('ば', 'b', 0), ('び', 'b', 1), ('ぶ', 'b', 2), ('べ', 'b', 3), ('ぼ', 'b', 4),
        ('ぱ', 'p', 0), ('ぴ', 'p', 1), ('ぷ', 'p', 2), ('ぺ', 'p', 3), ('ぽ', 'p', 4),
        ('ま', 'm', 0), ('み', 'm', 1), ('む', 'm', 2), ('め', 'm', 3), ('も', 'm', 4),
        ('や', 'y', 0), ('ゆ', 'y', 2), ('よ', 'y', 4),
        ('ら', 'r', 0), ('り', 'r', 1), ('る', 'r', 2), ('れ', 'r', 3), ('ろ', 'r', 4),
        ('わ', 'w', 0), ('を', 'w', 4),
        // 小書き (拗音の第2要素として単独で来た場合の保険)
        ('ぁ', '-', 0), ('ぃ', '-', 1), ('ぅ', '-', 2), ('ぇ', '-', 3), ('ぉ', '-', 4),
        ('ゃ', 'y', 0), ('ゅ', 'y', 2), ('ょ', 'y', 4),
    ];
    table.iter().find(|&&(k, _, _)| k == c).map(|&(_, r, v)| (r, v))
}

/// カタカナ → ひらがな (U+30A1..U+30F6 を U+3041..U+3096 に写す)
fn to_hiragana(c: char) -> char {
    let u = c as u32;
    if (0x30A1..=0x30F6).contains(&u) {
        char::from_u32(u - 0x60).unwrap_or(c)
    } else {
        c
    }
}

/// かな列 → モーラ列。
///
/// 拗音 (きゃ・しゅ 等) は**2 文字で 1 モーラ**なので、小書きが続いたら
/// 直前のモーラの母音を差し替える。
/// 表に無い文字 (漢字・記号・英数) は**黙って捨てず**、呼び出し側が
/// 気づけるよう返り値の第 2 要素で数を返す。
pub fn moras_from_kana(s: &str) -> (Vec<Mora>, usize) {
    let mut out: Vec<Mora> = Vec::new();
    let mut skipped = 0usize;
    for raw in s.chars() {
        let c = to_hiragana(raw);
        match c {
            'ー' => out.push(Mora::Long),
            'っ' => out.push(Mora::Sokuon),
            'ん' => out.push(Mora::Moraic),
            'ゃ' | 'ゅ' | 'ょ' => {
                // 拗音: 直前の CV の母音を差し替える (2 文字で 1 モーラ)
                let v = match c {
                    'ゃ' => 0,
                    'ゅ' => 2,
                    _ => 4,
                };
                if let Some(Mora::Cv { consonant, locus, vot_ms, .. }) = out.last().copied() {
                    let n = out.len();
                    out[n - 1] = Mora::Cv { consonant, vowel: vowel_of(v), locus, vot_ms };
                } else {
                    skipped += 1;
                }
            }
            _ => match kana_to_cv(c) {
                Some((row, vi)) => out.push(Mora::Cv {
                    consonant: consonant_of(row),
                    vowel: vowel_of(vi),
                    locus: locus_of(row),
                    vot_ms: vot_ms_of(row),
                }),
                None => skipped += 1,
            },
        }
    }
    (out, skipped)
}

/// 1 モーラの標準長 [ms]。日本語のモーラは等時性が強い。
pub const MORA_MS: f64 = 120.0;
/// モーラ内の子音区間 [ms]
pub const CONSONANT_MS: f64 = 30.0;

/// モーラ列 → 波形。
///
/// - `Cv`: 子音 (帯域つき) + 母音 (F0 つき)
/// - `Long`: 直前の母音をもう 1 モーラ伸ばす
/// - `Sokuon`: 無音 1 モーラ
/// - `Moraic`: 鼻音 1 モーラ
///
/// F0 は発話全体で一定 (抑揚は未実装・§14)。
pub fn synth_utterance(moras: &[Mora], f0_hz: f64, noise: &mut LfsrNoise) -> Vec<i32> {
    // **連続合成**(2026-08-27)。既定 OFF。DRPNN_CONTINUOUS=1 / set_continuous(true) で ON。
    if continuous_enabled() {
        return synth_utterance_continuous(moras, f0_hz, noise);
    }
    let mut out: Vec<i32> = Vec::new();
    let mut last_vowel: Option<Vowel> = None;
    for m in moras {
        match *m {
            Mora::Cv { consonant, vowel, locus, vot_ms } => {
                if consonant != Consonant::None {
                    out.extend(synth_consonant_banded(consonant, CONSONANT_MS, f0_hz, noise));
                    // **フォルマント遷移 + VOT** (2026-08-27):
                    //   遷移 = 子音の調音位置を後続母音のフォルマントの動きに転写する。
                    //   VOT  = 無声破裂音の解放後、声帯振動が始まるまでを気音にする。
                    // どちらも 0/None なら従来と同一波形。
                    out.extend(synth_vowel_f0_full(
                        &vowel, f0_hz, MORA_MS - CONSONANT_MS, locus, vot_ms, noise,
                    ));
                } else {
                    out.extend(synth_vowel_f0(&vowel, f0_hz, MORA_MS));
                }
                last_vowel = Some(vowel);
            }
            Mora::Long => {
                if let Some(v) = last_vowel {
                    out.extend(synth_vowel_f0(&v, f0_hz, MORA_MS));
                } else {
                    out.extend(std::iter::repeat(0).take(
                        (MORA_MS * SAMPLE_RATE_HZ / 1000.0) as usize,
                    ));
                }
            }
            Mora::Sokuon => {
                out.extend(
                    std::iter::repeat(0).take((MORA_MS * SAMPLE_RATE_HZ / 1000.0) as usize),
                );
            }
            Mora::Moraic => {
                out.extend(synth_consonant_banded(
                    // 撥音 ん は口蓋垂音 [ɴ]。**極は な と同じでも零点が位置を運ぶ** (Fujimura 1962)。
                    Consonant::Nasal { f1: 250.0, f2: 1700.0, zero_hz: 2800.0 },
                    MORA_MS,
                    f0_hz,
                    noise,
                ));
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic_kana_become_moras() {
        let (m, skipped) = moras_from_kana("あいうえお");
        assert_eq!(m.len(), 5);
        assert_eq!(skipped, 0);
        assert!(matches!(m[0], Mora::Cv { consonant: Consonant::None, .. }));
    }

    #[test]
    fn katakana_normalizes_to_hiragana() {
        let (h, _) = moras_from_kana("かきくけこ");
        let (k, _) = moras_from_kana("カキクケコ");
        assert_eq!(h, k, "カタカナがひらがなと同じモーラ列にならない");
    }

    /// 長音・促音・撥音はそれぞれ独立した 1 モーラ。
    #[test]
    fn special_moras_count_as_one_each() {
        let (m, skipped) = moras_from_kana("がっこう");
        // が / っ / こ / う = 4 モーラ
        assert_eq!(m.len(), 4, "{:?}", m);
        assert_eq!(skipped, 0);
        assert_eq!(m[1], Mora::Sokuon);

        let (m2, _) = moras_from_kana("かーん");
        assert_eq!(m2.len(), 3);
        assert_eq!(m2[1], Mora::Long);
        assert_eq!(m2[2], Mora::Moraic);
    }

    /// 拗音は 2 文字で 1 モーラ。
    #[test]
    fn youon_is_one_mora() {
        let (m, skipped) = moras_from_kana("きゃきゅきょ");
        assert_eq!(m.len(), 3, "拗音が 1 モーラになっていない: {:?}", m);
        assert_eq!(skipped, 0);
    }

    /// 表に無い文字は黙って捨てず、数を返す。
    #[test]
    fn unknown_chars_are_counted_not_swallowed() {
        let (m, skipped) = moras_from_kana("あ漢字い");
        assert_eq!(m.len(), 2);
        assert_eq!(skipped, 2, "漢字が黙って捨てられている");
    }

    /// 発話の長さがモーラ数に比例する (等時性)。
    #[test]
    fn utterance_length_tracks_mora_count() {
        let mut n = LfsrNoise::new(0xACE1);
        let (m3, _) = moras_from_kana("あいう");
        let w3 = synth_utterance(&m3, 150.0, &mut n);
        let mut n2 = LfsrNoise::new(0xACE1);
        let (m6, _) = moras_from_kana("あいうえおか");
        let w6 = synth_utterance(&m6, 150.0, &mut n2);
        let ratio = w6.len() as f64 / w3.len() as f64;
        assert!((ratio - 2.0).abs() < 0.15, "6 モーラ / 3 モーラ = {:.2}", ratio);
    }

    /// 決定論的であること (原理 3)。
    #[test]
    fn utterance_is_deterministic() {
        let run = || {
            let mut n = LfsrNoise::new(0xACE1);
            let (m, _) = moras_from_kana("こんにちは");
            synth_utterance(&m, 150.0, &mut n)
        };
        assert_eq!(run(), run());
    }

    /// 促音は無音であること。
    #[test]
    fn sokuon_is_silent() {
        let mut n = LfsrNoise::new(0xACE1);
        let w = synth_utterance(&[Mora::Sokuon], 150.0, &mut n);
        assert!(w.iter().all(|&v| v == 0), "促音が無音でない");
        assert_eq!(w.len(), (MORA_MS * SAMPLE_RATE_HZ / 1000.0) as usize);
    }
}


// ──────────────────────────────────────────────────────────────
// 連続合成 (2026-08-27)
// ──────────────────────────────────────────────────────────────

/// 連続合成の気音 (asp) 区間を、**放射後の RMS で有声駆動と同じ音量に合わせる**か。
/// **既定 ON** (2026-08-28・§14.51)。`DRPNN_ASP_MATCH=0` で従来 (整合なし)。
///
/// ## なぜ
///
/// G98f で実測: 無声/有声の最小対で、母音頭 30-45ms (気音区間) の RMS 比が **12〜52 倍**。
/// 雑音源 (±16384) は声帯パルス列 (峰 4096・duty 40%) よりはるかに強く、
/// さらに放射 (+6dB/oct) が平坦スペクトルの雑音を優遇するため。
/// **回帰テストの有声性 100.0% は、周期性でなくこの音量差を読んだ疑いが濃い。**
///
/// これは §14.41 の連続合成が最初から持っていた欠陥である (破裂音の VOT も同罪)。
/// **§14.34 で離散経路に施した是正 (「気音は放射の後で合わせる」) を、
/// 連続経路に適用し忘れていた。** 同じ原則の適用であり、発明ではない。
///
/// ## どう合わせるか
///
/// asp 区間の頭で、**同じ声道状態・同じ声道軌道**に対して
/// (a) 声帯パルスで駆動した放射後 RMS (基準) と (b) 雑音で駆動した放射後 RMS を
/// 局所 2 パスで測り、比で雑音源を補正する。系は線形なので 1 回で足りる。
/// **目標値は「同じ区間を有声で鳴らしたときの音量」そのもの** — 新しいパラメータは無い。
static ASP_MATCH: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(2);

/// 有効か。初回だけ環境変数を読む。**乱数は使わない。**
pub fn asp_match_enabled() -> bool {
    use std::sync::atomic::Ordering;
    let v = ASP_MATCH.load(Ordering::Relaxed);
    if v == 2 {
        let on = std::env::var("DRPNN_ASP_MATCH").map(|s| s != "0").unwrap_or(true);
        ASP_MATCH.store(on as u8, Ordering::Relaxed);
        return on;
    }
    v == 1
}

/// 実行時に切り替える (対照実験用)。
pub fn set_asp_match(on: bool) {
    ASP_MATCH.store(on as u8, std::sync::atomic::Ordering::Relaxed);
}

/// 連続合成を使うか。**既定 ON** (2026-08-27・§14.41 の A/B のあと)。
/// `DRPNN_CONTINUOUS=0` で旧経路 (断片連結)。
static CONTINUOUS: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(2);

/// 有効か。初回だけ環境変数を読む。**乱数は使わない。**
pub fn continuous_enabled() -> bool {
    use std::sync::atomic::Ordering;
    let v = CONTINUOUS.load(Ordering::Relaxed);
    if v == 2 {
        let on = std::env::var("DRPNN_CONTINUOUS").map(|s| s != "0").unwrap_or(true);
        CONTINUOUS.store(on as u8, Ordering::Relaxed);
        return on;
    }
    v == 1
}

/// 実行時に切り替える (対照実験用)。
pub fn set_continuous(on: bool) {
    CONTINUOUS.store(on as u8, std::sync::atomic::Ordering::Relaxed);
}

/// 摩擦枝に帯域通過した雑音を書き込む。**2ms の ramp はクリック防止の最小限。**
fn write_fric(buf: &mut [i32], noise: &mut LfsrNoise, a: usize, b: usize, lo: f64, hi: f64) {
    let b = b.min(buf.len());
    if b <= a { return; }
    let mut f = band_filter(lo, hi);
    let len = b - a;
    let ramp = ((2.0 * SAMPLE_RATE_HZ / 1000.0) as usize).min(len / 4).max(1);
    for i in 0..len {
        let x = f.process(noise.next_sample());
        let env = if i < ramp { (i * 1024 / ramp) as i32 }
                  else if i + ramp >= len { (((len - i) * 1024) / ramp) as i32 }
                  else { 1024 };
        buf[a + i] = buf[a + i].saturating_add(((x as i64 * env as i64) >> 10) as i32);
    }
}

/// **発話全体を 1 本の連続した声帯源と、連続的に変化する声道で合成する。** (2026-08-27)
///
/// ## なぜ
///
/// §14.38 で出荷コードによって確定した:
/// - **モーラ境界を音響が越えない** (連続合成と個別連結が**バイト同一**)
/// - **子音-母音境界にちょうど −35 dB の谷**があり、2窓の境界と完全一致
///
/// ユーザーの指摘: 「**ストリームで考えているのだから、かなごとに区切って
/// 非線形にしたら意味がない**」
///
/// **旧経路は「独立に合成した断片を、それぞれ ramp でゼロに落としてから連結」していた。**
/// 実音声は連続発声であり、**声道は前のモーラの形から次の形へ連続的に動く**。
///
/// ## 構造 (Klatt 型の並列合成)
///
/// - 声帯源は発話全体で 1 本 (位相連続) → 声道 (3 共鳴器・毎サンプル補間) → 放射
/// - 雑音 → 帯域通過 (破裂/摩擦の枝・声道を通さない) を足す
/// - **声道の形はアンカーの間を線形補間する。** アンカーは
///   (子音の解放時点 = locus) と (遷移の終わり = 母音の目標値)。
///   **前のモーラの母音から次の子音の locus へも補間されるので、
///   協調調音がモーラ境界を越える。** これが旧経路との決定的な違い。
/// - **ramp は発話の先頭と末尾だけ。** モーラ境界にも子音-母音境界にも置かない。
/// - **正規化は発話全体で 1 回。**
/// - 無声破裂音の閉鎖が無音なのは**実音声でも正しい**のでそのまま残す。
///
/// ## 直していないこと
///
/// 摩擦の枝は声道を通さない (Klatt と同じ近似) / 先行母音長・閉鎖長の
/// 有声-無声非対称はまだ無い / F4・F5 は無い。
pub fn synth_utterance_continuous(moras: &[Mora], f0_hz: f64, noise: &mut LfsrNoise) -> Vec<i32> {
    let spm = (MORA_MS * SAMPLE_RATE_HZ / 1000.0) as usize;
    let cn_s = (CONSONANT_MS * SAMPLE_RATE_HZ / 1000.0) as usize;
    let tr_s = (TRANSITION_MS * SAMPLE_RATE_HZ / 1000.0) as usize;
    let n = moras.len() * spm;
    if n == 0 { return Vec::new(); }
    const BW: [f64; 3] = [60.0, 90.0, 150.0];

    // ---- 1) 計画を立てる ----
    let mut anchors: Vec<(usize, [f64; 3])> = Vec::new();
    let mut vg = vec![0i32; n];        // 声帯源の利得 (0..4096)
    let mut asp = vec![false; n];      // 声道を雑音で駆動する (気音)
    let mut fric = vec![0i32; n];      // 摩擦枝
    let mut az = vec![0f64; n];        // 反共鳴の周波数 (0 = なし)
    let mut last_v: Option<Vowel> = None;

    for (mi, m) in moras.iter().enumerate() {
        let s = mi * spm;
        match *m {
            Mora::Cv { consonant, vowel, locus, vot_ms } => {
                let vot_n = ((vot_ms * SAMPLE_RATE_HZ / 1000.0) as usize).min(spm - cn_s);
                match consonant {
                    Consonant::None => {
                        anchors.push((s + tr_s, vowel.formants_hz));
                        for i in s..(s + spm).min(n) { vg[i] = 4096; }
                    }
                    Consonant::Plosive { burst_freq_low, burst_freq_high, voiced } => {
                        let cl = cn_s * CLOSURE_FRACTION_PERCENT / 100;
                        for i in s..(s + cl).min(n) { vg[i] = if voiced { 1024 } else { 0 }; }
                        write_fric(&mut fric, noise, s + cl, s + cn_s, burst_freq_low, burst_freq_high);
                        if let Some(l) = locus { anchors.push((s + cn_s, l)); }
                        anchors.push((s + cn_s + tr_s, vowel.formants_hz));
                        for i in (s + cl)..(s + spm).min(n) { vg[i] = 4096; }
                        for i in (s + cn_s)..(s + cn_s + vot_n).min(n) { asp[i] = true; }
                    }
                    Consonant::Fricative { freq_low, freq_high, voiced } => {
                        // §14.51 (2026-08-28): **/h/ = 無声摩擦音で口腔の locus を持たない
                        // 唯一の子音** (声門なので位置が無い・§14.27 の設計判断がそのまま
                        // 判別子になる。enum に手を入れる必要が無い)。
                        // 物理: /h/ は雑音が**後続母音の声道全体**を通ったもの = 無声化した母音。
                        // 歯擦音には適用しない (前腔だけの励振なので全声道は物理的に誤り)。
                        let is_glottal = !voiced && locus.is_none() && glottal_h_enabled();
                        if is_glottal {
                            // 帯域ノイズは使わない。**気音と同じ機構**で声道を雑音駆動する。
                            for i in s..(s + cn_s).min(n) { asp[i] = true; vg[i] = 4096; }
                            // /h/ は自分の声道形を持たないので、最初から後続母音へ向かう
                            anchors.push((s + tr_s, vowel.formants_hz));
                        } else {
                            write_fric(&mut fric, noise, s, s + cn_s, freq_low, freq_high);
                            for i in s..(s + cn_s).min(n) { vg[i] = if voiced { 1024 } else { 0 }; }
                            if let Some(l) = locus { anchors.push((s + cn_s, l)); }
                            anchors.push((s + cn_s + tr_s, vowel.formants_hz));
                        }
                        for i in (s + cn_s)..(s + spm).min(n) { vg[i] = 4096; }
                        for i in (s + cn_s)..(s + cn_s + vot_n).min(n) { asp[i] = true; }
                    }
                    Consonant::Affricate { burst_freq_low, burst_freq_high,
                                           fric_freq_low, fric_freq_high } => {
                        let cl = cn_s * 2 / 5;
                        for i in s..(s + cl).min(n) { vg[i] = 0; }
                        write_fric(&mut fric, noise, s + cl, s + cl + cn_s / 5,
                                   burst_freq_low, burst_freq_high);
                        write_fric(&mut fric, noise, s + cl + cn_s / 5, s + cn_s,
                                   fric_freq_low, fric_freq_high);
                        if let Some(l) = locus { anchors.push((s + cn_s, l)); }
                        anchors.push((s + cn_s + tr_s, vowel.formants_hz));
                        for i in (s + cn_s)..(s + spm).min(n) { vg[i] = 4096; }
                        for i in (s + cn_s)..(s + cn_s + vot_n).min(n) { asp[i] = true; }
                    }
                    Consonant::Nasal { f1, f2, zero_hz } => {
                        anchors.push((s + cn_s / 2, [f1, f2, 2700.0]));
                        anchors.push((s + cn_s + tr_s, vowel.formants_hz));
                        for i in s..(s + spm).min(n) { vg[i] = 4096; }
                        for i in s..(s + cn_s).min(n) { az[i] = zero_hz; }
                    }
                    Consonant::Approximant { f1, f2 } => {
                        anchors.push((s + cn_s / 2, [f1, f2, 2700.0]));
                        anchors.push((s + cn_s + tr_s, vowel.formants_hz));
                        for i in s..(s + spm).min(n) { vg[i] = 4096; }
                    }
                }
                anchors.push((s + spm, vowel.formants_hz));
                last_v = Some(vowel);
            }
            Mora::Long => {
                if let Some(v) = last_v {
                    anchors.push((s + spm, v.formants_hz));
                    for i in s..(s + spm).min(n) { vg[i] = 4096; }
                }
            }
            Mora::Sokuon => { /* 無音。声道の形は保持 (アンカーを置かない) */ }
            Mora::Moraic => {
                anchors.push((s + cn_s, [250.0, 1700.0, 2700.0]));
                anchors.push((s + spm, [250.0, 1700.0, 2700.0]));
                for i in s..(s + spm).min(n) { vg[i] = 4096; az[i] = 2800.0; }
            }
        }
    }
    if anchors.is_empty() { return vec![0i32; n]; }
    anchors.sort_by_key(|&(t, _)| t);

    // ---- 2) アンカーの間を線形補間して毎サンプルの声道の形を作る ----
    let mut form = vec![[0f64; 3]; n];
    let (mut k, mut prev_t, mut prev_f) = (0usize, 0usize, anchors[0].1);
    for i in 0..n {
        while k < anchors.len() && anchors[k].0 <= i {
            prev_t = anchors[k].0;
            prev_f = anchors[k].1;
            k += 1;
        }
        form[i] = match anchors.get(k) {
            Some(&(t, f)) if t > prev_t => {
                let a = ((i.saturating_sub(prev_t)) as f64 / (t - prev_t) as f64).min(1.0);
                [prev_f[0] + (f[0] - prev_f[0]) * a,
                 prev_f[1] + (f[1] - prev_f[1]) * a,
                 prev_f[2] + (f[2] - prev_f[2]) * a]
            }
            _ => prev_f,
        };
    }

    // ---- 3) 走らせる ----
    let pulse = glottal_pulse_train(f0_hz, n, 4096);
    let mut rs: Vec<FormantResonator> = (0..3)
        .map(|j| FormantResonator::new(form[0][j], BW[j], 4.0, SAMPLE_RATE_HZ))
        .collect();
    let (mut ax1, mut ax2) = (0i64, 0i64);
    let mut a_cache: (f64, i64, i64, i64) = (0.0, 0, 0, 0);
    let mut voiced: Vec<i32> = Vec::with_capacity(n);
    // 気音の音量整合 (§14.51): 現在の asp 区間に適用する雑音の倍率 (分子/分母)。
    let (mut asp_num, mut asp_den) = (1i64, 1i64);
    for i in 0..n {
        for (j, r) in rs.iter_mut().enumerate() {
            r.retune(form[i][j], BW[j], 4.0, SAMPLE_RATE_HZ);
        }
        // ---- asp 区間の頭で局所 2 パス較正する (§14.51・原則は §14.34 と同一) ----
        if asp[i] && (i == 0 || !asp[i - 1]) {
            if asp_match_enabled() {
                let i1 = (i..n).find(|&k| !asp[k]).unwrap_or(n);
                // (基準) 同じ声道状態・同じ軌道を**声帯パルス**で駆動した放射後 RMS
                let run = |use_noise: bool, nz: &mut LfsrNoise, rs0: &Vec<FormantResonator>| -> i64 {
                    let mut rc = rs0.clone();
                    let mut prev = *voiced.last().unwrap_or(&0);
                    let mut sq: i64 = 0;
                    for k in i..i1 {
                        for (j, r) in rc.iter_mut().enumerate() {
                            r.retune(form[k][j], BW[j], 4.0, SAMPLE_RATE_HZ);
                        }
                        let src = if use_noise { nz.next_sample() } else { pulse[k] };
                        let x = ((src as i64 * vg[k] as i64) >> 12) as i32;
                        let mut y = 0i32;
                        for r in rc.iter_mut() { y = y.saturating_add(r.process(x)); }
                        let d = y.saturating_sub(prev) as i64;
                        prev = y;
                        sq += d * d;
                    }
                    sq
                };
                let mut nz = noise.clone();   // 実走行と同じ雑音列で測る
                let ref_sq = run(false, &mut nz.clone(), &rs);
                let probe_sq = run(true, &mut nz, &rs);
                if probe_sq > 0 && ref_sq > 0 {
                    // 倍率 = sqrt(ref/probe)。整数で保持 (分子/分母)。
                    asp_num = ((ref_sq as f64 / probe_sq as f64).sqrt() * 4096.0) as i64;
                    asp_den = 4096;
                    if asp_num == 0 { asp_num = 1; }
                } else {
                    asp_num = 1; asp_den = 1;
                }
            } else {
                asp_num = 1; asp_den = 1;
            }
        }
        let src = if asp[i] {
            ((noise.next_sample() as i64 * asp_num) / asp_den) as i32
        } else { pulse[i] };
        let x = ((src as i64 * vg[i] as i64) >> 12) as i32;
        let mut y = 0i32;
        for r in rs.iter_mut() { y = y.saturating_add(r.process(x)); }
        if az[i] > 0.0 {
            if a_cache.0 != az[i] {
                let r = (-std::f64::consts::PI * ANTIFORMANT_BW_HZ / SAMPLE_RATE_HZ).exp();
                let th = 2.0 * std::f64::consts::PI * az[i] / SAMPLE_RATE_HZ;
                let (b1, b2) = (-2.0 * r * th.cos(), r * r);
                let dc = 1.0 + b1 + b2;
                a_cache = (az[i], (b1 * 32768.0).round() as i64, (b2 * 32768.0).round() as i64,
                           if dc.abs() < 1e-6 { 32768 } else { (32768.0 / dc).round() as i64 });
            }
            let acc = (y as i64) * 32768 + a_cache.1 * ax1 + a_cache.2 * ax2;
            ax2 = ax1;
            ax1 = y as i64;
            y = ((acc.saturating_mul(a_cache.3)) >> 30) as i32;
        } else {
            ax2 = ax1;
            ax1 = y as i64;
        }
        voiced.push(y.saturating_add(fric[i]));
    }

    // ---- 4) 唇からの放射 (一次差分) ----
    let mut raw = Vec::with_capacity(n);
    let mut prev = 0i32;
    for &v in voiced.iter() { raw.push(v.saturating_sub(prev)); prev = v; }

    // ---- 5) **ramp は発話の先頭と末尾だけ** ----
    let ramp = ((5.0 * SAMPLE_RATE_HZ / 1000.0) as usize).min(n / 4).max(1);
    for i in 0..n {
        let env = if i < ramp { (i * 1024 / ramp) as i32 }
                  else if i + ramp >= n { (((n - i) * 1024) / ramp) as i32 }
                  else { 1024 };
        raw[i] = ((raw[i] as i64 * env as i64) >> 10) as i32;
    }

    // ---- 6) **正規化は発話全体で 1 回** ----
    normalize_rms(raw, UTTERANCE_TARGET_RMS)
}
