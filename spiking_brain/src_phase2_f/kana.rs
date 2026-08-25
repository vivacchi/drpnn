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
    synth_consonant_banded, synth_vowel_f0, vowels, Consonant, LfsrNoise, Vowel, SAMPLE_RATE_HZ,
};

/// 日本語のモーラ。
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Mora {
    /// 子音 + 母音 (子音なしの「あいうえお」は `Consonant::None`)
    Cv { consonant: Consonant, vowel: Vowel },
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
fn consonant_of(row: char) -> Consonant {
    match row {
        'k' | 'g' => Consonant::Plosive { burst_freq_low: 2000.0, burst_freq_high: 4000.0 },
        't' | 'd' => Consonant::Plosive { burst_freq_low: 1500.0, burst_freq_high: 3500.0 },
        'p' | 'b' => Consonant::Plosive { burst_freq_low: 500.0, burst_freq_high: 2000.0 },
        's' | 'z' => Consonant::Fricative { freq_low: 3000.0, freq_high: 8000.0 },
        'S' => Consonant::Fricative { freq_low: 2000.0, freq_high: 6000.0 }, // し・しゃ行
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
        'h' => Consonant::Fricative { freq_low: 500.0, freq_high: 4000.0 },
        'm' => Consonant::Nasal { f1: 250.0, f2: 1500.0 },
        'n' => Consonant::Nasal { f1: 250.0, f2: 1700.0 },
        // ラ行 (弾き音) は破裂音で近似する。**正確でない**ことを明記 (§14)。
        'r' => Consonant::Plosive { burst_freq_low: 1200.0, burst_freq_high: 2800.0 },
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
        ('ざ', 'z', 0), ('じ', 'S', 1), ('ず', 'z', 2), ('ぜ', 'z', 3), ('ぞ', 'z', 4),
        ('た', 't', 0), ('ち', 'C', 1), ('つ', 'c', 2), ('て', 't', 3), ('と', 't', 4),
        ('だ', 'd', 0), ('ぢ', 'S', 1), ('づ', 'z', 2), ('で', 'd', 3), ('ど', 'd', 4),
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
                if let Some(Mora::Cv { consonant, .. }) = out.last().copied() {
                    let n = out.len();
                    out[n - 1] = Mora::Cv { consonant, vowel: vowel_of(v) };
                } else {
                    skipped += 1;
                }
            }
            _ => match kana_to_cv(c) {
                Some((row, vi)) => out.push(Mora::Cv {
                    consonant: consonant_of(row),
                    vowel: vowel_of(vi),
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
    let mut out: Vec<i32> = Vec::new();
    let mut last_vowel: Option<Vowel> = None;
    for m in moras {
        match *m {
            Mora::Cv { consonant, vowel } => {
                if consonant != Consonant::None {
                    out.extend(synth_consonant_banded(consonant, CONSONANT_MS, noise));
                    out.extend(synth_vowel_f0(&vowel, f0_hz, MORA_MS - CONSONANT_MS));
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
                    Consonant::Nasal { f1: 250.0, f2: 1700.0 },
                    MORA_MS,
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
