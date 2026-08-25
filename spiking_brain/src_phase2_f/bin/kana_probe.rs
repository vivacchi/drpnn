//! かなは音として通るか (2026-08-26)
//!
//! 元の目的「コーパスを流してストリーミングで学ぶ」への復帰。
//! かな → モーラ → 波形 → 蝸牛 → M0.5 が実際に通るかを測る。
//!
//! ## ゲート (実測前に固定)
//!
//! 正解の出どころは §2.2 の形 (同じものに同じく、違うものに違って応じるか)。
//!
//!   G65a かなが音になる  : 各かな 1 文字が M0.5 に届く (無音でない)
//!        正解 = 音を出したのは実験者
//!   G65b かなが区別される: 五十音の各行が互いに異なる M0.5 応答を持つ
//!        正解 = 違うかなを与えたのは実験者
//!   G65c 決定論性        : 同じ発話を 2 回流して完全一致
//!
//! さらに参考として、**モーラの識別率** (どのかなだったかを 1-NN で当てる) も出す。
//! チャンスレベルは 1/(かな数)。
//!
//! CLI: kana_probe

use spiking_brain::phase2_f::cochlea::{Cochlea, SAMPLES_PER_STEP};
use spiking_brain::phase2_f::cochlear_nucleus::{CochlearNucleus, N_CN_OUTPUT};
use spiking_brain::phase2_f::kana::{moras_from_kana, synth_utterance};
use spiking_brain::phase2_f::phoneme_synth::LfsrNoise;

const F0: f64 = 150.0;
const SEED: u16 = 0xACE1;

/// 五十音 (清音のみ・44 かな)
const GOJUON: &str = "あいうえおかきくけこさしすせそたちつてとなにぬねのはひふへほまみむめもやゆよらりるれろわを";

fn cn_counts(text: &str) -> Vec<u32> {
    let mut noise = LfsrNoise::new(SEED);
    let (moras, skipped) = moras_from_kana(text);
    debug_assert_eq!(skipped, 0);
    let wave = synth_utterance(&moras, F0, &mut noise);
    let mut co = Cochlea::new();
    let mut cn = CochlearNucleus::new();
    let mut counts = vec![0u32; N_CN_OUTPUT];
    for chunk in wave.chunks(SAMPLES_PER_STEP) {
        if chunk.len() < SAMPLES_PER_STEP {
            break;
        }
        let out = co.process_step(chunk);
        for (i, &v) in cn.process_step(&out).iter().enumerate() {
            if v != 0 {
                counts[i] += 1;
            }
        }
    }
    counts
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

fn main() {
    let kana: Vec<char> = GOJUON.chars().collect();
    println!("=== かなは音として通るか ===");
    println!("五十音 {} かな ・ F0 = {:.0} Hz ・ M0.5 出力 ({} ch) で測る",
             kana.len(), F0, N_CN_OUTPUT);
    println!();

    // --- G65a: 各かなが音になるか ---
    let profiles: Vec<(char, Vec<u32>)> = kana
        .iter()
        .map(|&c| (c, cn_counts(&c.to_string())))
        .collect();
    let silent: Vec<char> = profiles
        .iter()
        .filter(|(_, p)| p.iter().all(|&x| x == 0))
        .map(|(c, _)| *c)
        .collect();
    println!("--- G65a かなが音になるか ---");
    println!("無音のかな: {} 個 {}",
             silent.len(),
             if silent.is_empty() { "→ PASS".to_string() } else { format!("{:?} → **FAIL**", silent) });

    // 総スパイク数の分布
    let totals: Vec<u32> = profiles.iter().map(|(_, p)| p.iter().sum()).collect();
    let mn = totals.iter().min().unwrap();
    let mx = totals.iter().max().unwrap();
    println!("総スパイク数: 最小 {} / 最大 {} ", mn, mx);

    // --- G65b: かなが区別されるか ---
    println!();
    println!("--- G65b かなが区別されるか ---");
    let mut identical = 0usize;
    let mut worst = (f64::NEG_INFINITY, ' ', ' ');
    for i in 0..profiles.len() {
        for j in (i + 1)..profiles.len() {
            if profiles[i].1 == profiles[j].1 {
                identical += 1;
            }
            let c = cosine(&profiles[i].1, &profiles[j].1);
            if c > worst.0 {
                worst = (c, profiles[i].0, profiles[j].0);
            }
        }
    }
    println!("完全に同一の応答を持つかなの対: {} 組 {}",
             identical,
             if identical == 0 { "→ PASS" } else { "→ **FAIL**" });
    println!("最も似ている対: {} と {} (コサイン {:.4})", worst.1, worst.2, worst.0);

    // --- 参考: モーラの識別率 ---
    let mut hit = 0usize;
    for i in 0..profiles.len() {
        let mut best = (-2.0f64, ' ');
        for j in 0..profiles.len() {
            if i == j {
                continue;
            }
            let c = cosine(&profiles[i].1, &profiles[j].1);
            if c > best.0 {
                best = (c, profiles[j].0);
            }
        }
        // 同じ母音を持つかなが最近傍なら「母音は当たっている」
        if best.1 == profiles[i].0 {
            hit += 1;
        }
    }
    println!();
    println!("(参考) 自分自身が最近傍になった数: {} / {}", hit, profiles.len());

    // 母音別のまとまり: 各かなの最近傍が同じ母音行かどうか
    let vowel_of = |c: char| -> usize {
        let idx = GOJUON.chars().position(|k| k == c).unwrap_or(0);
        // 「や ゆ よ」(3つ) と「わ を」(2つ) があるので単純な %5 にはならない
        match c {
            'あ' | 'か' | 'さ' | 'た' | 'な' | 'は' | 'ま' | 'や' | 'ら' | 'わ' => 0,
            'い' | 'き' | 'し' | 'ち' | 'に' | 'ひ' | 'み' | 'り' => 1,
            'う' | 'く' | 'す' | 'つ' | 'ぬ' | 'ふ' | 'む' | 'ゆ' | 'る' => 2,
            'え' | 'け' | 'せ' | 'て' | 'ね' | 'へ' | 'め' | 'れ' => 3,
            _ => {
                let _ = idx;
                4
            }
        }
    };
    let mut vowel_hit = 0usize;
    for i in 0..profiles.len() {
        let mut best = (-2.0f64, ' ');
        for j in 0..profiles.len() {
            if i == j {
                continue;
            }
            let c = cosine(&profiles[i].1, &profiles[j].1);
            if c > best.0 {
                best = (c, profiles[j].0);
            }
        }
        if vowel_of(best.1) == vowel_of(profiles[i].0) {
            vowel_hit += 1;
        }
    }
    println!("最近傍が**同じ母音**だった数: {} / {} ({:.1}%・チャンス約 20%)",
             vowel_hit, profiles.len(),
             vowel_hit as f64 / profiles.len() as f64 * 100.0);

    // --- G65c: 決定論性 ---
    println!();
    println!("--- G65c 決定論性 ---");
    let a = cn_counts("こんにちは");
    let b = cn_counts("こんにちは");
    println!("同じ発話を 2 回: {}", if a == b { "完全一致 → PASS" } else { "**不一致 → FAIL**" });

    // --- 実際の文を通してみる ---
    println!();
    println!("--- 実際の文 ---");
    for text in ["こんにちは", "がっこう", "きょうはいいてんきですね", "とうきょう"].iter() {
        let (moras, skipped) = moras_from_kana(text);
        let c = cn_counts(text);
        let total: u32 = c.iter().sum();
        let active = c.iter().filter(|&&x| x > 0).count();
        println!("  {:<16} {} モーラ (未対応 {})  総スパイク {:>6}  発火ch {:>3}/{}",
                 text, moras.len(), skipped, total, active, N_CN_OUTPUT);
    }
}
