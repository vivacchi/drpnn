//! M0 のゲートにレベル軸を入れる (2026-08-25・独立監査の指摘を自分で確かめる)
//!
//! ## 指摘
//!
//! 「現行の M0 ゲート (m0_design / cochlea_gates_v2 / consonant_gate) は
//!  固定ゲインを 1 つだけ渡して評価している。**ゲート群にレベル軸が存在しない。**
//!  被覆 15/15・場所符号 10/10・母音精度 92.5% はすべて**単一レベルでの数字**。
//!  場所符号だけで母音同定が成立するレベル範囲は 14 dB で、
//!  それより下では /e/ と /o/ が cos=1.000 で完全一致する」
//!
//! これは 2026-08-25 に採用した N_BANDS=80 / Q×6 / 閾120 の結論
//! 「M0 はフォルマントで母音を区別できるようになった」を直撃するので、自分で測る。
//!
//! ## ゲート (実測前に固定)
//!
//!   G40 レベル横断の場所符号: 提示レベルを設計側で振ったとき、
//!        **どのレベルでも**5 母音の発火帯域集合が全 10 対で相異なるか。
//!        正解の出どころ = 5 組の別々のフォルマントを与えたのは実験者。
//!        レベルを変えたのも実験者。母音の同一性はレベルに依存しないはず。
//!   G41 レベル横断の被覆: どのレベルでも指定 3 フォルマントが応答するか (15/15)。
//!   G42 順序の逆転: 「同一母音・別レベル」の類似度が
//!        「別母音・同レベル」の類似度を**必ず上回る**か。
//!        正解の出どころ = どれが同じ母音でどれが違う母音かは実験者が決めた。
//!        逆転していれば、その表現では母音とレベルを分離できない。
//!
//! **G42 が要**: 母音同定が成立するかどうかは、絶対的な類似度でなく
//! 「同じもの同士 > 違うもの同士」という**順序**で決まる。
//!
//! CLI: level_axis

use spiking_brain::phase2_f::cochlea::{Cochlea, N_BANDS, SAMPLES_PER_STEP};
use spiking_brain::phase2_f::phoneme_synth::{synth_vowel, vowels};

const VOWEL_MS: f64 = 170.0;
/// 提示レベル [dB] (0 = 母音テーブルのまま)
const LEVELS_DB: [f64; 9] = [0.0, -3.0, -6.0, -9.0, -12.0, -15.0, -18.0, -21.0, -24.0];

fn spike_cost_arg() -> i32 {
    std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(0)
}

fn band_spikes(wave: &[i32], gain_num: i32, gain_den: i32) -> [u32; N_BANDS] {
    let mut c = Cochlea::new();
    let sc = spike_cost_arg();
    if sc > 0 {
        for f in c.fire_gens.iter_mut() { f.spike_cost = sc; }
    }
    let mut counts = [0u32; N_BANDS];
    for chunk in wave.chunks(SAMPLES_PER_STEP) {
        if chunk.len() < SAMPLES_PER_STEP {
            break;
        }
        let amp: Vec<i32> = chunk
            .iter()
            .map(|&x| ((x as i64 * gain_num as i64) / gain_den as i64) as i32)
            .collect();
        let out = c.process_step(&amp);
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

fn nearest_band(freqs: &[f64], f_hz: f64) -> usize {
    freqs
        .iter()
        .enumerate()
        .min_by(|a, b| (a.1 - f_hz).abs().partial_cmp(&(b.1 - f_hz).abs()).unwrap())
        .unwrap()
        .0
}

/// G52: フォルマント**強度の順位**が蝸牛出力に残っているか。
///
/// 監査の指摘:「フォルマント振幅比 (1.0 : 0.7 : 0.3) が完全に消える。
/// 母音表の ×4 はこの副作用の対症療法だった」。
/// これは 2026-08-25 に入れた ×4 への直接の批判なので確かめる。
///
/// 正解の出どころ = **振幅比を決めたのは実験者**。
/// F1 > F2 > F3 と置いたのだから、出力の順位もそうなるべき。
fn formant_rank(spike_cost: i32) {
    let vs = vowels();
    let names = ["a", "i", "u", "e", "o"];
    let freqs = Cochlea::new().center_freqs.clone();
    println!();
    println!("--- G52 フォルマント強度の順位は残るか ---");
    println!("音素  指定振幅 (F1,F2,F3)      発火数 (F1,F2,F3)   順位一致");
    let mut ok = 0usize;
    for (k, v) in vs.iter().enumerate() {
        let counts = band_spikes(&synth_vowel(v, VOWEL_MS), 4096, 4096);
        let obs: Vec<u32> = (0..3)
            .map(|f| counts[nearest_band(&freqs, v.formants_hz[f])])
            .collect();
        // 指定振幅の順位と観測発火数の順位が一致するか (同順位は不一致扱い)
        let spec = v.amplitudes;
        let rank_ok = spec[0] > spec[1] && spec[1] > spec[2]
            && obs[0] > obs[1] && obs[1] > obs[2];
        if rank_ok {
            ok += 1;
        }
        println!(
            "{:>4}  ({:>5},{:>5},{:>5})   ({:>5},{:>5},{:>5})   {}",
            names[k], spec[0], spec[1], spec[2], obs[0], obs[1], obs[2],
            if rank_ok { "○" } else { "×" }
        );
        let _ = spike_cost;
    }
    println!("G52: {}/5 の母音で F1>F2>F3 の順位が保たれた {}",
             ok, if ok == 5 { "PASS" } else { "**FAIL**" });
    println!("  (指定振幅は F1>F2>F3 と置いてある。出力もそうなるべき)");
}

fn main() {
    formant_rank(spike_cost_arg());
    let vs = vowels();
    let names = ["a", "i", "u", "e", "o"];
    let freqs = Cochlea::new().center_freqs.clone();

    println!("=== M0 のゲートにレベル軸を入れる ===");
    println!("N_BANDS={} ・ 提示レベル {:?} dB", N_BANDS, LEVELS_DB);
    println!("(0 dB = 母音テーブルのまま) ・ spike_cost = {} (0=旧モード)", spike_cost_arg());
    println!();

    // レベルごとのプロファイル
    let mut profiles: Vec<Vec<[u32; N_BANDS]>> = Vec::new();
    for &db in LEVELS_DB.iter() {
        // 整数比でゲインを作る (決定論的)
        let g = 10f64.powf(db / 20.0);
        let den = 4096i32;
        let num = (g * den as f64).round() as i32;
        profiles.push(
            vs.iter()
                .map(|v| band_spikes(&synth_vowel(v, VOWEL_MS), num, den))
                .collect(),
        );
    }

    println!("レベル  母音ごとの発火帯域数   被覆/15  場所符号の相異/10  無音母音");
    let mut g40_ok = true;
    let mut g41_ok = true;
    for (li, &db) in LEVELS_DB.iter().enumerate() {
        let p = &profiles[li];
        let bands: Vec<usize> = p.iter().map(|x| x.iter().filter(|&&c| c > 0).count()).collect();
        let mut recall = 0usize;
        for (k, v) in vs.iter().enumerate() {
            for f in 0..3 {
                if p[k][nearest_band(&freqs, v.formants_hz[f])] > 0 {
                    recall += 1;
                }
            }
        }
        let sets: Vec<Vec<usize>> = p
            .iter()
            .map(|x| (0..N_BANDS).filter(|&i| x[i] > 0).collect())
            .collect();
        let mut distinct = 0usize;
        for i in 0..sets.len() {
            for j in (i + 1)..sets.len() {
                if sets[i] != sets[j] {
                    distinct += 1;
                }
            }
        }
        let silent = sets.iter().filter(|s| s.is_empty()).count();
        if distinct != 10 || silent != 0 {
            g40_ok = false;
        }
        if recall != 15 {
            g41_ok = false;
        }
        println!(
            "{:>5.0}  {:<20}  {:>7}  {:>17}  {:>8}",
            db,
            format!("{:?}", bands),
            format!("{}/15", recall),
            format!("{}/10", distinct),
            silent
        );
    }

    // G42: 同一母音・別レベル vs 別母音・同レベル
    println!();
    println!("--- G42 順序の逆転 ---");
    let mut min_same = f64::INFINITY;
    let mut min_same_desc = String::new();
    let mut max_diff = f64::NEG_INFINITY;
    let mut max_diff_desc = String::new();
    for k in 0..vs.len() {
        for li in 0..LEVELS_DB.len() {
            for lj in (li + 1)..LEVELS_DB.len() {
                let c = cosine(&profiles[li][k], &profiles[lj][k]);
                if c < min_same {
                    min_same = c;
                    min_same_desc =
                        format!("/{}/ {:.0}dB vs {:.0}dB", names[k], LEVELS_DB[li], LEVELS_DB[lj]);
                }
            }
        }
    }
    for li in 0..LEVELS_DB.len() {
        for i in 0..vs.len() {
            for j in (i + 1)..vs.len() {
                let c = cosine(&profiles[li][i], &profiles[li][j]);
                if c > max_diff {
                    max_diff = c;
                    max_diff_desc =
                        format!("/{}/ vs /{}/ @{:.0}dB", names[i], names[j], LEVELS_DB[li]);
                }
            }
        }
    }
    println!("同一母音・別レベルの**最小**類似度: {:.4}  ({})", min_same, min_same_desc);
    println!("別母音・同レベルの**最大**類似度: {:.4}  ({})", max_diff, max_diff_desc);
    let g42_ok = min_same > max_diff;
    println!(
        "G42 (同じもの同士 > 違うもの同士): {}",
        if g42_ok { "PASS" } else { "**FAIL — 逆転している**" }
    );

    println!();
    println!("--- 判定 ---");
    println!("G40 レベル横断の場所符号 (全レベルで 10/10・無音0): {}",
             if g40_ok { "PASS" } else { "**FAIL**" });
    println!("G41 レベル横断の被覆 (全レベルで 15/15): {}",
             if g41_ok { "PASS" } else { "**FAIL**" });
    println!("G42 順序: {}", if g42_ok { "PASS" } else { "**FAIL**" });
    println!();
    if !(g40_ok && g41_ok && g42_ok) {
        println!("**既存ゲートの「被覆15/15・場所符号10/10・精度92.5%」は単一レベル(0dB)の数字であり、");
        println!("  レベルを振ると成立しない。M0 はまだフォルマントを「レベルに依らず」区別できていない。**");
    }
}
