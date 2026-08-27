//! 入り口監査の主張を、出荷コードで検算する (2026-08-27)
//!
//! ## なぜ
//!
//! 独立監査 (6 観点 × 反証) が合成音声の入り口に 16 件の所見を返した。
//! **だが反証段が機能していなかった** (照合をタイトルの完全一致で行う実装にしたため、
//! 反証エージェントの返した題名と噛み合わず全件が「触れられなかった」扱いになった)。
//! **「反証を生き延びた」は「反証されなかった」であって「確認された」ではない。**
//!
//! さらに監査は **Python で整数意味論を再移植して**測っており、
//! **Rust の出荷コードを実行していない。**
//!
//! **したがって主張はすべて仮説である。ここで出荷コードによって検算する。**
//! 監査の数値を**予測として置き**、再現するかを見る。
//!
//! ## 検算する主張 (監査の予測値つき)
//!
//! - **V1 声帯源の櫛形零点がフォルマントを消す** [fatal]
//!   予測: /a/ の M0 argmax が F0=100 で 820Hz (F1) → F0=150 で 293Hz → F0=160 で 1257Hz と飛ぶ。
//!   *当たれば「音程を変えても包絡は動かない」という docstring が偽になり、
//!    F0 を変種軸に使った全測定が「母音の恒常性でなく零点の移動」を測っていたことになる。*
//! - **V2 「全子音を同一 RMS」が蝸牛では成立しない** [fatal]
//!   予測: 総 RMS は 11314〜11317 に揃うのに、M0 総スパイクは /h/ 258 vs /s/ 101 = **2.55 倍**。
//! - **V3 VOT が「音量」として母音に転写されている** [fatal]
//!   予測: モーラ末 30ms の RMS が VOT ON で **7.7 dB** ばらつき、OFF で 0.4 dB。
//!   *当たれば §14.30/§14.31 の VOT の結論が「周期性でなく音量を読んだ」ものになる。**本日の主張が崩れる。***
//! - **V4 阻害音がモーラ単位で cos ≥ 0.99 に縮退** [fatal]
//!   予測: た/か 0.9989・だ/ら 0.9995・し/ち 0.9994・す/つ 0.9996・ぱ/ば 0.9982。
//! - **V5 鼻音・接近音が声帯源を通らない (純音 2 本)** [fatal]
//!   予測: `f0_hz` がこの経路で使われないので、**な の子音区間が F0 を変えてもバイト同一**。
//!
//! ## この検算のゲート
//!
//! **判定は「監査の主張が再現するか」であり、良し悪しではない。**
//! 再現したら主張は本物。再現しなければ主張は棄却。**どちらでも記録する。**
//!
//! CLI: audit_verify  (DRPNN_M0_SPONTANEOUS=0 で自発発火 OFF アーム)

use spiking_brain::phase2_f::cochlea::{spontaneous_default_amplitude, Cochlea, N_BANDS, SAMPLES_PER_STEP};
use spiking_brain::phase2_f::kana::{moras_from_kana, set_vot, synth_utterance};
use spiking_brain::phase2_f::phoneme_synth::{
    synth_consonant_banded, synth_vowel_f0, vowels, Consonant, LfsrNoise,
};

const SEED: u16 = 0xACE1;

fn rms(w: &[i32]) -> f64 {
    (w.iter().map(|&s| (s as f64) * (s as f64)).sum::<f64>() / w.len().max(1) as f64).sqrt()
}

fn m0_counts(w: &[i32]) -> Vec<f64> {
    let mut co = Cochlea::new();
    let mut out = vec![0f64; N_BANDS];
    for chunk in w.chunks(SAMPLES_PER_STEP) {
        if chunk.len() < SAMPLES_PER_STEP { break; }
        for (i, &v) in co.process_step(chunk).iter().enumerate() { if v != 0 { out[i] += 1.0; } }
    }
    out
}

fn cosine(a: &[f64], b: &[f64]) -> f64 {
    let d: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let na: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
    let nb: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();
    if na == 0.0 || nb == 0.0 { 0.0 } else { d / (na * nb) }
}

fn wave(k: &str) -> Vec<i32> {
    let mut n = LfsrNoise::new(SEED);
    let (m, sk) = moras_from_kana(k);
    assert_eq!(sk, 0, "未対応: {}", k);
    synth_utterance(&m, 150.0, &mut n)
}

fn main() {
    println!("=== 入り口監査の主張を、出荷コードで検算する ===");
    println!();
    println!("【なぜ】独立監査(6観点×反証)が16件を返したが **反証段が機能していなかった**");
    println!("(照合をタイトル完全一致にした私の実装の欠陥で全件が「触れられなかった」扱い)。");
    println!("**「反証を生き延びた」は「反証されなかった」であって「確認された」ではない。**");
    println!("さらに監査は **Python で整数意味論を再移植**して測っており **Rust の出荷コードを実行していない。**");
    println!("**したがって主張はすべて仮説。ここで出荷コードで検算する。**");
    println!();
    println!("自発発火の振幅 = {} (DRPNN_M0_SPONTANEOUS で切替)", spontaneous_default_amplitude());

    let freqs = Cochlea::new().center_freqs.clone();
    let vs = vowels();

    // ---------- V1 櫛形零点 ----------
    println!();
    println!("--- V1 声帯源の櫛形零点がフォルマントを消すか [fatal] ---");
    println!("  監査の予測: /a/ の argmax が F0=100 で 820Hz(F1) -> 150 で 293Hz -> 160 で 1257Hz と飛ぶ");
    println!();
    println!("  {:<6} {:>10} | {}", "母音", "F1(設計)", "F0ごとの argmax 帯域中心 [Hz]");
    let f0s = [100.0, 130.0, 150.0, 160.0, 200.0];
    print!("  {:<6} {:>10} | ", "", "");
    for f in f0s.iter() { print!("{:>9.0}", f); }
    println!();
    let mut jump = 0usize;
    for v in vs.iter() {
        print!("  {:<6} {:>10.0} | ", v.label, v.formants_hz[0]);
        let mut peaks = Vec::new();
        for &f0 in f0s.iter() {
            let c = m0_counts(&synth_vowel_f0(v, f0, 90.0));
            let am = (0..N_BANDS).max_by(|&a, &b| c[a].partial_cmp(&c[b]).unwrap()).unwrap();
            peaks.push(freqs[am]);
            print!("{:>9.0}", freqs[am]);
        }
        let (lo, hi) = (peaks.iter().cloned().fold(f64::INFINITY, f64::min),
                        peaks.iter().cloned().fold(f64::NEG_INFINITY, f64::max));
        let moved = hi / lo.max(1.0);
        if moved > 1.5 { jump += 1; }
        println!("   最大/最小 = {:.2}{}", moved, if moved > 1.5 { " **飛んだ**" } else { "" });
    }
    println!();
    println!("  **V1 判定**: F0 を変えて argmax が 1.5 倍以上動いた母音 = {}/5 -> {}",
             jump, if jump > 0 { "**主張は再現した**" } else { "**主張は再現しない — 棄却**" });
    println!("  (docstring「音程を変えても包絡は動かない」が偽かどうかを見ている)");

    // ---------- V2 等 RMS ----------
    println!();
    println!("--- V2 「全子音を同一 RMS」は蝸牛でも成立するか [fatal] ---");
    println!("  監査の予測: 総RMS は 11314-11317 に揃うのに M0 総スパイクは /h/ 258 vs /s/ 101 = 2.55倍");
    println!();
    let cons: &[(&str, Consonant)] = &[
        ("h 500-4000",  Consonant::Fricative { freq_low: 500.0, freq_high: 4000.0, voiced: false }),
        ("S 2000-6000", Consonant::Fricative { freq_low: 2000.0, freq_high: 6000.0, voiced: false }),
        ("p 500-2000",  Consonant::Plosive { burst_freq_low: 500.0, burst_freq_high: 2000.0, voiced: false }),
        ("r 1200-2800", Consonant::Plosive { burst_freq_low: 1200.0, burst_freq_high: 2800.0, voiced: true }),
        ("t 1500-3500", Consonant::Plosive { burst_freq_low: 1500.0, burst_freq_high: 3500.0, voiced: false }),
        ("k 2000-4000", Consonant::Plosive { burst_freq_low: 2000.0, burst_freq_high: 4000.0, voiced: false }),
        ("s 3000-8000", Consonant::Fricative { freq_low: 3000.0, freq_high: 8000.0, voiced: false }),
    ];
    println!("  {:<14} {:>10} {:>14}", "子音(宣言帯域)", "総RMS", "**M0総スパイク**");
    let (mut smin, mut smax) = (f64::INFINITY, f64::NEG_INFINITY);
    let (mut rmin, mut rmax) = (f64::INFINITY, f64::NEG_INFINITY);
    for (nm, c) in cons.iter() {
        let mut n = LfsrNoise::new(SEED);
        let w = synth_consonant_banded(*c, 30.0, 150.0, &mut n);
        let (r, sp) = (rms(&w), m0_counts(&w).iter().sum::<f64>());
        smin = smin.min(sp); smax = smax.max(sp);
        rmin = rmin.min(r);  rmax = rmax.max(r);
        println!("  {:<14} {:>10.0} {:>14.0}", nm, r, sp);
    }
    println!();
    println!("  総RMS の比 {:.4} / **M0総スパイクの比 {:.2} 倍**", rmax / rmin, smax / smin);
    println!("  **V2 判定**: {}",
             if smax / smin > 1.5 { "**主張は再現した — 等RMS は蝸牛では成立していない**" }
             else { "**主張は再現しない — 棄却**" });

    // ---------- V3 VOT が音量として転写されているか ----------
    println!();
    println!("--- V3 VOT は「音量」として母音に転写されているか [fatal] ---");
    println!("  監査の予測: モーラ末30msのRMSが VOT ON で **7.7dB** ばらつき OFF で 0.4dB");
    println!("  *当たれば §14.30/§14.31 の VOT の結論が『周期性でなく音量を読んだ』ものになる。*");
    println!();
    let test = ["か", "た", "ぱ", "つ", "が", "だ", "ば", "さ", "ま", "ら"];
    println!("  {:<8} {:>12} {:>12}", "かな", "OFF 末30ms", "ON 末30ms");
    let mut off_v: Vec<f64> = Vec::new();
    let mut on_v: Vec<f64> = Vec::new();
    for k in test.iter() {
        set_vot(false);
        let wo = wave(k);
        set_vot(true);
        let wn = wave(k);
        let tail = 30 * 16;
        let (a, b) = (rms(&wo[wo.len() - tail..]), rms(&wn[wn.len() - tail..]));
        off_v.push(a); on_v.push(b);
        println!("  {:<8} {:>12.0} {:>12.0}", k, a, b);
    }
    let db = |v: &Vec<f64>| {
        let lo = v.iter().cloned().fold(f64::INFINITY, f64::min);
        let hi = v.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        20.0 * (hi / lo.max(1e-9)).log10()
    };
    let (doff, don) = (db(&off_v), db(&on_v));
    println!();
    println!("  ばらつき: OFF **{:.1} dB** / ON **{:.1} dB**", doff, don);
    println!("  **V3 判定**: {}",
             if don > doff + 3.0 { "**主張は再現した — VOT が音量差を作っている**" }
             else { "**主張は再現しない — 棄却**" });

    // ---------- V4 縮退 ----------
    println!();
    println!("--- V4 阻害音はモーラ単位で cos >= 0.99 に縮退しているか [fatal] ---");
    println!("  監査の予測: た/か 0.9989 · だ/ら 0.9995 · し/ち 0.9994 · す/つ 0.9996 · ぱ/ば 0.9982");
    println!();
    let pairs = [("た","か"),("だ","ら"),("し","ち"),("す","つ"),("ぱ","ば"),("た","だ"),("か","が")];
    let mut over = 0usize;
    for (a, b) in pairs.iter() {
        let c = cosine(&m0_counts(&wave(a)), &m0_counts(&wave(b)));
        if c >= 0.99 { over += 1; }
        println!("  {}/{}  cos = {:.4}{}", a, b, c, if c >= 0.99 { "  **0.99 超**" } else { "" });
    }
    println!();
    println!("  **V4 判定**: 0.99 を超えた対 {}/{} -> {}", over, pairs.len(),
             if over >= pairs.len() / 2 { "**主張は再現した**" } else { "**主張は再現しない — 棄却**" });

    // ---------- V5 鼻音は声帯源を通らないか ----------
    println!();
    println!("--- V5 鼻音・接近音は声帯源を通らないか (純音2本) [fatal] ---");
    println!("  監査の予測: f0_hz がこの経路で使われないので **な の子音区間が F0 を変えてもバイト同一**");
    println!();
    for (nm, c) in [("鼻音 n", Consonant::Nasal { f1: 250.0, f2: 1700.0, zero_hz: 1800.0 }),
                    ("接近 y", Consonant::Approximant { f1: 300.0, f2: 2200.0 }),
                    ("破裂 k", Consonant::Plosive { burst_freq_low: 2000.0, burst_freq_high: 4000.0, voiced: false })].iter() {
        let mut n1 = LfsrNoise::new(SEED);
        let mut n2 = LfsrNoise::new(SEED);
        let w1 = synth_consonant_banded(*c, 30.0, 100.0, &mut n1);
        let w2 = synth_consonant_banded(*c, 30.0, 250.0, &mut n2);
        println!("  {:<8} F0=100 と F0=250 で -> {}", nm,
                 if w1 == w2 { "**バイト同一 = F0 を使っていない**" } else { "違う (F0 を使っている)" });
    }
    println!();
    println!("  【この検算が答えないこと】**直し方は決めていない。**");
    println!("  再現したかどうかだけを記録する。**どう直すかは人間の判断。**");
}
