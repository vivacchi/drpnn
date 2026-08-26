//! M0.5 を動的平衡で測る — これまでの測定は全部「冷開始」だった (2026-08-26)
//!
//! ## 発端 — ユーザーの指摘
//!
//! > **この M0 と M0.5 は動的平衡に落ち着いた後に測定しているよね?**
//!
//! **いいえ。全部の測定が冷開始である。**
//!
//! | | 適応 | これまでの測定 |
//! |---|---|---|
//! | **M0** | **無し** (G47a 保存率 100%) | 問題なし。数 ms で落ち着く |
//! | **M0.5** | **有り** (G47b 保存率 **83.1%**) | **毎回 `CochlearNucleus::new()` = 冷開始** |
//!
//! `kana_identify` / `baseline_subtraction` / `context_enhancement` /
//! `m0_design_v3` / `where_is_the_information` — **全部、条件ごとに作り直している。**
//!
//! §12.14 の記録では 提示1 で 2393 → 提示2 で 2036 → 以降平坦。
//! **平衡は冷開始より約 17% 低く、提示 2 回で達する。**
//!
//! **私が持っている M0.5 の数字は全部、過渡状態のものである。**
//!
//! ## 予測 (実測前に固定・機構つき)
//!
//! **平衡では子音の重心が上がる (冷開始より良くなる)。**
//!
//! 機構: **適応はよく駆動されるチャネルほど強く効く** (`local_entropy` は発火で溜まる)。
//! 中域は常に駆動されるので適応が進み、/s/ が使う高域は駆動が稀なので適応が浅い。
//! **§14.17 で見つけた文脈強調と同じ機構が、試行をまたいで効くはず。**
//!
//! **数値は置かない。** 方向だけ。これは**事前の予測**である。
//!
//! ## ゲート (実測前に固定・以後動かさない)
//!
//! **正解の出どころ**: 何回提示したかも、無音を挟んだのも実験者が決めた。
//!
//! - **G79a 平衡到達**: 総スパイク数が提示を重ねて平坦になる。
//!   *平坦の判定は「連続する 2 チェックポイントの差が、最初の 2 点の差の 1/10 未満」*
//!   (§14.10 の G73b と同じ相対の形)。
//! - **G79b 子音の重心**: 平衡での重心が冷開始より**上がる**。
//!   *順序 pa < tu < ki < se が保たれることも要求する。*
//! - **G79c 床**: 平衡での自発発火の床が冷開始より**下がる**。
//! - **G79d 信号対床の比**: 平衡での (子音の総スパイク / 床の総スパイク) が
//!   冷開始より**上がる**。*これが本当に見たい量。* 重心も床も、これの現れ方の一つ。
//! - **G79e 決定論性**: 2 回実行して完全一致。
//!
//! ## 測り方の限界 (先に書く)
//!
//! - 測定そのものが状態を動かす。無音を挟んで床を測り、続けて子音を出すので、
//!   **子音の測定は直前の無音の影響を受ける**。これは避けられず、実運用も同じ。
//! - 順序依存がある。決定論的な順序を使い、**順序を固定して比較する**。
//!
//! ## 第 1 回で見つかった自分のプローブの欠陥 (2026-08-27)
//!
//! 1. **「床」が床になっていなかった。** 無音を刺激の**直後**に 1 回置いて測っていたので、
//!    測っていたのは**前の刺激からの回復の途中**。冷開始では直前のかなが強く (2252発)
//!    抑制が深く、平衡では弱く (1686発) 抑制が浅い。
//!    **「平衡で床が上がった (+29.9%)」は、適応が浅いぶん回復が早いだけかもしれない。**
//!    → **無音を連続で出して回復曲線として測り、落ち着いた側を床とする。**
//! 2. **チェックポイントごとに違うかなを測っていた。** LCG で選ぶので提示1 と提示100 で
//!    別のかな。**「かな総」の変動が適応でなく「どのかなだったか」で決まっていた。**
//!    G79a の「未到達」判定はこれに汚染されている。
//!    → **固定の参照かな (あ) を測る。**
//!
//! 子音は全チェックポイントで同一なので、そちら側の結果 (重心が上がらなかった) は
//! 欠陥の影響を受けない。**予測が外れたことは変わらない。**
//!
//! CLI: equilibrium_measurement

use spiking_brain::phase2_f::cochlea::{Cochlea, N_BANDS, SAMPLES_PER_STEP};
use spiking_brain::phase2_f::cochlear_nucleus::{CochlearNucleus, N_CN_OUTPUT};
use spiking_brain::phase2_f::kana::{moras_from_kana, synth_utterance};
use spiking_brain::phase2_f::phoneme_synth::{synth_consonant_banded, Consonant, LfsrNoise};

const KANA: &[&str] = &[
    "あ","い","う","え","お","か","き","く","け","こ","さ","し","す","せ","そ",
    "た","ち","つ","て","と","な","に","ぬ","ね","の","は","ひ","ふ","へ","ほ",
    "ま","み","む","め","も","や","ゆ","よ","ら","り","る","れ","ろ","わ","を","ん",
];
const F0: f64 = 150.0;
const SEED: u16 = 0xACE1;
const CONSONANT_MS: f64 = 30.0;
const FLOOR_MS: f64 = 170.0;
const CHECKPOINTS: [usize; 7] = [1, 2, 5, 10, 20, 50, 100];

fn consonants() -> Vec<(&'static str, Consonant)> {
    vec![
        ("pa", Consonant::Plosive { burst_freq_low: 500.0, burst_freq_high: 2000.0 }),
        ("tu", Consonant::Plosive { burst_freq_low: 1500.0, burst_freq_high: 3500.0 }),
        ("ki", Consonant::Plosive { burst_freq_low: 2000.0, burst_freq_high: 4000.0 }),
        ("se", Consonant::Fricative { freq_low: 3000.0, freq_high: 8000.0 }),
    ]
}

fn kana_waves() -> Vec<Vec<i32>> {
    KANA.iter().map(|k| {
        let mut n = LfsrNoise::new(SEED);
        let (m, s) = moras_from_kana(k);
        assert_eq!(s, 0);
        synth_utterance(&m, F0, &mut n)
    }).collect()
}

/// 状態を持ち越したまま 1 刺激を流し、M0.5 の発火数を返す。
fn present(co: &mut Cochlea, cn: &mut CochlearNucleus, wave: &[i32]) -> Vec<f64> {
    let mut counts = vec![0f64; N_CN_OUTPUT];
    for chunk in wave.chunks(SAMPLES_PER_STEP) {
        if chunk.len() < SAMPLES_PER_STEP { break; }
        let m0 = co.process_step(chunk);
        for (i, &v) in cn.process_step(&m0).iter().enumerate() {
            if v != 0 { counts[i] += 1.0; }
        }
    }
    counts
}

fn centroid_bushy(c: &[f64], freqs: &[f64]) -> f64 {
    // M0.5 の 84ch のうち Bushy 部分 (4..4+N_BANDS) を帯域として扱う
    let b: Vec<f64> = (0..N_BANDS).map(|i| c[4 + i]).collect();
    let tot: f64 = b.iter().sum();
    if tot == 0.0 { return 0.0; }
    b.iter().zip(freqs.iter()).map(|(&x, &f)| x * f).sum::<f64>() / tot
}

const FLOOR_SEGMENTS: usize = 4;

struct Row {
    trial: usize,
    kana_total: f64,
    floor_total: f64,
    floor_curve: Vec<f64>,
    cons_total: Vec<f64>,
    cons_centroid: Vec<f64>,
}

fn run() -> Vec<Row> {
    let waves = kana_waves();
    let freqs = Cochlea::new().center_freqs.clone();
    let cs = consonants();
    let silence = vec![0i32; (FLOOR_MS * 16000.0 / 1000.0) as usize];
    let cons_waves: Vec<Vec<i32>> = cs.iter().map(|&(_, c)| {
        let mut n = LfsrNoise::new(SEED);
        synth_consonant_banded(c, CONSONANT_MS, &mut n)
    }).collect();

    let mut co = Cochlea::new();
    let mut cn = CochlearNucleus::new();
    let mut rows = Vec::new();
    let last = *CHECKPOINTS.last().unwrap();
    let mut order = 0x1234_5678_9ABC_DEF0u64;

    for trial in 1..=last {
        // 環境: かなを決定論的な順序で 1 つ流す (**リセットしない**)
        order = order.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let si = ((order >> 33) as usize) % waves.len();
        let kt: f64 = present(&mut co, &mut cn, &waves[si]).iter().sum();

        if CHECKPOINTS.contains(&trial) {
            // 【修正1】固定の参照かな (KANA[0] = あ) を測る。
            // 旧版はチェックポイントごとに違うかなを測っており、
            // 「かな総」の変動が適応でなく**どのかなだったか**で決まっていた。
            let refk: f64 = present(&mut co, &mut cn, &waves[0]).iter().sum();

            // 【修正2】床を**回復曲線**として測る。
            // 旧版は刺激の直後に無音を 1 回だけ置いており、測っていたのは
            // 「前の刺激からの回復の途中」であって床ではなかった。
            // 無音を連続で出して、各区間の発火数を並べる。
            let mut floor_curve = Vec::new();
            for _ in 0..FLOOR_SEGMENTS {
                floor_curve.push(present(&mut co, &mut cn, &silence).iter().sum::<f64>());
            }
            let ft = *floor_curve.last().unwrap(); // 落ち着いた側を床とする

            // 子音: 続けて出す (状態は持ち越したまま)
            let mut ct = Vec::new();
            let mut cc = Vec::new();
            for w in cons_waves.iter() {
                let c = present(&mut co, &mut cn, w);
                ct.push(c.iter().sum());
                cc.push(centroid_bushy(&c, &freqs));
            }
            rows.push(Row { trial, kana_total: refk, floor_total: ft,
                            floor_curve, cons_total: ct, cons_centroid: cc });
        }
    }
    rows
}

fn main() {
    println!("=== M0.5 を動的平衡で測る — これまでの測定は全部『冷開始』だった ===");
    println!();
    println!("【ユーザーの指摘】この M0 と M0.5 は動的平衡に落ち着いた後に測定しているか?");
    println!("→ **いいえ。全部の測定が冷開始。**");
    println!("  M0 : 適応なし (G47a 保存率 100%) → 平衡の問題は無い");
    println!("  M0.5: 適応あり (G47b 保存率 83.1%) → **毎回 CochlearNucleus::new()**");
    println!("  §12.14 の記録: 提示1 2393 → 提示2 2036 → 以降平坦。**平衡は冷開始より約17%低い。**");
    println!();
    println!("【予測・事前・機構つき】**平衡では子音の重心が上がる。**");
    println!("  適応はよく駆動されるチャネルほど強く効く (local_entropy は発火で溜まる)。");
    println!("  中域は常に駆動されて適応が進み、/s/ の高域は駆動が稀で適応が浅い。");
    println!("  **§14.17 の文脈強調と同じ機構が、試行をまたいで効くはず。** 数値は置かない。");
    println!();
    println!("【ゲート・実測前に固定】正解の出どころ = 提示回数も無音を挟んだのも実験者が決めた");
    println!("  G79a 平衡到達 (総スパイクが平坦・相対 1/10)   G79b 子音の重心が上がる (順序保持)");
    println!("  G79c 床が下がる   G79d **信号対床の比が上がる (本命)**   G79e 決定論性");
    println!();
    println!("【限界】測定そのものが状態を動かす。無音→子音の順で出すので子音は直前の無音の");
    println!("影響を受ける。避けられず、実運用も同じ。順序は決定論的に固定して比較する。");

    // 【修正3】真の冷開始アーム。他プローブと同じ「まっさらな CochlearNucleus」。
    // 第2回で気づいた欠陥: 提示1のチェックポイントは既にかな1+参照かな+無音4区間を
    // 通しており **約6刺激ぶん進んでいた**。§12.14 では適応は提示2回で平衡に達するので、
    // **両アームとも平衡**だった。道理で差が出ないわけである。
    let cs = consonants();
    let freqs0 = Cochlea::new().center_freqs.clone();
    let (cold_tot, cold_cent): (Vec<f64>, Vec<f64>) = {
        let mut t = Vec::new();
        let mut c = Vec::new();
        for &(_, cons) in cs.iter() {
            let mut n = LfsrNoise::new(SEED);
            let w = synth_consonant_banded(cons, CONSONANT_MS, &mut n);
            // **毎回まっさら** = consonant_probe / kana_identify と同じ条件
            let mut co = Cochlea::new();
            let mut cn = CochlearNucleus::new();
            let r = present(&mut co, &mut cn, &w);
            t.push(r.iter().sum::<f64>());
            c.push(centroid_bushy(&r, &freqs0));
        }
        (t, c)
    };
    println!();
    println!("--- 真の冷開始 (まっさらな CochlearNucleus・他プローブと同条件) ---");
    println!("  総スパイク [{}]",
             cold_tot.iter().map(|x| format!("{:.0}", x)).collect::<Vec<_>>().join(", "));
    println!("  重心 [Hz]  [{}]",
             cold_cent.iter().map(|x| format!("{:.0}", x)).collect::<Vec<_>>().join(", "));

    let rows = run();

    println!();
    println!("--- 提示を重ねたときの推移 (リセットなし) ---");
    println!("  提示  参照かな  床の回復曲線 (無音 {}ms x {})       床  信号/床   子音の重心 [Hz]", FLOOR_MS, FLOOR_SEGMENTS);
    for r in rows.iter() {
        let ct: Vec<String> = r.cons_total.iter().map(|x| format!("{:.0}", x)).collect();
        let cc: Vec<String> = r.cons_centroid.iter().map(|x| format!("{:.0}", x)).collect();
        let ratio = r.cons_total.iter().sum::<f64>() / r.floor_total.max(1.0);
        let fc: Vec<String> = r.floor_curve.iter().map(|x| format!("{:.0}", x)).collect();
        let _ = &ct;
        println!("  {:>4} {:>9.0}  [{}] {:>6.0} {:>7.3}   [{}]",
                 r.trial, r.kana_total, fc.join(", "), r.floor_total, ratio, cc.join(", "));
    }
    println!("  (子音の順: {})", cs.iter().map(|c| c.0).collect::<Vec<_>>().join(", "));

    let cold = &rows[0];
    let eq = rows.last().unwrap();

    // G79a 平衡到達
    let mut diffs = Vec::new();
    for i in 1..rows.len() {
        diffs.push((rows[i].kana_total - rows[i - 1].kana_total).abs());
    }
    let g79a = *diffs.last().unwrap() < diffs[0] / 10.0;

    // G79b 重心
    let order_ok = eq.cons_centroid[0] < eq.cons_centroid[1]
        && eq.cons_centroid[1] < eq.cons_centroid[2]
        && eq.cons_centroid[2] < eq.cons_centroid[3];
    let up = eq.cons_centroid.iter().zip(cold.cons_centroid.iter()).filter(|(a, b)| a > b).count();
    let g79b = order_ok && up >= 3;

    // G79c 床
    let g79c = eq.floor_total < cold.floor_total;

    // G79d 信号対床
    let r_cold = cold.cons_total.iter().sum::<f64>() / cold.floor_total.max(1.0);
    let r_eq = eq.cons_total.iter().sum::<f64>() / eq.floor_total.max(1.0);
    let g79d = r_eq > r_cold;

    println!();
    println!("=== 冷開始 (提示1) vs 平衡 (提示{}) ===", eq.trial);
    println!("  床の総スパイク    : {:.0} → {:.0}  ({:+.1}%)",
             cold.floor_total, eq.floor_total,
             (eq.floor_total - cold.floor_total) / cold.floor_total * 100.0);
    println!("  子音の総スパイク  : {:.0} → {:.0}  ({:+.1}%)",
             cold.cons_total.iter().sum::<f64>(), eq.cons_total.iter().sum::<f64>(),
             (eq.cons_total.iter().sum::<f64>() - cold.cons_total.iter().sum::<f64>())
                 / cold.cons_total.iter().sum::<f64>() * 100.0);
    println!("  **信号/床の比**   : {:.3} → {:.3}  ({:+.1}%)",
             r_cold, r_eq, (r_eq - r_cold) / r_cold * 100.0);
    println!();
    println!("  子音の重心:");
    for (i, (nm, _)) in cs.iter().enumerate() {
        println!("    {:<4} {:>7.0}Hz → {:>7.0}Hz  ({:+.0}Hz)",
                 nm, cold.cons_centroid[i], eq.cons_centroid[i],
                 eq.cons_centroid[i] - cold.cons_centroid[i]);
    }

    println!();
    println!("=== 判定 (規則は実測前に固定) ===");
    println!("  G79a 平衡到達      -> {}", if g79a { "**到達**" } else { "**未到達**" });
    println!("  G79b 子音の重心    -> {} (順序 {} / 上がった {}/4)",
             if g79b { "**PASS**" } else { "**FAIL**" },
             if order_ok { "保持" } else { "崩れ" }, up);
    println!("  G79c 床が下がる    -> {}", if g79c { "**PASS**" } else { "**FAIL**" });
    println!("  G79d 信号/床の比   -> {}", if g79d { "**PASS**" } else { "**FAIL**" });

    // G79e
    let r2 = run();
    let same = rows.len() == r2.len()
        && rows.iter().zip(r2.iter()).all(|(a, b)|
            a.floor_total == b.floor_total && a.cons_total == b.cons_total);
    println!("  G79e 決定論性      -> {}", if same { "PASS" } else { "**FAIL**" });

    println!();
    println!("=== **真の冷開始** vs 平衡 (これが本当の比較) ===");
    println!("  子音    真の冷開始      平衡(提示{})      差", eq.trial);
    for (i, (nm, _)) in cs.iter().enumerate() {
        println!("    {:<4} {:>8.0}発 {:>7.0}Hz   {:>6.0}発 {:>7.0}Hz   {:+.0}発 {:+.0}Hz",
                 nm, cold_tot[i], cold_cent[i], eq.cons_total[i], eq.cons_centroid[i],
                 eq.cons_total[i] - cold_tot[i], eq.cons_centroid[i] - cold_cent[i]);
    }
    let ct: f64 = cold_tot.iter().sum();
    let et: f64 = eq.cons_total.iter().sum();
    println!("  総計: {:.0} → {:.0} ({:+.1}%)", ct, et, (et - ct) / ct * 100.0);
    let up_true = eq.cons_centroid.iter().zip(cold_cent.iter()).filter(|(a, b)| a > b).count();
    println!("  重心が上がった子音: {}/4 -> 予測 (平衡で上がる) は {}",
             up_true, if up_true >= 3 { "**当たり**" } else { "**外れ**" });

    println!();
    println!("  【この測定が答えないこと】かな同定率を平衡で測り直してはいない。");
    println!("  ここで測ったのは床・子音の総スパイク・重心だけ。**既定は変えていない。**");
}
