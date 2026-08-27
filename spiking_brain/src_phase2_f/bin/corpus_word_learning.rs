//! コーパスを聞かせると、M1 は単語をよく区別できるようになるか (2026-08-27)
//!
//! ## なぜ — ユーザーの定義そのものの検証
//!
//! ユーザーの定義: **「シナプスの動的平衡に至ることを学習と言う。」**
//!
//! §14.42 で確認したとおり、`word_stream` は **M0 → M0.5 までしか通していない。**
//! **シナプスは 1 本も学習していない。**「平衡」と呼んでいたのは適応状態だけだった。
//!
//! **したがってユーザーの定義する学習は、単語という単位でまだ一度も測っていない。**
//! ここで初めて **M1 を通し**、**コーパスを聞かせた量に対して単語の弁別が上がるか**を測る。
//!
//! ## 測り方 — 本線を乱さずに分岐する
//!
//! コーパスを 1 回流しながら、決めた地点で **(蝸牛・神経核・M1) を丸ごと複製**し、
//! **複製の側にだけ**単語テストを流す。本線はテストの影響を受けない。
//!
//! 単語は **§14.40 の最小対 32 語**、**フレーム列** (窓で切らない・時間を潰さない)。
//! **変種ごとに並び順を無作為化する** (§14.42 の欠陥を持ち込まないため)。
//!
//! ## 設計監査が指摘した 4 つの致命的欠陥を、最初から潰してある
//!
//! 1. **次元をそろえた対照**: M1 出力は 40、M0.5 は 84。同じ統計で比べると
//!    「M1 が情報を落とした」と「出力層が小さい」を分離できない。
//!    → **CN40 (bushy 40ch だけ) の列を足す。** M1 の比較相手はこれ。
//! 2. **棄却域が空**: 解析チャンス値と最大値を比べると、情報ゼロでも
//!    **8 地点なら 97% で「超えた」と出る** (実際に算術で確認した)。
//!    → **置換帰無** (ラベルを混ぜて同じ統計を計算) の分布と比べ、
//!      **地点数で Bonferroni 補正**した閾値を使う。
//! 3. **用量反応がコイン投げ**: `last > first` は帰無で 46% 成立する。
//!    → **8 地点の単調性 (Spearman) と、その置換帰無**で判定する。
//! 4. **M1 の伝導遅延**: `delay_range=(2,40)` step なので M1 の応答は遅れる。
//!    → **遅延 0 と、網自身の定数から決まる遅延**の 2 列を必ず出す。
//!
//! さらに: **コーパスも F0 を変えて流す** (単一 F0 に適応しただけ、を防ぐ)。
//! **密度と伝達可能シナプス数を各地点で出す** (崩壊が見えるように)。
//!
//! ## ゲート (実測前に固定・以後動かさない)
//!
//! - **G96a M1 は単語を区別できるか**: 置換帰無 (Bonferroni 補正) を超えるか。
//! - **G96b M1 vs CN40 (次元をそろえた対照)**: どちらが高いか。
//! - **G96c 用量反応**: 8 地点の Spearman と置換帰無。***これがユーザーの定義の検証。***
//! - **G96d 密度で説明できないか**: 同定率と M1 の発火密度が同方向に動いていないか。
//! - **G96e 決定論性 / G96f コーパスの内容は一切出力しない**
//!
//! ## 予測 (実測前・数値は置かない)
//!
//! 1. **M1 は置換帰無を超える。** M0.5 を受けているので何かは残る。
//! 2. **M1 < CN40。** 段を足せば情報は減る + M1 は単語を区別する目的を知らない。
//! 3. **用量反応は出ない。** M1 の可塑性は弁別を目的にしていない。
//!    **これが本命の予測であり、外れたら大きい。**
//!
//! CLI: corpus_word_learning  (DRPNN_CORPUS_MORAS でモーラ数・既定 12000)

use spiking_brain::phase2_f::cochlea::{Cochlea, SAMPLES_PER_STEP};
use spiking_brain::phase2_f::cochlear_nucleus::{CochlearNucleus, N_CN_OUTPUT};
use spiking_brain::phase2_f::kana::{moras_from_kana, synth_utterance, Mora, MORA_MS};
use spiking_brain::phase2_f::phoneme_synth::{LfsrNoise, SAMPLE_RATE_HZ};
use spiking_brain::phase2_f::thermo_network::{ThermoNetwork, ThermoNetworkConfig, SIGNAL_SCALE_DIVISOR};
use std::io::Read;

const CORPUS: &str = "../data/corpus/roleplay_kana.txt";
const F0S: [f64; 4] = [100.0, 130.0, 160.0, 200.0];
const N_VAR: usize = 4;
const STEPS_PER_FRAME: usize = 20;   // 10ms
const N_PERM: usize = 400;
/// M1 の伝導遅延の代表値 = delay_range (2,40) の中央 21 step。**網自身の定数から決まる。**
const M1_DELAY_STEPS: usize = 21;

/// §14.40 と同じ最小対 16 組 = 32 語
const PAIRS: &[(&str, &str)] = &[
    ("こころ", "ところ"), ("からだ", "かなだ"), ("たまご", "たなご"), ("てがみ", "てあみ"),
    ("せかい", "せたい"), ("みどり", "みのり"), ("かたち", "かたな"), ("ひかり", "ひかる"),
    ("さかな", "さかや"), ("くるま", "くるみ"), ("なまえ", "なまり"), ("ちから", "ちかく"),
    ("いのち", "いのり"), ("みかん", "みかた"), ("あたま", "あたり"), ("からす", "からて"),
];

fn words() -> Vec<&'static str> { PAIRS.iter().flat_map(|&(a, b)| [a, b]).collect() }

fn lcg(s: &mut u64) -> u64 {
    *s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    *s >> 33
}

fn shuffled(n: usize, seed: u64) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..n).collect();
    let mut s = seed | 1;
    for i in (1..n).rev() { let r = lcg(&mut s) as usize % (i + 1); idx.swap(i, r); }
    idx
}

fn utterance_seed(w: usize, v: usize) -> u16 {
    ((w as u16).wrapping_mul(131).wrapping_add(v as u16).wrapping_mul(4099)) | 1
}

/// コーパスの先頭から `n` モーラ。**内容は返さない・出力もしない。**
fn load_moras(n: usize) -> (Vec<Mora>, usize) {
    let want = n * 9 + 4096;
    let mut f = std::fs::File::open(CORPUS)
        .unwrap_or_else(|e| panic!("コーパスが開けない ({}): {}", CORPUS, e));
    let mut buf = vec![0u8; want];
    let mut filled = 0usize;
    while filled < want {
        match f.read(&mut buf[filled..]) { Ok(0) => break, Ok(k) => filled += k, Err(e) => panic!("{}", e) }
    }
    buf.truncate(filled);
    let text = loop {
        match std::str::from_utf8(&buf) { Ok(s) => break s.to_string(), Err(e) => buf.truncate(e.valid_up_to()) }
    };
    let mut out = Vec::new();
    let mut kinds = std::collections::BTreeSet::new();
    for c in text.chars() {
        if out.len() >= n { break; }
        if c == '\n' || c == ' ' { continue; }
        let (m, _) = moras_from_kana(&c.to_string());
        if !m.is_empty() { kinds.insert(c); }
        out.extend(m);
    }
    out.truncate(n);
    (out, kinds.len())
}

struct Readout { cn84: Vec<(usize, Vec<f64>)>, cn40: Vec<(usize, Vec<f64>)>,
                 m1: Vec<(usize, Vec<f64>)>, m1d: Vec<(usize, Vec<f64>)>,
                 m1_density: f64 }

/// **複製の側に**単語テストを流す。本線は触らない。
fn battery(net: &ThermoNetwork, co: &Cochlea, cn: &CochlearNucleus) -> Readout {
    let ws = words();
    let n_out = net.output_neurons.len();
    let (mut c84, mut c40, mut m1v, mut m1d) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    let mut m1_total = 0f64;
    for v in 0..N_VAR {
        // **変種ごとに複製する。**変種どうしが互いを汚さない。
        let (mut net, mut co, mut cn) = (net.clone(), co.clone(), cn.clone());
        for &w in shuffled(ws.len(), 0xA5A5 ^ ((v as u64) << 20)).iter() {
            let mut noise = LfsrNoise::new(utterance_seed(w, v));
            let (m, sk) = moras_from_kana(ws[w]);
            assert_eq!(sk, 0, "未対応の単語: {}", ws[w]);
            let wave = synth_utterance(&m, F0S[v], &mut noise);
            let n_steps = wave.len() / SAMPLES_PER_STEP;
            let n_frames = n_steps / STEPS_PER_FRAME;
            let mut f84 = vec![0f64; n_frames * N_CN_OUTPUT];
            let mut f40 = vec![0f64; n_frames * 40];
            let mut fm1 = vec![0f64; n_frames * n_out];
            let mut fm1d = vec![0f64; n_frames * n_out];
            for (step, chunk) in wave.chunks(SAMPLES_PER_STEP).enumerate() {
                if chunk.len() < SAMPLES_PER_STEP { break; }
                let m0 = co.process_step(chunk);
                let cno = cn.process_step(&m0);
                let fr = step / STEPS_PER_FRAME;
                if fr < n_frames {
                    for (i, &x) in cno.iter().enumerate() { if x != 0 { f84[fr * N_CN_OUTPUT + i] += 1.0; } }
                    // CN40 = bushy 40ch だけ (**次元をそろえた対照**)
                    for i in 0..40 { if cno[4 + i] != 0 { f40[fr * 40 + i] += 1.0; } }
                }
                for nid in net.step(&cno) {
                    if let Some(oi) = net.output_index_of(nid) {
                        m1_total += 1.0;
                        if fr < n_frames { fm1[fr * n_out + oi] += 1.0; }
                        // **伝導遅延の補正列**: 応答を M1_DELAY_STEPS ぶん前に戻して数える
                        let sd = step.saturating_sub(M1_DELAY_STEPS) / STEPS_PER_FRAME;
                        if sd < n_frames && step >= M1_DELAY_STEPS { fm1d[sd * n_out + oi] += 1.0; }
                    }
                }
            }
            c84.push((w, f84)); c40.push((w, f40)); m1v.push((w, fm1)); m1d.push((w, fm1d));
        }
    }
    let n = c84.len() as f64;
    Readout { cn84: c84, cn40: c40, m1: m1v, m1d, m1_density: m1_total / n }
}

fn cosine(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len().min(b.len());
    let d: f64 = (0..n).map(|i| a[i] * b[i]).sum();
    let na: f64 = a[..n].iter().map(|x| x * x).sum::<f64>().sqrt();
    let nb: f64 = b[..n].iter().map(|x| x * x).sum::<f64>().sqrt();
    if na == 0.0 || nb == 0.0 { 0.0 } else { d / (na * nb) }
}

/// 近傍のクラス列 (同点棄却つき)。ラベルを差し替えて再利用できるよう index を返す。
fn neighbors(v: &[(usize, Vec<f64>)]) -> Vec<Option<usize>> {
    let n = v.len();
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let mut best = f64::NEG_INFINITY;
        for j in 0..n { if j != i { let c = cosine(&v[i].1, &v[j].1); if c > best { best = c; } } }
        let tied: Vec<usize> = (0..n).filter(|&j| j != i && cosine(&v[i].1, &v[j].1) == best).collect();
        if tied.is_empty() { out.push(None); }
        else if tied.iter().all(|&j| v[j].0 == v[tied[0]].0) { out.push(Some(tied[0])); }
        else { out.push(None); }
    }
    out
}

fn acc_from(nb: &[Option<usize>], lab: &[usize]) -> f64 {
    let ok = nb.iter().enumerate()
        .filter(|(i, x)| x.map_or(false, |j| lab[j] == lab[*i])).count();
    ok as f64 / nb.len() as f64 * 100.0
}

/// **置換帰無**: ラベルを混ぜて同じ統計を計算する。近傍は使い回す (幾何は変えない)。
fn perm_null(v: &[(usize, Vec<f64>)], nb: &[Option<usize>]) -> (f64, f64) {
    let lab: Vec<usize> = v.iter().map(|x| x.0).collect();
    let mut accs: Vec<f64> = Vec::with_capacity(N_PERM);
    for p in 0..N_PERM {
        let perm = shuffled(lab.len(), 0xBEEF ^ ((p as u64) << 16));
        let shuffled_lab: Vec<usize> = perm.iter().map(|&i| lab[i]).collect();
        accs.push(acc_from(nb, &shuffled_lab));
    }
    accs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let p95 = accs[(N_PERM as f64 * 0.95) as usize];
    // 8 地点ぶんの Bonferroni 補正 (1 - 0.05/8 = 0.99375)
    let pb = accs[((N_PERM as f64 * 0.99375) as usize).min(N_PERM - 1)];
    (p95, pb)
}

/// Spearman の順位相関 (地点の順序 vs 同定率)
fn spearman(y: &[f64]) -> f64 {
    let n = y.len();
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&a, &b| y[a].partial_cmp(&y[b]).unwrap());
    let mut rank = vec![0f64; n];
    for (r, &i) in idx.iter().enumerate() { rank[i] = r as f64; }
    let mx = (n - 1) as f64 / 2.0;
    let (mut num, mut dx, mut dy) = (0f64, 0f64, 0f64);
    for i in 0..n {
        let (a, b) = (i as f64 - mx, rank[i] - mx);
        num += a * b; dx += a * a; dy += b * b;
    }
    if dx == 0.0 || dy == 0.0 { 0.0 } else { num / (dx * dy).sqrt() }
}

/// Spearman の置換帰無 (順序を混ぜる)
fn spearman_p(y: &[f64]) -> f64 {
    let obs = spearman(y);
    let mut ge = 0usize;
    for p in 0..N_PERM {
        let perm = shuffled(y.len(), 0xF00D ^ ((p as u64) << 12));
        let z: Vec<f64> = perm.iter().map(|&i| y[i]).collect();
        if spearman(&z).abs() >= obs.abs() { ge += 1; }
    }
    (ge + 1) as f64 / (N_PERM + 1) as f64
}

fn main() {
    let n_moras: usize = std::env::var("DRPNN_CORPUS_MORAS").ok()
        .and_then(|v| v.parse().ok()).unwrap_or(12000);
    let cps: Vec<usize> = vec![0, 250, 1000, 2000, 4000, 6000, 9000, n_moras]
        .into_iter().filter(|&c| c <= n_moras).collect();

    println!("=== コーパスを聞かせると、M1 は単語をよく区別できるようになるか ===");
    println!();
    println!("【なぜ】ユーザーの定義「**シナプスの動的平衡に至ることを学習と言う**」。");
    println!("§14.42 で確認したとおり word_stream は **M0->M0.5 までしか通しておらず");
    println!("シナプスは1本も学習していない**。**ユーザーの定義する学習は単語という単位で");
    println!("まだ一度も測っていない。**ここで初めて M1 を通す。");
    println!();
    println!("【設計監査が指摘した4つの致命的欠陥を最初から潰してある】");
    println!("  ① **次元をそろえた対照 CN40**(bushy 40ch)。M1(40) の比較相手はこれ");
    println!("  ② **置換帰無 + Bonferroni**(解析チャンスと最大値の比較は情報ゼロでも97%通る)");
    println!("  ③ **用量反応は Spearman + 置換帰無**(last>first は帰無で46%成立)");
    println!("  ④ **M1 の伝導遅延**(delay_range 中央 {} step)の補正列を必ず併記", M1_DELAY_STEPS);
    println!("  さらに **コーパスも F0 を変えて流す** / **密度と伝達可能シナプス数を各地点で出す**");
    println!();
    println!("【予測・実測前・数値は置かない】");
    println!("  ① M1 は置換帰無を超える  ② **M1 < CN40**(段を足せば情報は減る)");
    println!("  ③ **用量反応は出ない**(M1 の可塑性は弁別を目的にしていない)。**本命。外れたら大きい。**");

    let t0 = std::time::Instant::now();
    let (moras, kinds) = load_moras(n_moras);
    println!();
    println!("  コーパス {} モーラ / **{} 種類のかなが鳴った**。**内容は出力しない。**", moras.len(), kinds);

    let cfg = if N_CN_OUTPUT == 164 { ThermoNetworkConfig::for_m1_cn_80() }
              else { ThermoNetworkConfig::for_m1_cn_40() };
    assert_eq!(cfg.n_input, N_CN_OUTPUT, "M1 の入力数と M0.5 の出力数が一致しない");
    let mut net = ThermoNetwork::new(cfg);
    let (mut co, mut cn) = (Cochlea::new(), CochlearNucleus::new());
    let mut noise = LfsrNoise::new(0xACE1);

    let mut rows: Vec<(usize, f64, f64, f64, f64, f64, usize, usize)> = Vec::new();
    let mut null95 = 0f64;
    let mut nullb = 0f64;
    let mut next = 0usize;
    for i in 0..=moras.len() {
        if next < cps.len() && cps[next] == i {
            let r = battery(&net, &co, &cn);
            let (a84, a40) = (acc(&r.cn84), acc(&r.cn40));
            let (am1, am1d) = (acc(&r.m1), acc(&r.m1d));
            if next == 0 {
                let nb = neighbors(&r.cn40);
                let p = perm_null(&r.cn40, &nb);
                null95 = p.0; nullb = p.1;
            }
            let alive = net.n_open_synapses();
            let live = net.synapses.iter()
                .filter(|s| s.alive && s.conductance >= SIGNAL_SCALE_DIVISOR).count();
            rows.push((i, a84, a40, am1, am1d, r.m1_density, alive, live));
            next += 1;
        }
        if i == moras.len() { break; }
        let f0 = F0S[i % N_VAR];   // **コーパスも F0 を変える**
        let w = synth_utterance(std::slice::from_ref(&moras[i]), f0, &mut noise);
        for chunk in w.chunks(SAMPLES_PER_STEP) {
            if chunk.len() < SAMPLES_PER_STEP { break; }
            let m0 = co.process_step(chunk);
            let cno = cn.process_step(&m0);
            let _ = net.step(&cno);
        }
    }

    println!();
    println!("--- 単語の同定率 (最小対 32 語 × 4 変種 = 128 条件・フレーム列・窓なし) ---");
    println!("  {:>7} | {:>8} {:>8} | {:>9} {:>10} | {:>9} {:>9} {:>9}",
             "聞いた", "M0.5(84)", "**CN40**", "**M1(40)**", "M1(遅延補正)", "M1発火/語", "alive", "伝達可");
    for (m, a84, a40, am1, am1d, d, al, lv) in rows.iter() {
        println!("  {:>7} | {:>7.1}% {:>7.1}% | {:>8.1}% {:>9.1}% | {:>9.1} {:>9} {:>9}",
                 m, a84, a40, am1, am1d, d, al, lv);
    }
    println!("  **置換帰無: 95%点 {:.1}% / Bonferroni(8地点) {:.1}%**", null95, nullb);

    let m1s: Vec<f64> = rows.iter().map(|r| r.3).collect();
    let a40s: Vec<f64> = rows.iter().map(|r| r.2).collect();
    let dens: Vec<f64> = rows.iter().map(|r| r.5).collect();

    println!();
    let over = m1s.iter().filter(|&&x| x > nullb).count();
    println!("  **G96a M1 は単語を区別できるか** -> {}/{} 地点が Bonferroni 帰無を超えた -> {}",
             over, m1s.len(), if over > 0 { "**超えた**" } else { "**超えない — M1 出力は単語を表現していない**" });

    let m1_best = m1s.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let a40_best = a40s.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    println!("  **G96b M1 vs CN40 (次元をそろえた対照)** -> M1 最良 {:.1}% / CN40 最良 {:.1}% -> {}",
             m1_best, a40_best,
             if m1_best > a40_best { "**M1 が上 — 帰無が破れた**" } else { "**CN40 が上 — 帰無は保たれた**" });

    let rho = spearman(&m1s);
    let p = spearman_p(&m1s);
    println!();
    println!("  **G96c 用量反応 (ユーザーの定義の検証)** -> Spearman ρ = {:+.3} / 置換帰無 p = {:.3} -> {}",
             rho, p, if p < 0.05 && rho > 0.0 { "**上がっている — 予測③が外れた = 学習が読み出しを良くしている**" }
                     else if p < 0.05 && rho < 0.0 { "**下がっている**" }
                     else { "**有意な用量反応なし (予測どおり)**" });

    let rho_d = spearman(&dens);
    println!("  **G96d 密度で説明できないか** -> M1 発火密度の Spearman ρ = {:+.3} -> {}",
             rho_d, if (rho > 0.0) == (rho_d > 0.0) && rho.abs() > 0.3 && rho_d.abs() > 0.3
                    { "**同方向。密度の変化で説明できてしまう可能性がある**" }
                    else { "**同定率と密度は同方向に動いていない**" });

    let r2 = battery(&net, &co, &cn);
    println!();
    println!("  G96e 決定論性 -> {}",
             if (acc(&r2.m1) - rows.last().unwrap().3).abs() < 1e-12 { "PASS" } else { "**FAIL**" });
    println!("  G96f コーパスの内容 -> **一切出力していない (数値のみ)**");
    println!();
    println!("  所要 {:.1} 秒。", t0.elapsed().as_secs_f64());
    println!("  【この測定が答えないこと】**M1 の出力層 40 個の発火数しか見ていない。**");
    println!("  10ms フレームには分けたが、**フレーム内の時間パターンは捨てている。**");
}

fn acc(v: &[(usize, Vec<f64>)]) -> f64 {
    let nb = neighbors(v);
    let lab: Vec<usize> = v.iter().map(|x| x.0).collect();
    acc_from(&nb, &lab)
}

#[allow(dead_code)]
fn unused(_: f64) { let _ = MORA_MS; let _ = SAMPLE_RATE_HZ; }
