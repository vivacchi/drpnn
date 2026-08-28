//! モーラ境界と子音-母音境界を検算する (2026-08-27)
//!
//! ## なぜ
//!
//! 入り口監査が 2 つ主張した (**まだ出荷コードで確かめていない**):
//!
//! - **モーラ境界を音響が越えない** — 各モーラの母音が 10ms の release ramp でゼロに落ち、
//!   次のモーラがゼロから始まるので、**「連続発話」は「孤立モーラの連結」と同一**である。
//!   本当なら §14.19/§14.26 の「文脈」軸は**協調調音ではなく適応だけ**を測っていたことになる。
//! - **子音-母音境界に谷ができる** — 子音にも母音にも ramp があるので、
//!   **その谷の位置は `CONSONANT_STEPS = 60` (= 30ms) = 2窓の境界とちょうど同じ。**
//!   本当なら **2窓の利得の一部は「谷を見ているだけ」**かもしれない。
//!
//! ## ゲート (実測前に固定・以後動かさない)
//!
//! - **V6 モーラ境界を音響が越えるか**: 連続合成と個別連結が**バイト同一か**。
//!   *同一なら越えていない。* 母音だけのモーラを使う (雑音を消費しないので条件が揃う)。
//! - **V7 子音-母音境界に谷があるか**: 1ms ごとの包絡を dB で出す。
//!   谷の**位置**と**深さ**を記録する。*深さは判定しない。記述である。*
//! - **V8 谷は 2窓の境界と一致するか**: 谷の最小点が 30ms ± 3ms に入るか。
//!
//! ## 予測 (実測前)
//!
//! - **V6 はバイト同一になるはず** (各モーラが独立に合成され連結されるだけ)。
//! - **V7 谷はあるはず** (母音に 10ms の attack、子音にも 5ms の release)。**深さは分からない。**
//!
//! CLI: boundary_check

use spiking_brain::phase2_f::kana::{moras_from_kana, set_fric_vot, set_glottal_h, synth_utterance, MORA_MS};
use spiking_brain::phase2_f::phoneme_synth::{LfsrNoise, SAMPLE_RATE_HZ};

const F0: f64 = 150.0;
const SEED: u16 = 0xACE1;

fn synth(k: &str, noise: &mut LfsrNoise) -> Vec<i32> {
    let (m, sk) = moras_from_kana(k);
    assert_eq!(sk, 0, "未対応: {}", k);
    synth_utterance(&m, F0, noise)
}

fn rms(w: &[i32]) -> f64 {
    (w.iter().map(|&s| (s as f64) * (s as f64)).sum::<f64>() / w.len().max(1) as f64).sqrt()
}

fn main() {
    println!("=== モーラ境界と子音-母音境界を検算する ===");
    println!();
    println!("【なぜ】入り口監査が2つ主張した (**まだ出荷コードで確かめていない**):");
    println!("  ・**モーラ境界を音響が越えない** -> 本当なら §14.19/§14.26 の「文脈」軸は");
    println!("    **協調調音ではなく適応だけ**を測っていたことになる");
    println!("  ・**子音-母音境界に谷** -> その位置は CONSONANT_STEPS=60 (=30ms) = **2窓の境界と同じ**");
    println!("    本当なら **2窓の利得の一部は『谷を見ているだけ』**かもしれない");
    println!();
    println!("【予測・事前】V6 はバイト同一になるはず / V7 谷はあるはず・**深さは分からない**");

    // ---------- V6 モーラ境界 ----------
    println!();
    println!("--- V6 モーラ境界を音響が越えるか ---");
    for pair in ["あい", "おえ", "いあ"].iter() {
        let mut n0 = LfsrNoise::new(SEED);
        let cont = synth(pair, &mut n0);
        let cs: Vec<char> = pair.chars().collect();
        let mut cat: Vec<i32> = Vec::new();
        for c in cs.iter() {
            let mut n = LfsrNoise::new(SEED);
            cat.extend(synth(&c.to_string(), &mut n));
        }
        println!("  {} : 連続合成 {} サンプル / 個別連結 {} サンプル -> {}",
                 pair, cont.len(), cat.len(),
                 if cont == cat { "**バイト同一 = 音響は境界を越えていない**" } else { "違う (越えている)" });
    }

    // ---------- V7 / V8 子音-母音境界 ----------
    println!();
    println!("--- V7/V8 子音-母音境界の包絡 (1ms ごと・最大を 0dB とする) ---");
    println!("  子音区間 = 0-30ms / 母音区間 = 30-120ms。**2窓の境界は 30ms。**");
    println!();
    let step = (SAMPLE_RATE_HZ / 1000.0) as usize;   // 1ms
    for k in ["か", "な", "さ", "あ"].iter() {
        let mut n = LfsrNoise::new(SEED);
        let w = synth(k, &mut n);
        let nb = (MORA_MS as usize).min(w.len() / step);
        let env: Vec<f64> = (0..nb).map(|i| rms(&w[i * step..((i + 1) * step).min(w.len())])).collect();
        let peak = env.iter().cloned().fold(0f64, f64::max).max(1e-9);
        let db: Vec<f64> = env.iter().map(|&e| 20.0 * (e / peak).max(1e-9).log10()).collect();
        // 20-45ms の最小点 (境界の谷)
        let lo = 20usize.min(nb - 1);
        let hi = 45usize.min(nb);
        let (mut vi, mut vmin) = (lo, f64::INFINITY);
        for i in lo..hi { if db[i] < vmin { vmin = db[i]; vi = i; } }
        print!("  {} 20-45ms: ", k);
        for i in lo..hi.min(lo + 25) { print!("{:>5.0}", db[i]); }
        println!();
        println!("      -> **谷は {}ms で {:.1} dB**  (2窓の境界 30ms との差 {}ms)",
                 vi, vmin, (vi as i64 - 30));
    }

    // ================= §14.51 G98 ゲート (2026-08-28・実測前に固定) =================
    println!();
    println!("=== G98: 摩擦音の接合部修正 (案2: 無声摩擦音に気音10ms / h: 声門物理) ===");
    println!("  【予測・実測前】G98a は通るはず (雑音は毎サンプル声道を叩くので即座に立ち上がる)。");
    println!("  G98c で は行の同定が変わる方向は不明 (母音情報が乗る=上がる / 母音と混同=別の壊れ方)。");
    println!();
    let cseg = 30 * 16;   // 子音区間 = 30ms = 480 サンプル

    // --- G98a: さ の谷の深さ (ON = 既定) ---
    set_fric_vot(true);
    set_glottal_h(true);
    {
        let mut nz = LfsrNoise::new(SEED);
        let w = synth("さ", &mut nz);
        let step = (SAMPLE_RATE_HZ / 1000.0) as usize;
        let nb = (MORA_MS as usize).min(w.len() / step);
        let env: Vec<f64> = (0..nb).map(|i| rms(&w[i * step..((i + 1) * step).min(w.len())])).collect();
        let peak = env.iter().cloned().fold(0f64, f64::max).max(1e-9);
        let vmin = (20..45.min(nb)).map(|i| 20.0 * (env[i] / peak).max(1e-9).log10())
            .fold(f64::INFINITY, f64::min);
        println!("  G98a さ の 20-45ms の谷 = {:.1} dB (旧経路の基準 -35dB / 修正前 -180dB)", vmin);
        println!("       -> {}", if vmin >= -35.0 { "**PASS — 谷は埋まった**" } else { "**FAIL**" });
    }

    // --- G98b: 歯擦音の子音区間はバイト同一 (OFF vs ON) ---
    let sib = ["さ", "し", "す", "せ", "そ"];
    let mut same = true;
    for k in sib.iter() {
        set_fric_vot(false); set_glottal_h(false);
        let mut n1 = LfsrNoise::new(SEED);
        let off = synth(k, &mut n1);
        set_fric_vot(true); set_glottal_h(true);
        let mut n2 = LfsrNoise::new(SEED);
        let on = synth(k, &mut n2);
        if off[..cseg] != on[..cseg] { same = false; println!("  **{} の子音区間が変わってしまった**", k); }
    }
    println!("  G98b 歯擦音 5 かなの子音区間 0-30ms が OFF/ON でバイト同一 -> {}",
             if same { "**PASS — 変更は解放後に局在**" } else { "**FAIL**" });

    // --- G98c: は行の子音区間が後続母音に依存するようになったか ---
    let hs = ["は", "ひ", "ふ", "へ", "ほ"];
    for (label, on) in [("OFF (従来)", false), ("**ON (声門物理)**", true)].iter() {
        set_fric_vot(*on); set_glottal_h(*on);
        let ws: Vec<Vec<i32>> = hs.iter().map(|k| {
            let mut nz = LfsrNoise::new(SEED);
            synth(k, &mut nz)
        }).collect();
        let mut distinct = 0usize;
        let mut total = 0usize;
        for i in 0..hs.len() { for j in (i + 1)..hs.len() {
            total += 1;
            if ws[i][..cseg] != ws[j][..cseg] { distinct += 1; }
        }}
        println!("  G98c は行 5 モーラの子音区間・相異なる対 = {}/{}  [{}]", distinct, total, label);
    }
    println!("       予測: OFF では 0/10 (全て同一 = 監査所見 M6) / ON では 10/10");

    // --- G98b' / G98c' (2026-08-28): **G98b/G98c の設計の誤りを訂正した測り直し** ---
    //
    // G98b は FAIL し、G98c の OFF 側予測 (0/10) も外れた。**原因は両方同じ**:
    // `normalize_rms` は**発話全体に 1 回**掛かるので、気音を足すと総 RMS が変わり、
    // **倍率が変わって、触っていない 0-30ms もバイトでは変わる。**
    // バイト同一というゲートは正規化の存在と構造的に両立しなかった。
    // **宣言した基準では落ちたので落ちたと記録し、本来見るべき量 =
    // 倍率を除いた形の同一性 (コサイン) を別に測る。** (§14.27 G87c と同じ型の訂正)
    fn seg_cosine(a: &[i32], b: &[i32]) -> f64 {
        let n = a.len().min(b.len());
        let d: f64 = (0..n).map(|i| a[i] as f64 * b[i] as f64).sum();
        let na: f64 = a[..n].iter().map(|&x| (x as f64) * (x as f64)).sum::<f64>().sqrt();
        let nb: f64 = b[..n].iter().map(|&x| (x as f64) * (x as f64)).sum::<f64>().sqrt();
        if na == 0.0 || nb == 0.0 { 0.0 } else { d / (na * nb) }
    }
    println!();
    let mut min_cos = f64::INFINITY;
    for k in sib.iter() {
        set_fric_vot(false); set_glottal_h(false);
        let mut n1 = LfsrNoise::new(SEED);
        let off = synth(k, &mut n1);
        set_fric_vot(true); set_glottal_h(true);
        let mut n2 = LfsrNoise::new(SEED);
        let on = synth(k, &mut n2);
        let c = seg_cosine(&off[..cseg], &on[..cseg]);
        if c < min_cos { min_cos = c; }
    }
    println!("  G98b' 歯擦音の子音区間の形 (倍率を除く): OFF/ON コサイン最小 = {:.6}", min_cos);
    println!("       -> {}", if min_cos > 0.9999 { "**PASS — 形は同一。変わったのは発話全体の倍率だけ**" }
             else { "**FAIL — 形そのものが変わっている**" });

    let mut hc_off = f64::INFINITY;
    let mut hc_on = f64::INFINITY;
    for (arm_on, out) in [(false, &mut hc_off), (true, &mut hc_on)] {
        set_fric_vot(arm_on); set_glottal_h(arm_on);
        let ws: Vec<Vec<i32>> = hs.iter().map(|k| {
            let mut nz = LfsrNoise::new(SEED);
            synth(k, &mut nz)
        }).collect();
        let mut mn = f64::INFINITY;
        for i in 0..hs.len() { for j in (i + 1)..hs.len() {
            let c = seg_cosine(&ws[i][..cseg], &ws[j][..cseg]);
            if c < mn { mn = c; }
        }}
        *out = mn;
    }
    println!("  G98c' は行 5 モーラの子音区間の形・対の最小コサイン:");
    println!("       OFF (従来)     = {:.6}   (1.0 なら形は全て同一 = 監査所見 M6)", hc_off);
    println!("       **ON (声門物理)** = {:.6}   (低いほど後続母音に依存)", hc_on);
    println!("       -> {}", if hc_off > 0.9999 && hc_on < 0.99 {
        "**PASS — 従来は形が同一・修正後は母音依存**" } else { "**判定は上の数値を見よ**" });

    // --- G98f (2026-08-28): **有声性 100.0% の音量交絡チェック** ---
    //
    // 回帰テストで有声性の伝達情報量が 100.0% (満点) になった。**満点は疑う。**
    // 連続合成の気音は離散経路 (§14.34 = 放射後 RMS を合わせた) と違って音量整合を
    // していない。**「周期性の有無」でなく「音量」で取れていないかを、祝う前に確かめる。**
    // 見る量: 有声/無声の最小対で、母音頭 30-45ms (気音の乗る区間) の RMS 比。
    println!();
    set_fric_vot(true); set_glottal_h(true);
    let vpairs = [("さ", "ざ"), ("し", "じ"), ("す", "ず"), ("か", "が"), ("た", "だ")];
    let (a0, a1) = (30 * 16, 45 * 16);
    let mut worst: f64 = 1.0;
    println!("  G98f 母音頭 30-45ms の RMS 比 (無声 ÷ 有声・1.0 なら音量では当てられない):");
    for (u, v) in vpairs.iter() {
        let mut n1 = LfsrNoise::new(SEED);
        let mut n2 = LfsrNoise::new(SEED);
        let (wu, wv) = (synth(u, &mut n1), synth(v, &mut n2));
        let r = rms(&wu[a0..a1]) / rms(&wv[a0..a1]).max(1e-9);
        if (r - 1.0).abs() > (worst - 1.0).abs() { worst = r; }
        println!("     {}/{} = {:.3}", u, v, r);
    }
    println!("  最大のずれ {:.3} -> {}", worst,
             if worst > 0.667 && worst < 1.5 { "**音量だけでは系統的に当てられない範囲**" }
             else { "**FAIL — 音量交絡の疑い。有声性 100% はこの区間の音量を読んだ可能性**" });

    // --- G98e: 決定論性 ---
    set_fric_vot(true); set_glottal_h(true);
    let mut na = LfsrNoise::new(SEED);
    let mut nb2 = LfsrNoise::new(SEED);
    let (wa, wb) = (synth("は", &mut na), synth("は", &mut nb2));
    println!("  G98e 決定論性 -> {}", if wa == wb { "PASS" } else { "**FAIL**" });

    println!();
    println!("  V8 谷は 2窓の境界 (30ms±3ms) と一致するか -> 上の「差」を見よ。");
    println!();
    println!("  【この検算が答えないこと】**谷があることと、2窓の利得が谷由来であることは別。**");
    println!("  それを分けるには**谷を埋めて 2窓を測り直す**必要がある。**まだやっていない。**");
}
