//! DRAM サーモメータ読み出し 物理検証
//!
//! ユーザー提案: 同一ビット線で WL を順々に増やし、 SA が 1 を感知した WL 本数で多値化。
//! これを DRAM の電荷共有物理で忠実にシミュレートし、 多値読み出しが可能かを検証。
//!
//! 検証項目:
//!   1. count 読み出し: K 個の充電セル → サーモメータ N は K を符号化するか
//!   2. Vref 掃引: 基準電圧で分解能がどう変わるか
//!   3. Cbl/Cs 比の影響: 線形和領域の範囲
//!   4. アナログ和読み出し: 連続値セルで N が総電荷を符号化するか
//!   5. 実効分解能 (何ビット相当か)

use spiking_brain::phase2_dram::bitline_physics::{Bitline, VDD, VPRE, CS, CBL_DEFAULT};

fn main() {
    println!("== DRAM サーモメータ読み出し 物理検証 ==");
    println!("  電荷共有則: V_N = (Cbl·Vpre + Cs·ΣVcell) / (Cbl + N·Cs)");
    println!("  SA: V_N > Vref で 1、 サーモメータ = 初めて 1 になる N");
    println!("  パラメータ: VDD={}, Vpre={}, Cs={}, Cbl={} (Cbl/Cs={})",
        VDD, VPRE, CS, CBL_DEFAULT, CBL_DEFAULT / CS);

    let m = 64usize;  // ビット線あたりセル数

    // ── 検証 1: count 読み出し (二値セル、 K 個充電) ──
    // Vref を Vpre より下げて「充電セルがあれば flip」 する設定で、
    // 充電セルを先頭に詰めた場合の N を見る
    println!("\n── 検証 1: count 読み出し (Vref 掃引) ──");
    for &vref_frac in &[0.5, 0.55, 0.6, 0.65, 0.7] {
        let vref = VDD * vref_frac;
        print!("  Vref={:.2}V (VDD×{:.2}): K→N  ", vref, vref_frac);
        let mut prev_n: Option<usize> = None;
        let mut distinct = std::collections::BTreeSet::new();
        for k in 0..=m {
            let mut bl = Bitline::new(m, CBL_DEFAULT);
            bl.vref = vref;
            // 先頭 K セルを満充電
            for i in 0..k { bl.set_cell(i, VDD); }
            let n = bl.thermometer_read();
            if let Some(nn) = n { distinct.insert(nn); }
            // K の代表点だけ表示
            if [1, 4, 8, 16, 32, 64].contains(&k) {
                print!("K{}→{}  ", k, n.map(|x| x.to_string()).unwrap_or("∞".into()));
            }
            let _ = prev_n; prev_n = n;
        }
        println!("[異なる N 値: {}]", distinct.len());
    }

    // ── 検証 2: アナログ和読み出し (連続値セル) ──
    // 全セルに均一の電圧 v を入れ、 v を変えて N がどう変わるか
    println!("\n── 検証 2: アナログ和読み出し (全セル均一電圧 v) ──");
    println!("  Vref を Vpre より少し下に設定 (充電を検出しやすく)");
    let vref = VDD * 0.52;
    println!("  v(V) | total_charge | サーモメータ N");
    for &vfrac in &[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0] {
        let v = VDD * vfrac;
        let mut bl = Bitline::new(m, CBL_DEFAULT);
        bl.vref = vref;
        for i in 0..m { bl.set_cell(i, v); }
        let n = bl.thermometer_read();
        println!("  {:.2} |    {:6.1}    | {}",
            v, bl.total_charge(), n.map(|x| x.to_string()).unwrap_or("∞ (検出なし)".into()));
    }

    // ── 検証 3: Cbl/Cs 比の影響 ──
    println!("\n── 検証 3: Cbl/Cs 比と線形和領域 ──");
    println!("  K 個充電 (先頭) で N を測る、 Vref=VDD×0.52");
    for &cbl in &[3.0, 7.0, 15.0, 31.0] {
        print!("  Cbl/Cs={:>2}: ", cbl);
        let vref = VDD * 0.52;
        for &k in &[1, 2, 4, 8, 16, 32] {
            let mut bl = Bitline::new(m, cbl);
            bl.vref = vref;
            for i in 0..k { bl.set_cell(i, VDD); }
            let n = bl.thermometer_read();
            print!("K{}→{}  ", k, n.map(|x| x.to_string()).unwrap_or("∞".into()));
        }
        println!();
    }

    // ── 検証 4: 実効分解能 (混合電荷で N の単調性) ──
    // ランダムな部分充電パターンで、 total_charge と N の相関を見る
    println!("\n── 検証 4: total_charge vs サーモメータ N の単調性 ──");
    println!("  ランダム充電パターン 20 種、 Vref=VDD×0.52");
    let vref = VDD * 0.52;
    let mut lcg: u64 = 12345;
    let mut pairs: Vec<(f64, usize)> = Vec::new();
    for _ in 0..20 {
        let mut bl = Bitline::new(m, CBL_DEFAULT);
        bl.vref = vref;
        for i in 0..m {
            lcg = lcg.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let r = ((lcg >> 33) as f64) / (1u64 << 31) as f64;  // 0..1
            bl.set_cell(i, VDD * r);
        }
        if let Some(n) = bl.thermometer_read() {
            pairs.push((bl.total_charge(), n));
        }
    }
    pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    println!("  total_charge → N (総電荷でソート):");
    for (tc, n) in &pairs {
        println!("    {:6.1} → N={}", tc, n);
    }
    // 単調性チェック (総電荷大 → N 小 を期待)
    let mut monotonic_violations = 0;
    for w in pairs.windows(2) {
        if w[1].1 > w[0].1 { monotonic_violations += 1; }  // 総電荷増で N が増えたら違反
    }
    println!("  単調性違反 (総電荷増で N 増): {} / {}", monotonic_violations, pairs.len().saturating_sub(1));

    println!("\n── 結論 ──");
    println!("  上記から、 二値 SA + WL 本数カウントで多値読み出しが");
    println!("  どの程度の分解能・単調性で成立するかを判定。");
    println!("  単調なら graded spike (発火強度) のアナログ読み出しが物理的に成立。");
}
