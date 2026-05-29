//! DRAM ビット線 物理挙動シミュレーション (サーモメータ読み出し検証)
//!
//! 目的: ユーザー提案の「WL を順々に増やし、 SA が 1 を感知した WL 本数で多値化」
//!       を、 DRAM の実際の電荷共有物理で忠実に検証する。
//!
//! ★ 重要: このモジュールは「アナログハードウェアの物理」 を再現するため浮動小数を使う。
//!    SNN アルゴリズム本体 (整数演算、 DRP 制約) とは別の関心事。
//!    ここで検証するのは「二値 SA + WL 本数カウントで多値読み出しが可能か」 という
//!    ハードウェア実現性の物理シミュレーション。
//!
//! 物理モデル (DDR4 相当、 正規化単位):
//!   - 各セル: キャパシタ Cs に電圧 Vcell (0..VDD) を保持
//!   - ビット線: 容量 Cbl、 読み出し前に Vpre にプリチャージ
//!   - N セル活性化時の電荷共有 (電荷保存則):
//!       V_N = (Cbl·Vpre + Cs·ΣVcell_i) / (Cbl + N·Cs)
//!   - センスアンプ: V_N > Vref で 1 出力 (二値)
//!   - サーモメータ読み出し: N=1,2,... と増やし、 SA が 1 になる最小 N を読む

/// 電源電圧 (DDR4 相当)
pub const VDD: f64 = 1.2;
/// プリチャージ電圧 (VDD/2)
pub const VPRE: f64 = 0.6;
/// セルキャパシタンス (正規化、 Cs=1)
pub const CS: f64 = 1.0;
/// ビット線キャパシタンス (Cbl/Cs ≈ 7、 DDR4 典型)
pub const CBL_DEFAULT: f64 = 7.0;

/// ビット線 1 本 (複数セルがぶら下がる)
#[derive(Clone, Debug)]
pub struct Bitline {
    /// 各セルの保持電圧 (0..VDD)
    pub cells: Vec<f64>,
    /// ビット線容量 (Cs 単位)
    pub cbl: f64,
    /// プリチャージ電圧
    pub vpre: f64,
    /// SA 基準電圧
    pub vref: f64,
}

impl Bitline {
    pub fn new(n_cells: usize, cbl: f64) -> Self {
        Self {
            cells: vec![0.0; n_cells],
            cbl,
            vpre: VPRE,
            vref: VPRE,  // 標準 DRAM は Vref = Vpre
        }
    }

    /// PWM 書き込み: パルス幅 (0..1 正規化) → セル電圧
    /// キャパシタ充電カーブ V = VDD·(1 - exp(-t/τ))、 τ で飽和
    /// pulse_width=1.0 で約 95% 充電
    pub fn write_pwm(&mut self, cell: usize, pulse_width: f64) {
        let tau = 0.33;  // 時定数 (pulse_width=1 で 1/0.33≈3τ → 95%)
        let v = VDD * (1.0 - (-pulse_width / tau).exp());
        self.cells[cell] += v;
        if self.cells[cell] > VDD { self.cells[cell] = VDD; }
    }

    /// セルに直接電圧を設定 (テスト用)
    pub fn set_cell(&mut self, cell: usize, v: f64) {
        self.cells[cell] = v.clamp(0.0, VDD);
    }

    /// 自然リーク (保持時間 τ_leak で減衰)
    pub fn leak(&mut self, dt_over_tau: f64) {
        let factor = (-dt_over_tau).exp();
        for c in &mut self.cells { *c *= factor; }
    }

    /// 先頭 N セルを活性化したときのビット線電圧 (電荷共有則)
    /// V_N = (Cbl·Vpre + Cs·ΣVcell[0..N]) / (Cbl + N·Cs)
    pub fn shared_voltage(&self, n: usize) -> f64 {
        let sum: f64 = self.cells[..n.min(self.cells.len())].iter().sum();
        (self.cbl * self.vpre + CS * sum) / (self.cbl + n as f64 * CS)
    }

    /// センスアンプ: V_N > Vref で true (1)
    pub fn sense(&self, n: usize) -> bool {
        self.shared_voltage(n) > self.vref
    }

    /// サーモメータ読み出し: N=1,2,... で SA が初めて 1 になる N を返す
    /// (ユーザー提案の多値読み出し)
    pub fn thermometer_read(&self) -> Option<usize> {
        for n in 1..=self.cells.len() {
            if self.sense(n) { return Some(n); }
        }
        None
    }

    /// 全セルの電荷総和 (検証用の真値)
    pub fn total_charge(&self) -> f64 {
        self.cells.iter().sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn charge_sharing_telescopes() {
        // 順次活性化と一括活性化が同じ結果 (電荷保存の telescoping)
        let mut bl = Bitline::new(3, CBL_DEFAULT);
        bl.set_cell(0, 1.0);
        bl.set_cell(1, 0.8);
        bl.set_cell(2, 0.5);
        // V_3 = (7·0.6 + 1.0+0.8+0.5)/(7+3) = (4.2+2.3)/10 = 0.65
        let v3 = bl.shared_voltage(3);
        assert!((v3 - 0.65).abs() < 1e-9, "V_3={}", v3);
    }

    #[test]
    fn empty_cells_never_sense() {
        let bl = Bitline::new(10, CBL_DEFAULT);
        // 全セル 0 → ビット線は Vpre 以下に留まる → SA flip しない
        assert_eq!(bl.thermometer_read(), None);
    }

    #[test]
    fn charged_cells_sense_early() {
        let mut bl = Bitline::new(10, CBL_DEFAULT);
        for i in 0..10 { bl.set_cell(i, VDD); }  // 全セル満充電
        // 全セル VDD なら平均 VDD > Vref → N=1 で flip
        assert_eq!(bl.thermometer_read(), Some(1));
    }

    #[test]
    fn pwm_write_monotonic() {
        let mut bl = Bitline::new(1, CBL_DEFAULT);
        bl.write_pwm(0, 0.3);
        let low = bl.cells[0];
        let mut bl2 = Bitline::new(1, CBL_DEFAULT);
        bl2.write_pwm(0, 1.0);
        let high = bl2.cells[0];
        assert!(high > low, "longer pulse → more charge");
    }
}
