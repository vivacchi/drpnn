//! DRAM セルとしてのニューロン (STAGE2GAMMA V2 §1 原理 1, 2, 4)
//!
//! 物理対応:
//!   - charge:        キャパシタ電荷 (整数 0..CHARGE_MAX、 0 = 完全放電、 MAX = VDD)
//!   - vth_threshold: アクセストランジスタ Vth = 発火閾値 (製造時固定、 セルごと個別)
//!   - leak_per_step: 自然な電荷漏れ (DDR4 64ms 保持時間に対応、 LIF の τ_m)
//!
//! 単位 (実機との対応、 整数演算で表現):
//!   - CHARGE_MAX = 1000 (= VDD、 完全充電)
//!   - VTH_MEAN   = 600  (約 0.6 VDD、 平均的なニューロン)
//!   - VTH_STD    = 80   (Gaussian 標準偏差、 製造ばらつき)
//!   - LEAK_PER_STEP = 5 (約 0.5%/step、 60 step (=30ms) で半減程度)
//!
//! 1 step = 0.5ms (Phase 2 f と同じ DT)

use rand::prelude::*;
use rand_distr::Normal;

/// キャパシタ電荷の最大値 (= VDD)
pub const CHARGE_MAX: i32 = 1000;
/// Vth Gaussian 分布の平均
pub const VTH_MEAN: i32 = 600;
/// Vth Gaussian 分布の標準偏差 (製造ばらつき)
pub const VTH_STD: i32 = 80;
/// Vth の範囲下限 (Gaussian の裾を切り詰める)
pub const VTH_MIN: i32 = 300;
/// Vth の範囲上限
pub const VTH_MAX: i32 = 900;
/// 1 step あたりの自然リーク量
pub const LEAK_PER_STEP: i32 = 5;
/// 発火後の spike_trace 値 (STDP 因果窓 = 160 step = 80ms)
pub const SPIKE_TRACE_INIT: i32 = 160;

#[derive(Clone, Debug)]
pub struct DramCell {
    /// キャパシタ電荷 (膜電位の物理表現)
    pub charge: i32,

    /// 発火閾値 (Vth、 製造時固定 = 個性)
    pub vth_threshold: i32,

    /// 最終発火時刻 (STDP 用、 V2 §3.4 簡略化に従い別領域として保持)
    pub last_spike_time: i32,

    /// 発火直後の経過カウンタ (Phase 2 と同じ spike_trace)
    pub spike_trace: i32,

    /// 入力ニューロンか (true なら外部入力のみ受け、 reset_state でクリア)
    pub is_input: bool,

    /// 抑制性か (V2 §9.2.g 開放問題: 興奮側に negative weight 加算で表現)
    pub is_inhibitory: bool,
}

impl DramCell {
    /// 指定 Vth でセル作成
    pub fn new_with_vth(vth: i32) -> Self {
        Self {
            charge: 0,
            vth_threshold: vth.clamp(VTH_MIN, VTH_MAX),
            last_spike_time: i32::MIN,
            spike_trace: 0,
            is_input: false,
            is_inhibitory: false,
        }
    }

    /// Gaussian 分布からランダムに Vth をサンプリングしてセル作成
    /// (実機の製造ばらつきをシミュレート)
    pub fn new_random_vth(rng: &mut StdRng) -> Self {
        let normal = Normal::new(VTH_MEAN as f64, VTH_STD as f64).unwrap();
        let vth = normal.sample(rng).round() as i32;
        Self::new_with_vth(vth)
    }

    /// 入力ニューロン版 (Vth は低めに設定して刺激に応答しやすく)
    pub fn new_input(rng: &mut StdRng) -> Self {
        let mut cell = Self::new_random_vth(rng);
        cell.vth_threshold = (cell.vth_threshold * 7) / 10;  // 平均的に 70% に下げる
        cell.is_input = true;
        cell
    }

    /// 抑制性ニューロン版
    pub fn new_inhibitory(rng: &mut StdRng) -> Self {
        let mut cell = Self::new_random_vth(rng);
        cell.vth_threshold = (cell.vth_threshold * 8) / 10;  // 平均的に 80% に下げる (抑制性は閾値低め)
        cell.is_inhibitory = true;
        cell
    }

    /// 1 step 進行
    ///
    /// input_charge: シナプス入力による電荷加算 (正 or 負、 V2 §9.2.g 開放問題への暫定対応)
    /// current_time: グローバル SNN クロック
    ///
    /// 戻り値: 発火したか
    pub fn update(&mut self, input_charge: i32, current_time: i32) -> bool {
        if self.spike_trace > 0 {
            self.spike_trace -= 1;
        }

        // 1. シナプス入力を電荷に加算
        self.charge = self.charge.saturating_add(input_charge);

        // 2. 自然リーク (DRAM 物理: キャパシタからの自然電荷漏れ)
        self.charge = self.charge.saturating_sub(LEAK_PER_STEP);
        if self.charge < 0 {
            self.charge = 0;
        }
        if self.charge > CHARGE_MAX {
            self.charge = CHARGE_MAX;
        }

        // 3. 発火判定: Vth 超え (= センスアンプが ON と判定)
        if self.charge >= self.vth_threshold {
            // 破壊読み出し: 発火後はキャパシタが放電
            self.charge = 0;
            self.last_spike_time = current_time;
            self.spike_trace = SPIKE_TRACE_INIT;
            true
        } else {
            false
        }
    }

    /// 試行間リセット (Phase 2 reset_trial_state と同等)
    pub fn reset_state(&mut self) {
        self.charge = 0;
        self.last_spike_time = i32::MIN;
        self.spike_trace = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cell_new_with_vth_clamps() {
        let c1 = DramCell::new_with_vth(50);    // below MIN
        assert_eq!(c1.vth_threshold, VTH_MIN);
        let c2 = DramCell::new_with_vth(2000);  // above MAX
        assert_eq!(c2.vth_threshold, VTH_MAX);
        let c3 = DramCell::new_with_vth(500);
        assert_eq!(c3.vth_threshold, 500);
    }

    #[test]
    fn cell_random_vth_diversity() {
        // 1000 個サンプリングして範囲と分布を確認
        let mut rng = StdRng::seed_from_u64(42);
        let cells: Vec<DramCell> = (0..1000)
            .map(|_| DramCell::new_random_vth(&mut rng))
            .collect();

        // 全部 VTH_MIN..=VTH_MAX 範囲
        for c in &cells {
            assert!(c.vth_threshold >= VTH_MIN);
            assert!(c.vth_threshold <= VTH_MAX);
        }

        // 平均が VTH_MEAN 近く
        let sum: i64 = cells.iter().map(|c| c.vth_threshold as i64).sum();
        let mean = sum as f64 / cells.len() as f64;
        assert!((mean - VTH_MEAN as f64).abs() < 20.0,
            "Vth mean {} should be near {}", mean, VTH_MEAN);

        // 標準偏差が VTH_STD 近く
        let var: f64 = cells.iter()
            .map(|c| { let d = c.vth_threshold as f64 - mean; d * d })
            .sum::<f64>() / cells.len() as f64;
        let std = var.sqrt();
        assert!((std - VTH_STD as f64).abs() < 20.0,
            "Vth std {} should be near {}", std, VTH_STD);
    }

    #[test]
    fn cell_leak_decays_charge() {
        let mut c = DramCell::new_with_vth(500);
        c.charge = 100;
        // 入力なし、 leak のみ
        for _ in 0..20 {
            c.update(0, 0);
        }
        // 20 step × LEAK_PER_STEP = 100 → 0
        assert_eq!(c.charge, 0);
    }

    #[test]
    fn cell_fires_at_threshold() {
        let mut c = DramCell::new_with_vth(500);
        // 一度の入力で閾値超え
        let fired = c.update(600, 100);
        assert!(fired);
        assert_eq!(c.charge, 0);  // 破壊読み出し
        assert_eq!(c.last_spike_time, 100);
        assert_eq!(c.spike_trace, SPIKE_TRACE_INIT);
    }

    #[test]
    fn cell_accumulates_below_threshold() {
        let mut c = DramCell::new_with_vth(500);
        c.charge = 0;
        // leak を相殺する程度の小さい入力
        for _ in 0..50 {
            c.update(10, 0);  // 10 - 5 = 5 net gain per step
        }
        // 50 × 5 = 250、 まだ閾値 500 未満
        assert!(c.charge >= 200 && c.charge < 500);
    }

    #[test]
    fn diverse_vth_gives_diverse_firing() {
        // 同じ入力刺激に対して、 Vth 多様性で発火頻度が変わることを確認
        let mut rng = StdRng::seed_from_u64(123);
        let mut cells: Vec<DramCell> = (0..50)
            .map(|_| DramCell::new_random_vth(&mut rng))
            .collect();

        let mut fire_counts = vec![0u32; cells.len()];
        // 200 step 一定刺激
        for t in 0..200 {
            for (i, c) in cells.iter_mut().enumerate() {
                if c.update(20, t) {  // 入力 20、 leak 5 = net +15/step
                    fire_counts[i] += 1;
                }
            }
        }

        // 発火回数に多様性があるはず (全部同じではない)
        let min_fires = *fire_counts.iter().min().unwrap();
        let max_fires = *fire_counts.iter().max().unwrap();
        assert!(max_fires > min_fires,
            "Vth diversity should produce diverse firing rates, but min=max={}", min_fires);
        // 大体の細胞が発火する (一定刺激下で)
        let active_count = fire_counts.iter().filter(|&&c| c > 0).count();
        assert!(active_count > cells.len() / 2,
            "Most cells should fire at least once, only {}/{} fired",
            active_count, cells.len());
    }
}
