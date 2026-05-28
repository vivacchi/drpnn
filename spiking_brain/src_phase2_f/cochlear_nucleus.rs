//! M0.5 蝸牛神経核 (Cochlear Nucleus)
//!
//! 設計: M0_5_COCHLEAR_NUCLEUS_DESIGN.md
//!
//! M0 蝸牛 (有毛細胞、 20 帯域スパイク) と M1 (A1) の間に欠落していた信号分解ステージ。
//! 3 細胞型 (Octopus / Bushy / Stellate) で音を時間・周波数・オンセットに並列分解。
//!
//! Step 1 (本実装): Octopus 細胞のみ (広帯域オンセット検出)
//!   - 全 20 帯域の同時発火数を見て、 音のオンセット (多帯域同時立ち上がり) を検出
//!   - ThermoNeuron で実装 (entropy 適応がオンセットのみ発火を物理的に生む)
//!
//! Step 2-3 で Bushy (周波数別 transient)、 Stellate (包絡 rate) を追加予定。

use super::thermo_neuron::ThermoNeuron;
use super::cochlea::{N_BANDS, FIRE_CURRENT};

/// Octopus 細胞数 (オンセット感度の異なる 4 細胞)
pub const N_OCTOPUS: usize = 4;
/// 各 Octopus の同時発火閾値 (帯域数): 3, 5, 8, 12 帯域同時で段階的に発火
pub const OCTOPUS_COINCIDENCE_TH: [i32; N_OCTOPUS] = [3, 5, 8, 12];

/// Octopus 細胞用 ThermoNeuron を作る
/// coincidence_th 帯域が同時発火 (= coincidence_th × FIRE_CURRENT の入力) で発火するよう調整
fn make_octopus(coincidence_th: i32) -> ThermoNeuron {
    let mut n = ThermoNeuron::excitatory((0, 0));
    // 閾値 = 同時発火帯域数 × FIRE_CURRENT (例: 3 帯域 → 180)
    n.threshold_base = coincidence_th * FIRE_CURRENT;
    // 中程度の leak: 数 step の積分 (オンセットは数 ms かけて立ち上がる)
    n.leak = 40;
    // エンタルピー: 速い回復で連続検出を許可
    n.enthalpy_max = 10;
    n.enthalpy_recovery_rate = 10;
    // 強い entropy 適応: 一度発火したら閾値上昇 → 持続音では再発火せずオンセットのみ
    n.entropy_per_spike = 80;
    n.entropy_decay_rate = 1;
    n.entropy_decay_interval = 5;  // 80 を 400 step (200ms) で散逸
    // 自発活動なし (純粋な検出器)
    n.spontaneous_input = 0;
    n.generates_entropy = true;
    n
}

#[derive(Clone)]
pub struct CochlearNucleus {
    /// Octopus 細胞 (広帯域オンセット検出)
    pub octopus: Vec<ThermoNeuron>,
    pub current_time: i32,
}

impl CochlearNucleus {
    pub fn new() -> Self {
        let octopus: Vec<ThermoNeuron> = OCTOPUS_COINCIDENCE_TH.iter()
            .map(|&th| make_octopus(th))
            .collect();
        Self { octopus, current_time: 0 }
    }

    /// 1 step 処理
    /// cochlea_out: 蝸牛 20 帯域出力 (各 ch: 発火で FIRE_CURRENT、 他 0)
    /// 戻り値: Octopus 出力 [N_OCTOPUS] (発火で FIRE_CURRENT、 他 0)
    pub fn process_step(&mut self, cochlea_out: &[i32]) -> [i32; N_OCTOPUS] {
        debug_assert_eq!(cochlea_out.len(), N_BANDS);
        // 全帯域の総入力 (同時発火の強度)
        let total_input: i32 = cochlea_out.iter().sum();

        let mut out = [0i32; N_OCTOPUS];
        for (k, oct) in self.octopus.iter_mut().enumerate() {
            // 各 Octopus は全帯域の総和を入力として受ける (広帯域横断)
            if oct.update(total_input, self.current_time) {
                out[k] = FIRE_CURRENT;
            }
        }
        self.current_time += 1;
        out
    }

    /// 試行間リセット (entropy は持ち越し、 Phase 2 と同じ方針)
    pub fn reset(&mut self) {
        for n in &mut self.octopus {
            n.reset_state();
        }
    }
}

impl Default for CochlearNucleus {
    fn default() -> Self { Self::new() }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn octopus_fires_on_broadband_onset() {
        let mut cn = CochlearNucleus::new();
        // 12 帯域同時発火 (強いオンセット) → 全 Octopus が発火するはず
        let mut strong = [0i32; N_BANDS];
        for i in 0..12 { strong[i] = FIRE_CURRENT; }
        let out = cn.process_step(&strong);
        // 少なくとも低閾値 Octopus (th=3,5,8) は発火
        assert!(out[0] > 0, "octopus[0] (th=3) should fire on 12-band onset");
        assert!(out[1] > 0, "octopus[1] (th=5) should fire");
    }

    #[test]
    fn octopus_silent_on_weak_input() {
        let mut cn = CochlearNucleus::new();
        // 2 帯域のみ発火 (弱い) → 高閾値 Octopus は沈黙
        let mut weak = [0i32; N_BANDS];
        weak[0] = FIRE_CURRENT;
        weak[1] = FIRE_CURRENT;
        let out = cn.process_step(&weak);
        // th=12 の Octopus[3] は発火しない (2 帯域では足りない)
        assert_eq!(out[3], 0, "octopus[3] (th=12) should be silent on 2-band input");
    }

    #[test]
    fn octopus_adapts_to_sustained() {
        let mut cn = CochlearNucleus::new();
        // 持続的な広帯域入力 → 最初は発火、 entropy 適応で減少するはず
        let mut strong = [0i32; N_BANDS];
        for i in 0..12 { strong[i] = FIRE_CURRENT; }
        let mut fire_count = 0;
        for _ in 0..200 {
            let out = cn.process_step(&strong);
            if out[0] > 0 { fire_count += 1; }
        }
        // 持続 200 step でも、 適応により発火回数は制限される (全 step 発火しない)
        assert!(fire_count < 200, "octopus should adapt, not fire every step");
        assert!(fire_count > 0, "octopus should fire at least at onset");
    }
}
