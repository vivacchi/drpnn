//! 熱力学的シナプス 3D 版 (Phase 3)
//!
//! src_phase2_f/thermo_synapse.rs から派生。
//! シナプスは位置を持たない (pre/post idx のみ) ので、 2D 版と完全に同一の物理プロセス。
//! 3D 化に伴うコード変更は無し (位置非依存)。
//!
//! 別ファイルにする理由:
//!   - phase2_f 側と独立に進化させる余地を残す (3D 特有のパラメータ調整など)
//!   - phase3_3d モジュールを self-contained に保つ

/// 因果窓 (step 単位、 0.5ms/step → 80ms = 160 step)
pub const CAUSAL_WINDOW: i32 = 160;
/// LTP 量
pub const LTP_AMOUNT: i32 = 5;
/// LTD 量 (LTD > LTP で安定)
pub const LTD_AMOUNT: i32 = 6;
/// conductance 上限
pub const CONDUCTANCE_MAX: i32 = 100;
/// 開放判定閾値
pub const OPEN_THRESHOLD: i32 = 30;
/// conductance 自然減衰の周期
pub const DECAY_INTERVAL: i32 = 1000;

/// vitality 初期値
pub const VITALITY_INITIAL: i32 = 100;
/// vitality 上限
pub const VITALITY_MAX: i32 = 200;
/// 信号通過 1 回あたりの vitality 増加
pub const VITALITY_GAIN: i32 = 1;
/// vitality 自然減衰の周期
pub const VITALITY_DECAY_INTERVAL: i32 = 10000;

#[derive(Clone, Debug)]
pub struct ThermoSynapse3d {
    pub pre: usize,
    pub post: usize,
    pub delay: i32,
    // ─── 機能的可塑性 (短中期、STDP で変化) ───
    pub conductance: i32,
    // ─── 構造的可塑性 (長期、使用頻度ベース) ───
    pub vitality: i32,
    pub alive: bool,
    // ─── 内部カウンタ ───
    pub decay_counter: i32,
    pub vitality_counter: i32,
}

impl ThermoSynapse3d {
    pub fn new(pre: usize, post: usize, delay: i32, conductance: i32) -> Self {
        Self {
            pre,
            post,
            delay,
            conductance,
            vitality: VITALITY_INITIAL,
            alive: VITALITY_INITIAL > 0,
            decay_counter: 0,
            vitality_counter: 0,
        }
    }

    pub fn update_on_post_spike_trace(&mut self, pre_spike_trace: i32) {
        if !self.alive { return; }
        if pre_spike_trace > 0 && pre_spike_trace < CAUSAL_WINDOW {
            self.conductance += LTP_AMOUNT;
            if self.conductance > CONDUCTANCE_MAX { self.conductance = CONDUCTANCE_MAX; }
        }
    }

    pub fn update_on_pre_spike_trace(&mut self, post_spike_trace: i32) {
        if !self.alive { return; }
        if post_spike_trace > 0 && post_spike_trace < CAUSAL_WINDOW {
            self.conductance -= LTD_AMOUNT;
            if self.conductance < 0 { self.conductance = 0; }
        }
    }

    pub fn on_transmission(&mut self) {
        self.vitality += VITALITY_GAIN;
        if self.vitality > VITALITY_MAX { self.vitality = VITALITY_MAX; }
    }

    pub fn decay(&mut self) {
        if !self.alive { return; }
        self.decay_counter += 1;
        if self.decay_counter >= DECAY_INTERVAL {
            self.decay_counter = 0;
            if self.conductance > 0 {
                self.conductance -= 1;
            }
        }
        self.vitality_counter += 1;
        if self.vitality_counter >= VITALITY_DECAY_INTERVAL {
            self.vitality_counter = 0;
            if self.vitality > 0 {
                self.vitality -= 1;
            }
        }
    }

    pub fn update_alive(&mut self) {
        self.alive = self.vitality > 0;
    }

    pub fn can_transmit(&self) -> bool {
        self.alive
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_synapse_default_state() {
        let s = ThermoSynapse3d::new(0, 1, 4, 50);
        assert!(s.alive);
        assert!(s.can_transmit());
        assert_eq!(s.vitality, VITALITY_INITIAL);
    }

    #[test]
    fn ltp_on_causal_pair() {
        let mut s = ThermoSynapse3d::new(0, 1, 4, 50);
        s.update_on_post_spike_trace(80);
        assert_eq!(s.conductance, 55);
    }

    #[test]
    fn ltd_on_anticausal_pair() {
        let mut s = ThermoSynapse3d::new(0, 1, 4, 50);
        s.update_on_pre_spike_trace(80);
        assert_eq!(s.conductance, 44);
    }
}
