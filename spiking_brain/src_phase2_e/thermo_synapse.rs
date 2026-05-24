//! 熱力学的シナプス (ThermoSynapse)
//!
//! 1 ビット結線 + 整数 conductance (熱伝導度)。
//! conductance >= OPEN_THRESHOLD なら自動で exists = true。
//!
//! STDP の熱力学的解釈:
//!   - 因果ペア (pre 先、post 後): 経路がエネルギーをうまく伝えた → conductance += LTP
//!   - 反因果ペア (post 先、pre 後): 経路が役に立たなかった → conductance -= LTD
//!   - 時間経過: 使われない経路は緩やかに減衰 (整数演算実現: decay_counter)

/// 因果窓 (ms 単位ではなく step 単位、0.5ms/step → 80ms = 160 step)
pub const CAUSAL_WINDOW: i32 = 160;
/// LTP 量 (因果ペアで conductance += LTP_AMOUNT)
pub const LTP_AMOUNT: i32 = 5;
/// LTD 量 (反因果ペアで conductance -= LTD_AMOUNT)、LTD > LTP で安定 (Song et al. 2000)
pub const LTD_AMOUNT: i32 = 6;
/// conductance 上限
pub const CONDUCTANCE_MAX: i32 = 100;
/// 開放判定閾値 (conductance >= OPEN_THRESHOLD なら exists = true)
pub const OPEN_THRESHOLD: i32 = 30;
/// 自然減衰の周期 (N step に 1 回 -1)
pub const DECAY_INTERVAL: i32 = 50;

/// 熱力学的シナプス
#[derive(Clone, Debug)]
pub struct ThermoSynapse {
    pub pre: usize,
    pub post: usize,
    pub delay: i32,
    /// 熱伝導度 (信号伝達効率、整数)
    pub conductance: i32,
    /// 結線の存在 (conductance >= OPEN_THRESHOLD で自動 true)
    pub exists: bool,
    pub is_inhibitory: bool,
    /// 可塑性: false なら STDP も減衰も起きない (入力→皮質の固定結線などに使う)
    pub plastic: bool,
    /// 自然減衰用カウンタ (DECAY_INTERVAL step に 1 回 -1 する整数実装)
    pub decay_counter: i32,
}

impl ThermoSynapse {
    /// 新規プラスチック結線 (軸索成長で作られるもの)
    pub fn new_plastic(pre: usize, post: usize, delay: i32, conductance: i32, is_inhibitory: bool) -> Self {
        Self {
            pre,
            post,
            delay,
            conductance,
            exists: conductance >= OPEN_THRESHOLD,
            is_inhibitory,
            plastic: true,
            decay_counter: 0,
        }
    }

    /// 固定結線 (入力→皮質、抑制→興奮など、可塑性なし、最初から open)
    pub fn new_fixed(pre: usize, post: usize, delay: i32, conductance: i32, is_inhibitory: bool) -> Self {
        Self {
            pre,
            post,
            delay,
            conductance,
            exists: true,
            is_inhibitory,
            plastic: false,
            decay_counter: 0,
        }
    }

    /// pre が発火したときの STDP-:
    /// post が直近に発火していたら反因果 → LTD で conductance を下げる
    /// (旧版、last_spike_time ベース、verify_determinism 用に保持)
    pub fn update_on_pre_spike(&mut self, post_last_spike: i32, current_time: i32) {
        if !self.plastic { return; }
        if post_last_spike == i32::MIN { return; }
        let dt = current_time - post_last_spike;
        if dt > 0 && dt < CAUSAL_WINDOW {
            self.conductance -= LTD_AMOUNT;
            if self.conductance < 0 { self.conductance = 0; }
        }
    }

    /// post が発火したときの STDP+:
    /// pre が直近に発火していたら因果 → LTP で conductance を上げる
    /// (旧版、last_spike_time ベース、verify_determinism 用に保持)
    pub fn update_on_post_spike(&mut self, pre_last_spike: i32, current_time: i32) {
        if !self.plastic { return; }
        if pre_last_spike == i32::MIN { return; }
        let dt = current_time - pre_last_spike;
        if dt > 0 && dt < CAUSAL_WINDOW {
            self.conductance += LTP_AMOUNT;
            if self.conductance > CONDUCTANCE_MAX { self.conductance = CONDUCTANCE_MAX; }
        }
    }

    // ─── B4 新版: spike_trace ベースの STDP ──────
    // 判定 `0 < trace < CAUSAL_WINDOW` は旧版の `0 < dt < CAUSAL_WINDOW` と等価。
    // trace = CAUSAL_WINDOW は「発火直後 (dt=0)」を意味し、同 step 内 STDP は除外。

    /// pre 発火時 (spike_trace 版): post の spike_trace を見て LTD
    pub fn update_on_pre_spike_trace(&mut self, post_spike_trace: i32) {
        if !self.plastic { return; }
        if post_spike_trace > 0 && post_spike_trace < CAUSAL_WINDOW {
            self.conductance -= LTD_AMOUNT;
            if self.conductance < 0 { self.conductance = 0; }
        }
    }

    /// post 発火時 (spike_trace 版): pre の spike_trace を見て LTP
    pub fn update_on_post_spike_trace(&mut self, pre_spike_trace: i32) {
        if !self.plastic { return; }
        if pre_spike_trace > 0 && pre_spike_trace < CAUSAL_WINDOW {
            self.conductance += LTP_AMOUNT;
            if self.conductance > CONDUCTANCE_MAX { self.conductance = CONDUCTANCE_MAX; }
        }
    }

    /// 自然減衰 (毎クロック呼ばれる)。整数実装: DECAY_INTERVAL クロックに 1 回 -1。
    pub fn decay(&mut self) {
        if !self.plastic { return; }
        self.decay_counter += 1;
        if self.decay_counter >= DECAY_INTERVAL {
            self.decay_counter = 0;
            if self.conductance > 0 {
                self.conductance -= 1;
            }
        }
    }

    /// conductance に基づいて exists を自動更新
    pub fn update_existence(&mut self) {
        self.exists = self.conductance >= OPEN_THRESHOLD;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ltp_on_causal_pair() {
        let mut s = ThermoSynapse::new_plastic(0, 1, 4, 50, false);
        // pre が時刻 10 に発火、post が時刻 20 に発火 → 因果
        s.update_on_post_spike(10, 20);
        assert_eq!(s.conductance, 55); // 50 + LTP_AMOUNT(5)
    }

    #[test]
    fn ltd_on_anticausal_pair() {
        let mut s = ThermoSynapse::new_plastic(0, 1, 4, 50, false);
        // post が時刻 10 に発火、pre が時刻 20 に発火 → 反因果
        s.update_on_pre_spike(10, 20);
        assert_eq!(s.conductance, 44); // 50 - LTD_AMOUNT(6)
    }

    #[test]
    fn no_update_outside_window() {
        let mut s = ThermoSynapse::new_plastic(0, 1, 4, 50, false);
        // dt = 200 > CAUSAL_WINDOW (160) なので更新なし
        s.update_on_post_spike(0, 200);
        assert_eq!(s.conductance, 50);
    }

    #[test]
    fn fixed_synapse_no_plasticity() {
        let mut s = ThermoSynapse::new_fixed(0, 1, 4, 100, false);
        s.update_on_post_spike(10, 20);
        assert_eq!(s.conductance, 100);
    }

    #[test]
    fn natural_decay() {
        let mut s = ThermoSynapse::new_plastic(0, 1, 4, 50, false);
        for _ in 0..DECAY_INTERVAL {
            s.decay();
        }
        assert_eq!(s.conductance, 49);
    }

    #[test]
    fn existence_auto_update() {
        let mut s = ThermoSynapse::new_plastic(0, 1, 4, 50, false);
        assert!(s.exists);
        s.conductance = 20;
        s.update_existence();
        assert!(!s.exists);
    }
}
