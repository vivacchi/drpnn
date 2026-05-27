//! PWM 重みシナプス (STAGE2GAMMA V2 §1 原理 3, §3.2 行間転写の物理)
//!
//! 物理対応:
//!   - pulse_width:  WL ON 時間 (整数 ns、 シナプス重みの物理表現)
//!   - charge_added: pulse_width に応じてキャパシタへ加算される電荷量
//!   - delay:        軸索遅延 (行アドレスのオフセット = step 数)
//!   - last_spike_time ベースの STDP (V2 §3.4 簡略化)
//!
//! 重みの物理意味:
//!   - pulse_width 短い (例: 10) → わずかな充電 (charge += 10)
//!   - pulse_width 長い (例: 100) → 大きな充電 (charge += 100、 ただし CHARGE_MAX で飽和)
//!   - 同じセルに複数の入力 → 累積充電 (LIF の空間統合)
//!
//! V2 設計とは違って、 ここでは pulse_width = charge_added の単純な線形対応にしている。
//! 実機ではキャパシタ充電カーブが非線形 (指数飽和) なので、 必要なら後で精緻化。

/// 因果窓 (STDP 用、 Phase 2 と同じ 160 step = 80ms)
pub const CAUSAL_WINDOW: i32 = 160;
/// LTP 量 (pulse_width に加算)
pub const LTP_AMOUNT: i32 = 5;
/// LTD 量 (pulse_width から減算、 Phase 2 fork F-G1-R1 と同じ LTD > LTP)
pub const LTD_AMOUNT: i32 = 6;
/// pulse_width の最大値 (= CHARGE_MAX に近い値、 ただし飽和回避で 100 程度)
pub const PULSE_WIDTH_MAX: i32 = 100;
/// pulse_width 自然減衰の周期 (Phase 2 DECAY_INTERVAL と同じ)
pub const DECAY_INTERVAL: i32 = 1000;
/// 開放判定の閾値 (低 pulse_width のシナプスは vitality 切れで死ぬ)
pub const OPEN_THRESHOLD: i32 = 30;

/// vitality 初期値 (Phase 2 fork F と同じ)
pub const VITALITY_INITIAL: i32 = 100;
pub const VITALITY_MAX: i32 = 200;
pub const VITALITY_GAIN: i32 = 1;
pub const VITALITY_DECAY_INTERVAL: i32 = 10000;

#[derive(Clone, Debug)]
pub struct DramSynapse {
    pub pre: usize,
    pub post: usize,
    /// 軸索遅延 (V2 §2.3 のリングバッファ行アドレスオフセット)
    pub delay: i32,
    /// PWM パルス幅 (シナプス重み、 0..PULSE_WIDTH_MAX)
    pub pulse_width: i32,
    /// 構造的可塑性 (Phase 2 fork F と同じ vitality)
    pub vitality: i32,
    /// シナプス生存判定
    pub alive: bool,
    /// pulse_width 減衰カウンタ
    pub decay_counter: i32,
    /// vitality 減衰カウンタ
    pub vitality_counter: i32,
}

impl DramSynapse {
    pub fn new(pre: usize, post: usize, delay: i32, pulse_width: i32) -> Self {
        Self {
            pre,
            post,
            delay,
            pulse_width: pulse_width.clamp(0, PULSE_WIDTH_MAX),
            vitality: VITALITY_INITIAL,
            alive: true,
            decay_counter: 0,
            vitality_counter: 0,
        }
    }

    /// pulse_width → charge_added 変換
    /// 単純線形: charge_added = pulse_width (実機では非線形だが POC では十分)
    pub fn charge_per_spike(&self) -> i32 {
        if !self.alive { return 0; }
        self.pulse_width
    }

    /// LTP (post 発火時に pre が直近に発火していたら強化)
    pub fn update_on_post_spike(&mut self, pre_spike_trace: i32) {
        if !self.alive { return; }
        if pre_spike_trace > 0 && pre_spike_trace < CAUSAL_WINDOW {
            self.pulse_width = (self.pulse_width + LTP_AMOUNT).min(PULSE_WIDTH_MAX);
        }
    }

    /// LTD (pre 発火時に post が直近に発火していたら弱化、 反因果)
    pub fn update_on_pre_spike(&mut self, post_spike_trace: i32) {
        if !self.alive { return; }
        if post_spike_trace > 0 && post_spike_trace < CAUSAL_WINDOW {
            self.pulse_width = (self.pulse_width - LTD_AMOUNT).max(0);
        }
    }

    /// 信号通過時の vitality 増加 (Phase 2 use-dependent maintenance)
    pub fn on_transmission(&mut self) {
        self.vitality = (self.vitality + VITALITY_GAIN).min(VITALITY_MAX);
    }

    /// 1 step 経過: 自然減衰
    pub fn decay(&mut self) {
        if !self.alive { return; }
        self.decay_counter += 1;
        if self.decay_counter >= DECAY_INTERVAL {
            self.decay_counter = 0;
            if self.pulse_width > 0 { self.pulse_width -= 1; }
        }
        self.vitality_counter += 1;
        if self.vitality_counter >= VITALITY_DECAY_INTERVAL {
            self.vitality_counter = 0;
            if self.vitality > 0 { self.vitality -= 1; }
        }
    }

    /// alive 自動更新 (vitality > 0 で生存)
    pub fn update_alive(&mut self) {
        self.alive = self.vitality > 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn synapse_new_clamps_pulse_width() {
        let s = DramSynapse::new(0, 1, 4, 150);
        assert_eq!(s.pulse_width, PULSE_WIDTH_MAX);
        let s2 = DramSynapse::new(0, 1, 4, 50);
        assert_eq!(s2.pulse_width, 50);
    }

    #[test]
    fn charge_per_spike_returns_pulse_width() {
        let s = DramSynapse::new(0, 1, 4, 60);
        assert_eq!(s.charge_per_spike(), 60);
    }

    #[test]
    fn charge_per_spike_zero_when_dead() {
        let mut s = DramSynapse::new(0, 1, 4, 50);
        s.alive = false;
        assert_eq!(s.charge_per_spike(), 0);
    }

    #[test]
    fn ltp_on_causal_pair() {
        let mut s = DramSynapse::new(0, 1, 4, 50);
        s.update_on_post_spike(80);  // pre 発火 trace=80 (因果窓内)
        assert_eq!(s.pulse_width, 55);
    }

    #[test]
    fn ltd_on_anticausal_pair() {
        let mut s = DramSynapse::new(0, 1, 4, 50);
        s.update_on_pre_spike(80);  // post 発火 trace=80 (反因果)
        assert_eq!(s.pulse_width, 44);
    }

    #[test]
    fn vitality_decays_unused_synapse() {
        let mut s = DramSynapse::new(0, 1, 4, 50);
        // 1 回も on_transmission 呼ばずに 100 step
        for _ in 0..VITALITY_DECAY_INTERVAL {
            s.decay();
        }
        s.update_alive();
        // vitality 初期 100 から VITALITY_DECAY_INTERVAL 経過で -1
        assert_eq!(s.vitality, VITALITY_INITIAL - 1);
        assert!(s.alive);
    }
}
