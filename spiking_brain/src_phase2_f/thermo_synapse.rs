//! 熱力学的シナプス Fork F-G1-R1 (発生学的・原理厳格版、Tier 2 整理済 2026-05-25)
//!
//! Fork F の核心変更:
//!   1. is_inhibitory 削除 (デールの原則: pre neuron の属性を継承)
//!   2. plastic 削除 (全シナプス可塑性、固定/可塑の区別なし)
//!   3. vitality 追加 (構造的可塑性: 使用頻度ベースの存続)
//!   4. exists 削除 (F-fix: 信号配送は alive のみで判定、conductance は強度のみ)
//!
//! R1 の追加:
//!   - LTD 復活 (反因果ペアで弱化、Bi & Poo 1998 / Song et al. 2000)
//!   - LTD_AMOUNT > LTP_AMOUNT で安定 (シナプス強度の発散を防ぐ)
//!
//! 2 層の可塑性 (生物学的に分離):
//!   - 機能的可塑性 (短中期): conductance (STDP +/- で変化、0-100)
//!   - 構造的可塑性 (長期): vitality, alive (使用頻度ベース)
//!
//! 信号オン/オフ判定 (F-fix): alive のみで判定
//!   conductance は信号強度のアナログ多値 (低くても弱信号は流れる)

/// 因果窓 (step 単位、0.5ms/step → 80ms = 160 step)
pub const CAUSAL_WINDOW: i32 = 160;
/// LTP 量 (因果ペアで conductance += LTP_AMOUNT)
pub const LTP_AMOUNT: i32 = 5;
/// LTD 量 (R1 復活: 生物的に確立された反因果弱化、Bi & Poo 1998 / Song et al. 2000)
/// LTD > LTP で安定 (シナプス強度の発散を防ぐ)
pub const LTD_AMOUNT: i32 = 6;
/// conductance 上限
pub const CONDUCTANCE_MAX: i32 = 100;
/// 開放判定閾値
pub const OPEN_THRESHOLD: i32 = 30;
/// conductance 自然減衰の周期 (N step に 1 回 -1)
/// S3 スケーリング: 50 → 1000 (20倍緩和、機能可塑性 vs 構造可塑性の比率 1:10 を維持)
/// 1000 step = 500ms に 1 回 -1、生物の AMPA receptor turnover (短中期) に近いスケール
pub const DECAY_INTERVAL: i32 = 1000;

/// Fork F: vitality 初期値 (新生シナプスのスタート値)
pub const VITALITY_INITIAL: i32 = 100;
/// vitality 上限
pub const VITALITY_MAX: i32 = 200;
/// 信号通過 1 回あたりの vitality 増加
pub const VITALITY_GAIN: i32 = 1;
/// vitality 自然減衰の周期 (N step に 1 回 -1)
/// S3 スケーリング (1:10000): 10,000 step = 5 秒に 1 回 -1
/// → vitality 消失まで: 100 × 10,000 = 1×10⁶ step ≒ 1,700 trial で「使われない結線」が刈り込まれる
/// 生物の数日 (数百万 step) を Phase 2 では数千 trial に圧縮、相関関係は維持
///
/// 試行: 30,000 (3 倍緩和) でも最終動的平衡点は同じ (pruned=11784, active=10) を確認。
/// 時間スケールは「学習速度」を変えるが「平衡点」は変えない、という Prigogine 散逸構造論の予測通り
pub const VITALITY_DECAY_INTERVAL: i32 = 10000;

/// 熱力学的シナプス Fork F-fix (発生学的・原理厳格版)
///
/// F-fix の核心: 信号配送の 1 ビット判定は alive (vitality ベース) のみ。
/// conductance は信号強度のアナログ多値 (0-100)、配送判定には使わない。
/// これにより「機能 (conductance) と存続 (vitality) を真に分離」する。
///
/// 生物学的解釈:
///   - alive=true (vitality > 0): シナプスが「存在している」(まだ刈り込まれていない)
///   - conductance: 「神経伝達物質の放出効率」(低くても放出はされる)
///   - 信号 = conductance / 10 (conductance=20 でも信号=2 が流れる)
#[derive(Clone, Debug)]
pub struct ThermoSynapse {
    pub pre: usize,
    pub post: usize,
    pub delay: i32,
    // ─── 機能的可塑性 (短中期、STDP で変化) ───
    /// 熱伝導度 (信号伝達効率、整数 0-100)
    /// この値がそのまま信号強度の元 (低くても弱い信号は流れる)
    pub conductance: i32,
    // ─── 構造的可塑性 (長期、使用頻度ベース) ───
    /// シナプスの「生きてる度」(使用頻度に応じた存続値)
    /// 信号通過で +1、時間で -1、>0 で alive=true
    pub vitality: i32,
    /// シナプスの生死 = 信号配送の 1 ビット判定 (vitality > 0 で true)
    pub alive: bool,
    // ─── 内部カウンタ ───
    /// 自然減衰用カウンタ (DECAY_INTERVAL ごとに conductance -= 1)
    pub decay_counter: i32,
    /// vitality 減衰用カウンタ (VITALITY_DECAY_INTERVAL ごとに vitality -= 1)
    pub vitality_counter: i32,
    /// 因果窓 (この値より小さい spike_trace が「最近発火」 と判定される)
    /// M1: 160 step (80ms)、 M2: 320 step (160ms 音節スケール)
    pub causal_window: i32,
}

impl ThermoSynapse {
    /// 新規シナプス (発生時、または軸索成長時)
    /// 全シナプスが同じ機構で生まれる (Fork F: fixed/plastic の区別なし)
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
            causal_window: CAUSAL_WINDOW,  // M1 default、 M2 では 320 に書き換える
        }
    }

    /// post 発火時の STDP+ (LTP):
    /// pre が直近に発火していたら因果 → conductance を上げる
    pub fn update_on_post_spike_trace(&mut self, pre_spike_trace: i32) {
        if !self.alive { return; }
        if pre_spike_trace > 0 && pre_spike_trace < self.causal_window {
            self.conductance += LTP_AMOUNT;
            if self.conductance > CONDUCTANCE_MAX { self.conductance = CONDUCTANCE_MAX; }
        }
    }

    /// pre 発火時の STDP- (LTD): R1 復活
    /// post が直近に発火していたら反因果 → conductance を下げる
    /// Bi & Poo 1998 で実験的に確立された機構、生物的に必要
    pub fn update_on_pre_spike_trace(&mut self, post_spike_trace: i32) {
        if !self.alive { return; }
        if post_spike_trace > 0 && post_spike_trace < self.causal_window {
            self.conductance -= LTD_AMOUNT;
            if self.conductance < 0 { self.conductance = 0; }
        }
    }

    /// 信号通過時の vitality 増加 (生物の use-dependent maintenance)
    /// 実際に信号が流れたシナプスの存続値が上がる
    pub fn on_transmission(&mut self) {
        self.vitality += VITALITY_GAIN;
        if self.vitality > VITALITY_MAX { self.vitality = VITALITY_MAX; }
    }

    /// 毎クロックの自然減衰 (conductance と vitality 両方)
    pub fn decay(&mut self) {
        if !self.alive { return; }
        // conductance の自然減衰 (機能的弱化、ただし 0 までは下がる)
        self.decay_counter += 1;
        if self.decay_counter >= DECAY_INTERVAL {
            self.decay_counter = 0;
            if self.conductance > 0 {
                self.conductance -= 1;
            }
        }
        // vitality の自然減衰 (構造的忘却)
        self.vitality_counter += 1;
        if self.vitality_counter >= VITALITY_DECAY_INTERVAL {
            self.vitality_counter = 0;
            if self.vitality > 0 {
                self.vitality -= 1;
            }
        }
    }

    /// alive の自動更新 (vitality ベース)
    pub fn update_alive(&mut self) {
        self.alive = self.vitality > 0;
    }

    /// 信号配送可能か (F-fix: alive のみで判定)
    pub fn can_transmit(&self) -> bool {
        self.alive
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_synapse_default_state() {
        let s = ThermoSynapse::new(0, 1, 4, 50);
        assert!(s.alive);
        assert!(s.can_transmit());
        assert_eq!(s.vitality, VITALITY_INITIAL);
    }

    #[test]
    fn ltp_on_causal_pair() {
        let mut s = ThermoSynapse::new(0, 1, 4, 50);
        s.update_on_post_spike_trace(80);
        assert_eq!(s.conductance, 55);
    }

    #[test]
    fn vitality_increases_on_transmission() {
        let mut s = ThermoSynapse::new(0, 1, 4, 50);
        let v0 = s.vitality;
        s.on_transmission();
        assert_eq!(s.vitality, v0 + VITALITY_GAIN);
    }
}
