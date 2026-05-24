//! Izhikevich neuron model.
//!
//! 連続時間方程式:
//!     dv/dt = 0.04 v² + 5v + 140 - u + I
//!     du/dt = a (b v - u)
//!     if v >= 30 then v ← c, u ← u + d
//!
//! 興奮性(RS = regular spiking): a=0.02, b=0.2, c=-65, d=8
//! 抑制性(FS = fast spiking)    : a=0.10, b=0.2, c=-65, d=2

#[derive(Clone, Copy, Debug)]
pub enum NeuronKind {
    /// 入力(視床)ニューロン: 外部刺激で発火、可塑性なし
    Input,
    /// 興奮性皮質ニューロン: STDP の対象
    ExcitatoryCortex,
    /// 抑制性皮質ニューロン: 結合は固定
    InhibitoryCortex,
}

#[derive(Clone, Copy, Debug)]
pub struct Neuron {
    pub kind: NeuronKind,
    /// 膜電位
    pub v: f64,
    /// 回復変数
    pub u: f64,
    pub a: f64,
    pub b: f64,
    pub c: f64,
    pub d: f64,
    /// 最終発火時刻 (ms). `f64::NEG_INFINITY` は未発火
    pub last_spike: f64,
}

impl Neuron {
    pub fn new(kind: NeuronKind) -> Self {
        let (a, b, c, d) = match kind {
            // Input は cortex の RS と同じ動力学で良い(駆動するだけ)
            NeuronKind::Input | NeuronKind::ExcitatoryCortex => (0.02, 0.2, -65.0, 8.0),
            NeuronKind::InhibitoryCortex => (0.10, 0.2, -65.0, 2.0),
        };
        let v = -65.0;
        let u = b * v;
        Self { kind, v, u, a, b, c, d, last_spike: f64::NEG_INFINITY }
    }

    /// 1ステップ進める。`true` を返したら発火。
    /// 数値安定性のため v は半ステップ × 2 で更新する。
    pub fn step(&mut self, dt: f64, current: f64, t_now: f64) -> bool {
        // v half-step 1
        let dv1 = 0.04 * self.v * self.v + 5.0 * self.v + 140.0 - self.u + current;
        self.v += 0.5 * dt * dv1;
        // v half-step 2
        let dv2 = 0.04 * self.v * self.v + 5.0 * self.v + 140.0 - self.u + current;
        self.v += 0.5 * dt * dv2;
        // u full step
        self.u += dt * self.a * (self.b * self.v - self.u);

        if self.v >= 30.0 {
            self.v = self.c;
            self.u += self.d;
            self.last_spike = t_now;
            true
        } else {
            false
        }
    }
}
