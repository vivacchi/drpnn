//! 1ビット版 SNN: シリコン現実解バージョン。
//!
//! ニューロン: 8bit カウンタ + 閾値発火 (Leaky Integrate-and-Fire)
//!           膜電位は内部状態として保持するが、表現精度は粗い
//! シナプス: 1bit (存在 or 非存在) + 遅延 + eligibility カウンタ (8bit)
//!           「重み」は同じ pre→post 間の冗長結線数で表現可能
//! 学習:    R-STDP の構造的可塑性版
//!           - 因果ペアで eligibility 蓄積
//!           - 正報酬: 高 eligibility の非存在結線を「開く」
//!           - 負報酬: 高 eligibility の存在結線を「閉じる」
//!
//! DRP に直結する設計:
//!   - 全演算が XOR/AND/カウンタなのでハードウェア面積が極小
//!   - 「結線あり/なし」=ルーティングコンフィグそのもの
//!   - 報酬イベント時の構造書き換え = DRP コンフィグレーション書き換え

use crate::trace::OutputTrace;
use rand::prelude::*;
use rand_distr::{Distribution, Uniform};

/// 1ビットニューロン: 蓄積カウンタが閾値を超えたら発火、リセット、不応期
#[derive(Clone, Copy, Debug)]
pub struct BinaryNeuron {
    /// 蓄積カウンタ (0..=255、飽和加算)
    pub counter: u16,
    /// 発火閾値
    pub threshold: u16,
    /// 毎ステップ抜けるリーク量
    pub leak: u16,
    /// 不応期 (発火後このステップ数だけ抑制)
    pub refractory_remaining: u8,
    /// 不応期の長さ (リセット時にここからカウントダウン)
    pub refractory_period: u8,
    /// 最終発火時刻 (ms)
    pub last_spike: f64,
}

impl BinaryNeuron {
    pub fn excitatory() -> Self {
        Self {
            counter: 0,
            threshold: 80,
            leak: 2,
            refractory_remaining: 0,
            refractory_period: 4,
            last_spike: f64::NEG_INFINITY,
        }
    }
    pub fn inhibitory() -> Self {
        Self {
            counter: 0,
            threshold: 40, // 興奮性より発火しやすい (高速制御)
            leak: 3,
            refractory_remaining: 0,
            refractory_period: 2,
            last_spike: f64::NEG_INFINITY,
        }
    }

    /// `arriving_spikes` = このステップに届いた興奮性スパイク数
    /// `inhibitory_spikes` = 抑制性スパイク数 (差し引き)
    pub fn step(&mut self, arriving: u16, inhibition: u16, t: f64) -> bool {
        if self.refractory_remaining > 0 {
            self.refractory_remaining -= 1;
            return false;
        }
        // 累積入力 (飽和算術)
        self.counter = self.counter.saturating_add(arriving * 10);
        // 抑制は引き算 (saturating)
        self.counter = self.counter.saturating_sub(inhibition * 15);
        // リーク
        self.counter = self.counter.saturating_sub(self.leak);

        if self.counter >= self.threshold {
            self.counter = 0;
            self.refractory_remaining = self.refractory_period;
            self.last_spike = t;
            true
        } else {
            false
        }
    }
}

/// 1ビット版シナプス
/// 1ビットモデルでは weight が無いので、「結線が存在するか」と
/// 「タスクへの寄与の eligibility」だけを持つ。
#[derive(Clone, Debug)]
pub struct BinarySynapse {
    pub pre: usize,
    pub post: usize,
    pub delay: usize,
    pub exists: bool,
    pub is_inhibitory: bool,
    pub plastic: bool,
    pub eligibility: f32,
    pub last_elig_update: f64,
}

pub struct BinaryNetworkConfig {
    pub n_cortex: usize,
    pub n_input: usize,
    pub n_output: usize,
    pub input_fanout: usize,
    pub cortex_fanout: usize,
    pub dt_ms: f64,
    pub seed: u64,
}

impl Default for BinaryNetworkConfig {
    fn default() -> Self {
        Self {
            n_cortex: 400,
            n_input: 20,
            n_output: 40,
            input_fanout: 80,
            cortex_fanout: 40,
            dt_ms: 0.5,
            seed: 7,
        }
    }
}

pub struct BinaryStdpParams {
    pub tau_stdp_ms: f64,
    pub tau_e_ms: f64,
    pub r_a_plus: f32,
    /// 報酬で結線を「開く」閾値: eligibility がこれを超え、かつ報酬>0 なら open
    pub open_threshold: f32,
    /// 報酬で結線を「閉じる」閾値
    pub close_threshold: f32,
    pub causal_window_ms: f64,
}

impl Default for BinaryStdpParams {
    fn default() -> Self {
        Self {
            tau_stdp_ms: 20.0,
            tau_e_ms: 1000.0,
            r_a_plus: 0.1,
            open_threshold: 1.5,
            close_threshold: 1.5,
            causal_window_ms: 80.0,
        }
    }
}

pub struct BinaryNetwork {
    pub config: BinaryNetworkConfig,
    pub stdp: BinaryStdpParams,
    pub neurons: Vec<BinaryNeuron>,
    pub synapses: Vec<BinarySynapse>,
    out_syn: Vec<Vec<usize>>,
    in_syn: Vec<Vec<usize>>,
    /// 配送リングバッファ: delivery[slot][neuron] = (excitatory, inhibitory) のカウント
    delivery_exc: Vec<Vec<u16>>,
    delivery_inh: Vec<Vec<u16>>,
    delivery_head: usize,
    max_delay: usize,
    pub input_neurons: Vec<usize>,
    pub output_neurons: Vec<usize>,
    is_inh: Vec<bool>,
    pub t: f64,
    pub noise_rng: StdRng,
    /// 再構成回数 (DRP 上での書き換えイベント数の見積もり用)
    pub reconfig_count: u64,
}

impl BinaryNetwork {
    pub fn new(config: BinaryNetworkConfig, stdp: BinaryStdpParams) -> Self {
        let mut rng = StdRng::seed_from_u64(config.seed);
        let n_exc = (config.n_cortex as f64 * 0.8) as usize;
        let n_inh = config.n_cortex - n_exc;
        let n_total = config.n_input + config.n_cortex;

        let mut neurons: Vec<BinaryNeuron> = Vec::with_capacity(n_total);
        for _ in 0..config.n_input { neurons.push(BinaryNeuron::excitatory()); }
        for _ in 0..n_exc { neurons.push(BinaryNeuron::excitatory()); }
        for _ in 0..n_inh { neurons.push(BinaryNeuron::inhibitory()); }

        let mut is_inh = vec![false; n_total];
        for i in (config.n_input + n_exc)..n_total { is_inh[i] = true; }

        let input_neurons: Vec<usize> = (0..config.n_input).collect();
        let exc_neurons: Vec<usize> = (config.n_input..config.n_input + n_exc).collect();
        let inh_neurons: Vec<usize> = (config.n_input + n_exc..n_total).collect();
        let output_neurons: Vec<usize> = exc_neurons.iter().take(config.n_output).copied().collect();
        let cortex_all: Vec<usize> = exc_neurons.iter().chain(inh_neurons.iter()).copied().collect();

        let mut synapses: Vec<BinarySynapse> = Vec::new();
        let delay_dist = Uniform::from(2..=40usize);

        // (1) 入力 → 皮質 (強駆動、固定)
        for &inp in &input_neurons {
            let targets = sample_wr(&cortex_all, config.input_fanout, &mut rng);
            for &t_ in &targets {
                synapses.push(BinarySynapse {
                    pre: inp, post: t_, delay: rng.gen_range(2..=20),
                    exists: true, is_inhibitory: false, plastic: false,
                    eligibility: 0.0, last_elig_update: 0.0,
                });
            }
        }
        // (2) 興奮性皮質 → 全皮質 (可塑、初期は半分くらい結線)
        for &pre in &exc_neurons {
            let cand: Vec<usize> = cortex_all.iter().copied().filter(|&n| n != pre).collect();
            let targets = sample_wr(&cand, config.cortex_fanout, &mut rng);
            for &tgt in &targets {
                synapses.push(BinarySynapse {
                    pre, post: tgt, delay: delay_dist.sample(&mut rng),
                    exists: rng.gen::<f64>() < 0.5, // 初期は半分
                    is_inhibitory: false, plastic: true,
                    eligibility: 0.0, last_elig_update: 0.0,
                });
            }
        }
        // (3) 抑制性皮質 → 興奮性皮質 (固定、常時開)
        for &pre in &inh_neurons {
            let targets = sample_wr(&exc_neurons, config.cortex_fanout, &mut rng);
            for &tgt in &targets {
                synapses.push(BinarySynapse {
                    pre, post: tgt, delay: 2,
                    exists: true, is_inhibitory: true, plastic: false,
                    eligibility: 0.0, last_elig_update: 0.0,
                });
            }
        }

        let mut out_syn: Vec<Vec<usize>> = vec![Vec::new(); n_total];
        let mut in_syn: Vec<Vec<usize>> = vec![Vec::new(); n_total];
        for (i, s) in synapses.iter().enumerate() {
            out_syn[s.pre].push(i);
            in_syn[s.post].push(i);
        }

        let max_delay = synapses.iter().map(|s| s.delay).max().unwrap_or(2) + 1;
        let delivery_exc = (0..max_delay).map(|_| vec![0u16; n_total]).collect();
        let delivery_inh = (0..max_delay).map(|_| vec![0u16; n_total]).collect();

        Self {
            config, stdp, neurons, synapses,
            out_syn, in_syn,
            delivery_exc, delivery_inh, delivery_head: 0, max_delay,
            input_neurons, output_neurons, is_inh,
            t: 0.0, noise_rng: StdRng::seed_from_u64(99),
            reconfig_count: 0,
        }
    }

    pub fn n_neurons(&self) -> usize { self.neurons.len() }
    pub fn n_synapses(&self) -> usize { self.synapses.len() }
    pub fn n_open_synapses(&self) -> usize {
        self.synapses.iter().filter(|s| s.exists).count()
    }
    pub fn n_plastic_synapses(&self) -> usize {
        self.synapses.iter().filter(|s| s.plastic).count()
    }

    pub fn reset_state(&mut self) {
        for n in &mut self.neurons {
            n.counter = 0;
            n.refractory_remaining = 0;
            n.last_spike = f64::NEG_INFINITY;
        }
        for slot in &mut self.delivery_exc { for x in slot { *x = 0; } }
        for slot in &mut self.delivery_inh { for x in slot { *x = 0; } }
        self.delivery_head = 0;
        self.t = 0.0;
    }

    pub fn set_noise_seed(&mut self, seed: u64) {
        self.noise_rng = StdRng::seed_from_u64(seed);
    }

    /// `ext_pulses[k]` = この時間ステップで input_neurons[k] に注入するスパイク数
    pub fn step(&mut self, ext_pulses: &[u16]) -> Vec<usize> {
        let n = self.neurons.len();

        let mut exc_in = std::mem::take(&mut self.delivery_exc[self.delivery_head]);
        let mut inh_in = std::mem::take(&mut self.delivery_inh[self.delivery_head]);

        for (k, &p) in ext_pulses.iter().enumerate() {
            if k < self.input_neurons.len() {
                exc_in[self.input_neurons[k]] = exc_in[self.input_neurons[k]].saturating_add(p);
            }
        }
        // 視床ノイズ: ベースライン発火を保つ
        for i in 0..n {
            if self.noise_rng.gen::<f64>() < 0.005 {
                exc_in[i] = exc_in[i].saturating_add(1);
            }
        }

        let mut fired = Vec::new();
        for i in 0..n {
            if self.neurons[i].step(exc_in[i], inh_in[i], self.t) {
                fired.push(i);
            }
        }

        for &pre in &fired {
            // 配送 (存在する結線のみ)
            for &s_idx in &self.out_syn[pre] {
                let syn = &self.synapses[s_idx];
                if !syn.exists { continue; }
                let slot = (self.delivery_head + syn.delay) % self.max_delay;
                if syn.is_inhibitory {
                    let v = &mut self.delivery_inh[slot][syn.post];
                    *v = v.saturating_add(1);
                } else {
                    let v = &mut self.delivery_exc[slot][syn.post];
                    *v = v.saturating_add(1);
                }
            }

            // R-STDP: 因果ペア → eligibility 蓄積 (borrow checker のため inline)
            // STDP+: 入ってくる可塑結線、 pre が直近で発火したものに +
            for &s_idx in &self.in_syn[pre] {
                if !self.synapses[s_idx].plastic { continue; }
                let pre_n = self.synapses[s_idx].pre;
                let dt = self.t - self.neurons[pre_n].last_spike;
                if dt > 0.0 && dt < self.stdp.causal_window_ms {
                    let k = (-dt / self.stdp.tau_stdp_ms).exp() as f32;
                    let delta = self.stdp.r_a_plus * k;
                    let t_now = self.t;
                    let tau = self.stdp.tau_e_ms as f32;
                    let s = &mut self.synapses[s_idx];
                    let elapsed = (t_now - s.last_elig_update) as f32;
                    if elapsed > 0.0 {
                        s.eligibility *= (-elapsed / tau).exp();
                    }
                    s.eligibility += delta;
                    s.last_elig_update = t_now;
                }
            }
            // STDP-: 出ていく可塑結線、 post が直近で発火していたものに -
            for &s_idx in &self.out_syn[pre] {
                if !self.synapses[s_idx].plastic { continue; }
                let post_n = self.synapses[s_idx].post;
                let dt = self.t - self.neurons[post_n].last_spike;
                if dt > 0.0 && dt < self.stdp.causal_window_ms {
                    let k = (-dt / self.stdp.tau_stdp_ms).exp() as f32;
                    let delta = -self.stdp.r_a_plus * k;
                    let t_now = self.t;
                    let tau = self.stdp.tau_e_ms as f32;
                    let s = &mut self.synapses[s_idx];
                    let elapsed = (t_now - s.last_elig_update) as f32;
                    if elapsed > 0.0 {
                        s.eligibility *= (-elapsed / tau).exp();
                    }
                    s.eligibility += delta;
                    s.last_elig_update = t_now;
                }
            }
        }

        for x in exc_in.iter_mut() { *x = 0; }
        for x in inh_in.iter_mut() { *x = 0; }
        self.delivery_exc[self.delivery_head] = exc_in;
        self.delivery_inh[self.delivery_head] = inh_in;
        self.delivery_head = (self.delivery_head + 1) % self.max_delay;
        self.t += self.config.dt_ms;
        fired
    }

    fn accumulate_eligibility(&mut self, s_idx: usize, delta: f32) {
        let s = &mut self.synapses[s_idx];
        let elapsed = (self.t - s.last_elig_update) as f32;
        if elapsed > 0.0 {
            s.eligibility *= (-elapsed / self.stdp.tau_e_ms as f32).exp();
        }
        s.eligibility += delta;
        s.last_elig_update = self.t;
    }

    /// 報酬イベント: ここが「DRP コンフィグレーション書き換え相」に相当。
    /// eligibility に応じて結線を open / close する (構造的可塑性)
    pub fn apply_reward(&mut self, r: f32) {
        let mut reconfigs = 0u64;
        for s in &mut self.synapses {
            if !s.plastic { continue; }
            let elapsed = (self.t - s.last_elig_update) as f32;
            let e_now = if elapsed > 0.0 {
                s.eligibility * (-elapsed / self.stdp.tau_e_ms as f32).exp()
            } else {
                s.eligibility
            };
            let signed_e = r * e_now;
            // 正のシグナル (r>0 かつ e>0、または r<0 かつ e<0) → open
            // 負のシグナル → close
            if signed_e > self.stdp.open_threshold && !s.exists {
                s.exists = true;
                reconfigs += 1;
            } else if signed_e < -self.stdp.close_threshold && s.exists {
                s.exists = false;
                reconfigs += 1;
            }
            s.eligibility = 0.0;
            s.last_elig_update = self.t;
        }
        self.reconfig_count += reconfigs;
    }

    pub fn output_index(&self, neuron_id: usize) -> Option<usize> {
        self.output_neurons.iter().position(|&n| n == neuron_id)
    }

    pub fn delay_range_ms(&self) -> (f64, f64) {
        let mn = self.synapses.iter().map(|s| s.delay).min().unwrap_or(0);
        let mx = self.synapses.iter().map(|s| s.delay).max().unwrap_or(0);
        (mn as f64 * self.config.dt_ms, mx as f64 * self.config.dt_ms)
    }

    /// バイト見積もり (DRP 上のメモリ使用量計算用)
    pub fn memory_bytes(&self) -> usize {
        let neuron_bytes = self.neurons.len() * 6;
        let synapse_bytes = self.synapses.len() * 6;
        neuron_bytes + synapse_bytes
    }
}

fn sample_wr<T: Copy>(pool: &[T], k: usize, rng: &mut StdRng) -> Vec<T> {
    let k = k.min(pool.len());
    let mut idx: Vec<usize> = (0..pool.len()).collect();
    let mut picked = Vec::with_capacity(k);
    for i in 0..k {
        let j = rng.gen_range(i..idx.len());
        idx.swap(i, j);
        picked.push(pool[idx[i]]);
    }
    picked
}

pub fn make_pulse_pattern(n_input: usize, seed: u64) -> Vec<f64> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..n_input).map(|_| rng.gen_range(0.0..20.0)).collect()
}

pub fn present_pulse_pattern(
    net: &mut BinaryNetwork,
    pattern_times: &[f64],
    duration_ms: f64,
    pulses_per_event: u16,
    pulse_width_ms: f64,
    trial_seed: u64,
) -> Vec<(usize, f64)> {
    net.reset_state();
    net.set_noise_seed(trial_seed.wrapping_add(999));

    let dt = net.config.dt_ms;
    let pulse_steps = (pulse_width_ms / dt).max(1.0) as usize;
    let n_steps = (duration_ms / dt) as usize;
    let fire_steps: Vec<i64> = pattern_times.iter()
        .map(|&t| (t / dt) as i64).collect();

    let mut out_log = Vec::new();
    let mut ext = vec![0u16; pattern_times.len()];
    for step in 0..n_steps {
        for k in 0..pattern_times.len() {
            let fs = fire_steps[k];
            ext[k] = if (step as i64) >= fs && (step as i64) < fs + pulse_steps as i64 {
                pulses_per_event
            } else { 0 };
        }
        let fired = net.step(&ext);
        for nid in fired {
            if let Some(oi) = net.output_index(nid) {
                out_log.push((oi, net.t));
            }
        }
    }
    out_log
}

pub fn fingerprint_from_log(
    output_log: &[(usize, f64)], n_outputs: usize, t_end: f64, tau: f64,
) -> Vec<f64> {
    let mut tr = OutputTrace::new(n_outputs, tau);
    for &(oi, t) in output_log { tr.record_spike(oi, t); }
    tr.fingerprint(t_end)
}
