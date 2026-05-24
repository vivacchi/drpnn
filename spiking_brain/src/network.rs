//! 動的シナプス可塑性を持つスパイキングニューラルネットワーク。
//!
//! アーキテクチャ:
//!   入力層(蝸牛模擬) → 皮質層(再帰的、軸索遅延 + STDP) → 出力読出し
//!
//! 遅延スパイク配送はリングバッファ方式。各時間ステップで配送スロットを進めるので
//! O(発火数 × ファンアウト) で済む。
//!
//! STDP は spike-pair 型(指数窓):
//!   pre が post の Δt 前に発火: Δw = +A⁺ exp(-Δt/τ⁺)  (LTP)
//!   post が pre の Δt 前に発火: Δw = -A⁻ exp(-Δt/τ⁻)  (LTD)

use crate::neuron::{Neuron, NeuronKind};
use crate::trace::OutputTrace;
use rand::prelude::*;
use rand_distr::{Distribution, Normal, Uniform};

/// シナプス: 動的に重みが変わる、遅延付き接続
///
/// R-STDP 拡張: eligibility trace を保持。因果ペアイベントで蓄積し、
/// 外部報酬信号到来時に weight 更新に「現像」される。
#[derive(Clone, Debug)]
pub struct Synapse {
    pub pre: usize,
    pub post: usize,
    pub weight: f64,
    /// 配送遅延 (シミュレーションステップ数)
    pub delay: usize,
    /// このシナプスは STDP / R-STDP の対象か
    pub plastic: bool,
    /// Eligibility trace: 因果的に寄与した量。報酬で現像される
    pub eligibility: f64,
    /// eligibility の最終更新時刻 (lazy decay 用) [ms]
    pub last_elig_update: f64,
}

pub struct NetworkConfig {
    pub n_cortex: usize,
    pub n_input: usize,
    pub n_output: usize,
    /// 入力 → 皮質の各入力ニューロンが投射する皮質ニューロン数
    pub input_fanout: usize,
    /// 興奮性皮質ニューロンの再帰ファンアウト
    pub cortex_fanout: usize,
    pub dt_ms: f64,
    pub seed: u64,
}

impl Default for NetworkConfig {
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

/// STDP / R-STDP パラメータ
pub struct StdpParams {
    // 即時 STDP (Hebbian) — weight に直接効く
    pub a_plus: f64,
    pub a_minus: f64,
    pub tau_plus: f64,
    pub tau_minus: f64,
    pub w_max: f64,
    pub w_min: f64,
    /// この時間幅(ms)を超える間隔は無視する
    pub window_ms: f64,
    /// 即時 STDP の効きの強さ (0.0 にすると pure R-STDP モード)
    pub immediate_scale: f64,

    // R-STDP — eligibility trace に効き、報酬で現像される
    /// 因果ペアで eligibility がもらう増分
    pub r_a_plus: f64,
    /// eligibility trace の減衰時定数 [ms]
    pub tau_e: f64,
    /// 報酬 → weight 更新の学習率
    pub reward_lr: f64,
}

impl Default for StdpParams {
    fn default() -> Self {
        Self {
            a_plus: 0.10,
            a_minus: 0.12,
            tau_plus: 20.0,
            tau_minus: 20.0,
            w_max: 10.0,
            w_min: 0.0,
            window_ms: 80.0,
            immediate_scale: 1.0,
            r_a_plus: 0.10,
            tau_e: 1000.0,
            reward_lr: 0.5,
        }
    }
}

pub struct Network {
    pub config: NetworkConfig,
    pub stdp: StdpParams,
    pub neurons: Vec<Neuron>,
    pub synapses: Vec<Synapse>,
    /// 各ニューロンの出ていくシナプスインデックス
    out_syn: Vec<Vec<usize>>,
    /// 各ニューロンに入ってくるシナプスインデックス
    in_syn: Vec<Vec<usize>>,
    /// 遅延スパイク配送用リングバッファ: `delivery[slot][neuron]` = 蓄積電流
    delivery: Vec<Vec<f64>>,
    delivery_head: usize,
    max_delay: usize,
    pub input_neurons: Vec<usize>,
    pub output_neurons: Vec<usize>,
    /// 現在時刻 (ms)
    pub t: f64,
    pub noise_rng: StdRng,
}

impl Network {
    pub fn new(config: NetworkConfig, stdp: StdpParams) -> Self {
        let mut build_rng = StdRng::seed_from_u64(config.seed);

        // ニューロン構成: [入力 | 興奮性皮質 | 抑制性皮質]
        let n_exc = (config.n_cortex as f64 * 0.8) as usize;
        let n_inh = config.n_cortex - n_exc;
        let n_total = config.n_input + config.n_cortex;

        let mut neurons = Vec::with_capacity(n_total);
        let input_neurons: Vec<usize> = (0..config.n_input).collect();
        for _ in 0..config.n_input {
            neurons.push(Neuron::new(NeuronKind::Input));
        }
        let exc_start = config.n_input;
        let exc_end = exc_start + n_exc;
        for _ in 0..n_exc {
            neurons.push(Neuron::new(NeuronKind::ExcitatoryCortex));
        }
        for _ in 0..n_inh {
            neurons.push(Neuron::new(NeuronKind::InhibitoryCortex));
        }
        let exc_neurons: Vec<usize> = (exc_start..exc_end).collect();
        let inh_neurons: Vec<usize> = (exc_end..n_total).collect();
        let output_neurons: Vec<usize> = exc_neurons.iter().take(config.n_output).copied().collect();
        let cortex_all: Vec<usize> = exc_neurons
            .iter()
            .chain(inh_neurons.iter())
            .copied()
            .collect();

        // シナプス構築
        let mut synapses: Vec<Synapse> = Vec::new();
        let delay_dist = Uniform::from(2..=40usize); // 1.0 - 20.0 ms (dt=0.5)
        let inp_w = Uniform::from(6.0..9.0);
        let exc_w = Uniform::from(3.0..5.0);
        let inh_w = Uniform::from(2.0..5.0);

        // (1) 入力 → 皮質: 強い、遅延あり、可塑性なし
        for &inp in &input_neurons {
            let targets =
                sample_without_replacement(&cortex_all, config.input_fanout, &mut build_rng);
            for &t in &targets {
                synapses.push(Synapse {
                    pre: inp,
                    post: t,
                    weight: inp_w.sample(&mut build_rng),
                    delay: build_rng.gen_range(2..=20), // 1-10 ms
                    plastic: false,
                    eligibility: 0.0,
                    last_elig_update: 0.0,
                });
            }
        }

        // (2) 興奮性皮質 → 全皮質(再帰): 中程度の重み、広い遅延、STDP対象
        for &pre in &exc_neurons {
            let candidates: Vec<usize> = cortex_all.iter().copied().filter(|&n| n != pre).collect();
            let targets =
                sample_without_replacement(&candidates, config.cortex_fanout, &mut build_rng);
            for &tgt in &targets {
                synapses.push(Synapse {
                    pre,
                    post: tgt,
                    weight: exc_w.sample(&mut build_rng),
                    delay: delay_dist.sample(&mut build_rng),
                    plastic: true,
                    eligibility: 0.0,
                    last_elig_update: 0.0,
                });
            }
        }

        // (3) 抑制性皮質 → 興奮性皮質: 負の重み、短遅延、固定
        for &pre in &inh_neurons {
            let targets =
                sample_without_replacement(&exc_neurons, config.cortex_fanout, &mut build_rng);
            for &tgt in &targets {
                synapses.push(Synapse {
                    pre,
                    post: tgt,
                    weight: -inh_w.sample(&mut build_rng),
                    delay: 2, // 1 ms
                    plastic: false,
                    eligibility: 0.0,
                    last_elig_update: 0.0,
                });
            }
        }

        // インデックスキャッシュ
        let mut out_syn: Vec<Vec<usize>> = vec![Vec::new(); n_total];
        let mut in_syn: Vec<Vec<usize>> = vec![Vec::new(); n_total];
        for (i, s) in synapses.iter().enumerate() {
            out_syn[s.pre].push(i);
            in_syn[s.post].push(i);
        }

        let max_delay = synapses.iter().map(|s| s.delay).max().unwrap_or(2) + 1;
        let delivery = (0..max_delay).map(|_| vec![0.0; n_total]).collect();

        Self {
            config,
            stdp,
            neurons,
            synapses,
            out_syn,
            in_syn,
            delivery,
            delivery_head: 0,
            max_delay,
            input_neurons,
            output_neurons,
            t: 0.0,
            noise_rng: StdRng::seed_from_u64(99),
        }
    }

    pub fn n_neurons(&self) -> usize {
        self.neurons.len()
    }

    pub fn n_synapses(&self) -> usize {
        self.synapses.len()
    }

    pub fn n_plastic_synapses(&self) -> usize {
        self.synapses.iter().filter(|s| s.plastic).count()
    }

    pub fn delay_range_ms(&self) -> (f64, f64) {
        let mn = self.synapses.iter().map(|s| s.delay).min().unwrap_or(0);
        let mx = self.synapses.iter().map(|s| s.delay).max().unwrap_or(0);
        (mn as f64 * self.config.dt_ms, mx as f64 * self.config.dt_ms)
    }

    /// 膜状態と配送バッファのみリセット(学習した重みは保持)
    pub fn reset_state(&mut self) {
        for n in &mut self.neurons {
            n.v = -65.0;
            n.u = n.b * n.v;
            n.last_spike = f64::NEG_INFINITY;
        }
        for slot in &mut self.delivery {
            for x in slot.iter_mut() {
                *x = 0.0;
            }
        }
        self.delivery_head = 0;
        self.t = 0.0;
    }

    pub fn set_noise_seed(&mut self, seed: u64) {
        self.noise_rng = StdRng::seed_from_u64(seed);
    }

    /// 1ステップ。`ext_input_currents[i]` を `input_neurons[i]` に注入。
    /// 戻り値はこのステップで発火した全ニューロン ID。
    pub fn step(&mut self, ext_input_currents: &[f64]) -> Vec<usize> {
        let dt = self.config.dt_ms;
        let n = self.neurons.len();

        // 1. 入力電流を組み立てる: 配送スロット + 外部刺激 + 熱雑音
        let mut current = std::mem::take(&mut self.delivery[self.delivery_head]);
        // ↑ 取り出した後はゼロベクトルで戻す必要がある
        for (k, &c) in ext_input_currents.iter().enumerate() {
            if k < self.input_neurons.len() {
                current[self.input_neurons[k]] += c;
            }
        }
        let noise_dist = Normal::new(0.0, 1.0).unwrap();
        for i in 0..n {
            let raw = noise_dist.sample(&mut self.noise_rng);
            let scale = match self.neurons[i].kind {
                NeuronKind::InhibitoryCortex => 0.5,
                NeuronKind::Input => 0.5,
                NeuronKind::ExcitatoryCortex => 1.5,
            };
            current[i] += raw * scale;
        }

        // 2. すべてのニューロンを更新
        let mut fired = Vec::new();
        for i in 0..n {
            if self.neurons[i].step(dt, current[i], self.t) {
                fired.push(i);
            }
        }

        // 3. 発火による波及効果(配送 + STDP + eligibility)
        for &pre in &fired {
            // 出ていくスパイクを未来のスロットへスケジュール
            for &s_idx in &self.out_syn[pre] {
                let syn = &self.synapses[s_idx];
                let slot = (self.delivery_head + syn.delay) % self.max_delay;
                self.delivery[slot][syn.post] += syn.weight;
            }

            // STDP+: pre から見た「今のpost発火」→ 入ってくる可塑シナプスを LTP
            // 同時に eligibility trace も加算 (R-STDP)
            for &s_idx in &self.in_syn[pre] {
                if !self.synapses[s_idx].plastic {
                    continue;
                }
                let pre_n = self.synapses[s_idx].pre;
                let dt_back = self.t - self.neurons[pre_n].last_spike;
                if dt_back > 0.0 && dt_back < self.stdp.window_ms {
                    let kernel = (-dt_back / self.stdp.tau_plus).exp();
                    let dw = self.stdp.a_plus * kernel * self.stdp.immediate_scale;
                    let delta_elig = self.stdp.r_a_plus * kernel;
                    // 借用チェッカ対策: メソッド呼び出しではなく直接フィールド操作
                    {
                        let s = &mut self.synapses[s_idx];
                        // 即時 STDP
                        s.weight = (s.weight + dw).min(self.stdp.w_max);
                        // Eligibility trace (lazy decay → 加算)
                        let elapsed = self.t - s.last_elig_update;
                        if elapsed > 0.0 {
                            s.eligibility *= (-elapsed / self.stdp.tau_e).exp();
                        }
                        s.eligibility += delta_elig;
                        s.last_elig_update = self.t;
                    }
                }
            }

            // STDP-: pre 発火時点で「すでに発火済みの post」がある可塑シナプス → LTD
            // 同時に eligibility に負の寄与
            for &s_idx in &self.out_syn[pre] {
                if !self.synapses[s_idx].plastic {
                    continue;
                }
                let post_n = self.synapses[s_idx].post;
                let dt_back = self.t - self.neurons[post_n].last_spike;
                if dt_back > 0.0 && dt_back < self.stdp.window_ms {
                    let kernel = (-dt_back / self.stdp.tau_minus).exp();
                    let dw = self.stdp.a_minus * kernel * self.stdp.immediate_scale;
                    let delta_elig = -self.stdp.r_a_plus * kernel;
                    {
                        let s = &mut self.synapses[s_idx];
                        s.weight = (s.weight - dw).max(self.stdp.w_min);
                        let elapsed = self.t - s.last_elig_update;
                        if elapsed > 0.0 {
                            s.eligibility *= (-elapsed / self.stdp.tau_e).exp();
                        }
                        s.eligibility += delta_elig;
                        s.last_elig_update = self.t;
                    }
                }
            }
        }

        // 4. 取り出していた配送スロットをゼロで戻す
        for x in current.iter_mut() {
            *x = 0.0;
        }
        self.delivery[self.delivery_head] = current;
        self.delivery_head = (self.delivery_head + 1) % self.max_delay;
        self.t += dt;
        fired
    }

    /// 出力ニューロン id (グローバル) → 出力層内インデックス
    pub fn output_index(&self, neuron_id: usize) -> Option<usize> {
        self.output_neurons.iter().position(|&n| n == neuron_id)
    }

    /// Eligibility trace に増分を加算。lazy decay を行ってから足す。
    /// (毎ステップ全シナプスを走査する代わりに、触ったときだけ減衰させる)
    fn accumulate_eligibility(&mut self, s_idx: usize, delta: f64) {
        let s = &mut self.synapses[s_idx];
        let elapsed = self.t - s.last_elig_update;
        if elapsed > 0.0 {
            s.eligibility *= (-elapsed / self.stdp.tau_e).exp();
        }
        s.eligibility += delta;
        s.last_elig_update = self.t;
    }

    /// 報酬信号 `r` (正: 強化、負: 抑制) を適用。
    /// 全可塑シナプスについて: weight ← weight + lr · r · eligibility(now)
    /// その後 eligibility は消化されてゼロに。
    ///
    /// 報酬は外部 (タスク評価器) から非同期に呼ぶ。
    pub fn apply_reward(&mut self, r: f64) {
        for s in &mut self.synapses {
            if !s.plastic {
                continue;
            }
            let elapsed = self.t - s.last_elig_update;
            let e_now = if elapsed > 0.0 {
                s.eligibility * (-elapsed / self.stdp.tau_e).exp()
            } else {
                s.eligibility
            };
            let dw = self.stdp.reward_lr * r * e_now;
            s.weight = (s.weight + dw).clamp(self.stdp.w_min, self.stdp.w_max);
            // 「現像」完了: タグはリセット
            s.eligibility = 0.0;
            s.last_elig_update = self.t;
        }
    }

    /// 可塑シナプス重みの平均
    pub fn mean_plastic_weight(&self) -> f64 {
        let (sum, count) = self
            .synapses
            .iter()
            .filter(|s| s.plastic)
            .fold((0.0, 0usize), |(s, c), syn| (s + syn.weight, c + 1));
        if count == 0 { 0.0 } else { sum / count as f64 }
    }

    /// 全可塑シナプスの eligibility 絶対値合計 (現在時点で減衰済みの値)
    pub fn total_eligibility_magnitude(&self) -> f64 {
        self.synapses.iter()
            .filter(|s| s.plastic)
            .map(|s| {
                let elapsed = self.t - s.last_elig_update;
                let e = if elapsed > 0.0 {
                    s.eligibility * (-elapsed / self.stdp.tau_e).exp()
                } else {
                    s.eligibility
                };
                e.abs()
            })
            .sum()
    }
}

fn sample_without_replacement<T: Copy>(pool: &[T], k: usize, rng: &mut StdRng) -> Vec<T> {
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

/// 入力パターン(各入力ニューロンの発火開始時刻 ms)を生成
pub fn make_pattern(n_input: usize, seed: u64) -> Vec<f64> {
    let mut rng = StdRng::seed_from_u64(seed);
    let dist = Uniform::from(0.0..20.0);
    (0..n_input).map(|_| dist.sample(&mut rng)).collect()
}

/// パターンを 1 回提示。各入力ニューロンは `pattern_times[i]` から `pulse_width_ms` の間 `amplitude` で駆動。
/// 出力層の発火イベント (出力層内インデックス, 時刻) を全て返す。
pub fn present_pattern(
    net: &mut Network,
    pattern_times: &[f64],
    duration_ms: f64,
    amplitude: f64,
    pulse_width_ms: f64,
    jitter_ms: f64,
    trial_seed: u64,
) -> Vec<(usize, f64)> {
    net.reset_state();
    net.set_noise_seed(trial_seed.wrapping_add(999));

    let mut jitter_rng = StdRng::seed_from_u64(trial_seed);
    let jitter_dist = Normal::new(0.0, jitter_ms).unwrap();
    let dt = net.config.dt_ms;
    let pulse_steps = (pulse_width_ms / dt) as usize;
    let n_steps = (duration_ms / dt) as usize;

    let fire_steps: Vec<i64> = pattern_times
        .iter()
        .map(|&t| ((t + jitter_dist.sample(&mut jitter_rng)) / dt) as i64)
        .collect();

    let mut output_log: Vec<(usize, f64)> = Vec::new();
    let n_input = net.input_neurons.len();
    let mut ext = vec![0.0; n_input];

    for step in 0..n_steps {
        for k in 0..n_input {
            let fs = fire_steps[k];
            ext[k] = if (step as i64) >= fs && (step as i64) < fs + pulse_steps as i64 {
                amplitude
            } else {
                0.0
            };
        }
        let fired = net.step(&ext);
        let t_after = net.t;
        for nid in fired {
            if let Some(oi) = net.output_index(nid) {
                output_log.push((oi, t_after));
            }
        }
    }
    output_log
}

/// 出力スパイクログから減衰フィンガープリント(N_out次元ベクトル)を計算
pub fn fingerprint_from_log(
    output_log: &[(usize, f64)],
    n_outputs: usize,
    t_end: f64,
    tau: f64,
) -> Vec<f64> {
    let mut trace = OutputTrace::new(n_outputs, tau);
    for &(oi, t) in output_log {
        trace.record_spike(oi, t);
    }
    trace.fingerprint(t_end)
}
