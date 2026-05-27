//! DRAM 物理モデル SNN ネットワーク統合 (STAGE2GAMMA V2 §3.1 全体動作)
//!
//! Phase 2 fork F (ThermoNetwork) の DRAM 物理対応版。
//!
//! 1 step の動作 (V2 §3.1 に従う):
//!   1. リングバッファの現在行から「到達した電荷」を取得
//!   2. 各 DramCell に電荷を加算、 leak 適用、 発火判定 (Vth 比較)
//!   3. 発火したセルについて STDP 更新 (LTP + LTD)
//!   4. 発火したセルから、 接続先 (post) のリングバッファ行 (現在+1+delay) に電荷加算
//!   5. シナプス自然減衰 + vitality 更新
//!   6. head 進行、 時刻 +1
//!
//! V2 §9.2.g (抑制性 ニューロン問題) は POC では暫定的に
//! 「抑制性 pre から post への伝達は negative charge」 で表現。
//! 実機では別機構が必要。

use super::dram_cell::{DramCell, CHARGE_MAX};
use super::dram_synapse::{DramSynapse, PULSE_WIDTH_MAX};
use super::ring_buffer::RingBuffer;
use rand::prelude::*;
use std::sync::atomic::{AtomicU64, Ordering};
use rayon::prelude::*;

#[derive(Debug, Clone)]
pub struct DramNetworkConfig {
    /// 入力ニューロン数
    pub n_input: usize,
    /// 興奮性 (出力含む)
    pub n_excitatory: usize,
    /// 抑制性
    pub n_inhibitory: usize,
    /// 出力ニューロン数 (interp: 末尾 n_output 個の興奮性が出力役)
    pub n_output: usize,
    /// 入力→皮質 1 入力あたり fanout
    pub input_fanout: usize,
    /// 興奮過剰接続 1 ニューロンあたり fanout
    pub overconnect_fanout: usize,
    /// 抑制→興奮 1 抑制あたり fanout
    pub inhibitory_fanout: usize,
    /// 初期遅延範囲
    pub delay_range: (i32, i32),
    /// 初期 pulse_width (シナプス強度の初期値)
    pub initial_pulse_width: i32,
    /// 入力結線の pulse_width
    pub input_pulse_width: i32,
    /// シナプス決定論性のシード
    pub seed: u64,
    /// リングサイズ (= 最大遅延 + 余裕)
    pub ring_size: usize,
}

impl Default for DramNetworkConfig {
    fn default() -> Self {
        // Phase 2 fork F と類似の構成 (cortex 420 cell、 抑制 76)
        Self {
            n_input: 20,
            n_excitatory: 344,
            n_inhibitory: 76,
            n_output: 40,
            input_fanout: 80,
            overconnect_fanout: 40,
            inhibitory_fanout: 20,
            delay_range: (2, 40),
            initial_pulse_width: 50,
            input_pulse_width: 80,
            seed: 300,
            ring_size: 64,  // > max_delay (40)
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct DramMacroObservables {
    pub charge_mean: f64,
    pub charge_max: i32,
    pub pulse_width_mean: f64,
    pub pulse_width_max: i32,
    pub sparsity: f64,
    pub vth_mean: f64,
    pub vth_std: f64,
}

pub struct DramNetwork {
    pub config: DramNetworkConfig,
    pub cells: Vec<DramCell>,
    pub synapses: Vec<DramSynapse>,
    pub ring: RingBuffer,

    pub current_time: i32,

    pub input_neurons: Vec<usize>,
    pub output_neurons: Vec<usize>,

    pub out_syn: Vec<Vec<usize>>,
    pub in_syn: Vec<Vec<usize>>,

    pub spikes_total: u64,
    pub pruned_total: u64,
}

impl DramNetwork {
    pub fn new(config: DramNetworkConfig) -> Self {
        let mut rng = StdRng::seed_from_u64(config.seed);

        let n_total = config.n_input + config.n_excitatory + config.n_inhibitory;
        let mut cells: Vec<DramCell> = Vec::with_capacity(n_total);
        let mut input_neurons: Vec<usize> = Vec::new();
        let mut output_neurons: Vec<usize> = Vec::new();

        // (1) 入力ニューロン (Vth 低め、 刺激応答性高)
        for _ in 0..config.n_input {
            cells.push(DramCell::new_input(&mut rng));
            input_neurons.push(cells.len() - 1);
        }

        // (2) 興奮性 (Vth 標準分布)、 末尾 n_output が出力役
        let exc_start = cells.len();
        for _ in 0..config.n_excitatory {
            cells.push(DramCell::new_random_vth(&mut rng));
        }
        for i in 0..config.n_output {
            output_neurons.push(exc_start + config.n_excitatory - config.n_output + i);
        }

        // (3) 抑制性 (is_inhibitory フラグ立て、 Vth 低め)
        let inh_start = cells.len();
        for _ in 0..config.n_inhibitory {
            cells.push(DramCell::new_inhibitory(&mut rng));
        }

        let exc_indices: Vec<usize> = (exc_start..exc_start + config.n_excitatory).collect();
        let inh_indices: Vec<usize> = (inh_start..inh_start + config.n_inhibitory).collect();
        let cortex_indices: Vec<usize> = (config.n_input..cells.len()).collect();

        let mut synapses: Vec<DramSynapse> = Vec::new();
        let delay_dist = config.delay_range;

        // (4) 入力 → 皮質 fanout
        for &inp in &input_neurons {
            let mut targets = cortex_indices.clone();
            targets.shuffle(&mut rng);
            for &t in targets.iter().take(config.input_fanout) {
                let d = rng.gen_range(delay_dist.0..=delay_dist.1);
                synapses.push(DramSynapse::new(inp, t, d, config.input_pulse_width));
            }
        }

        // (5) 抑制 → 興奮 (デールの原則: 抑制 pre は negative)
        for &inh in &inh_indices {
            let mut targets = exc_indices.clone();
            targets.shuffle(&mut rng);
            for &t in targets.iter().take(config.inhibitory_fanout) {
                let d = rng.gen_range(delay_dist.0..=delay_dist.1);
                synapses.push(DramSynapse::new(inh, t, d, config.input_pulse_width));
            }
        }

        // (6) 興奮過剰接続 (Phase 2 fork F と同じ発生学的)
        for &pre in &exc_indices {
            let mut cand: Vec<usize> = cortex_indices.iter().copied()
                .filter(|&t| t != pre).collect();
            cand.shuffle(&mut rng);
            for &post in cand.iter().take(config.overconnect_fanout) {
                let d = rng.gen_range(delay_dist.0..=delay_dist.1);
                synapses.push(DramSynapse::new(pre, post, d, config.initial_pulse_width));
            }
        }

        // out_syn / in_syn
        let mut out_syn: Vec<Vec<usize>> = vec![Vec::new(); cells.len()];
        let mut in_syn: Vec<Vec<usize>> = vec![Vec::new(); cells.len()];
        for (s_idx, s) in synapses.iter().enumerate() {
            out_syn[s.pre].push(s_idx);
            in_syn[s.post].push(s_idx);
        }

        let ring = RingBuffer::new(config.ring_size, cells.len());

        Self {
            config,
            cells,
            synapses,
            ring,
            current_time: 0,
            input_neurons,
            output_neurons,
            out_syn,
            in_syn,
            spikes_total: 0,
            pruned_total: 0,
        }
    }

    pub fn n_neurons(&self) -> usize { self.cells.len() }
    pub fn n_synapses(&self) -> usize { self.synapses.len() }
    pub fn n_alive_synapses(&self) -> usize {
        self.synapses.iter().filter(|s| s.alive).count()
    }

    /// 1 SNN クロック動作 (V2 §3.1)
    /// external_input は入力ニューロンへの外部電荷 (cochlea 出力等)
    /// 戻り値: 発火ニューロン idx リスト
    pub fn step(&mut self, external_input: &[i32]) -> Vec<usize> {
        let t = self.current_time;
        let n = self.cells.len();

        // 1. リングバッファ現在行を排出 (V2 §3.1 ステップ 1 の破壊読み出し)
        let mut arriving_charges = self.ring.drain_current_row();

        // 2. 外部入力を入力ニューロンに加算
        for (k, &v) in external_input.iter().enumerate() {
            if k < self.input_neurons.len() {
                let target = self.input_neurons[k];
                arriving_charges[target] = arriving_charges[target].saturating_add(v);
            }
        }

        // 3. 各セルを update (Vth 比較 + leak + 発火判定)
        let charges_ref: &[i32] = &arriving_charges;
        let fired: Vec<usize> = self.cells.par_iter_mut().enumerate()
            .filter_map(|(i, cell)| {
                if cell.update(charges_ref[i], t) { Some(i) } else { None }
            })
            .collect();
        self.spikes_total += fired.len() as u64;
        let _ = n;

        // 4. STDP 更新 (Phase 2 と同じパターン、 spike_trace 利用)
        for &pre in &fired {
            // LTD: pre 発火、 post が直近発火 (反因果)
            for &s_idx in &self.out_syn[pre] {
                let post = self.synapses[s_idx].post;
                let post_trace = self.cells[post].spike_trace;
                self.synapses[s_idx].update_on_pre_spike(post_trace);
            }
            // LTP: pre 発火 → 自分への入射シナプス更新 (因果)
            for &s_idx in &self.in_syn[pre] {
                let pre_n = self.synapses[s_idx].pre;
                let pre_trace = self.cells[pre_n].spike_trace;
                self.synapses[s_idx].update_on_post_spike(pre_trace);
            }
        }

        // 5. シナプス配送 (V2 §3.1 ステップ 4: 個別セル書き込み)
        //    抑制性 pre → 負の電荷加算 (V2 §9.2.g 暫定対応)
        for &pre in &fired {
            let is_inh = self.cells[pre].is_inhibitory;
            for &s_idx in &self.out_syn[pre] {
                if !self.synapses[s_idx].alive { continue; }
                let charge = self.synapses[s_idx].charge_per_spike();
                let signed = if is_inh { -charge } else { charge };
                let delay = self.synapses[s_idx].delay as usize;
                let post = self.synapses[s_idx].post;
                // V2 §3.1 ステップ 4: ring[(t+1+delay) mod N][post] += charge
                self.ring.add_charge_at_offset(1 + delay, post, signed);
                self.synapses[s_idx].on_transmission();
            }
        }

        // 6. シナプス自然減衰 + alive 更新 (rayon 並列化)
        let pruned_counter = AtomicU64::new(0);
        self.synapses.par_iter_mut().for_each(|s| {
            s.decay();
            let was_alive = s.alive;
            s.update_alive();
            if was_alive && !s.alive {
                pruned_counter.fetch_add(1, Ordering::Relaxed);
            }
        });
        self.pruned_total += pruned_counter.into_inner();

        // 7. リングバッファ head 進行 + 時刻 +1
        self.ring.advance();
        self.current_time += 1;

        fired
    }

    /// 試行間リセット (charge と spike_trace、 シナプス重み・vitality は保持)
    pub fn reset_trial_state(&mut self) {
        for c in &mut self.cells {
            c.reset_state();
        }
        self.ring.reset_all();
    }

    /// 出力ニューロンとして何番目か
    pub fn output_index_of(&self, neuron_id: usize) -> Option<usize> {
        self.output_neurons.iter().position(|&n| n == neuron_id)
    }

    /// マクロ観察量 (Phase 2 と同様)
    pub fn macro_observables(&self) -> DramMacroObservables {
        let n = self.cells.len();
        let charge_sum: i64 = self.cells.iter().map(|c| c.charge as i64).sum();
        let charge_max = self.cells.iter().map(|c| c.charge).max().unwrap_or(0);

        let pw_alive: Vec<i32> = self.synapses.iter()
            .filter(|s| s.alive)
            .map(|s| s.pulse_width).collect();
        let pw_sum: i64 = pw_alive.iter().map(|&v| v as i64).sum();
        let pw_max = pw_alive.iter().copied().max().unwrap_or(0);
        let pw_mean = if pw_alive.is_empty() { 0.0 } else {
            pw_sum as f64 / pw_alive.len() as f64
        };

        let vth_sum: i64 = self.cells.iter().map(|c| c.vth_threshold as i64).sum();
        let vth_mean = vth_sum as f64 / n as f64;
        let vth_var: f64 = self.cells.iter()
            .map(|c| { let d = c.vth_threshold as f64 - vth_mean; d * d })
            .sum::<f64>() / n as f64;
        let vth_std = vth_var.sqrt();

        let alive_total = self.synapses.iter().filter(|s| s.alive).count();
        let sparsity = if self.synapses.is_empty() { 0.0 } else {
            alive_total as f64 / self.synapses.len() as f64
        };

        DramMacroObservables {
            charge_mean: charge_sum as f64 / n as f64,
            charge_max,
            pulse_width_mean: pw_mean,
            pulse_width_max: pw_max,
            sparsity,
            vth_mean,
            vth_std,
        }
    }

    /// CHARGE_MAX を返す (外部からスケール参照用)
    pub fn charge_max() -> i32 { CHARGE_MAX }
    /// PULSE_WIDTH_MAX を返す
    pub fn pulse_width_max() -> i32 { PULSE_WIDTH_MAX }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_small_network() {
        let mut cfg = DramNetworkConfig::default();
        cfg.n_input = 5;
        cfg.n_excitatory = 20;
        cfg.n_inhibitory = 5;
        cfg.n_output = 5;
        cfg.input_fanout = 10;
        cfg.overconnect_fanout = 5;
        cfg.inhibitory_fanout = 5;
        let net = DramNetwork::new(cfg);
        assert_eq!(net.n_neurons(), 30);
        assert_eq!(net.input_neurons.len(), 5);
        assert_eq!(net.output_neurons.len(), 5);
        assert!(net.n_synapses() > 0);
    }

    #[test]
    fn one_step_no_panic() {
        let mut cfg = DramNetworkConfig::default();
        cfg.n_input = 5;
        cfg.n_excitatory = 20;
        cfg.n_inhibitory = 5;
        cfg.n_output = 5;
        cfg.input_fanout = 10;
        cfg.overconnect_fanout = 5;
        cfg.inhibitory_fanout = 5;
        let mut net = DramNetwork::new(cfg);
        let ext = vec![0i32; 5];
        let _ = net.step(&ext);
        assert_eq!(net.current_time, 1);
    }

    #[test]
    fn strong_input_drives_firing() {
        let mut cfg = DramNetworkConfig::default();
        cfg.n_input = 5;
        cfg.n_excitatory = 20;
        cfg.n_inhibitory = 5;
        cfg.n_output = 5;
        cfg.input_fanout = 10;
        cfg.overconnect_fanout = 5;
        cfg.inhibitory_fanout = 5;
        let mut net = DramNetwork::new(cfg);

        let mut total_fired = 0u64;
        let strong_ext = vec![800i32; 5];  // 強い外部電荷
        for _ in 0..50 {
            let fired = net.step(&strong_ext);
            total_fired += fired.len() as u64;
        }
        // 強い刺激で何度か発火する
        assert!(total_fired > 0, "Strong input should produce firings, got 0");
    }
}
