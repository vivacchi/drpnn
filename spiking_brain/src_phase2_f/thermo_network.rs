//! 熱力学的ネットワーク統合 (ThermoNetwork)
//!
//! ニューロン群、シナプス群、トポロジーを統合し、各クロックの物理プロセスを進行する。
//! 明示的な「学習関数」は存在しない。step() を呼び続けるだけで学習が散逸構造として形成される。

use super::axon_growth::{
    axon_growth_step, build_position_index, GROWTH_THRESHOLD,
};
use super::thermo_neuron::ThermoNeuron;
use super::thermo_synapse::{ThermoSynapse, OPEN_THRESHOLD};
use super::topology::Topology;
use rand::prelude::*;
use rayon::prelude::*;
use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicU64, Ordering};

/// 軸索成長を試みる周期 (step 数)。指示書 §4 では 200。
pub const GROWTH_INTERVAL: i32 = 200;
/// シナプス信号スケール: 1 発火あたりの寄与 = conductance / SIGNAL_SCALE_DIVISOR
/// 案A 調整: Phase 1 spike_gain=10 とのスケール整合
/// 固定結線 conductance=80 → 信号 8 (Phase 1 と同じ)
/// 軸索成長結線 conductance=20 → 信号 2 (弱い結線として正しく機能)
pub const SIGNAL_SCALE_DIVISOR: i32 = 10;

/// ネットワーク構成設定
#[derive(Debug, Clone)]
pub struct ThermoNetworkConfig {
    /// グリッド幅・高さ
    pub grid_width: i32,
    pub grid_height: i32,
    /// 入力ニューロン数
    pub n_input: usize,
    /// 出力ニューロン数
    pub n_output: usize,
    /// 興奮性皮質ニューロン数
    pub n_excitatory: usize,
    /// 抑制性皮質ニューロン数
    pub n_inhibitory: usize,
    /// 入力→皮質の fanout (1 入力あたり何本の固定結線を張るか)
    pub input_fanout: usize,
    /// 初期遅延の範囲 [min, max] (step)
    pub delay_range: (i32, i32),
    /// 乱数シード (初期構造のみで使用、ランタイムは決定論的)
    pub seed: u64,
    /// 軸索成長を試みる周期
    pub axon_growth_interval: i32,
    /// dt (ms / step) — 配送遅延の解釈用
    pub dt_ms: f64,
    /// UP/DOWN 状態を有効化するか (PAPER §5.12.7-A、池谷 2005)
    /// false で既存動作 (up_offset=0 と等価)、true で多重アトラクター物理実装
    pub enable_up_down: bool,
}

impl Default for ThermoNetworkConfig {
    fn default() -> Self {
        // Fork F-G1-R1 v2: 1セル1ニューロン原則 + 皮質的均一密度
        //   grid 20x22=440、入力 y=0 (20)、出力 y=20,21 (40)、内部 y=1..19 (380)
        //   内部 380 = 興奮 304 + 抑制 76 (抑制比 18.1% = 皮質典型値 15-20% のど真ん中)
        //   抑制配置ルール: (y+x) % 5 == 0 のセル (1-in-5 均一分散)
        //   n_excitatory = 304 (内部) + 40 (出力) = 344
        Self {
            grid_width: 20,
            grid_height: 22,
            n_input: 20,
            n_output: 40,
            n_excitatory: 344, // 内部 304 + 出力 40
            n_inhibitory: 76,  // 内部 380 セル中 1/5 = 76 (18.1%)
            input_fanout: 80,
            delay_range: (2, 40),
            seed: 300,
            axon_growth_interval: GROWTH_INTERVAL,
            dt_ms: 0.5,
            enable_up_down: false, // デフォルト OFF (既存実験との互換性、有効化は明示的に)
        }
    }
}

/// C8: マクロ観察量 (snapshot 用)
/// 学習則には一切使わない — 系の状態を「外部観察者」が測る指標。
#[derive(Debug, Clone, Copy)]
pub struct MacroObservables {
    pub entropy_mean: f64,
    pub entropy_max: i32,
    pub entropy_std: f64,
    pub conductance_mean: f64,
    pub conductance_max: i32,
    pub conductance_std: f64,
    pub enthalpy_mean: f64,
    pub sparsity: f64,
    pub syn_growth_rate: f64,
}

/// 熱力学的 SNN ネットワーク
pub struct ThermoNetwork {
    pub config: ThermoNetworkConfig,
    pub neurons: Vec<ThermoNeuron>,
    pub synapses: Vec<ThermoSynapse>,
    pub topology: Topology,
    pub position_index: HashMap<(i32, i32), Vec<usize>>,

    /// 遅延配送リングバッファ。delivery_queue[t % max_delay][post_idx] = 到達する信号合計
    delivery_queue: Vec<Vec<i32>>,
    delivery_head: usize,
    max_delay: usize,

    pub current_time: i32,
    pub last_growth_time: i32,

    pub input_neurons: Vec<usize>,
    pub output_neurons: Vec<usize>,

    /// 出射シナプス (pre 別)
    pub out_syn: Vec<Vec<usize>>,
    /// 入射シナプス (post 別)
    pub in_syn: Vec<Vec<usize>>,

    /// 軸索成長で作られた累積シナプス数
    pub axons_grown: u64,
    /// 刈り取られた (exists が一度 true から false に落ちた) 累積カウント
    pub axons_pruned: u64,
}

impl ThermoNetwork {
    pub fn new(config: ThermoNetworkConfig) -> Self {
        let mut rng = StdRng::seed_from_u64(config.seed);

        let grid_w = config.grid_width;
        let grid_h = config.grid_height;
        let topology = Topology::new(grid_w, grid_h);

        // ニューロン配置 (Fork F-G1-R1 v2: 1セル1ニューロン原則):
        //   入力 20  : y=0,             x=0..19
        //   出力 40  : y=20, 21,        x=0..19 (各行 20 個、興奮性)
        //   内部 380 : y=1..19,         x=0..19 (19 行 x 20 = 380 セル)
        //     抑制 76 = (y+x) % 5 == 0 のセル (1-in-5 均一分散、各行 4 個 × 19 行 = 76)
        //     興奮 304 = 残り (内部の 80% = 304 セル)
        //
        // すべてのセルに 1 ニューロン、重なりなし、DRP の PE = 1 ニューロン原則を遵守。
        let internal_rows = (grid_h as usize) - 3; // y=0 入力 / y=H-2, H-1 出力 を除く
        let internal_cells = internal_rows * (grid_w as usize);
        let expected_internal = config.n_excitatory - config.n_output + config.n_inhibitory;
        assert_eq!(
            internal_cells, expected_internal,
            "internal cell count mismatch: grid 内部 {} セル vs 配置必要 {} (exc-out + inh)",
            internal_cells, expected_internal
        );
        assert_eq!(
            config.n_input as i32, grid_w,
            "input count ({}) must match grid_width ({})", config.n_input, grid_w
        );
        assert_eq!(
            config.n_output as i32, grid_w * 2,
            "output count ({}) must be 2 × grid_width ({}× 2 = {})",
            config.n_output, grid_w, grid_w * 2
        );

        let n_total = config.n_input + config.n_excitatory + config.n_inhibitory;
        assert_eq!(
            (grid_w * grid_h) as usize, n_total,
            "grid full-fill: {}×{}={} cells, n_total={}",
            grid_w, grid_h, grid_w * grid_h, n_total
        );

        let mut neurons: Vec<ThermoNeuron> = Vec::with_capacity(n_total);
        let mut input_neurons: Vec<usize> = Vec::new();
        let mut output_neurons: Vec<usize> = Vec::new();

        // (1) 入力ニューロン: y=0、x=0..grid_w-1 (1 セル 1 個)
        for x in 0..grid_w {
            neurons.push(ThermoNeuron::input((x, 0)));
            input_neurons.push(neurons.len() - 1);
        }

        // (2) 出力ニューロン (興奮性): y=grid_h-2, y=grid_h-1 の 2 行、各 grid_w 個
        for dy in 0..2 {
            let y = grid_h - 2 + dy;
            for x in 0..grid_w {
                neurons.push(ThermoNeuron::excitatory((x, y)));
                output_neurons.push(neurons.len() - 1);
            }
        }

        // (3) 内部: y=1..grid_h-3 の 19 行を埋める
        //   全 380 セルから seed ベースで 76 セルをランダム選択 → 抑制
        //   残り 304 セルに興奮
        //   均一密度 (Rockel et al. 1980) + 配置はランダム (生物の皮質と整合)
        //   seed 固定なので決定論性は保たれる
        let mut internal_positions: Vec<(i32, i32)> = Vec::new();
        for y in 1..=(grid_h - 3) {
            for x in 0..grid_w {
                internal_positions.push((x, y));
            }
        }
        // 抑制ニューロンを配置する position のインデックス集合 (シャッフルで決定論的乱択)
        let mut idx_pool: Vec<usize> = (0..internal_positions.len()).collect();
        idx_pool.shuffle(&mut rng);
        let inh_set: std::collections::HashSet<usize> =
            idx_pool.into_iter().take(config.n_inhibitory).collect();

        let mut placed_inh = 0usize;
        let mut placed_internal_exc = 0usize;
        for (i, &pos) in internal_positions.iter().enumerate() {
            if inh_set.contains(&i) {
                neurons.push(ThermoNeuron::inhibitory(pos));
                placed_inh += 1;
            } else {
                neurons.push(ThermoNeuron::excitatory(pos));
                placed_internal_exc += 1;
            }
        }
        assert_eq!(placed_inh, config.n_inhibitory,
            "inh placement: {} placed vs {} expected", placed_inh, config.n_inhibitory);
        assert_eq!(placed_internal_exc, config.n_excitatory - config.n_output,
            "internal exc placement: {} placed vs {} expected",
            placed_internal_exc, config.n_excitatory - config.n_output);

        // ── Spontaneous activity の個体差を決定論的に割り当て ──
        // 各ニューロン (入力以外) に index 由来の spontaneous_input を設定
        // 生物の Na/K ポンプ密度差を整数で表現: 0..=3 の 4 段階個体差
        //   spontaneous=0: 完全 silent 候補 (leak で減衰のみ)
        //   spontaneous=1: 弱発火傾向 (leak と平衡で sub-threshold)
        //   spontaneous=2: 中間 (leak=2 なので入力依存)
        //   spontaneous=3: 自発発火傾向 (+1/step → 80step で発火)
        // leak は全体で 2/step (細胞膜の自然リーク)
        for (idx, n) in neurons.iter_mut().enumerate() {
            if n.spontaneous_input == 0 && n.leak == 0 {
                // 入力ニューロンは個体差なし (パターン入力のみで動く)
                continue;
            }
            n.spontaneous_input = (idx as i32) % 4;
            // leak は excitatory()/inhibitory() で既に 2 設定済み
        }

        // ── UP/DOWN 状態の有効化 (PAPER §5.12.7-A) ──
        // 各ニューロンに個体差付きの周期を割り当て、初期状態は idx で決定
        // 確率なし、完全決定論的 (原理 3)
        if config.enable_up_down {
            for (idx, n) in neurons.iter_mut().enumerate() {
                // 入力ニューロンは UP/DOWN なし (純粋トランスデューサ維持)
                if !n.generates_entropy && n.spontaneous_input == 0 {
                    continue;
                }
                // UP/DOWN 周期 (生物の UP 状態 100-500ms, DT_MS=0.5ms → 200-1000step)
                // 個体差を素数で揃えて互いに同期しないようにする
                n.up_period = 200 + ((idx as i32) * 31) % 200;   // 200-399 step (100-200ms)
                n.down_period = 300 + ((idx as i32) * 37) % 300; // 300-599 step (150-300ms)
                // 初期状態を idx 由来で散らす (半数が UP、半数が DOWN から開始)
                n.up_state = (idx % 2) == 0;
                n.up_down_counter = (idx as i32) % 100;
                // up_offset: 閾値 80 に対して 20 程度 (脱分極相当、生物 10mV ≒ 25% 寄与)
                n.up_offset = 20;
            }
        }

        // 位置→インデックスマップ
        let position_index = build_position_index(&neurons);

        // (5) 固定結線: 入力→皮質 (input_fanout 本ずつ)
        let mut synapses: Vec<ThermoSynapse> = Vec::new();
        let delay_dist = config.delay_range;
        // 皮質ニューロン (入力以外) 全体から選ぶ
        let cortex_indices: Vec<usize> = (config.n_input..neurons.len()).collect();
        for &inp in &input_neurons {
            // 入力 fanout 分の皮質ターゲットをランダム選択 (初期化時のみ rand 使用)
            let mut targets: Vec<usize> = cortex_indices.clone();
            targets.shuffle(&mut rng);
            for &t in targets.iter().take(config.input_fanout) {
                let d = rng.gen_range(delay_dist.0..=delay_dist.1);
                // Fork F: 全シナプス可塑性、is_inhibitory は pre neuron 属性で決まる
                synapses.push(ThermoSynapse::new(inp, t, d, OPEN_THRESHOLD + 50));
            }
        }

        // (6) 抑制→興奮の初期結線 (Fork F: 固定ではなく可塑性、デールの原則で is_inhibitory は pre 属性)
        let exc_indices: Vec<usize> = neurons.iter().enumerate()
            .filter(|(i, n)| !n.is_inhibitory && !input_neurons.contains(i))
            .map(|(i, _)| i)
            .collect();
        for (idx, n) in neurons.iter().enumerate() {
            if !n.is_inhibitory { continue; }
            let mut targets: Vec<usize> = exc_indices.clone();
            targets.shuffle(&mut rng);
            for &t in targets.iter().take(20) {
                synapses.push(ThermoSynapse::new(idx, t, 2, OPEN_THRESHOLD + 50));
            }
        }

        // (7) Fork F 過剰接続: cortex_exc neuron から大量の plastic 結線 (発生学的)
        //     12800 本 (320 exc × 40 fanout) を OPEN_THRESHOLD + 20 = 50 で生成
        let cortex_exc_indices: Vec<usize> = neurons.iter().enumerate()
            .filter(|(i, n)| !n.is_inhibitory && !input_neurons.contains(i))
            .map(|(i, _)| i)
            .collect();
        let cortex_all: Vec<usize> = (config.n_input..neurons.len()).collect();
        for &pre in &cortex_exc_indices {
            let mut cand: Vec<usize> = cortex_all.iter().copied().filter(|&t| t != pre).collect();
            cand.shuffle(&mut rng);
            for &post in cand.iter().take(40) {
                let d = rng.gen_range(delay_dist.0..=delay_dist.1);
                synapses.push(ThermoSynapse::new(pre, post, d, 50));
            }
        }

        // out_syn / in_syn の構築
        let mut out_syn: Vec<Vec<usize>> = vec![Vec::new(); neurons.len()];
        let mut in_syn: Vec<Vec<usize>> = vec![Vec::new(); neurons.len()];
        for (s_idx, s) in synapses.iter().enumerate() {
            out_syn[s.pre].push(s_idx);
            in_syn[s.post].push(s_idx);
        }

        // 配送リングバッファ
        let max_delay = (delay_dist.1 as usize + 2).max(20);
        let delivery_queue = vec![vec![0i32; neurons.len()]; max_delay];

        Self {
            config,
            neurons,
            synapses,
            topology,
            position_index,
            delivery_queue,
            delivery_head: 0,
            max_delay,
            current_time: 0,
            last_growth_time: 0,
            input_neurons,
            output_neurons,
            out_syn,
            in_syn,
            axons_grown: 0,
            axons_pruned: 0,
        }
    }

    pub fn n_neurons(&self) -> usize { self.neurons.len() }
    pub fn n_synapses(&self) -> usize { self.synapses.len() }
    pub fn n_open_synapses(&self) -> usize {
        self.synapses.iter().filter(|s| s.alive).count()
    }
    pub fn n_plastic_synapses(&self) -> usize {
        self.synapses.iter().count()
    }

    /// 全ニューロンの local_entropy 平均と最大
    pub fn entropy_stats(&self) -> (f64, i32) {
        let n = self.neurons.len();
        if n == 0 { return (0.0, 0); }
        let sum: i64 = self.neurons.iter().map(|n| n.local_entropy as i64).sum();
        let max = self.neurons.iter().map(|n| n.local_entropy).max().unwrap_or(0);
        (sum as f64 / n as f64, max)
    }

    /// available_enthalpy 平均
    pub fn enthalpy_mean(&self) -> f64 {
        let n = self.neurons.len();
        if n == 0 { return 0.0; }
        let sum: i64 = self.neurons.iter().map(|n| n.available_enthalpy as i64).sum();
        sum as f64 / n as f64
    }

    /// conductance 分布 (plastic シナプスのみ): (mean, max)
    pub fn conductance_stats(&self) -> (f64, i32) {
        let plastic: Vec<i32> = self.synapses.iter()
            .map(|s| s.conductance).collect();
        if plastic.is_empty() { return (0.0, 0); }
        let sum: i64 = plastic.iter().map(|&v| v as i64).sum();
        let max = plastic.iter().copied().max().unwrap_or(0);
        (sum as f64 / plastic.len() as f64, max)
    }

    // ─── C8: マクロ観察量 (添付コード提案 #8 由来、学習則には使わない) ──────

    /// エントロピー分布の標準偏差 (熱分布の不均一性)
    /// 大きいほど「一部が熱い、他は冷たい」状態
    pub fn entropy_std(&self) -> f64 {
        let n = self.neurons.len();
        if n < 2 { return 0.0; }
        let mean: f64 = self.neurons.iter().map(|n| n.local_entropy as f64).sum::<f64>() / n as f64;
        let var: f64 = self.neurons.iter()
            .map(|n| { let d = n.local_entropy as f64 - mean; d * d })
            .sum::<f64>() / n as f64;
        var.sqrt()
    }

    /// conductance の標準偏差 (plastic シナプスのみ、結線強度の分布幅)
    pub fn conductance_std(&self) -> f64 {
        let plastic: Vec<i32> = self.synapses.iter()
            .map(|s| s.conductance).collect();
        if plastic.len() < 2 { return 0.0; }
        let mean: f64 = plastic.iter().map(|&v| v as f64).sum::<f64>() / plastic.len() as f64;
        let var: f64 = plastic.iter()
            .map(|&v| { let d = v as f64 - mean; d * d })
            .sum::<f64>() / plastic.len() as f64;
        var.sqrt()
    }

    /// 構造のスパース率 = exists=true な plastic シナプスの比率
    /// (固定結線は常に exists=true なので分母に含めるとバイアス。plastic のみ対象)
    pub fn sparsity(&self) -> f64 {
        let plastic_total = self.synapses.iter().count();
        if plastic_total == 0 { return 0.0; }
        let plastic_open = self.synapses.iter().filter(|s| s.alive).count();
        plastic_open as f64 / plastic_total as f64
    }

    /// 単位 step あたりの軸索成長率 (×1000 でスケール表示しやすく)
    pub fn syn_growth_rate(&self) -> f64 {
        if self.current_time <= 0 { return 0.0; }
        (self.axons_grown as f64 * 1000.0) / self.current_time as f64
    }

    /// すべてのマクロ観察量を一度に取得 (snapshot 用に効率化)
    /// 戻り値: (entropy_mean, entropy_max, entropy_std,
    ///         conductance_mean, conductance_max, conductance_std,
    ///         enthalpy_mean, sparsity, syn_growth_rate)
    pub fn macro_observables(&self) -> MacroObservables {
        let (ent_mean, ent_max) = self.entropy_stats();
        let (cond_mean, cond_max) = self.conductance_stats();
        MacroObservables {
            entropy_mean: ent_mean,
            entropy_max: ent_max,
            entropy_std: self.entropy_std(),
            conductance_mean: cond_mean,
            conductance_max: cond_max,
            conductance_std: self.conductance_std(),
            enthalpy_mean: self.enthalpy_mean(),
            sparsity: self.sparsity(),
            syn_growth_rate: self.syn_growth_rate(),
        }
    }

    /// 試行間の状態リセット (membrane と refractory のみクリア、enthalpy と entropy は持ち越し)
    pub fn reset_trial_state(&mut self) {
        for n in &mut self.neurons {
            n.reset_state();
        }
        for slot in &mut self.delivery_queue {
            for v in slot.iter_mut() { *v = 0; }
        }
        self.delivery_head = 0;
    }

    /// 1 クロック進行。external_input は input_neurons に対する外部電流。
    /// 戻り値: 発火したニューロンのインデックスリスト
    /// 直列版 step (検証用、並列版と同じ結果を出す参照実装)。
    /// rayon 並列化前の素直な実装。決定論性検証のためのみ使用。
    pub fn step_sequential(&mut self, external_input: &[i32]) -> Vec<usize> {
        let n = self.neurons.len();
        let mut current_inputs = std::mem::take(&mut self.delivery_queue[self.delivery_head]);
        for (k, &v) in external_input.iter().enumerate() {
            if k < self.input_neurons.len() {
                let target = self.input_neurons[k];
                current_inputs[target] = current_inputs[target].saturating_add(v);
            }
        }
        let t_now = self.current_time;
        let mut fired: Vec<usize> = Vec::new();
        for i in 0..n {
            if self.neurons[i].update(current_inputs[i], t_now) {
                fired.push(i);
            }
        }
        // Fork F-G1-R1: STDP の両方向 (LTP + LTD 復活、Bi & Poo 1998)
        for &pre in &fired {
            // STDP- (LTD): pre 発火、out_syn の post が直近に発火していたら反因果
            for &s_idx in &self.out_syn[pre] {
                let post = self.synapses[s_idx].post;
                let post_trace = self.neurons[post].spike_trace;
                self.synapses[s_idx].update_on_pre_spike_trace(post_trace);
            }
            // STDP+ (LTP): pre 発火を受けて、自分への入射シナプスを更新
            for &s_idx in &self.in_syn[pre] {
                let pre_n = self.synapses[s_idx].pre;
                let pre_trace = self.neurons[pre_n].spike_trace;
                self.synapses[s_idx].update_on_post_spike_trace(pre_trace);
            }
        }
        // 配送: pre neuron 属性で符号判定 (デールの原則)、信号通過で vitality 増加
        for &pre in &fired {
            let pre_is_inh = self.neurons[pre].is_inhibitory;
            for &s_idx in &self.out_syn[pre] {
                // F-fix: 配送判定は alive のみ (信号オン/オフは vitality ベース、conductance は強度のみ)
                let alive = self.synapses[s_idx].alive;
                if !alive { continue; }
                let s_delay = self.synapses[s_idx].delay;
                let s_post = self.synapses[s_idx].post;
                let s_cond = self.synapses[s_idx].conductance;
                let arrival_slot = (self.delivery_head + s_delay as usize) % self.max_delay;
                let scaled = s_cond / SIGNAL_SCALE_DIVISOR;
                let signal = if pre_is_inh { -scaled } else { scaled };
                self.delivery_queue[arrival_slot][s_post] =
                    self.delivery_queue[arrival_slot][s_post].saturating_add(signal);
                // Fork F: 信号通過で vitality 増加 (use-dependent maintenance)
                self.synapses[s_idx].on_transmission();
            }
        }
        // 自然減衰 + exists/alive 更新 (Fork F: 全シナプスが対象、plastic 区別なし)
        for s in &mut self.synapses {
            s.decay();
            let was_alive = s.alive;
            s.update_alive();
            if was_alive && !s.alive {
                self.axons_pruned += 1;
            }
        }
        if t_now - self.last_growth_time >= self.config.axon_growth_interval {
            let prev_total = self.synapses.len();
            let (created, _attempted) = axon_growth_step(
                &mut self.neurons,
                &mut self.synapses,
                &self.topology,
                &self.position_index,
            );
            for new_idx in prev_total..self.synapses.len() {
                let s = &self.synapses[new_idx];
                self.out_syn[s.pre].push(new_idx);
                self.in_syn[s.post].push(new_idx);
            }
            self.axons_grown += created as u64;
            self.last_growth_time = t_now;
        }
        for v in current_inputs.iter_mut() { *v = 0; }
        self.delivery_queue[self.delivery_head] = current_inputs;
        self.delivery_head = (self.delivery_head + 1) % self.max_delay;
        self.current_time += 1;
        fired
    }

    pub fn step(&mut self, external_input: &[i32]) -> Vec<usize> {
        let n = self.neurons.len();

        // 1) 配送スロットから今クロックの入力を取り出す (drain して使い切る)
        let mut current_inputs = std::mem::take(&mut self.delivery_queue[self.delivery_head]);

        // 2) 外部入力 (input_neurons に対応) を加算
        for (k, &v) in external_input.iter().enumerate() {
            if k < self.input_neurons.len() {
                let target = self.input_neurons[k];
                current_inputs[target] = current_inputs[target].saturating_add(v);
            }
        }

        // 3) 各ニューロンの update を呼び、発火を集める (rayon 並列化)
        //    各ニューロン独立、整数演算なので加算順序によらず結果一意
        //    par_iter_mut().enumerate().filter_map().collect() は元の順序を保つ
        let t_now = self.current_time;
        let inputs_ref: &[i32] = &current_inputs;
        let fired: Vec<usize> = self.neurons.par_iter_mut().enumerate()
            .filter_map(|(i, neuron)| {
                if neuron.update(inputs_ref[i], t_now) { Some(i) } else { None }
            })
            .collect();
        let _ = n;

        // 4) 発火したニューロンのスパイク配送 + STDP 更新
        // Fork F-G1-R1: STDP の両方向 (LTP + LTD 復活、Bi & Poo 1998)
        for &pre in &fired {
            // STDP- (LTD): pre 発火、out_syn の post が直近に発火していたら反因果
            for &s_idx in &self.out_syn[pre] {
                let post = self.synapses[s_idx].post;
                let post_trace = self.neurons[post].spike_trace;
                self.synapses[s_idx].update_on_pre_spike_trace(post_trace);
            }
            // STDP+ (LTP): pre 発火を受けて、自分への入射シナプスを更新
            for &s_idx in &self.in_syn[pre] {
                let pre_n = self.synapses[s_idx].pre;
                let pre_trace = self.neurons[pre_n].spike_trace;
                self.synapses[s_idx].update_on_post_spike_trace(pre_trace);
            }
        }

        // 5) スパイク配送 (Fork F: pre neuron 属性で符号判定、vitality 更新)
        for &pre in &fired {
            let pre_is_inh = self.neurons[pre].is_inhibitory;
            for &s_idx in &self.out_syn[pre] {
                // F-fix: 配送判定は alive のみ (信号オン/オフは vitality ベース、conductance は強度のみ)
                let alive = self.synapses[s_idx].alive;
                if !alive { continue; }
                let s_delay = self.synapses[s_idx].delay;
                let s_post = self.synapses[s_idx].post;
                let s_cond = self.synapses[s_idx].conductance;
                let arrival_slot = (self.delivery_head + s_delay as usize) % self.max_delay;
                let scaled = s_cond / SIGNAL_SCALE_DIVISOR;
                let signal = if pre_is_inh { -scaled } else { scaled };
                self.delivery_queue[arrival_slot][s_post] =
                    self.delivery_queue[arrival_slot][s_post].saturating_add(signal);
                // Fork F: 信号通過で vitality 増加
                self.synapses[s_idx].on_transmission();
            }
        }

        // 6) 全シナプスの自然減衰 + exists 更新 + プルーン検出 (rayon 並列化)
        //    各シナプス独立、axons_pruned は AtomicU64 で集約
        let pruned_counter = AtomicU64::new(0);
        self.synapses.par_iter_mut().for_each(|s| {
            s.decay();
            let was_alive = s.alive;
            s.update_alive();
            if was_alive && !s.alive {
                pruned_counter.fetch_add(1, Ordering::Relaxed);
            }
        });
        self.axons_pruned += pruned_counter.into_inner();

        // 7) 軸索成長 (N step に 1 回)
        if t_now - self.last_growth_time >= self.config.axon_growth_interval {
            let prev_total = self.synapses.len();
            let (created, _attempted) = axon_growth_step(
                &mut self.neurons,
                &mut self.synapses,
                &self.topology,
                &self.position_index,
            );
            // 新規シナプスのインデックスを out_syn / in_syn に登録
            for new_idx in prev_total..self.synapses.len() {
                let s = &self.synapses[new_idx];
                self.out_syn[s.pre].push(new_idx);
                self.in_syn[s.post].push(new_idx);
            }
            self.axons_grown += created as u64;
            self.last_growth_time = t_now;
        }

        // 8) 配送スロットをクリアして戻す + リングバッファ進行
        for v in current_inputs.iter_mut() { *v = 0; }
        self.delivery_queue[self.delivery_head] = current_inputs;
        self.delivery_head = (self.delivery_head + 1) % self.max_delay;

        // 9) 時間進行
        self.current_time += 1;
        fired
    }

    /// 出力ニューロンとして何番目かを返す
    pub fn output_index_of(&self, neuron_id: usize) -> Option<usize> {
        self.output_neurons.iter().position(|&n| n == neuron_id)
    }

    /// 概算メモリ
    pub fn memory_bytes(&self) -> usize {
        let neuron_b = self.neurons.len() * std::mem::size_of::<ThermoNeuron>();
        let syn_b = self.synapses.len() * std::mem::size_of::<ThermoSynapse>();
        let buf_b = self.delivery_queue.len() * self.neurons.len() * 4;
        neuron_b + syn_b + buf_b
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_small_network() {
        let mut cfg = ThermoNetworkConfig::default();
        cfg.grid_width = 10;
        cfg.grid_height = 10;
        cfg.n_input = 5;
        cfg.n_output = 10;
        cfg.n_excitatory = 60;
        cfg.n_inhibitory = 15;
        cfg.input_fanout = 10;
        let net = ThermoNetwork::new(cfg);
        assert!(net.n_neurons() >= 5 + 60 + 15);
        assert!(net.n_synapses() > 0);
    }
}
