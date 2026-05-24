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
}

impl Default for ThermoNetworkConfig {
    fn default() -> Self {
        Self {
            grid_width: 20,
            grid_height: 20,
            n_input: 20,
            n_output: 40,
            n_excitatory: 320, // 残り 380 個 (400 - 20入力) のうち 320 興奮
            n_inhibitory: 60,  // 60 抑制
            input_fanout: 80,
            delay_range: (2, 40),
            seed: 300,
            axon_growth_interval: GROWTH_INTERVAL,
            dt_ms: 0.5,
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

        // ニューロン配置:
        //   入力ニューロン: 上端の左 n_input 個 (x=0..n_input, y=0)
        //   出力ニューロン: 下端の左 n_output 個 (x=0..n_output, y=grid_h-1)
        //   抑制性: 上端の残り or 出力の下、または内部に分散
        //   興奮性: 残りの内部セルに分散
        //
        // 簡単のため: 1 セル 1 ニューロン制で配置。grid_w * grid_h >= n_total 必須。
        let n_total = config.n_input + config.n_excitatory + config.n_inhibitory;
        assert!(
            (grid_w * grid_h) as usize >= n_total,
            "grid too small: {} cells but {} neurons", grid_w * grid_h, n_total
        );

        let mut neurons: Vec<ThermoNeuron> = Vec::with_capacity(n_total);
        let mut input_neurons: Vec<usize> = Vec::new();
        let mut output_neurons: Vec<usize> = Vec::new();

        // (1) 入力ニューロン: 上端の左端から配置 (y=0, x=0..n_input)
        for i in 0..config.n_input {
            let x = i as i32 % grid_w;
            let y = 0;
            neurons.push(ThermoNeuron::input((x, y)));
            input_neurons.push(neurons.len() - 1);
        }

        // (2) 出力ニューロン (興奮性) は下端から (y=grid_h-1, x=0..n_output)
        // n_output 個を興奮性として下端に配置。これは興奮性カウントに含める。
        let mut placed_exc_at_bottom = 0;
        for i in 0..config.n_output {
            let x = (i as i32) % grid_w;
            let y = grid_h - 1;
            neurons.push(ThermoNeuron::excitatory((x, y)));
            output_neurons.push(neurons.len() - 1);
            placed_exc_at_bottom += 1;
        }

        // (3) 残りの興奮性ニューロンを内部 (y=1..grid_h-1) に配置
        let remaining_exc = config.n_excitatory - placed_exc_at_bottom;
        let mut cursor_x = 0i32;
        let mut cursor_y = 1i32;
        let mut placed_remaining_exc = 0;
        while placed_remaining_exc < remaining_exc {
            if cursor_y >= grid_h - 1 {
                // 下端は出力で使ったので、内部行で配置
                cursor_y = 1;
                cursor_x += 1;
                if cursor_x >= grid_w { break; }
            }
            neurons.push(ThermoNeuron::excitatory((cursor_x, cursor_y)));
            placed_remaining_exc += 1;
            cursor_y += 1;
        }

        // (4) 抑制性ニューロンを残りの内部セルに配置
        let mut placed_inh = 0;
        cursor_x = 0;
        cursor_y = 1;
        while placed_inh < config.n_inhibitory {
            if cursor_y >= grid_h - 1 {
                cursor_y = 1;
                cursor_x += 1;
                if cursor_x >= grid_w { break; }
            }
            // 既存ニューロンとの衝突を避ける (簡易: position が既出ならスキップ)
            let occupied = neurons.iter().any(|n| n.position == (cursor_x, cursor_y));
            if !occupied {
                neurons.push(ThermoNeuron::inhibitory((cursor_x, cursor_y)));
                placed_inh += 1;
            }
            cursor_y += 1;
        }

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
                // 固定結線: 最初から exists=true、conductance も十分高め
                synapses.push(ThermoSynapse::new_fixed(inp, t, d, OPEN_THRESHOLD + 50, false));
            }
        }

        // (6) 抑制→興奮の固定結線 (各抑制ニューロンが近傍の興奮を抑制)
        // ここではシンプルに、各抑制ニューロンが grid 上の近傍興奮に固定結線
        let exc_indices: Vec<usize> = neurons.iter().enumerate()
            .filter(|(i, n)| !n.is_inhibitory && !input_neurons.contains(i))
            .map(|(i, _)| i)
            .collect();
        for (idx, n) in neurons.iter().enumerate() {
            if !n.is_inhibitory { continue; }
            // 興奮性のうちランダムに 20 個に抑制結線
            let mut targets: Vec<usize> = exc_indices.clone();
            targets.shuffle(&mut rng);
            for &t in targets.iter().take(20) {
                synapses.push(ThermoSynapse::new_fixed(idx, t, 2, OPEN_THRESHOLD + 50, true));
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
        self.synapses.iter().filter(|s| s.exists).count()
    }
    pub fn n_plastic_synapses(&self) -> usize {
        self.synapses.iter().filter(|s| s.plastic).count()
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
            .filter(|s| s.plastic).map(|s| s.conductance).collect();
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
            .filter(|s| s.plastic).map(|s| s.conductance).collect();
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
        let plastic_total = self.synapses.iter().filter(|s| s.plastic).count();
        if plastic_total == 0 { return 0.0; }
        let plastic_open = self.synapses.iter().filter(|s| s.plastic && s.exists).count();
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
        // step_sequential: 旧版 last_spike_time ベース STDP (B4 等価性検証用に保持)
        for &pre in &fired {
            for &s_idx in &self.out_syn[pre] {
                let post = self.synapses[s_idx].post;
                let post_last = self.neurons[post].last_spike_time;
                self.synapses[s_idx].update_on_pre_spike(post_last, t_now);
            }
            for &s_idx in &self.in_syn[pre] {
                let pre_n = self.synapses[s_idx].pre;
                let pre_last = self.neurons[pre_n].last_spike_time;
                self.synapses[s_idx].update_on_post_spike(pre_last, t_now);
            }
        }
        for &pre in &fired {
            for &s_idx in &self.out_syn[pre] {
                let s = &self.synapses[s_idx];
                if !s.exists { continue; }
                let arrival_slot = (self.delivery_head + s.delay as usize) % self.max_delay;
                let scaled = s.conductance / SIGNAL_SCALE_DIVISOR;
                let signal = if s.is_inhibitory { -scaled } else { scaled };
                let target = s.post;
                self.delivery_queue[arrival_slot][target] =
                    self.delivery_queue[arrival_slot][target].saturating_add(signal);
            }
        }
        for s in &mut self.synapses {
            s.decay();
            let was_open = s.exists;
            s.update_existence();
            if was_open && !s.exists {
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
        for &pre in &fired {
            // (a) STDP+: post 発火を受けて、自分への入射シナプスを更新 (pre が直近因果か)
            // ここでは pre が発火したので、「pre 発火」イベント
            // (b) STDP-: pre 発火、out シナプスを処理 (post が直近に発火していたら LTD)
            for &s_idx in &self.out_syn[pre] {
                let post = self.synapses[s_idx].post;
                let post_last = self.neurons[post].last_spike_time;
                self.synapses[s_idx].update_on_pre_spike(post_last, t_now);
            }
            // STDP+: 自分が post として、入射シナプスの pre が直近因果か
            for &s_idx in &self.in_syn[pre] {
                let pre_n = self.synapses[s_idx].pre;
                let pre_last = self.neurons[pre_n].last_spike_time;
                self.synapses[s_idx].update_on_post_spike(pre_last, t_now);
            }
        }

        // 5) スパイク配送 (遅延付き) — 出射シナプスを通じて配送
        //    案A 調整: signal = conductance / SIGNAL_SCALE_DIVISOR
        for &pre in &fired {
            for &s_idx in &self.out_syn[pre] {
                let s = &self.synapses[s_idx];
                if !s.exists { continue; }
                let arrival_slot = (self.delivery_head + s.delay as usize) % self.max_delay;
                let scaled = s.conductance / SIGNAL_SCALE_DIVISOR;
                let signal = if s.is_inhibitory { -scaled } else { scaled };
                let target = s.post;
                self.delivery_queue[arrival_slot][target] =
                    self.delivery_queue[arrival_slot][target].saturating_add(signal);
            }
        }

        // 6) 全シナプスの自然減衰 + exists 更新 + プルーン検出 (rayon 並列化)
        //    各シナプス独立、axons_pruned は AtomicU64 で集約
        let pruned_counter = AtomicU64::new(0);
        self.synapses.par_iter_mut().for_each(|s| {
            s.decay();
            let was_open = s.exists;
            s.update_existence();
            if was_open && !s.exists {
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
