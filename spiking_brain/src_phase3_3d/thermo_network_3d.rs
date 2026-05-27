//! 熱力学的ネットワーク 3D 統合 (Phase 3)
//!
//! src_phase2_f/thermo_network.rs から派生。 2D 8 近傍 → 3D HCP 12 近傍 + 柱状配置。
//!
//! 配置 (M1_3D_DESIGN.md):
//!   - grid 5×6×22 = 660 セル
//!   - 底面 z=0 (30 セル): 入力 20 (中央 x∈[0..5], y∈[1..5]) + 内部 10 (y=0, y=5)
//!   - 内部層 z=1..20 (600 セル): 興奮 + 抑制 18%
//!   - 上面 z=21 (30 セル): 出力 30 完全充填
//!
//! 物理プロセス (update / STDP / 配送 / 軸索成長 / 自然減衰) は 2D 版と完全に同一。
//! 違いは「近傍トポロジー」と「配置」のみ。

use super::axon_growth_3d::{axon_growth_step_3d, build_position_index_3d};
use super::thermo_neuron_3d::ThermoNeuron3d;
use super::thermo_synapse_3d::{ThermoSynapse3d, OPEN_THRESHOLD};
use super::topology3d::Topology3d;
use rand::prelude::*;
use rayon::prelude::*;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};

pub const GROWTH_INTERVAL: i32 = 200;
pub const SIGNAL_SCALE_DIVISOR: i32 = 10;

/// 過剰接続 (cortex 内 plastic 結線) の幾何モード
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OverconnectMode {
    /// 全 cortex からランダム fanout (2D 版互換、 3D では希薄になる)
    Random,
    /// HCP 12 近傍のみ (純粋局所、 仮説 C-1)
    HcpLocal,
    /// HCP 近傍 + 2-hop (z±1, x/y±1 セル) で〜50 候補プール (仮説 C-2)
    HcpLocalPlus2hop,
    /// 全 cortex ペア (発生学的全結合、 仮説 D-1)
    /// 530 exc × 639 post ≒ 339,000 結線。 STDP + vitality 刈り取りで自然な sparse 構造選択を期待
    FullPair,
}

#[derive(Debug, Clone)]
pub struct ThermoNetwork3dConfig {
    pub grid_w: i32,
    pub grid_h: i32,
    pub grid_d: i32,
    pub n_input: usize,
    pub n_output: usize,
    pub n_excitatory: usize,
    pub n_inhibitory: usize,
    pub input_fanout: usize,
    pub delay_range: (i32, i32),
    pub seed: u64,
    pub axon_growth_interval: i32,
    pub dt_ms: f64,
    pub enable_up_down: bool,
    /// 過剰接続モード (デフォルト Random、 2D 互換)
    pub overconnect_mode: OverconnectMode,
    /// 過剰接続 1 ニューロンあたりの fanout (Random では 40、 HCP では 12 が天井)
    pub overconnect_fanout: usize,
    /// 層ごとの (w, h)。 None なら全層 (grid_w, grid_h) full block (default)
    /// Some(v) なら v[z] = (w, h) で cone shape (入力から出力へ徐々に増える)
    /// 各層は (x ∈ [0..w], y ∈ [0..h]) の左下原点矩形
    pub layer_grids: Option<Vec<(i32, i32)>>,
}

impl ThermoNetwork3dConfig {
    /// 指定 grid 寸法から派生する設定 (抑制比 18%、 配置ルールは default と同じ)
    ///   - 入力: 底面 z=0 中央 grid_w × (grid_h - 2) セル
    ///   - 内部底面: z=0 の y=0 / y=grid_h-1 (2 × grid_w セル)
    ///   - 内部中層: z=1..grid_d-1 全体 (grid_w × grid_h × (grid_d - 2) セル)
    ///   - 出力: 上面 z=grid_d-1 完全充填 (grid_w × grid_h セル)
    /// 例: for_grid(5, 6, 22) → 660 セル (デフォルト相当)
    ///     for_grid(5, 6, 11) → 330 セル (柱半分、 E-2 用)
    pub fn for_grid(grid_w: i32, grid_h: i32, grid_d: i32) -> Self {
        assert!(grid_w >= 3 && grid_h >= 3 && grid_d >= 3,
            "grid 寸法は各軸 3 以上が必要");
        let n_input = (grid_w * (grid_h - 2)) as usize;
        let n_output = (grid_w * grid_h) as usize;
        let total = (grid_w * grid_h * grid_d) as usize;
        let internal = total - n_input - n_output;
        // 抑制 18% (四捨五入)
        let n_inhibitory = ((internal as f64) * 0.18).round() as usize;
        let n_excitatory = n_output + (internal - n_inhibitory);
        Self {
            grid_w, grid_h, grid_d,
            n_input, n_output, n_excitatory, n_inhibitory,
            input_fanout: 80,
            delay_range: (2, 40),
            seed: 300,
            axon_growth_interval: GROWTH_INTERVAL,
            dt_ms: 0.5,
            enable_up_down: false,
            overconnect_mode: OverconnectMode::Random,
            overconnect_fanout: 40,
            layer_grids: None,
        }
    }

    /// Cone 形状: 入力層 (z=0) から出力層 (z=grid_d-1) へ徐々にニューロン数を増やす
    /// layer_grids[z] = (w, h) で z 層の cells は x∈[0..w], y∈[0..h] に配置
    /// - 自動計算: grid_w, grid_h は全層 max、 grid_d = len(layer_grids)
    /// - n_input = layer_grids[0].0 × layer_grids[0].1 (z=0 全て入力)
    /// - n_output = layer_grids[N-1].0 × layer_grids[N-1].1 (z=N-1 全て出力)
    /// - 抑制比 18%
    pub fn for_cone(layer_grids: Vec<(i32, i32)>) -> Self {
        assert!(!layer_grids.is_empty(), "layer_grids must not be empty");
        let grid_d = layer_grids.len() as i32;
        let grid_w = layer_grids.iter().map(|&(w, _)| w).max().unwrap();
        let grid_h = layer_grids.iter().map(|&(_, h)| h).max().unwrap();

        let (z0_w, z0_h) = layer_grids[0];
        let n_input = (z0_w * z0_h) as usize;
        let (zN_w, zN_h) = layer_grids[(grid_d - 1) as usize];
        let n_output = (zN_w * zN_h) as usize;

        let total: usize = layer_grids.iter()
            .map(|&(w, h)| (w * h) as usize)
            .sum();
        let internal = total - n_input - n_output;
        let n_inhibitory = ((internal as f64) * 0.18).round() as usize;
        let n_excitatory = n_output + (internal - n_inhibitory);

        Self {
            grid_w, grid_h, grid_d,
            n_input, n_output, n_excitatory, n_inhibitory,
            input_fanout: 80,
            delay_range: (2, 40),
            seed: 300,
            axon_growth_interval: GROWTH_INTERVAL,
            dt_ms: 0.5,
            enable_up_down: false,
            overconnect_mode: OverconnectMode::Random,
            overconnect_fanout: 40,
            layer_grids: Some(layer_grids),
        }
    }

    /// デフォルト cone shape (22 層、 入力 20 → 出力 30、 561 cells)
    /// z=0: 4×5=20 (入力)
    /// z=1-3: 4×5=20 (内部)
    /// z=4-7: 4×6=24 (内部)
    /// z=8-14: 5×5=25 (内部)
    /// z=15-20: 5×6=30 (内部)
    /// z=21: 5×6=30 (出力)
    pub fn cone_default() -> Self {
        let mut g = Vec::with_capacity(22);
        // z=0..=3: 4×5
        for _ in 0..=3 { g.push((4, 5)); }
        // z=4..=7: 4×6
        for _ in 4..=7 { g.push((4, 6)); }
        // z=8..=14: 5×5
        for _ in 8..=14 { g.push((5, 5)); }
        // z=15..=21: 5×6
        for _ in 15..=21 { g.push((5, 6)); }
        assert_eq!(g.len(), 22);
        Self::for_cone(g)
    }

    /// Spindle shape (紡錘形): 入力 20 → 中間ピーク 40 → 出力 30
    /// 生物の皮質 L4 (granular layer) が最も密という構造を模倣
    /// 22 層、 775 cells
    /// z=0: 4×5=20 (入力)
    /// z=1: 5×5=25
    /// z=2: 5×6=30
    /// z=3: 5×7=35
    /// z=4-15: 5×8=40 (ピーク 12 層)
    /// z=16: 5×7=35
    /// z=17-20: 5×6=30
    /// z=21: 5×6=30 (出力)
    pub fn spindle_default() -> Self {
        let mut g = Vec::with_capacity(22);
        g.push((4, 5));   // z=0: 20 (input)
        g.push((5, 5));   // z=1: 25
        g.push((5, 6));   // z=2: 30
        g.push((5, 7));   // z=3: 35
        for _ in 4..=15 { g.push((5, 8)); }  // z=4-15: 40 (peak)
        g.push((5, 7));   // z=16: 35
        for _ in 17..=20 { g.push((5, 6)); }  // z=17-20: 30
        g.push((5, 6));   // z=21: 30 (output)
        assert_eq!(g.len(), 22);
        Self::for_cone(g)  // for_cone でも spindle でも内部実装は同じ (layer_grids 使用)
    }
}

impl Default for ThermoNetwork3dConfig {
    fn default() -> Self {
        // 5×6×22 = 660、 入力 20、 出力 30、 内部 610
        //   内部抑制 110 (18% of 610 ≒ 18.0%)
        //   内部興奮 500
        //   出力 30 (興奮)
        //   n_excitatory = 500 + 30 = 530
        //   n_inhibitory = 110
        //   合計 = 20 + 530 + 110 = 660 ✓
        Self {
            grid_w: 5,
            grid_h: 6,
            grid_d: 22,
            n_input: 20,
            n_output: 30,
            n_excitatory: 530,
            n_inhibitory: 110,
            input_fanout: 80,
            delay_range: (2, 40),
            seed: 300,
            axon_growth_interval: GROWTH_INTERVAL,
            dt_ms: 0.5,
            enable_up_down: false,
            overconnect_mode: OverconnectMode::Random,
            overconnect_fanout: 40,
            layer_grids: None,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct MacroObservables3d {
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

pub struct ThermoNetwork3d {
    pub config: ThermoNetwork3dConfig,
    pub neurons: Vec<ThermoNeuron3d>,
    pub synapses: Vec<ThermoSynapse3d>,
    pub topology: Topology3d,
    pub position_index: HashMap<(i32, i32, i32), Vec<usize>>,

    delivery_queue: Vec<Vec<i32>>,
    delivery_head: usize,
    max_delay: usize,

    pub current_time: i32,
    pub last_growth_time: i32,

    pub input_neurons: Vec<usize>,
    pub output_neurons: Vec<usize>,

    pub out_syn: Vec<Vec<usize>>,
    pub in_syn: Vec<Vec<usize>>,

    pub axons_grown: u64,
    pub axons_pruned: u64,
}

impl ThermoNetwork3d {
    pub fn new(config: ThermoNetwork3dConfig) -> Self {
        let mut rng = StdRng::seed_from_u64(config.seed);

        let topology = Topology3d::new(config.grid_w, config.grid_h, config.grid_d);
        let n_total = config.n_input + config.n_excitatory + config.n_inhibitory;
        // Block モード (layer_grids=None) は grid 全充填、 Cone モードは部分充填
        if config.layer_grids.is_none() {
            assert_eq!(
                topology.total_cells(),
                n_total,
                "grid full-fill: {}×{}×{}={} cells, n_total={}",
                config.grid_w, config.grid_h, config.grid_d,
                topology.total_cells(), n_total
            );
        } else {
            // Cone は grid に対し sparse
            assert!(n_total <= topology.total_cells(),
                "cone n_total {} must not exceed grid cells {}",
                n_total, topology.total_cells());
        }

        // ─── 位置リスト構築 ───
        // 2 モード:
        //   Block (layer_grids=None): 旧来の 5×6×22、 z=0 中央 4×5=20 入力 + 角 10 内部、
        //                              z=1..20 全 30 cell 内部、 z=21 全 30 出力
        //   Cone (layer_grids=Some):  per-layer (w, h)、 z=0 全て入力、
        //                              z=1..N-1 内部、 z=N-1 全て出力 (左下原点矩形)
        let (input_positions, internal_positions, output_positions) =
            if let Some(layer_grids) = &config.layer_grids {
                // Cone shape
                let (z0_w, z0_h) = layer_grids[0];
                let mut inp = Vec::new();
                for y in 0..z0_h {
                    for x in 0..z0_w {
                        inp.push((x, y, 0));
                    }
                }
                let mut intl = Vec::new();
                for z in 1..(config.grid_d - 1) {
                    let (w, h) = layer_grids[z as usize];
                    for y in 0..h {
                        for x in 0..w {
                            intl.push((x, y, z));
                        }
                    }
                }
                let (zN_w, zN_h) = layer_grids[(config.grid_d - 1) as usize];
                let mut outp = Vec::new();
                for y in 0..zN_h {
                    for x in 0..zN_w {
                        outp.push((x, y, config.grid_d - 1));
                    }
                }
                (inp, intl, outp)
            } else {
                // Block (default): 旧来の hardcoded layout
                let mut inp: Vec<(i32, i32, i32)> = Vec::with_capacity(20);
                for y in 1..(config.grid_h - 1) {
                    for x in 0..config.grid_w {
                        inp.push((x, y, 0));
                    }
                }
                let mut bottom_internal: Vec<(i32, i32, i32)> = Vec::new();
                for x in 0..config.grid_w {
                    bottom_internal.push((x, 0, 0));
                    bottom_internal.push((x, config.grid_h - 1, 0));
                }
                let mut middle_internal: Vec<(i32, i32, i32)> = Vec::new();
                for z in 1..(config.grid_d - 1) {
                    for y in 0..config.grid_h {
                        for x in 0..config.grid_w {
                            middle_internal.push((x, y, z));
                        }
                    }
                }
                let mut intl = Vec::new();
                intl.extend(bottom_internal.iter().copied());
                intl.extend(middle_internal.iter().copied());

                let mut outp = Vec::new();
                for y in 0..config.grid_h {
                    for x in 0..config.grid_w {
                        outp.push((x, y, config.grid_d - 1));
                    }
                }
                (inp, intl, outp)
            };
        assert_eq!(input_positions.len(), config.n_input,
            "input positions: {} vs n_input {}", input_positions.len(), config.n_input);
        let expected_internal = config.n_excitatory - config.n_output + config.n_inhibitory;
        assert_eq!(internal_positions.len(), expected_internal,
            "internal positions: {} vs expected {}",
            internal_positions.len(), expected_internal);
        assert_eq!(output_positions.len(), config.n_output,
            "output positions: {} vs n_output {}",
            output_positions.len(), config.n_output);

        // ─── ニューロン生成 ───
        let mut neurons: Vec<ThermoNeuron3d> = Vec::with_capacity(n_total);
        let mut input_neurons: Vec<usize> = Vec::new();
        let mut output_neurons: Vec<usize> = Vec::new();

        // (1) 入力
        for &pos in &input_positions {
            neurons.push(ThermoNeuron3d::input(pos));
            input_neurons.push(neurons.len() - 1);
        }

        // (2) 出力 (興奮)
        for &pos in &output_positions {
            neurons.push(ThermoNeuron3d::excitatory(pos));
            output_neurons.push(neurons.len() - 1);
        }

        // (3) 内部 (興奮 + 抑制 18%)
        //     internal_positions から決定論的乱択で抑制配置
        let mut idx_pool: Vec<usize> = (0..internal_positions.len()).collect();
        idx_pool.shuffle(&mut rng);
        let inh_set: std::collections::HashSet<usize> =
            idx_pool.into_iter().take(config.n_inhibitory).collect();

        let mut placed_inh = 0usize;
        let mut placed_internal_exc = 0usize;
        for (i, &pos) in internal_positions.iter().enumerate() {
            if inh_set.contains(&i) {
                neurons.push(ThermoNeuron3d::inhibitory(pos));
                placed_inh += 1;
            } else {
                neurons.push(ThermoNeuron3d::excitatory(pos));
                placed_internal_exc += 1;
            }
        }
        assert_eq!(placed_inh, config.n_inhibitory);
        assert_eq!(placed_internal_exc, config.n_excitatory - config.n_output);

        // ─── Spontaneous activity 個体差 ───
        for (idx, n) in neurons.iter_mut().enumerate() {
            if n.spontaneous_input == 0 && n.leak == 0 {
                continue;
            }
            n.spontaneous_input = (idx as i32) % 4;
        }

        // ─── UP/DOWN 状態 (有効化時のみ) ───
        if config.enable_up_down {
            for (idx, n) in neurons.iter_mut().enumerate() {
                if !n.generates_entropy && n.spontaneous_input == 0 {
                    continue;
                }
                n.up_period = 200 + ((idx as i32) * 31) % 200;
                n.down_period = 300 + ((idx as i32) * 37) % 300;
                n.up_state = (idx % 2) == 0;
                n.up_down_counter = (idx as i32) % 100;
                n.up_offset = 20;
            }
        }

        let position_index = build_position_index_3d(&neurons);

        // ─── 固定結線: 入力→皮質 ───
        let mut synapses: Vec<ThermoSynapse3d> = Vec::new();
        let delay_dist = config.delay_range;
        let cortex_indices: Vec<usize> = (config.n_input..neurons.len()).collect();
        for &inp in &input_neurons {
            let mut targets: Vec<usize> = cortex_indices.clone();
            targets.shuffle(&mut rng);
            for &t in targets.iter().take(config.input_fanout) {
                let d = rng.gen_range(delay_dist.0..=delay_dist.1);
                synapses.push(ThermoSynapse3d::new(inp, t, d, OPEN_THRESHOLD + 50));
            }
        }

        // ─── 抑制→興奮 初期結線 ───
        let exc_indices: Vec<usize> = neurons.iter().enumerate()
            .filter(|(i, n)| !n.is_inhibitory && !input_neurons.contains(i))
            .map(|(i, _)| i)
            .collect();
        for (idx, n) in neurons.iter().enumerate() {
            if !n.is_inhibitory { continue; }
            let mut targets: Vec<usize> = exc_indices.clone();
            targets.shuffle(&mut rng);
            for &t in targets.iter().take(20) {
                synapses.push(ThermoSynapse3d::new(idx, t, 2, OPEN_THRESHOLD + 50));
            }
        }

        // ─── 過剰接続 (cortex_exc → 候補プール) ───
        // モードで候補プールが変わる:
        //   Random:           cortex_all 全体 (2D 互換、 3D では希薄)
        //   HcpLocal:         pre の HCP 12 近傍のみ (純粋局所、 C-1)
        //   HcpLocalPlus2hop: HCP 近傍 + 2-hop (〜50 候補、 C-2)
        let cortex_exc_indices: Vec<usize> = neurons.iter().enumerate()
            .filter(|(i, n)| !n.is_inhibitory && !input_neurons.contains(i))
            .map(|(i, _)| i)
            .collect();
        let cortex_all: Vec<usize> = (config.n_input..neurons.len()).collect();
        let cortex_set: std::collections::HashSet<usize> =
            cortex_all.iter().copied().collect();
        let input_set: std::collections::HashSet<usize> =
            input_neurons.iter().copied().collect();

        for &pre in &cortex_exc_indices {
            let mut cand: Vec<usize> = match config.overconnect_mode {
                OverconnectMode::Random => {
                    cortex_all.iter().copied().filter(|&t| t != pre).collect()
                }
                OverconnectMode::HcpLocal => {
                    // pre の物理位置から HCP 12 近傍を取得 → cortex メンバのみ抽出
                    let pos = neurons[pre].position;
                    topology.neighbors(pos).into_iter()
                        .filter_map(|np| {
                            position_index.get(&np).map(|v| v.iter().copied())
                        })
                        .flatten()
                        .filter(|&t| t != pre
                            && cortex_set.contains(&t)
                            && !input_set.contains(&t))
                        .collect()
                }
                OverconnectMode::FullPair => {
                    // 全 cortex ペア (input は除外、 自身も除外)
                    cortex_all.iter().copied().filter(|&t| t != pre).collect()
                }
                OverconnectMode::HcpLocalPlus2hop => {
                    // HCP 12 近傍 + 2-hop (各 HCP 近傍からさらに HCP 近傍)
                    let pos = neurons[pre].position;
                    let mut pool: std::collections::HashSet<usize> = std::collections::HashSet::new();
                    let hop1 = topology.neighbors(pos);
                    for np in &hop1 {
                        if let Some(v) = position_index.get(np) {
                            for &nidx in v {
                                if nidx != pre
                                    && cortex_set.contains(&nidx)
                                    && !input_set.contains(&nidx)
                                {
                                    pool.insert(nidx);
                                }
                            }
                        }
                        // 2-hop: np の近傍も追加
                        for np2 in topology.neighbors(*np) {
                            if let Some(v) = position_index.get(&np2) {
                                for &nidx in v {
                                    if nidx != pre
                                        && cortex_set.contains(&nidx)
                                        && !input_set.contains(&nidx)
                                    {
                                        pool.insert(nidx);
                                    }
                                }
                            }
                        }
                    }
                    pool.into_iter().collect()
                }
            };
            cand.shuffle(&mut rng);
            for &post in cand.iter().take(config.overconnect_fanout) {
                let d = rng.gen_range(delay_dist.0..=delay_dist.1);
                synapses.push(ThermoSynapse3d::new(pre, post, d, 50));
            }
        }

        // out_syn / in_syn
        let mut out_syn: Vec<Vec<usize>> = vec![Vec::new(); neurons.len()];
        let mut in_syn: Vec<Vec<usize>> = vec![Vec::new(); neurons.len()];
        for (s_idx, s) in synapses.iter().enumerate() {
            out_syn[s.pre].push(s_idx);
            in_syn[s.post].push(s_idx);
        }

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

    pub fn entropy_stats(&self) -> (f64, i32) {
        let n = self.neurons.len();
        if n == 0 { return (0.0, 0); }
        let sum: i64 = self.neurons.iter().map(|n| n.local_entropy as i64).sum();
        let max = self.neurons.iter().map(|n| n.local_entropy).max().unwrap_or(0);
        (sum as f64 / n as f64, max)
    }

    pub fn enthalpy_mean(&self) -> f64 {
        let n = self.neurons.len();
        if n == 0 { return 0.0; }
        let sum: i64 = self.neurons.iter().map(|n| n.available_enthalpy as i64).sum();
        sum as f64 / n as f64
    }

    pub fn conductance_stats(&self) -> (f64, i32) {
        if self.synapses.is_empty() { return (0.0, 0); }
        let sum: i64 = self.synapses.iter().map(|s| s.conductance as i64).sum();
        let max = self.synapses.iter().map(|s| s.conductance).max().unwrap_or(0);
        (sum as f64 / self.synapses.len() as f64, max)
    }

    pub fn entropy_std(&self) -> f64 {
        let n = self.neurons.len();
        if n < 2 { return 0.0; }
        let mean: f64 = self.neurons.iter().map(|n| n.local_entropy as f64).sum::<f64>() / n as f64;
        let var: f64 = self.neurons.iter()
            .map(|n| { let d = n.local_entropy as f64 - mean; d * d })
            .sum::<f64>() / n as f64;
        var.sqrt()
    }

    pub fn conductance_std(&self) -> f64 {
        if self.synapses.len() < 2 { return 0.0; }
        let mean: f64 = self.synapses.iter().map(|s| s.conductance as f64).sum::<f64>()
            / self.synapses.len() as f64;
        let var: f64 = self.synapses.iter()
            .map(|s| { let d = s.conductance as f64 - mean; d * d })
            .sum::<f64>() / self.synapses.len() as f64;
        var.sqrt()
    }

    pub fn sparsity(&self) -> f64 {
        let total = self.synapses.len();
        if total == 0 { return 0.0; }
        let open = self.synapses.iter().filter(|s| s.alive).count();
        open as f64 / total as f64
    }

    pub fn syn_growth_rate(&self) -> f64 {
        if self.current_time <= 0 { return 0.0; }
        (self.axons_grown as f64 * 1000.0) / self.current_time as f64
    }

    pub fn macro_observables(&self) -> MacroObservables3d {
        let (ent_mean, ent_max) = self.entropy_stats();
        let (cond_mean, cond_max) = self.conductance_stats();
        MacroObservables3d {
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

    pub fn reset_trial_state(&mut self) {
        for n in &mut self.neurons {
            n.reset_state();
        }
        for slot in &mut self.delivery_queue {
            for v in slot.iter_mut() { *v = 0; }
        }
        self.delivery_head = 0;
    }

    /// 1 クロック進行 (並列版、 2D 版と完全に同じ物理プロセス)
    pub fn step(&mut self, external_input: &[i32]) -> Vec<usize> {
        let n = self.neurons.len();

        let mut current_inputs = std::mem::take(&mut self.delivery_queue[self.delivery_head]);
        for (k, &v) in external_input.iter().enumerate() {
            if k < self.input_neurons.len() {
                let target = self.input_neurons[k];
                current_inputs[target] = current_inputs[target].saturating_add(v);
            }
        }

        let t_now = self.current_time;
        let inputs_ref: &[i32] = &current_inputs;
        let fired: Vec<usize> = self.neurons.par_iter_mut().enumerate()
            .filter_map(|(i, neuron)| {
                if neuron.update(inputs_ref[i], t_now) { Some(i) } else { None }
            })
            .collect();
        let _ = n;

        for &pre in &fired {
            for &s_idx in &self.out_syn[pre] {
                let post = self.synapses[s_idx].post;
                let post_trace = self.neurons[post].spike_trace;
                self.synapses[s_idx].update_on_pre_spike_trace(post_trace);
            }
            for &s_idx in &self.in_syn[pre] {
                let pre_n = self.synapses[s_idx].pre;
                let pre_trace = self.neurons[pre_n].spike_trace;
                self.synapses[s_idx].update_on_post_spike_trace(pre_trace);
            }
        }

        for &pre in &fired {
            let pre_is_inh = self.neurons[pre].is_inhibitory;
            for &s_idx in &self.out_syn[pre] {
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
                self.synapses[s_idx].on_transmission();
            }
        }

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

        if t_now - self.last_growth_time >= self.config.axon_growth_interval {
            let prev_total = self.synapses.len();
            let (created, _attempted) = axon_growth_step_3d(
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

    pub fn output_index_of(&self, neuron_id: usize) -> Option<usize> {
        self.output_neurons.iter().position(|&n| n == neuron_id)
    }

    /// z 層ごとの (input, excitatory, inhibitory) ニューロン数を返す
    /// 抑制配置がランダム (層偏りなし) かを検証するための観察関数。
    pub fn layer_distribution(&self) -> Vec<(i32, usize, usize, usize)> {
        let input_set: std::collections::HashSet<usize> =
            self.input_neurons.iter().copied().collect();
        let mut result: Vec<(i32, usize, usize, usize)> =
            (0..self.config.grid_d).map(|z| (z, 0, 0, 0)).collect();
        for (i, n) in self.neurons.iter().enumerate() {
            let z = n.position.2 as usize;
            if input_set.contains(&i) {
                result[z].1 += 1;
            } else if n.is_inhibitory {
                result[z].3 += 1;
            } else {
                result[z].2 += 1;
            }
        }
        result
    }

    pub fn memory_bytes(&self) -> usize {
        let neuron_b = self.neurons.len() * std::mem::size_of::<ThermoNeuron3d>();
        let syn_b = self.synapses.len() * std::mem::size_of::<ThermoSynapse3d>();
        let buf_b = self.delivery_queue.len() * self.neurons.len() * 4;
        neuron_b + syn_b + buf_b
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_default_network_660_cells() {
        let cfg = ThermoNetwork3dConfig::default();
        let net = ThermoNetwork3d::new(cfg);
        assert_eq!(net.n_neurons(), 660);
        assert_eq!(net.input_neurons.len(), 20);
        assert_eq!(net.output_neurons.len(), 30);
        assert!(net.n_synapses() > 0);
    }

    #[test]
    fn input_neurons_at_bottom() {
        let cfg = ThermoNetwork3dConfig::default();
        let net = ThermoNetwork3d::new(cfg);
        for &i in &net.input_neurons {
            assert_eq!(net.neurons[i].position.2, 0,
                "input neuron at z={}, expected z=0", net.neurons[i].position.2);
        }
    }

    #[test]
    fn output_neurons_at_top() {
        let cfg = ThermoNetwork3dConfig::default();
        let net = ThermoNetwork3d::new(cfg);
        let top_z = cfg_top_z();
        for &i in &net.output_neurons {
            assert_eq!(net.neurons[i].position.2, top_z,
                "output neuron at z={}, expected z={}",
                net.neurons[i].position.2, top_z);
        }
    }

    fn cfg_top_z() -> i32 {
        ThermoNetwork3dConfig::default().grid_d - 1
    }

    #[test]
    fn one_step_no_panic() {
        let cfg = ThermoNetwork3dConfig::default();
        let mut net = ThermoNetwork3d::new(cfg);
        let ext = vec![0i32; 20];
        let _ = net.step(&ext);
        assert_eq!(net.current_time, 1);
    }
}
