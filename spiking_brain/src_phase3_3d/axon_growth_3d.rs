//! 軸索成長 3D 版 (熱勾配誘導、 HCP 12 近傍)
//!
//! src_phase2_f/axon_growth.rs から派生。 2D 8 近傍 → 3D HCP 12 近傍へ拡張。
//! 物理プロセス (熱勾配を下る・成長コスト・熱伝達) は 2D と完全に同一。
//!
//! 拡張点:
//!   - position: (i32, i32) → (i32, i32, i32)
//!   - topology: Topology → Topology3d (HCP 12 近傍)
//!   - position_index: HashMap<(i32,i32), Vec<usize>> → HashMap<(i32,i32,i32), Vec<usize>>

use super::thermo_neuron_3d::ThermoNeuron3d;
use super::thermo_synapse_3d::ThermoSynapse3d;
use super::topology3d::Topology3d;
use std::collections::HashMap;

/// 軸索成長を試みる閾値 (local_entropy がこれ以上で成長候補)
pub const GROWTH_THRESHOLD: i32 = 50;
/// 隣接 PE がこれだけ冷たければ結線を伸ばす
pub const GROWTH_DELTA: i32 = 20;
/// 成長で消費する自分のエントロピー
pub const GROWTH_COST: i32 = 15;
/// 接続先に伝わる熱
pub const HEAT_TRANSFER: i32 = 10;
/// 新規結線の初期 conductance
pub const MIN_CONDUCTANCE: i32 = 20;
/// 新規結線の初期遅延 (4 step = 2ms)
pub const INITIAL_DELAY: i32 = 4;

/// 軸索成長 1 ステップ (3D HCP 12 近傍版)
///
/// 戻り値: (新規作成数, 試行数)
pub fn axon_growth_step_3d(
    neurons: &mut [ThermoNeuron3d],
    synapses: &mut Vec<ThermoSynapse3d>,
    topology: &Topology3d,
    position_index: &HashMap<(i32, i32, i32), Vec<usize>>,
) -> (usize, usize) {
    let mut existing_pairs: std::collections::HashSet<(usize, usize)> =
        std::collections::HashSet::with_capacity(synapses.len());
    for s in synapses.iter() {
        existing_pairs.insert((s.pre, s.post));
    }

    let mut created = 0usize;
    let mut attempted = 0usize;
    let mut new_synapses: Vec<ThermoSynapse3d> = Vec::new();
    let mut entropy_deltas: Vec<(usize, i32)> = Vec::new();

    for i in 0..neurons.len() {
        if neurons[i].local_entropy < GROWTH_THRESHOLD {
            continue;
        }
        attempted += 1;

        let my_position = neurons[i].position;
        let my_entropy = neurons[i].local_entropy;

        // HCP 12 近傍を取得
        let neighbor_positions = topology.neighbors(my_position);

        let mut coldest_idx: Option<usize> = None;
        let mut coldest_entropy = my_entropy;

        for npos in &neighbor_positions {
            if let Some(neuron_indices) = position_index.get(npos) {
                for &j in neuron_indices {
                    if i == j { continue; }
                    if existing_pairs.contains(&(i, j)) { continue; }
                    if neurons[j].local_entropy < coldest_entropy - GROWTH_DELTA {
                        coldest_idx = Some(j);
                        coldest_entropy = neurons[j].local_entropy;
                    }
                }
            }
        }

        if let Some(j) = coldest_idx {
            new_synapses.push(ThermoSynapse3d::new(
                i, j, INITIAL_DELAY, MIN_CONDUCTANCE,
            ));
            existing_pairs.insert((i, j));
            entropy_deltas.push((i, -GROWTH_COST));
            entropy_deltas.push((j, HEAT_TRANSFER));
            created += 1;
        }
    }

    for (idx, delta) in entropy_deltas {
        let n = &mut neurons[idx];
        n.local_entropy = (n.local_entropy + delta).max(0);
    }

    synapses.extend(new_synapses);

    (created, attempted)
}

/// 位置→ニューロンインデックスの逆引きマップを作る (3D 版)
pub fn build_position_index_3d(
    neurons: &[ThermoNeuron3d],
) -> HashMap<(i32, i32, i32), Vec<usize>> {
    let mut map: HashMap<(i32, i32, i32), Vec<usize>> = HashMap::new();
    for (i, n) in neurons.iter().enumerate() {
        map.entry(n.position).or_default().push(i);
    }
    map
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_growth_when_cold() {
        let mut neurons = vec![
            ThermoNeuron3d::excitatory((2, 2, 5)),
            ThermoNeuron3d::excitatory((3, 2, 5)),
        ];
        neurons[0].local_entropy = 10;
        let mut synapses = Vec::new();
        let topo = Topology3d::new(5, 6, 22);
        let pos_idx = build_position_index_3d(&neurons);
        let (created, _) = axon_growth_step_3d(&mut neurons, &mut synapses, &topo, &pos_idx);
        assert_eq!(created, 0);
    }

    #[test]
    fn growth_toward_cold_same_layer() {
        // z=10 (偶数) 同層 (1,0) は近傍
        let mut neurons = vec![
            ThermoNeuron3d::excitatory((2, 3, 10)),
            ThermoNeuron3d::excitatory((3, 3, 10)),
        ];
        neurons[0].local_entropy = 100;
        neurons[1].local_entropy = 0;
        let mut synapses = Vec::new();
        let topo = Topology3d::new(5, 6, 22);
        let pos_idx = build_position_index_3d(&neurons);
        let (created, _) = axon_growth_step_3d(&mut neurons, &mut synapses, &topo, &pos_idx);
        assert_eq!(created, 1);
        assert_eq!(synapses[0].pre, 0);
        assert_eq!(synapses[0].post, 1);
        assert_eq!(neurons[0].local_entropy, 100 - GROWTH_COST);
        assert_eq!(neurons[1].local_entropy, HEAT_TRANSFER);
    }

    #[test]
    fn growth_toward_cold_upper_layer() {
        // z=10 (偶数) 上層 (0,0,11) は HCP 近傍
        let mut neurons = vec![
            ThermoNeuron3d::excitatory((2, 3, 10)),
            ThermoNeuron3d::excitatory((2, 3, 11)),
        ];
        neurons[0].local_entropy = 100;
        neurons[1].local_entropy = 0;
        let mut synapses = Vec::new();
        let topo = Topology3d::new(5, 6, 22);
        let pos_idx = build_position_index_3d(&neurons);
        let (created, _) = axon_growth_step_3d(&mut neurons, &mut synapses, &topo, &pos_idx);
        assert_eq!(created, 1);
    }

    #[test]
    fn no_duplicate_synapse() {
        let mut neurons = vec![
            ThermoNeuron3d::excitatory((2, 3, 10)),
            ThermoNeuron3d::excitatory((3, 3, 10)),
        ];
        neurons[0].local_entropy = 100;
        neurons[1].local_entropy = 0;
        let mut synapses = vec![ThermoSynapse3d::new(0, 1, 4, 50)];
        let topo = Topology3d::new(5, 6, 22);
        let pos_idx = build_position_index_3d(&neurons);
        let (created, _) = axon_growth_step_3d(&mut neurons, &mut synapses, &topo, &pos_idx);
        assert_eq!(created, 0);
    }

    #[test]
    fn non_adjacent_skipped() {
        // 距離 2 (非 HCP 近傍) なので結線できない
        let mut neurons = vec![
            ThermoNeuron3d::excitatory((2, 3, 10)),
            ThermoNeuron3d::excitatory((2, 3, 13)),  // z 方向 +3 は非近傍
        ];
        neurons[0].local_entropy = 100;
        neurons[1].local_entropy = 0;
        let mut synapses = Vec::new();
        let topo = Topology3d::new(5, 6, 22);
        let pos_idx = build_position_index_3d(&neurons);
        let (created, _) = axon_growth_step_3d(&mut neurons, &mut synapses, &topo, &pos_idx);
        assert_eq!(created, 0);
    }
}
