//! 軸索成長 (熱勾配誘導)
//!
//! Phase 2 の核心新機能。DESIGN_PHILOSOPHY.md §11.3 と PHASE2_INSTRUCTION.md §3.5。
//!
//! 各ニューロンは、自分の local_entropy が高い (熱い) 場合、隣接 PE の中で
//! 最も冷たい (local_entropy 最小) かつ未結線の相手へ新しい結線を伸ばす。
//!
//! 「最も冷たい」を選ぶのは判断機構ではない。これは物理的選択 (水が低きに流れる)。
//!
//! 3 つの飽和回避メカニズム:
//!   A. 成長コスト: 自分の entropy を消費 (集団的殺到を物理的にブロック)
//!   B. 維持コスト: 共鳴しない結線は STDP+decay で焼き切れる
//!   C. 空間的距離: 隣接限定 (結晶成長的な秩序ある拡大)
//!
//! 時間スケール: 軸索成長は遅いプロセス。N step (例: 200) に 1 回だけ判定。

use super::thermo_neuron::ThermoNeuron;
use super::thermo_synapse::ThermoSynapse;
use super::topology::Topology;
use std::collections::HashMap;

/// 軸索成長を試みる閾値 (local_entropy がこれ以上で成長候補)
pub const GROWTH_THRESHOLD: i32 = 50;
/// 隣接 PE がこれだけ冷たければ結線を伸ばす (n.entropy - neighbor.entropy > GROWTH_DELTA)
pub const GROWTH_DELTA: i32 = 20;
/// 成長で消費する自分のエントロピー
pub const GROWTH_COST: i32 = 15;
/// 接続先に伝わる熱
pub const HEAT_TRANSFER: i32 = 10;
/// 新規結線の初期 conductance
pub const MIN_CONDUCTANCE: i32 = 20;
/// 新規結線の初期遅延 (4 step = 2ms)
pub const INITIAL_DELAY: i32 = 4;

/// 軸索成長 1 ステップ。GROWTH_INTERVAL step に 1 回だけ呼ぶ。
///
/// 戻り値: (新規作成数, 試行数)
pub fn axon_growth_step(
    neurons: &mut [ThermoNeuron],
    synapses: &mut Vec<ThermoSynapse>,
    topology: &Topology,
    position_index: &HashMap<(i32, i32), Vec<usize>>,
) -> (usize, usize) {
    // 既存 (pre, post) ペアを高速チェックするための集合
    let mut existing_pairs: std::collections::HashSet<(usize, usize)> =
        std::collections::HashSet::with_capacity(synapses.len());
    for s in synapses.iter() {
        existing_pairs.insert((s.pre, s.post));
    }

    let mut created = 0usize;
    let mut attempted = 0usize;
    let mut new_synapses: Vec<ThermoSynapse> = Vec::new();
    // (idx, entropy_delta) のペアでまとめて適用 (借用ルール対策)
    let mut entropy_deltas: Vec<(usize, i32)> = Vec::new();

    for i in 0..neurons.len() {
        // 入力ニューロンは entropy 蓄積しないので、自然と成長候補から外れる
        if neurons[i].local_entropy < GROWTH_THRESHOLD {
            continue;
        }
        attempted += 1;

        let my_position = neurons[i].position;
        let my_entropy = neurons[i].local_entropy;
        let my_is_inh = neurons[i].is_inhibitory;

        // 隣接位置を取得
        let neighbor_positions = topology.neighbors(my_position);

        // 隣接 PE のうち、最も冷たい未結線相手を探す
        let mut coldest_idx: Option<usize> = None;
        let mut coldest_entropy = my_entropy;

        for npos in &neighbor_positions {
            // 同じ位置に複数ニューロンがある場合も対応 (position_index は Vec<usize>)
            if let Some(neuron_indices) = position_index.get(npos) {
                for &j in neuron_indices {
                    if i == j { continue; }
                    // 既結線スキップ
                    if existing_pairs.contains(&(i, j)) { continue; }
                    // 物理的選択: より冷たい相手を採用
                    if neurons[j].local_entropy < coldest_entropy - GROWTH_DELTA {
                        coldest_idx = Some(j);
                        coldest_entropy = neurons[j].local_entropy;
                    }
                }
            }
        }

        // 冷たい相手が見つかれば結線
        if let Some(j) = coldest_idx {
            new_synapses.push(ThermoSynapse::new_plastic(
                i, j, INITIAL_DELAY, MIN_CONDUCTANCE, my_is_inh,
            ));
            existing_pairs.insert((i, j));
            // 成長コスト + 接続先への熱伝達
            entropy_deltas.push((i, -GROWTH_COST));
            entropy_deltas.push((j, HEAT_TRANSFER));
            created += 1;
        }
    }

    // エントロピーの増減を一括適用
    for (idx, delta) in entropy_deltas {
        let n = &mut neurons[idx];
        n.local_entropy = (n.local_entropy + delta).max(0);
    }

    // 新規シナプスを既存配列に追加
    synapses.extend(new_synapses);

    (created, attempted)
}

/// 位置→ニューロンインデックスの逆引きマップを作る (毎回作り直すのではなく、ネットワーク構築時に作る)
pub fn build_position_index(neurons: &[ThermoNeuron]) -> HashMap<(i32, i32), Vec<usize>> {
    let mut map: HashMap<(i32, i32), Vec<usize>> = HashMap::new();
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
            ThermoNeuron::excitatory((5, 5)),
            ThermoNeuron::excitatory((6, 5)),
        ];
        neurons[0].local_entropy = 10; // GROWTH_THRESHOLD (50) 未満
        let mut synapses = Vec::new();
        let topo = Topology::new(20, 20);
        let pos_idx = build_position_index(&neurons);
        let (created, _) = axon_growth_step(&mut neurons, &mut synapses, &topo, &pos_idx);
        assert_eq!(created, 0);
    }

    #[test]
    fn growth_toward_cold_neighbor() {
        let mut neurons = vec![
            ThermoNeuron::excitatory((5, 5)),  // 熱い
            ThermoNeuron::excitatory((6, 5)),  // 冷たい (隣)
        ];
        neurons[0].local_entropy = 100; // GROWTH_THRESHOLD 超え
        neurons[1].local_entropy = 0;
        let mut synapses = Vec::new();
        let topo = Topology::new(20, 20);
        let pos_idx = build_position_index(&neurons);
        let (created, _) = axon_growth_step(&mut neurons, &mut synapses, &topo, &pos_idx);
        assert_eq!(created, 1);
        assert_eq!(synapses[0].pre, 0);
        assert_eq!(synapses[0].post, 1);
        // 成長コストで熱い側のエントロピー減少
        assert_eq!(neurons[0].local_entropy, 100 - GROWTH_COST);
        // 接続先に熱伝達
        assert_eq!(neurons[1].local_entropy, HEAT_TRANSFER);
    }

    #[test]
    fn no_duplicate_synapse() {
        let mut neurons = vec![
            ThermoNeuron::excitatory((5, 5)),
            ThermoNeuron::excitatory((6, 5)),
        ];
        neurons[0].local_entropy = 100;
        neurons[1].local_entropy = 0;
        let mut synapses = vec![ThermoSynapse::new_plastic(0, 1, 4, 50, false)];
        let topo = Topology::new(20, 20);
        let pos_idx = build_position_index(&neurons);
        let (created, _) = axon_growth_step(&mut neurons, &mut synapses, &topo, &pos_idx);
        assert_eq!(created, 0); // 既存結線があるので新規作成なし
    }

    #[test]
    fn non_adjacent_skipped() {
        // 距離 2 (非隣接) なので結線できない
        let mut neurons = vec![
            ThermoNeuron::excitatory((5, 5)),
            ThermoNeuron::excitatory((7, 5)),  // 隣の隣
        ];
        neurons[0].local_entropy = 100;
        neurons[1].local_entropy = 0;
        let mut synapses = Vec::new();
        let topo = Topology::new(20, 20);
        let pos_idx = build_position_index(&neurons);
        let (created, _) = axon_growth_step(&mut neurons, &mut synapses, &topo, &pos_idx);
        assert_eq!(created, 0);
    }
}
