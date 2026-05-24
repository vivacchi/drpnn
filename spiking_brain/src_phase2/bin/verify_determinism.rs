//! 決定論性検証: 旧版 step_sequential (last_spike_time STDP) と
//!                新版 step (spike_trace STDP, rayon 並列) で完全一致を確認。
//!
//! 検証内容:
//!   1. 6 原理 (原理 3 決定論性、原理 4 整数演算) が並列化後も維持されているか
//!   2. B4 spike_trace カウンタが last_spike_time と完全等価か
//!
//! 整数加算は順序によらず同じ結果、spike_trace と last_spike_time も等価判定
//! なので、ビット単位で完全一致するはず。
//!
//! テスト内容:
//!   1. 同じ seed で 2 つのネットワークを作る (seq, par)
//!   2. seq は step_sequential (旧版)、par は step (新版 spike_trace + 並列)
//!   3. ホワイトノイズ訓練 1000 試行 (= 600,000 step) を実行
//!   4. 全 neuron / synapse の状態が完全一致するか比較
//!
//! CLI:
//!   cargo run --release --bin verify_determinism -- [n_trials]
//! デフォルト n_trials = 1000

use spiking_brain::phase2::thermo_network::{ThermoNetwork, ThermoNetworkConfig};
use rand::prelude::*;

const TRIAL_DURATION_MS: f64 = 300.0;
const DT_MS: f64 = 0.5;
const PULSE_WIDTH_MS: f64 = 4.0;
const INPUT_CURRENT: i32 = 60;

/// ホワイトノイズ訓練: 各 step で各 input neuron が独立に p_per_step で発火 (パルス幅 8 step)
/// 直列版でも並列版でも同じシグナル列を提示するため、外部から spike pattern を渡す形にする
fn present_white_noise<F>(
    net: &mut ThermoNetwork,
    n_input: usize,
    p_per_step: f64,
    rng: &mut StdRng,
    mut step_fn: F,
) -> Vec<(usize, f64)>
where
    F: FnMut(&mut ThermoNetwork, &[i32]) -> Vec<usize>,
{
    net.reset_trial_state();
    let trial_start_t_ms = net.current_time as f64 * DT_MS;
    let n_steps = (TRIAL_DURATION_MS / DT_MS) as usize;
    let pulse_steps = (PULSE_WIDTH_MS / DT_MS).max(1.0) as i32;

    let mut pulse_remaining = vec![0i32; n_input];
    let mut external_input = vec![0i32; n_input];
    let mut out_log: Vec<(usize, f64)> = Vec::new();

    for _step in 0..n_steps {
        for k in 0..n_input {
            if pulse_remaining[k] > 0 {
                pulse_remaining[k] -= 1;
                external_input[k] = INPUT_CURRENT;
            } else if rng.gen::<f64>() < p_per_step {
                pulse_remaining[k] = pulse_steps - 1;
                external_input[k] = INPUT_CURRENT;
            } else {
                external_input[k] = 0;
            }
        }
        let fired = step_fn(net, &external_input);
        for nid in fired {
            if let Some(oi) = net.output_index_of(nid) {
                let t_abs_ms = net.current_time as f64 * DT_MS;
                out_log.push((oi, t_abs_ms - trial_start_t_ms));
            }
        }
    }
    out_log
}

fn networks_equal(a: &ThermoNetwork, b: &ThermoNetwork) -> Result<(), String> {
    if a.neurons.len() != b.neurons.len() {
        return Err(format!("neuron count differs: {} vs {}", a.neurons.len(), b.neurons.len()));
    }
    if a.synapses.len() != b.synapses.len() {
        return Err(format!("synapse count differs: {} vs {}", a.synapses.len(), b.synapses.len()));
    }
    if a.current_time != b.current_time {
        return Err(format!("current_time differs: {} vs {}", a.current_time, b.current_time));
    }
    if a.axons_grown != b.axons_grown {
        return Err(format!("axons_grown differs: {} vs {}", a.axons_grown, b.axons_grown));
    }
    if a.axons_pruned != b.axons_pruned {
        return Err(format!("axons_pruned differs: {} vs {}", a.axons_pruned, b.axons_pruned));
    }
    // ニューロン詳細比較
    for (i, (na, nb)) in a.neurons.iter().zip(b.neurons.iter()).enumerate() {
        if na.membrane != nb.membrane {
            return Err(format!("neuron[{i}].membrane differs: {} vs {}", na.membrane, nb.membrane));
        }
        if na.available_enthalpy != nb.available_enthalpy {
            return Err(format!("neuron[{i}].available_enthalpy differs: {} vs {}", na.available_enthalpy, nb.available_enthalpy));
        }
        if na.local_entropy != nb.local_entropy {
            return Err(format!("neuron[{i}].local_entropy differs: {} vs {}", na.local_entropy, nb.local_entropy));
        }
        if na.refractory_remaining != nb.refractory_remaining {
            return Err(format!("neuron[{i}].refractory_remaining differs", ));
        }
        if na.last_spike_time != nb.last_spike_time {
            return Err(format!("neuron[{i}].last_spike_time differs: {} vs {}", na.last_spike_time, nb.last_spike_time));
        }
    }
    // シナプス詳細比較
    for (i, (sa, sb)) in a.synapses.iter().zip(b.synapses.iter()).enumerate() {
        if sa.conductance != sb.conductance {
            return Err(format!("synapse[{i}].conductance differs: {} vs {}", sa.conductance, sb.conductance));
        }
        if sa.exists != sb.exists {
            return Err(format!("synapse[{i}].exists differs: {} vs {}", sa.exists, sb.exists));
        }
        if sa.decay_counter != sb.decay_counter {
            return Err(format!("synapse[{i}].decay_counter differs: {} vs {}", sa.decay_counter, sb.decay_counter));
        }
    }
    Ok(())
}

fn main() {
    let n_trials: usize = std::env::args().nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(1000);

    println!("== 決定論性検証 (直列版 vs 並列版) ==");
    println!("  6 原理 (特に整数演算 + 決定論性) が並列化後も維持されているか");
    println!("  ホワイトノイズ訓練 {n_trials} 試行で完全一致確認");
    println!();

    let cfg = ThermoNetworkConfig::default();
    let cfg2 = ThermoNetworkConfig {
        grid_width: cfg.grid_width,
        grid_height: cfg.grid_height,
        n_input: cfg.n_input,
        n_output: cfg.n_output,
        n_excitatory: cfg.n_excitatory,
        n_inhibitory: cfg.n_inhibitory,
        input_fanout: cfg.input_fanout,
        delay_range: cfg.delay_range,
        seed: cfg.seed,
        axon_growth_interval: cfg.axon_growth_interval,
        dt_ms: cfg.dt_ms,
    };
    let mut net_seq = ThermoNetwork::new(cfg);
    let mut net_par = ThermoNetwork::new(cfg2);

    println!("  初期状態:");
    println!("    neurons  : {}", net_seq.n_neurons());
    println!("    synapses : {} (open {})", net_seq.n_synapses(), net_seq.n_open_synapses());
    match networks_equal(&net_seq, &net_par) {
        Ok(()) => println!("    初期状態一致 ✓"),
        Err(e) => {
            println!("    初期状態不一致 ✗: {e}");
            std::process::exit(1);
        }
    }

    let p = 1.0 / 600.0;
    let mut rng_seq = StdRng::seed_from_u64(42);
    let mut rng_par = StdRng::seed_from_u64(42);
    let n_input = net_seq.input_neurons.len();

    let start = std::time::Instant::now();
    for trial in 1..=n_trials {
        // 直列版
        let _ = present_white_noise(&mut net_seq, n_input, p, &mut rng_seq, |net, input| {
            net.step_sequential(input)
        });
        // 並列版
        let _ = present_white_noise(&mut net_par, n_input, p, &mut rng_par, |net, input| {
            net.step(input)
        });

        if trial % (n_trials / 10).max(1) == 0 || trial == n_trials {
            match networks_equal(&net_seq, &net_par) {
                Ok(()) => {
                    println!("  trial {:>5}/{}: 完全一致 ✓  (current_time={}, open={})",
                        trial, n_trials, net_seq.current_time, net_seq.n_open_synapses());
                }
                Err(e) => {
                    println!("  trial {:>5}/{}: 不一致 ✗  {e}", trial, n_trials);
                    std::process::exit(1);
                }
            }
        }
    }
    let elapsed = start.elapsed();

    println!();
    println!("══════════════════════════════════════════════════════════");
    println!("  ✓✓✓ 決定論性検証 通過 ✓✓✓");
    println!("  全 {n_trials} 試行で直列版と並列版が完全一致");
    println!("  6 原理 (整数演算 + 決定論性) は並列化後も維持されている");
    println!("  実行時間: {:.2}s", elapsed.as_secs_f64());
    println!("══════════════════════════════════════════════════════════");
}
