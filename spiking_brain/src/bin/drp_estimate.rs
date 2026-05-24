//! DRP コスト見積もりツール。
//!
//! いまのプロトタイプ規模、想定スケール、フル聴覚野規模のそれぞれを
//! RZ/V2L / RZ/V2N / RZ/V2H に載せたときの見積もりを出す。
//!
//! マニュアル到着後、`drp_cost.rs` の TBD 値を更新するとここの数字も変わる。

use spiking_brain::drp_cost::DrpConfig;

fn estimate_for(label: &str, n: usize, syn: usize, rate: f64) {
    println!("\n████████████████████████████████████████████████████████████████");
    println!("█ Scenario: {}", label);
    println!("█   {} neurons, {} synapses, {:.1} Hz avg", n, syn, rate);
    println!("████████████████████████████████████████████████████████████████");
    for cfg in [DrpConfig::rzv2l_estimate(),
                DrpConfig::rzv2n_estimate(),
                DrpConfig::rzv2h_estimate()] {
        let est = cfg.estimate(n, syn, rate);
        cfg.report(&est);
    }
}

fn main() {
    println!("================================================================");
    println!(" DRP Cost Estimator");
    println!(" Spiking Brain prototype on Renesas RZ/V series");
    println!("================================================================");
    println!(" NOTE: figures use estimated values for parameters that are");
    println!(" not yet confirmed from Renesas hardware manuals.");
    println!(" Search for 'TBD' in drp_cost.rs and update after manual review.");

    // 1. 今のプロトタイプ規模
    estimate_for("Current prototype",
        420, 17_600, 10.0);

    // 2. ハチ脳規模(1個のチップで完結する現実的目標)
    estimate_for("Bee-brain scale (target)",
        100_000, 10_000_000, 10.0);

    // 3. マウス聴覚野規模 (RZ/V2H 複数で組む規模感)
    estimate_for("Mouse auditory cortex region",
        1_000_000, 100_000_000, 5.0);
}
