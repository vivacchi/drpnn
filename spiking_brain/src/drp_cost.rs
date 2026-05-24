//! DRP 上にネットワークを載せたときのコスト見積もりモデル。
//!
//! 既知パラメータ (Renesas 公開情報, 2024-2026):
//!   - RZ/V2N: 96 AI-PE + 2048 INT8 MAC, 15 TOPS (pruned), 10 TOPS/W
//!   - RZ/V2H: 80 TOPS (pruned) → 推定で MAC 数 ~8000, PE 数 ~200
//!   - 両者とも別途「無印 DRP」を持つ (再構成ロジックハードウェア用、PE数等は要マニュアル)
//!   - オンチップ共有 SRAM: RZ/V2H は 6MB (CPU と DRP-AI で共有), 専用部分は不明
//!
//! マニュアル待ちのパラメータ (TBD と書いてあるところ):
//!   - 無印 DRP の PE 数、接続トポロジー
//!   - 部分再構成 (Partial Reconfiguration) サイクル数
//!   - PE 間スループット、ローカルメモリ
//!
//! 使い方:
//! ```
//! use spiking_brain::drp_cost::DrpConfig;
//! let cfg = DrpConfig::rzv2h_estimate();
//! let est = cfg.estimate(50_000, 5_000_000, 10.0);  // 5万ニューロン, 500万シナプス, 平均10Hz
//! cfg.report(&est);
//! ```

#[derive(Debug, Clone)]
pub struct DrpConfig {
    pub name: &'static str,

    // 演算ユニット ----
    /// AI-PE 数 (DRP-AI3 内のプロセッシングエレメント)
    pub ai_pe_count: usize,
    /// MAC セル数 (INT8 換算)
    pub mac_count: usize,
    /// 動作周波数 [Hz]
    pub clock_hz: f64,

    // 無印 DRP (TBD: マニュアルで更新) ----
    /// 汎用 DRP の PE 数 (再構成ロジック用)
    pub generic_drp_pe_count: usize,
    /// ルーティング書き換え1回あたりのサイクル数 (TBD)
    pub reconfig_cycles: usize,

    // メモリ ----
    /// DRP 専用と仮定するオンチップ SRAM [bytes]
    pub onchip_mem_bytes: usize,
    /// 共有 SRAM 全体 [bytes] (DRP-AI と CPU で共有)
    pub shared_sram_bytes: usize,
    /// 外部 DRAM (LPDDR4) 上限 [bytes]
    pub external_mem_bytes: usize,

    // データ表現コスト (本プロジェクトの想定) ----
    /// 1ニューロンが保持するバイト数 (1ビット版で 6 bytes 程度)
    pub bytes_per_neuron: usize,
    /// 1シナプスが保持するバイト数
    pub bytes_per_synapse: usize,
    /// 1スパイクイベントを配送・処理するのに要する平均演算数
    pub ops_per_event: f64,
}

impl DrpConfig {
    /// RZ/V2L (低価格・低性能、PoC開発に向く)
    pub fn rzv2l_estimate() -> Self {
        Self {
            name: "RZ/V2L (estimated)",
            ai_pe_count: 32,
            mac_count: 256,
            clock_hz: 0.8e9,
            generic_drp_pe_count: 0, // RZ/V2L には無印DRPが無い (要確認)
            reconfig_cycles: 1000,
            onchip_mem_bytes: 512 * 1024,
            shared_sram_bytes: 2 * 1024 * 1024,
            external_mem_bytes: 4 * 1024 * 1024 * 1024,
            bytes_per_neuron: 6,
            bytes_per_synapse: 6,
            ops_per_event: 30.0,
        }
    }

    /// RZ/V2N (中位、量産向け、DRP-AI3 搭載)
    /// 既知: 96 AI-PE + 2048 INT8 MAC, 10 TOPS/W, 15 TOPS pruned
    pub fn rzv2n_estimate() -> Self {
        Self {
            name: "RZ/V2N",
            ai_pe_count: 96,
            mac_count: 2048,
            clock_hz: 1.0e9,
            generic_drp_pe_count: 48,
            reconfig_cycles: 1000, // TBD
            onchip_mem_bytes: 2 * 1024 * 1024,
            shared_sram_bytes: 4 * 1024 * 1024,
            external_mem_bytes: 4 * 1024 * 1024 * 1024,
            bytes_per_neuron: 6,
            bytes_per_synapse: 6,
            ops_per_event: 30.0,
        }
    }

    /// RZ/V2H (フラグシップ、80 TOPS pruned)
    /// 確定値 (データシート R01DS0429EJ0130 より):
    ///   - DRP-AI = DRP0 + AI-MAC, 8 dense TOPS / 80 sparse TOPS
    ///   - DRP Bus 直結 SRAM = SRAM4-11 = 8 × 512KB = 4 MB
    ///   - 外部 DRAM = LPDDR4/4X × 2ch、最大 16 GB
    ///   - DRP 専用電源ドメイン (PLLCLN_DTY_DRP) が独立
    /// 推定値 (要 R01UH1015 マニュアル確認):
    ///   - AI-PE 数、MAC 数、再構成サイクル数、無印 DRP の PE 数
    pub fn rzv2h_estimate() -> Self {
        Self {
            name: "RZ/V2H",
            ai_pe_count: 200, // 推定 (要マニュアル)
            mac_count: 8000,  // 推定 (RZ/V2N の 2048 から TOPS 比でスケール)
            clock_hz: 1.0e9,
            generic_drp_pe_count: 96, // 推定 (要マニュアル)
            reconfig_cycles: 1000,    // 推定 (要マニュアル)
            onchip_mem_bytes: 4 * 1024 * 1024, // 確定: SRAM4-11 = 4 MB
            shared_sram_bytes: 6 * 1024 * 1024, // 確定: 全体 6 MB
            external_mem_bytes: 16 * 1024 * 1024 * 1024, // 確定: LPDDR4 最大 16 GB
            bytes_per_neuron: 6,
            bytes_per_synapse: 6,
            ops_per_event: 30.0,
        }
    }
}

/// ネットワーク規模 × DRP 構成から得られる見積もり結果
#[derive(Debug, Clone)]
pub struct CostEstimate {
    pub n_neurons: usize,
    pub n_synapses: usize,
    pub avg_firing_rate_hz: f64,

    pub neuron_mem_bytes: usize,
    pub synapse_mem_bytes: usize,
    pub total_mem_bytes: usize,

    pub fits_onchip_dedicated: bool,
    pub fits_shared_sram: bool,
    pub fits_with_ddr: bool,

    /// 1秒シミュレートするのに DRP 上で何秒かかるか
    pub realtime_factor: f64,
    /// 1ms シミュレーションステップにかかる実時間 [μs]
    pub step_time_us: f64,
    /// 1回の構造的再構成にかかる時間 [μs]
    pub reconfig_time_us: f64,
    /// DRP が処理しきれるイベント数の上限 [events/sec]
    pub max_event_rate_hz: f64,
    /// 実際に必要なイベント数 [events/sec]
    pub required_event_rate_hz: f64,
}

impl DrpConfig {
    pub fn estimate(
        &self,
        n_neurons: usize,
        n_synapses: usize,
        avg_firing_rate_hz: f64,
    ) -> CostEstimate {
        let neuron_mem = n_neurons * self.bytes_per_neuron;
        let synapse_mem = n_synapses * self.bytes_per_synapse;
        let total = neuron_mem + synapse_mem;

        let required_event_rate = n_neurons as f64 * avg_firing_rate_hz;
        let ops_per_sec_needed = required_event_rate * self.ops_per_event;
        let ops_per_sec_available = self.mac_count as f64 * self.clock_hz;
        let max_event_rate = ops_per_sec_available / self.ops_per_event;

        // 1ms シミュレーションステップにかかる実時間
        let events_per_sim_ms = required_event_rate / 1000.0;
        let cycles_per_sim_ms = events_per_sim_ms * self.ops_per_event;
        let step_time_us = (cycles_per_sim_ms / self.clock_hz) * 1e6;

        let reconfig_time_us = (self.reconfig_cycles as f64 / self.clock_hz) * 1e6;

        // 1秒シミュレートに必要な実時間 [秒] = required / available
        let realtime_factor = ops_per_sec_needed / ops_per_sec_available;

        CostEstimate {
            n_neurons,
            n_synapses,
            avg_firing_rate_hz,
            neuron_mem_bytes: neuron_mem,
            synapse_mem_bytes: synapse_mem,
            total_mem_bytes: total,
            fits_onchip_dedicated: total <= self.onchip_mem_bytes,
            fits_shared_sram: total <= self.shared_sram_bytes,
            fits_with_ddr: total <= self.external_mem_bytes,
            realtime_factor,
            step_time_us,
            reconfig_time_us,
            max_event_rate_hz: max_event_rate,
            required_event_rate_hz: required_event_rate,
        }
    }

    pub fn report(&self, est: &CostEstimate) {
        println!("================================================================");
        println!(" DRP Cost Estimate: {}", self.name);
        println!("================================================================");
        println!(" Hardware:");
        println!("   AI-PE: {}, MAC: {}, clock: {:.2} GHz",
                 self.ai_pe_count, self.mac_count, self.clock_hz / 1e9);
        println!("   generic-DRP PE: {} (TBD)", self.generic_drp_pe_count);
        println!("   reconfig cycles: {} (TBD)", self.reconfig_cycles);
        println!("   memory: {} KB dedicated, {} KB shared, {} MB DRAM",
                 self.onchip_mem_bytes / 1024,
                 self.shared_sram_bytes / 1024,
                 self.external_mem_bytes / (1024 * 1024));
        println!();
        println!(" Workload:");
        println!("   neurons: {}, synapses: {}, avg firing rate: {:.1} Hz",
                 est.n_neurons, est.n_synapses, est.avg_firing_rate_hz);
        println!("   required ops: {:.2e} ops/sec",
                 est.required_event_rate_hz * self.ops_per_event);
        println!("   available ops: {:.2e} ops/sec",
                 self.mac_count as f64 * self.clock_hz);
        println!();
        println!(" Memory:");
        println!("   neuron data : {:>10} bytes", est.neuron_mem_bytes);
        println!("   synapse data: {:>10} bytes", est.synapse_mem_bytes);
        println!("   total       : {:>10} bytes ({:.2} MB)",
                 est.total_mem_bytes,
                 est.total_mem_bytes as f64 / (1024.0 * 1024.0));
        println!("   fits onchip dedicated?  {}", yn(est.fits_onchip_dedicated));
        println!("   fits shared SRAM?       {}", yn(est.fits_shared_sram));
        println!("   fits with DDR?          {}", yn(est.fits_with_ddr));
        println!();
        println!(" Performance:");
        println!("   realtime factor      : {:.4}x  (= sim_sec / wall_sec)",
                 1.0 / est.realtime_factor.max(1e-12));
        println!("   1 ms sim step takes  : {:.3} μs wall time",
                 est.step_time_us);
        println!("   single reconfig takes: {:.3} μs",
                 est.reconfig_time_us);
        println!("   max event rate       : {:.2e} events/sec",
                 est.max_event_rate_hz);
        println!("   required event rate  : {:.2e} events/sec",
                 est.required_event_rate_hz);
        let util = est.required_event_rate_hz / est.max_event_rate_hz * 100.0;
        println!("   DRP utilization      : {:.2} %", util);
        println!("================================================================");
    }
}

fn yn(b: bool) -> &'static str { if b { "YES ✓" } else { "no  ✗" } }
