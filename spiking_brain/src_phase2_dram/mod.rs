//! Phase 2 DRAM 物理モデル版 SNN
//!
//! STAGE2GAMMA_PWM_DESIGN_V2.md に基づく実装。
//!
//! 4 原理:
//!   1. 時間軸 = 行アドレス (リングバッファで時間を物理化)
//!   2. 1 セル = 1 ニューロン (キャパシタ電荷 = 膜電位)
//!   3. 個別セル書き込み (WL + BL + PWM パルス幅 = シナプス重み)
//!   4. Vth ばらつき = 発火閾値の個性 (製造ばらつきを個性として活用)
//!
//! Phase 2 (src_phase2_f) の熱力学物理を、 さらに DRAM の物理デバイス特性に
//! 即した形で再実装。 既存 Phase 2 と並行発展、 ハードウェア実装への橋渡し。
//!
//! Step 0: dram_cell.rs - 単セル POC (Vth 多様性 + 自然リーク + 発火)
//! Step 1: dram_synapse.rs - PWM 重みシナプス + STDP
//! Step 2: ring_buffer.rs + dram_network.rs - 時間軸 = 行アドレスのリングバッファ統合

pub mod dram_cell;
pub mod dram_synapse;
pub mod ring_buffer;
pub mod dram_network;
pub mod bitline_physics;
