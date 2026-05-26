//! Phase 3: 3D HCP 配置版 M1
//!
//! 設計: M1_3D_DESIGN.md
//!
//! 既存 src_phase2_f (Fork F-G1-R1 v2 2D) と完全独立。
//! 並行開発・並行評価で 2D vs 3D の効果を実証する。
//!
//! 採用: grid 5×6×22 = 660 セル (柱状)
//!   - 底面 5×6=30 セル: 入力 20 (中央) + 内部 10
//!   - 内部層 20 層 × 30 = 600 セル
//!   - 上面 5×6=30 セル: 出力 30 完全充填
//!   - HCP 12 等距離近傍 (z 偶奇オフセット)

pub mod topology3d;
pub mod thermo_neuron_3d;
pub mod thermo_synapse_3d;
pub mod axon_growth_3d;
pub mod thermo_network_3d;
