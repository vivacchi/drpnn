//! Phase 2: 非平衡熱力学系としての SNN
//!
//! 設計の中核は DESIGN_PHILOSOPHY.md §11、実装指示は PHASE2_INSTRUCTION.md。
//!
//! 各要素を熱力学量で記述:
//!   ニューロン発火     = エンタルピー消費 + エントロピー (熱) 生成
//!   慣化              = 局所エントロピー蓄積による実効閾値上昇 (明示機構なし)
//!   エントロピー散逸    = 熱の自然減衰 (放熱)
//!   シナプス強化 (LTP) = 経路の熱伝導度上昇
//!   軸索成長          = 熱勾配に沿った経路形成 (冷たい方へ伸びる)
//!   学習              = 散逸構造の形成 (Prigogine 1977)
//!
//! 絶対遵守の 6 原理 (DESIGN_PHILOSOPHY.md §2):
//!   1. 局所性 (隣接相手のみ)
//!   2. 物理性 (判断機構なし)
//!   3. 決定論性 (確率なし、初期化を除く)
//!   4. 整数演算 (浮動小数なし)
//!   5. 創発性 (トップダウン設計なし)
//!   6. 平衡としての学習 (学習関数なし)

pub mod thermo_neuron;
pub mod thermo_synapse;
pub mod topology;
pub mod axon_growth;
pub mod thermo_network;
pub mod cochlea;
pub mod cochlear_nucleus;
pub mod phoneme_synth;
pub mod sha256;
