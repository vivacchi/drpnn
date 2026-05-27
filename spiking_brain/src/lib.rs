//! Spiking Brain — ライブラリ層
//!
//! 3つの軸で実装されている:
//!   1. 連続値版 (Izhikevich + 連続シナプス重み)        — `network` モジュール
//!   2. 1ビット版 (LIF カウンタ + 二値シナプス)         — `binary_network` モジュール
//!   3. 学習機構 (即時STDP + 報酬変調 R-STDP)             — `network` / `binary_network` に統合
//!
//! 補助:
//!   - `neuron`    : Izhikevich モデル
//!   - `trace`     : 出力層の Spike Lifetime と類似度
//!   - `drp_cost`  : Renesas DRP に載せたときのリソース見積もり

pub mod neuron;
pub mod trace;
pub mod network;
pub mod binary_network;
pub mod drp_cost;

// Phase 2: 非平衡熱力学系としての SNN
#[path = "../src_phase2/mod.rs"]
pub mod phase2;

// Phase 2 fork A: 新生軸索の活性を高くスタート (MIN_CONDUCTANCE 20→40)
#[path = "../src_phase2_a/mod.rs"]
pub mod phase2_a;

// Phase 2 fork B: 新生結線の decay 保護期間 (MATURATION_PERIOD step は減衰免除)
#[path = "../src_phase2_b/mod.rs"]
pub mod phase2_b;

// Phase 2 fork C: Fork B + 新生結線 MIN_CONDUCTANCE=40 (A と B の組み合わせ)
#[path = "../src_phase2_c/mod.rs"]
pub mod phase2_c;

// Phase 2 fork D: 全シナプス (固定 + 新生) の初期 conductance=50 (信号強度均一化)
#[path = "../src_phase2_d/mod.rs"]
pub mod phase2_d;

// Phase 2 fork E: 過剰接続版 (発生学的) — cortex 内に 12800 本の plastic 結線を初期生成、
// 生物のシナプス過形成 → 刈り込みを物理シミュレート
#[path = "../src_phase2_e/mod.rs"]
pub mod phase2_e;

// Phase 2 fork F: 統合再設計 (発生学的・原理厳格版)
//   1. is_inhibitory を pre neuron 属性に (デールの原則)
//   2. plastic / fixed の区別撤廃 (全シナプス可塑性)
//   3. vitality 追加 (構造的可塑性: 使用頻度ベース存続)
//   4. LTD 廃止 (LTP のみ能動、自己強化型分岐)
#[path = "../src_phase2_f/mod.rs"]
pub mod phase2_f;

// Phase 3: 3D HCP 配置版 M1
//   - 5×6×22 = 660 セル (柱状)
//   - HCP 12 等距離近傍 (z 偶奇オフセット)
//   - 物理プロセスは Phase 2 fork F と完全同一 (近傍トポロジーと配置のみ変更)
//   - 2D vs 3D の並行比較で「物理空間次元が学習結果に与える影響」を検証
#[path = "../src_phase3_3d/mod.rs"]
pub mod phase3_3d;

// Phase 2 DRAM 物理モデル版 (STAGE2GAMMA V2 設計)
//   - 1 セル = 1 ニューロン (キャパシタ電荷 = 膜電位)
//   - Vth ばらつき = 発火閾値の個性 (製造ばらつきを活用)
//   - 個別セル書き込み (WL + BL + PWM パルス幅 = シナプス重み)
//   - 時間軸 = 行アドレス (リングバッファで時間を物理化)
//   - Phase 2 fork F と並存、 ハードウェア実装への橋渡し
#[path = "../src_phase2_dram/mod.rs"]
pub mod phase2_dram;
