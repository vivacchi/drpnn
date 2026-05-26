# DRPNN

**An autonomous, growth-capable 1-bit Spiking Neural Network for Renesas DRP.**
**Pure integer math. Deterministic physics. No probability. No floating point. An alternative to LLMs.**

[![License: CC BY-SA 4.0](https://img.shields.io/badge/License-CC%20BY--SA%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-sa/4.0/)
[![Rust](https://img.shields.io/badge/Rust-1.75%2B-orange?logo=rust)](https://www.rust-lang.org/)
[![GitHub Sponsors](https://img.shields.io/badge/Sponsor-GitHub-pink?logo=github-sponsors)](https://github.com/sponsors/vivacchi)
[![Status](https://img.shields.io/badge/Phase-2%20(M0%2BM1%20working)-brightgreen)](./PAPER_DRAFT.md)

---

[日本語](#日本語) | [English](#english)

---

## English

### What is this?

`drpnn` is an independent research project building a brain-inspired AI from
scratch, designed to run on **Renesas DRP (Dynamically Reconfigurable Processor)**
— a class of edge AI hardware that performs **dynamic circuit reconfiguration
at runtime**.

The core insight: in this design, **"learning" *is* the DRP reconfiguration**.
They are physically the same act. There is no separation between the network
weights and the hardware circuit. Synaptic strengthening = circuit being formed.
Synaptic pruning = circuit being torn down.

### Why this matters

The dominant AI paradigm (LLMs, gradient descent) requires:
- Floating-point arithmetic
- Probabilistic sampling
- Gigantic models with billions of parameters
- Enormous data centers consuming megawatts

`drpnn` explores a fundamentally different path:
- **Pure integer arithmetic** (no FP, anywhere)
- **Deterministic physical processes** (no probability at runtime)
- **Small networks** (current: 440 neurons in M1) that learn from environmental dynamics
- **Target: embedded devices** consuming milliwatts

This is not a competitor to GPT or Llama. It's an entirely different question:
*"What's the smallest, simplest brain-like system that exhibits genuine learning?"*

### Key results (current)

| Metric | Value | Notes |
|---|---|---|
| **M1 selectivity** (fixed pattern, 10k training) | **POST = 0.795** | Time-binned fingerprint evaluation |
| **Kenet 2003 internal state repertoire** | **t = 4.169, p < 0.001** | Spatially-preserved shuffle null distribution |
| State transition rate | **72.4%** | Biological observation: ~80% |
| Within-pattern reproducibility | **0.953** | Same stimulus → same response (deterministic) |
| Between-pattern separation | **0.158** | Different stimuli → different responses |
| **Phoneme recognition** (M0 cochlea + M1) | POST = 0.389 | 5 syllables (pa, ki, tu, se, mo) |
| Memory footprint | **~700 KB** | 440 neurons + 17,000 synapses |

These numbers are reproducible (deterministic, seed-fixed) and verified across
multiple training/evaluation runs.

### Architecture

```
[Sound wave 16 kHz]
        │
        ▼
┌──────────────────┐
│ M0 Cochlea       │  20-band ERB-spaced filter bank, Q1.15 fixed-point IIR
│ (✅ implemented) │  envelope detection + integer sqrt compression + threshold firing
└──────────────────┘  Pure algorithm, no learning, no neurons
        │
        ▼ (20 input neurons, electric current vector)
┌──────────────────┐
│ M0.5 SOC         │  ITD/ILD calculation (binaural integration)
│ (planned)        │
└──────────────────┘
        │
        ▼
┌──────────────────┐
│ M1 A1            │  Fork F-G1-R1: 440 neurons (20 input + 40 output + 304 exc + 76 inh)
│ (✅ implemented) │  STDP, structural plasticity (vitality), axon growth (thermal gradient)
│                  │  UP/DOWN states (multiple attractors), 8-neighbor topology
└──────────────────┘  Output: time-binned fingerprint (40 × 30 = 1200 dim)
        │
        ▼
┌──────────────────┐
│ M2 A2, M3, ...   │  Invariance, categorization, language
│ (future)         │
└──────────────────┘
```

### Documents

- **[DESIGN_PHILOSOPHY.md](./DESIGN_PHILOSOPHY.md)** — Core design principles (6 principles), thermodynamic framework
- **[PAPER_DRAFT.md](./spiking_brain/PAPER_DRAFT.md)** — Detailed paper draft (under active development, CC BY-SA 4.0)
- **[CONTEXT.md](./CONTEXT.md)** — Overall architecture (12 modules: M0-M11)
- **[M0_COCHLEA_DESIGN.md](./M0_COCHLEA_DESIGN.md)** — M0 cochlea design and implementation
- **[HANDOFF.md](./HANDOFF.md)** — Current status and next tasks

### Build & run

```bash
git clone https://github.com/vivacchi/drpnn.git
cd drpnn/spiking_brain

# Main M1 evaluation (5-30 min depending on iterations)
cargo run --release --bin thermo_m1_evaluation_f -- 10000 fixed

# M0 cochlea + M1 phoneme recognition
cargo run --release --bin m0_m1_pipeline -- 10000

# Internal state repertoire validation (Kenet 2003 verification)
cargo run --release --bin internal_state_probe -- 10000

# Real-time GPU visualization (requires display)
cargo run --release --features visualizer --bin thermo_visualizer
```

### Roadmap

- ✅ Phase 1: Physical-process based SNN (with habituation mechanism)
- ✅ Phase 2 Fork A-E: Iterative refinement (10+ forks tested)
- ✅ Phase 2 Fork F-G1-R1: Vitality + STDP + axon growth + UP/DOWN states
- ✅ M0 Cochlea: ERB filter bank + envelope + threshold firing
- ✅ Phoneme synthesis: 5 vowels + consonants (formant + LFSR noise)
- ✅ Time-binned fingerprint evaluation
- ✅ Kenet 2003 internal state repertoire (quantitatively reproduced)
- 🔄 M2 A2 (secondary auditory cortex): invariance acquisition
- 🔄 Higher modules (M3 categorization, M4 hippocampus, ...)
- ⏳ Renesas RZ/V2H (DRP1) implementation
- ⏳ Real-world audio input (microphone, WAV files)

### Design philosophy (6 principles)

1. **Locality** — each element communicates only with directly connected neighbors
2. **Physicality** — physical processes only, no decision/judgment mechanisms
3. **Determinism** — no probability or randomness at runtime (initialization excluded)
4. **Integer arithmetic** — no floating point
5. **Emergence** — bottom-up construction, no top-down "should-be" design
6. **Learning as equilibrium** — no `apply_learning()` function; learning emerges as dynamic equilibrium

### References (selected)

- Izhikevich E.M., Gally J.A., Edelman G.M. (2004) "Spike-timing dynamics of neuronal groups" *Cereb Cortex* 14:933
- Ikegaya Y. et al. (2004) "Synfire chains and cortical songs" *Science* 304:559
- Kenet T. et al. (2003) "Spontaneously emerging cortical representations of visual attributes" *Nature* 425:954
- Bi G.Q., Poo M.M. (1998) "Synaptic modifications in cultured hippocampal neurons" *J Neurosci* 18:10464
- Prigogine I. (1977) "Self-Organization in Non-Equilibrium Systems" (Nobel Chemistry Prize)
- 池谷裕二 (2005) 「自発活動の意味」実験医学誌

### Support this project

This is a personal research project, published under CC BY-SA 4.0 as a public good.
If you find this work valuable, please consider sponsoring:

[![GitHub Sponsors](https://img.shields.io/badge/Sponsor%20on%20GitHub-pink?style=for-the-badge&logo=github-sponsors)](https://github.com/sponsors/vivacchi)

See [SPONSORS_PROFILE.md](./SPONSORS_PROFILE.md) for tier details. Sponsorship enables:
- Independent research time
- Hardware (Renesas evaluation kits, FPGA boards)
- Continued open publication

I do not sell products, do not run ads, do not lock content behind paywalls.

### License

- **Code**: CC BY-SA 4.0 (sharing and modification allowed with attribution and sharealike)
- **Paper**: CC BY-SA 4.0 (see [PAPER_DRAFT.md](./spiking_brain/PAPER_DRAFT.md))

---

## 日本語

### これは何

`drpnn` は、**Renesas DRP (動的再構成プロセッサ)** 上で動作する、自律成長型の
1-bit Spiking Neural Network (SNN) を、ゼロから設計・実装している個人研究プロジェクトです。

中核アイデア: この設計では「**学習」と「DRP の動的再構成」を物理的に同一視**する。
シナプスの強化 = 回路が結ばれること。シナプスの刈り取り = 回路が解かれること。
ネットワークの重みとハードウェア回路は、別々の存在ではなく **同じ物理現象**。

### なぜこれが重要か

支配的な AI 手法 (LLM、勾配降下) は:
- 浮動小数演算が必要
- 確率的サンプリングを使う
- 数十億パラメータの巨大モデル
- メガワット級のデータセンター

`drpnn` は根本的に異なる道を探ります:
- **整数演算のみ** (浮動小数はどこにも使わない)
- **決定論的物理プロセス** (ランタイムでは確率も乱数も使わない)
- **小規模ネットワーク** (現在 M1 で 440 ニューロン) が環境ダイナミクスから学習
- **目標: ミリワット消費の組み込みデバイス**

GPT や Llama の競合ではありません。全く異なる問いです:
*「**本物の学習**を示す、最小限・最も単純な脳様システムとは何か?」*

### 主要成果 (現状)

| 指標 | 値 | 備考 |
|---|---|---|
| **M1 selectivity** (固定パターン、10k 訓練) | **POST = 0.795** | 時間 bin 化 fingerprint 評価 |
| **Kenet 2003 内部状態レパートリ** | **t = 4.169, p < 0.001** | 空間保存シャッフル帰無分布 |
| 状態遷移率 | **72.4%** | 生物観察値 ~80% |
| 同パターン応答再現性 | **0.953** | 同じ刺激 → 同じ応答 (決定論的) |
| 異パターン分離 | **0.158** | 異なる刺激 → 異なる応答 |
| **音素認識** (M0 蝸牛 + M1) | POST = 0.389 | 5 音節 (pa, ki, tu, se, mo) |
| メモリフットプリント | **~700 KB** | 440 ニューロン + 17,000 シナプス |

これらの数値は再現可能 (決定論、seed 固定)、複数回の訓練/評価で確認済みです。

### アーキテクチャ

(英語版と同じ図、上記参照)

### ドキュメント

- **[DESIGN_PHILOSOPHY.md](./DESIGN_PHILOSOPHY.md)** — 設計原理 (6 原理)、熱力学的描像
- **[PAPER_DRAFT.md](./spiking_brain/PAPER_DRAFT.md)** — 論文ドラフト (執筆中、CC BY-SA 4.0)
- **[CONTEXT.md](./CONTEXT.md)** — 全体アーキテクチャ (12 モジュール構成 M0-M11)
- **[M0_COCHLEA_DESIGN.md](./M0_COCHLEA_DESIGN.md)** — M0 蝸牛設計と実装
- **[HANDOFF.md](./HANDOFF.md)** — 現状と次のタスク

### ビルド & 実行

(英語版コマンドと同じ)

### ロードマップ

- ✅ Phase 1: 物理プロセス型 SNN (慣化機構あり)
- ✅ Phase 2 Fork A-E: 反復的洗練 (10+ フォーク試験)
- ✅ Phase 2 Fork F-G1-R1: vitality + STDP + 軸索成長 + UP/DOWN 状態
- ✅ M0 蝸牛: ERB スケール フィルタバンク + 包絡線 + 閾値発火
- ✅ 音素生成: 5 母音 + 子音 (フォルマント合成 + LFSR ノイズ)
- ✅ 時間 bin 化 fingerprint 評価
- ✅ Kenet 2003 内部状態レパートリ (定量再現)
- 🔄 M2 A2 (二次聴覚野): 不変性獲得
- 🔄 上位モジュール (M3 カテゴリ、M4 海馬、…)
- ⏳ Renesas RZ/V2H (DRP1) 実装
- ⏳ 実音響入力 (マイク、WAV)

### 設計哲学 (6 原理)

1. **局所性** — 各要素は接続された相手とのみ情報交換
2. **物理性** — 物理プロセスのみ、判断機構なし
3. **決定論性** — ランタイムで確率や乱数を使わない (初期化時を除く)
4. **整数演算** — 浮動小数を使わない
5. **創発性** — ボトムアップ構築、トップダウンの「こうあるべき」設計をしない
6. **平衡としての学習** — `apply_learning()` のような関数は存在しない、学習は動的平衡として立ち上がる

### このプロジェクトを支援する

これは個人研究プロジェクトで、公共財として CC BY-SA 4.0 で公開されています。
この研究を価値があると感じていただけたら、ご支援をお願いします:

[![GitHub Sponsors](https://img.shields.io/badge/GitHub%20Sponsors-pink?style=for-the-badge&logo=github-sponsors)](https://github.com/sponsors/vivacchi)

階層詳細は [SPONSORS_PROFILE.md](./SPONSORS_PROFILE.md) を参照。ご支援で可能になるもの:
- 独立した研究時間の確保
- ハードウェア (Renesas 評価キット、FPGA ボード)
- オープンな発表の継続

私は製品販売、広告、ペイウォール、いずれもしません。

### ライセンス

- **コード**: CC BY-SA 4.0 (帰属表示 + 同条件継承で改変・共有可)
- **論文**: CC BY-SA 4.0 ([PAPER_DRAFT.md](./spiking_brain/PAPER_DRAFT.md) 参照)
