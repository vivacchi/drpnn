# Phase 2 実装指示書: 熱力学版 SNN の構築

最終更新: 2026-05-22
対象: Claude Code
前提読了: DESIGN_PHILOSOPHY.md (特に §11)、CONTEXT.md、HANDOFF.md

---

## 1. 本指示書の位置づけ

本プロジェクトは Phase 1 (物理置換型) の実装と評価を進めてきた。慣化機構の導入で部分的成功 (POST selectivity 0.702、PERSIST 崩壊 97.4% → 15.9%) を得たが、実装が判断機構を含み、設計哲学に厳密に従っていなかった。

2026 年 5 月の議論で、本プロジェクトの全体を**非平衡熱力学系**として記述する設計基盤が確立された (DESIGN_PHILOSOPHY.md §11)。これは Phase 1 を否定するものではなく、その背後にある統一原理を明らかにするものである。

ユーザーの判断:

1. 熱力学的描像を正式採用
2. **Phase 2 (熱力学版) を 0 から構築し、Phase 1 と並行比較**
3. 概念実装で進める (整数値で熱力学量を表現)
4. 軸索成長は隣接 PE の比較で実装

本指示書は、上記の Phase 2 を実装するための作業指示である。Phase 1 のコードは保持する (上書きせず別ディレクトリに作る)。

---

## 2. ディレクトリ構造

```
spiking_brain/
├── src/                          # Phase 1 (現状のコード、保持)
│   ├── binary_network.rs         # 物理置換型
│   ├── bin/m1_evaluation.rs
│   └── ...
├── src_phase2/                   # Phase 2 (新規作成)
│   ├── lib.rs
│   ├── thermo_neuron.rs          # 熱力学的ニューロン
│   ├── thermo_synapse.rs         # 熱力学的シナプス
│   ├── thermo_network.rs         # ネットワーク統合
│   ├── topology.rs               # ニューロン物理配置と隣接関係
│   ├── axon_growth.rs            # 軸索成長機構
│   └── bin/
│       ├── thermo_m1_evaluation.rs   # Phase 2 評価ランナー
│       └── compare_phases.rs         # Phase 1 vs Phase 2 比較
├── Cargo.toml                    # 両 Phase のバイナリを定義
└── ...
```

Cargo.toml に Phase 2 用のバイナリ定義を追加すること。

---

## 3. 設計の核心

Phase 2 で実装すべき内容は、DESIGN_PHILOSOPHY.md §11 に詳述されている。本指示書では実装上の具体性を補う。

### 3.1 熱力学的ニューロン

```rust
pub struct ThermoNeuron {
    // 物理量 (すべて整数)
    pub membrane: i32,              // 膜電位
    pub available_enthalpy: i32,    // 利用可能エンタルピー (発火可能エネルギー)
    pub local_entropy: i32,         // 局所エントロピー (蓄積された熱)
    pub refractory_remaining: i32,  // 不応期残り
    pub last_spike_time: i32,
    
    // 固定パラメータ
    pub threshold_base: i32,        // 基準閾値
    pub enthalpy_max: i32,          // エンタルピーの上限
    pub entropy_decay_rate: i32,    // エントロピーの散逸速度 (per step)
    pub enthalpy_recovery_rate: i32, // エンタルピーの回復速度 (per step)
    pub refractory_period: i32,
    pub is_inhibitory: bool,
    
    // 物理配置 (軸索成長用)
    pub position: (i32, i32),       // 2D グリッド上の位置
}
```

### 3.2 各クロックでの物理プロセス

```rust
impl ThermoNeuron {
    pub fn update(&mut self, input_current: i32, current_time: i32) -> bool {
        // 1. 不応期処理
        if self.refractory_remaining > 0 {
            self.refractory_remaining -= 1;
            return false;
        }
        
        // 2. 膜電位への入力 (エンタルピー流入)
        self.membrane += input_current;
        if self.membrane < 0 { self.membrane = 0; }
        
        // 3. エンタルピーの自然回復
        if self.available_enthalpy < self.enthalpy_max {
            self.available_enthalpy += self.enthalpy_recovery_rate;
            if self.available_enthalpy > self.enthalpy_max {
                self.available_enthalpy = self.enthalpy_max;
            }
        }
        
        // 4. エントロピーの散逸 (放熱)
        if self.local_entropy > 0 {
            self.local_entropy -= self.entropy_decay_rate;
            if self.local_entropy < 0 { self.local_entropy = 0; }
        }
        
        // 5. 発火条件: 膜電位が「基準閾値 + 局所エントロピー」を超え、
        //              かつエンタルピーが残っている
        let effective_threshold = self.threshold_base + self.local_entropy;
        if self.membrane >= effective_threshold && self.available_enthalpy > 0 {
            // 発火 (エネルギー消費 + 熱生成)
            self.available_enthalpy -= 1;
            self.local_entropy += ENTROPY_PER_SPIKE;  // 例: 10
            self.membrane = 0;
            self.refractory_remaining = self.refractory_period;
            self.last_spike_time = current_time;
            return true;
        }
        false
    }
}
```

**重要なポイント**:
- target_rate との比較は一切ない
- 確率も乱数も使わない
- すべて整数演算
- 慣化機構は明示的でなく、エンタルピー消費とエントロピー蓄積の自然な帰結

### 3.3 熱力学的シナプス

```rust
pub struct ThermoSynapse {
    pub pre: usize,
    pub post: usize,
    pub delay: i32,
    pub conductance: i32,    // 熱伝導度 (信号伝達効率)
    pub exists: bool,
    pub is_inhibitory: bool,
    pub plastic: bool,
}
```

STDP の実装:

```rust
impl ThermoSynapse {
    // pre が発火したとき (post の last_spike_time を見る)
    pub fn update_on_pre_spike(&mut self, post_last_spike: i32, current_time: i32) {
        if !self.plastic { return; }
        let dt = current_time - post_last_spike;
        if dt > 0 && dt < CAUSAL_WINDOW {
            // post が先に発火 → 反因果 → LTD
            self.conductance -= LTD_AMOUNT;
            if self.conductance < 0 { self.conductance = 0; }
        }
    }
    
    // post が発火したとき (pre の last_spike_time を見る)
    pub fn update_on_post_spike(&mut self, pre_last_spike: i32, current_time: i32) {
        if !self.plastic { return; }
        let dt = current_time - pre_last_spike;
        if dt > 0 && dt < CAUSAL_WINDOW {
            // pre が先に発火 → 因果 → LTP
            self.conductance += LTP_AMOUNT;
            if self.conductance > CONDUCTANCE_MAX {
                self.conductance = CONDUCTANCE_MAX;
            }
        }
    }
    
    // 毎クロックの自然減衰
    pub fn decay(&mut self) {
        if self.conductance > 0 {
            // 緩やかな減衰 (整数演算で実現: N クロックに 1 回 -1 など)
            // 実装は decay カウンタを持たせる方が単純
        }
    }
    
    // exists の自動更新
    pub fn update_existence(&mut self) {
        self.exists = self.conductance >= OPEN_THRESHOLD;
    }
}
```

### 3.4 物理配置と隣接性

```rust
pub struct Topology {
    pub grid_width: i32,
    pub grid_height: i32,
    pub neurons_per_cell: i32,  // 1 セルに複数ニューロンを配置可
}

impl Topology {
    // 隣接する PE (4 近傍または 8 近傍) を返す
    pub fn neighbors(&self, position: (i32, i32)) -> Vec<(i32, i32)> {
        let (x, y) = position;
        let mut result = vec![];
        // 4 近傍
        for (dx, dy) in &[(-1, 0), (1, 0), (0, -1), (0, 1)] {
            let nx = x + dx;
            let ny = y + dy;
            if nx >= 0 && nx < self.grid_width && ny >= 0 && ny < self.grid_height {
                result.push((nx, ny));
            }
        }
        result
    }
}
```

ニューロン配置の方針:

- M1 (一次聴覚野) を例として、400 ニューロンを 20×20 グリッドに配置
- 入力ニューロン (20) はグリッドの一辺に集中配置 (例: 上端の 20 セル)
- 出力ニューロン (40) はグリッドの反対側に配置 (例: 下端の 40 セル、または 2 段)
- 残りの皮質ニューロンはグリッド内部

### 3.5 軸索成長機構

これが Phase 2 の核心新機能である。

```rust
pub fn axon_growth_step(
    neurons: &mut [ThermoNeuron],
    synapses: &mut Vec<ThermoSynapse>,
    topology: &Topology,
    current_time: i32,
) {
    // N 試行に 1 回だけ実行 (1 クロックごとではない)
    // 軸索成長は遅いプロセス
    
    for (i, neuron) in neurons.iter().enumerate() {
        // 成長条件: local_entropy が閾値を超えている (熱がこもっている)
        if neuron.local_entropy < GROWTH_THRESHOLD {
            continue;
        }
        
        // 隣接 PE を探す
        let neighbors_positions = topology.neighbors(neuron.position);
        
        // 隣接 PE のうち、最も冷たい (local_entropy が低い) を見つける
        let mut coldest_idx: Option<usize> = None;
        let mut coldest_entropy = neuron.local_entropy;
        
        for (j, other) in neurons.iter().enumerate() {
            if i == j { continue; }
            if !neighbors_positions.contains(&other.position) { continue; }
            
            // 既に結線がある場合はスキップ
            if synapses.iter().any(|s| s.pre == i && s.post == j) {
                continue;
            }
            
            if other.local_entropy < coldest_entropy - GROWTH_DELTA {
                coldest_idx = Some(j);
                coldest_entropy = other.local_entropy;
            }
        }
        
        // 冷たい隣接 PE が見つかった場合、結線を作る
        if let Some(j) = coldest_idx {
            synapses.push(ThermoSynapse {
                pre: i,
                post: j,
                delay: INITIAL_DELAY,
                conductance: MIN_CONDUCTANCE,  // 初期値、低めから始まる
                exists: false,  // すぐには有効化しない
                is_inhibitory: neurons[i].is_inhibitory,
                plastic: true,
            });
            
            // 成長のコスト: 自分のエントロピーを消費
            neurons[i].local_entropy -= GROWTH_COST;
            if neurons[i].local_entropy < 0 { neurons[i].local_entropy = 0; }
            
            // 接続先に熱が伝わる
            neurons[j].local_entropy += HEAT_TRANSFER;
        }
    }
}
```

**重要なポイント**:

1. **判断機構ではない**: 「最も冷たい」を探すのは比較演算であり、目的関数の最小化ではない。これは「水が傾斜に沿って流れる」のと同じ物理的選択である。

2. **タイムスケール分離**: axon_growth_step は毎クロックではなく、N クロック (例えば 100 や 1000) に 1 回だけ実行する。生物の軸索成長は遅いプロセスである。

3. **飽和回避**: GROWTH_COST と HEAT_TRANSFER により、冷たい場所が「冷たくなくなる」ため、集団的な殺到が物理的にブロックされる。

4. **既存結線の刈り取り**: 軸索成長で作った結線も、STDP と conductance 減衰により、共鳴しなければ自然に exists = false に戻る。

### 3.6 ネットワーク統合

```rust
pub struct ThermoNetwork {
    pub neurons: Vec<ThermoNeuron>,
    pub synapses: Vec<ThermoSynapse>,
    pub topology: Topology,
    pub delivery_queue: VecDeque<DeliverySlot>,  // 遅延配送用
    pub current_time: i32,
    pub axon_growth_interval: i32,
    pub last_growth_time: i32,
}

impl ThermoNetwork {
    pub fn step(&mut self, external_input: &[i32]) {
        // 1. 配送スロットから今クロックの入力を取り出す
        let current_inputs = self.delivery_queue.pop_front().unwrap_or_default();
        
        // 2. 外部入力を加算
        // 3. 各ニューロンの update を呼ぶ
        let mut fired = vec![false; self.neurons.len()];
        for (i, neuron) in self.neurons.iter_mut().enumerate() {
            let input = current_inputs.get(i).copied().unwrap_or(0)
                      + external_input.get(i).copied().unwrap_or(0);
            fired[i] = neuron.update(input, self.current_time);
        }
        
        // 4. 発火したニューロンのスパイクを配送 (遅延付き)
        for (i, &did_fire) in fired.iter().enumerate() {
            if !did_fire { continue; }
            
            // STDP 更新 (post 発火を受けて、pre 候補のシナプスを処理)
            for syn in self.synapses.iter_mut() {
                if syn.post == i && syn.exists {
                    let pre_last = self.neurons[syn.pre].last_spike_time;
                    syn.update_on_post_spike(pre_last, self.current_time);
                }
                if syn.pre == i && syn.exists {
                    let post_last = self.neurons[syn.post].last_spike_time;
                    syn.update_on_pre_spike(post_last, self.current_time);
                }
            }
            
            // スパイク配送 (遅延付き)
            for syn in self.synapses.iter() {
                if syn.pre != i || !syn.exists { continue; }
                let arrival_time = self.current_time + syn.delay;
                let signal = if syn.is_inhibitory { -syn.conductance } else { syn.conductance };
                schedule_delivery(&mut self.delivery_queue, arrival_time, syn.post, signal);
            }
        }
        
        // 5. シナプスの自然減衰と exists 更新
        for syn in self.synapses.iter_mut() {
            syn.decay();
            syn.update_existence();
        }
        
        // 6. 軸索成長 (N クロックに 1 回)
        if self.current_time - self.last_growth_time >= self.axon_growth_interval {
            axon_growth_step(
                &mut self.neurons,
                &mut self.synapses,
                &self.topology,
                self.current_time,
            );
            self.last_growth_time = self.current_time;
        }
        
        // 7. 時間進行
        self.current_time += 1;
    }
}
```

**判断機構が一切ない**ことを確認すること:
- target_rate との比較なし
- 報酬関数なし
- apply_self_organization のような明示的学習関数なし

学習は時間経過に伴う物理プロセスの結果として自然に進行する。

---

## 4. パラメータの初期値

実験で調整するため、以下を初期値として使う:

```rust
// ニューロン関連
const ENTHALPY_MAX: i32 = 10;           // エンタルピー上限
const ENTHALPY_RECOVERY_RATE: i32 = 1;  // 数クロックに 1 回回復
const ENTROPY_DECAY_RATE: i32 = 1;      // 数クロックに 1 回減衰
const ENTROPY_PER_SPIKE: i32 = 10;      // 発火 1 回で生じる熱
const THRESHOLD_BASE_EXC: i32 = 80;
const THRESHOLD_BASE_INH: i32 = 40;
const REFRACTORY_EXC: i32 = 4;
const REFRACTORY_INH: i32 = 2;

// シナプス関連
const CONDUCTANCE_MAX: i32 = 100;
const OPEN_THRESHOLD: i32 = 30;
const LTP_AMOUNT: i32 = 5;
const LTD_AMOUNT: i32 = 6;  // LTD > LTP で安定 (Song et al. 2000)
const CAUSAL_WINDOW: i32 = 160;  // 80 ms / 0.5 ms = 160 step

// 軸索成長関連
const GROWTH_INTERVAL: i32 = 200;       // 200 クロックに 1 回成長判定
const GROWTH_THRESHOLD: i32 = 50;       // local_entropy がこれ以上なら成長を試みる
const GROWTH_DELTA: i32 = 20;           // 隣接 PE がこれだけ冷たいなら結線
const GROWTH_COST: i32 = 15;            // 成長で消費するエントロピー
const HEAT_TRANSFER: i32 = 10;          // 接続先に伝わる熱
const MIN_CONDUCTANCE: i32 = 20;        // 新規結線の初期 conductance
const INITIAL_DELAY: i32 = 4;           // 新規結線の初期遅延 (2 ms 相当)
```

これらは初期値である。実験で観察しながら、必要に応じて調整する。ただし「あと一押し」モードに入らないこと。

---

## 5. 評価方法

Phase 1 と同じ枠組みで評価する。これにより比較が可能になる。

### 5.1 評価する 5 性質

1. **決定論性 (Determinism)**: 同じ入力に対する F(X) の安定性
2. **連続性 (Continuity)**: 入力摂動に対する F(X) の連続変化 (最重要)
3. **識別性 (Distinctness)**: 異なる入力に対する F(X) の違い
4. **安定性 (Stability)**: 追加経験後の F(X) の保持
5. **容量 (Capacity)**: 区別可能なパターン数

### 5.2 入力パターン

Phase 1 と同じパターン X を使う:
- 全 5 パターン同一時間スパン (0-95ms, 5ms step)
- A: ascending、B: descending、C: even-then-odd、D: phase shift、E: random

### 5.3 訓練設定

- 訓練試行数: 3000 (Phase 1 と同じ)
- 各試行で 5 パターンからランダムに選択
- 報酬なし、target 指定なし、外部評価なし
- ただ繰り返し提示するだけ

### 5.4 観察項目 (snapshot)

500 試行ごとに以下を記録:

- 各ニューロンの local_entropy 平均と分布
- 各ニューロンの available_enthalpy 平均
- シナプスの conductance 分布
- exists = true のシナプス数 (初期 vs 現在)
- 軸索成長で新規作成されたシナプス数 (累積)
- 刈り取られたシナプス数 (累積)
- 発火率分布 (target_rate との比較ではなく、観察のため)

これらを CSV に出力し、可視化スクリプトで時系列として表示する。

---

## 6. Phase 1 との比較

両 Phase を独立に評価した後、同じパターン・同じ初期条件で比較する。

### 比較項目

1. **収束性**: PRE → POST → PERSIST の selectivity 推移
2. **PERSIST 崩壊率**: Phase 2 で崩壊が起きないことが期待される
3. **active output 比率**: silent 比率の違い
4. **構造変化のダイナミクス**: シナプスの open/close の頻度と分布
5. **計算コスト**: 実行時間と消費メモリ

### 期待される結果

設計哲学に基づく予測:

- Phase 2 は Phase 1 より、長期的安定性で優れる (PERSIST 崩壊が起きにくい)
- Phase 2 は軸索成長により、構造が動的に変化する (Phase 1 はほぼ固定トポロジー)
- Phase 2 は判断機構を持たないため、より「自然」な振る舞いを示す
- ただし収束まで時間がかかる可能性 (軸索成長は遅いプロセス)

予測と異なる結果が出ても問題ない。それは設計哲学の検証である。

---

## 7. 報告タイミング

以下のタイミングで報告する:

1. **設計理解の確認後**: 実装を始める前に、本指示書の理解をユーザーに伝える
2. **基本構造実装後**: 1 試行が動く段階で、生成された snapshot を見せる
3. **500 試行ごと**: 推移を観察
4. **3000 試行完了後**: Phase 2 の評価結果
5. **Phase 1 との比較完了後**: 両 Phase の差異と考察

報告は数字だけでなく、振る舞いの質的な変化も含めること。

---

## 8. 注意事項

### 8.1 判断機構を入れない

実装中に「これ便利そう」と思って判断機構を追加しないこと。例えば:

- 「local_entropy が一定以上なら閾値を上げる」 → これは判断機構、すでに entropy が閾値に加算される設計で同じことが起きるはず
- 「シナプス数が多すぎたらプルーニングする」 → これも判断機構、conductance 減衰で自然に処理される

迷ったら DESIGN_PHILOSOPHY.md の 6 原理に戻る。

### 8.2 確率を使わない

確率的伝送、ランダムなシナプス選択など、確率を使う実装は禁止。すべて決定論的に。

ただし、初期化時に乱数を使うのは許容する (シードを固定すれば決定論的)。

### 8.3 「あと一押し」モードに入らない

パラメータをいじって selectivity を 0.05 上げる作業は禁止。それは設計の本質ではない。

### 8.4 浮動小数点を使わない

すべて整数演算。指数減衰のような「連続的な減衰」も、「N クロックに 1 回 -1」のような離散カウンタで実現する。

### 8.5 比較は公正に

Phase 1 と Phase 2 の比較で、Phase 1 を不利な条件にしないこと。同じ入力パターン、同じ試行数、同じ評価指標で比較する。

---

## 9. ファイル参照

- `DESIGN_PHILOSOPHY.md`: 設計哲学全体、特に §11 (熱力学的描像)
- `CONTEXT.md`: プロジェクト全体像
- `HANDOFF.md`: 現状と次のタスク
- `PAPER_DRAFT.md`: 論文ドラフト (ユーザーの公開意図)
- `src/binary_network.rs`: Phase 1 実装 (参考)
- `src/bin/m1_evaluation.rs`: Phase 1 評価ランナー (参考)

---

## 10. 最終確認

実装を始める前に、以下をユーザーに確認すること:

1. 本指示書の理解を伝える (核心となる物理プロセスを自分の言葉で説明)
2. ディレクトリ構造 (src_phase2) の合意
3. 初期パラメータの妥当性
4. 軸索成長機構の理解 (これが最も新規性が高い部分)
5. 評価方法と比較計画の合意

ユーザーが「進めて」と明示的に言うまで、コード作成を始めないこと。

---

これはユーザーが個人プロジェクトとして数ヶ月かけて練り上げた設計の、現時点での到達点である。慎重に、丁寧に実装してほしい。
