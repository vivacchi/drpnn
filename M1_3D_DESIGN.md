# M1 3D 配置 (HCP) 設計ドキュメント

作成日: 2026-05-25
ステータス: 設計検討中 (実装前)、 Phase 3 別バージョンとして位置付け

## 目的

現状の Fork F-G1-R1 v2 (2D grid 20×22) を **完全に保持**したまま、 並行で **3D 配置版 M1** を開発し、 両者を比較実験する。

ユーザー指摘事項を反映:
- 「実際の生物のニューロン配置 (3D 皮質構造) に近い形にしたい」
- 「ニューロン同士の距離は一定で均等に分布」 → HCP (六方最密充填)
- 「入力 20 / 出力 30 (= 入力 < 出力)」を実現
- DRP マッピングは「1 PE 時分多重」なので、 物理 PE 配置と論理ニューロン配置は分離可能 → 3D 論理配置は DRP 上で実装可能

## 既存設計との関係

### 完全独立

```
spiking_brain/
├── src_phase2_f/             ← Fork F-G1-R1 v2 (2D)、 完全保持
│   └── (POST 0.795, Kenet t=4.169 の基盤)
└── src_phase3_3d/             ← 新規 Phase 3 (3D HCP)
    └── (本ドキュメントの実装対象)
```

両者の数値を **同条件で比較** することで、 3D 化が学習性能に与える影響を直接検証する。

## 設計原理 (DESIGN_PHILOSOPHY.md 6 原理)

3D 化しても 6 原理は完全遵守:

1. **局所性**: HCP 12 近傍のみ情報交換、 3D でもローカル原則維持
2. **物理性**: 物理プロセスのみ、 判断機構なし
3. **決定論性**: 整数演算、 確率なし
4. **整数演算**: 浮動小数禁止
5. **創発性**: 配置が 3D 化しても局所機構から振る舞いを立ち上げる
6. **平衡としての学習**: apply_learning() なし

加えて:
- **タイプ 1 学習のみ** (M5 以降のタイプ 2 はなし)
- **1 セル 1 ニューロン原則** (DRP マッピング整合性、 3D 上でも 1 (x,y,z) 1 ニューロン)

---

## 第 1 部: 生物学的根拠

### 1.1 皮質の真の構造は 3D

- 大脳皮質: 面積 ~2,500 cm²、 厚み 2-4 mm
- ニューロン数 ~16 G (Herculano-Houzel 2009)
- **3D 構造**: 縦 (厚み) と横 (面積) で異なる組織化

縦方向 (厚み):
- 6 層構造 (Brodmann 1909): Layer I-VI
- 各層で細胞種・密度・役割が異なる

横方向 (面積):
- Minicolumn (Mountcastle 1957): 直径 30-50μm、 ~80-100 ニューロン
- Macrocolumn: 直径 ~500μm、 ~10,000 ニューロン
- 機能的単位として垂直に揃った発火パターン

### 1.2 ニューロン間距離

実際のニューロンは:
- 細胞体直径 10-20μm
- 隣接ニューロン間距離 ~20-50μm
- **3D で局所的に等距離分布**

これを HCP (六方最密充填) で擬似する:
- 各ニューロンの周囲 12 個が **等距離** (最近接)
- 球の最密充填 = 自然の規則的配置 (生物の細胞密度と整合)

### 1.3 関連文献

- **Mountcastle V.B. (1957)** "Modality and topographic properties of single neurons of cat's somatic sensory cortex" J Neurophysiol — カラム構造
- **Rockel A.J. et al. (1980)** "The basic uniformity in structure of the neocortex" Brain — 均一密度
- **Buxhoeveden D.P., Casanova M.F. (2002)** "The minicolumn hypothesis in neuroscience" Brain — minicolumn
- **Herculano-Houzel S. (2009)** "The human brain in numbers" Front Hum Neurosci

---

## 第 2 部: 3D 配置の具体設計

### 2.1 座標系 (HCP)

擬似 HCP を **整数座標** で表現:

```
position: (x, y, z) ∈ Z³

近傍計算で z の偶奇に応じてオフセット適用:
  z 偶数 (z % 2 == 0): 標準位置
  z 奇数 (z % 2 == 1): x 軸方向に +0.5 (整数では未表現、 近傍計算で考慮)
```

実装の単純化:
- 物理座標は整数 (x, y, z)
- 近傍判定は z の偶奇で **異なる近傍テーブル** を使用
- これにより HCP の 12 等距離近傍を表現

### 2.2 12 近傍 (HCP)

各ニューロンの 12 個の等距離近傍 (z 偶数の場合):

```
同層 (z): 6 個
  (x-1, y, z), (x+1, y, z)       横 2 個
  (x, y-1, z), (x, y+1, z)       縦 2 個
  (x-1, y-1, z), (x+1, y-1, z)   斜め (z 偶奇でずれ)

下層 (z-1): 3 個
  (x, y, z-1), (x-1, y-1, z-1), (x, y-1, z-1)

上層 (z+1): 3 個
  (x, y, z+1), (x-1, y-1, z+1), (x, y-1, z+1)
```

z 奇数の場合は x 軸方向に逆オフセット。 詳細は実装時に確定。

### 2.3 grid 規模

3 段階の規模で実装、 順次拡大:

| 規模 | 寸法 | セル総数 | 用途 |
|---|---|---|---|
| **小** | 10 × 10 × 5 | 500 | プロトタイプ、 動作確認 |
| 中 | 20 × 20 × 5 | 2,000 | 性能評価、 識別性 |
| 大 | 30 × 30 × 10 | 9,000 | 本評価、 大規模試験 |
| 最大 | 30 × 30 × 22 | 19,800 | 将来 (計算時間 8-12 時間/10k 試行) |

最初は **小規模 (500 セル)** で実装フィージビリティ確認。

### 2.4 入出力配置

ユーザー要望: 入力 20 / 出力 30 (= 入力 < 出力、 「層を超えて出力に集中する」生物的傾向)。

**中規模 (20 × 20 × 5 = 2,000 セル) の場合**:

```
z=0 (表面、 「Layer I 相当」):
  y=0  : 入力 20 (x=0..19、 1 行使い切り)
  y=1..19: 内部処理 380

z=1..3 (内部 3 層、 「Layer II-V 相当」):
  全 20×20×3 = 1,200 セル を内部処理

z=4 (背面、 「Layer V/VI 相当」):
  y=H-1: 出力 30 (... ここで grid_w=20 だと 30 個並ばない)
```

問題: 20×20×5 では出力 30 個 並べる「1 行」がない (grid_w=20)。

**解決策**:
- **(A) grid 拡張**: 30×20×5 = 3,000 セルにして grid_w=30
- **(B) 出力を分散配置**: 30 個を背面に散らす (z=4 の任意 30 セル)
- **(C) 出力を 2 行に**: z=4 の y=H-1, H-2 に分散 (元の「2 行問題」の 3D 版)

ユーザーの過去意図 (出力 1 行に統一) を尊重するなら **(A)** か **(B)**。

**推奨**: **(B) 出力を 1 z 層 (背面) に分散**
- 背面 z=Z-1 のすべて = 20×20=400 セルから 30 個を seed ベースで選択
- 「生物の A1 → A2 投射点が皮質背面で分散している」事実と整合

### 2.5 中規模設計 (推奨)

```
grid 20 × 20 × 5 = 2,000 セル

z=0 (表面):
  y=0   : 入力 20 (1 行)
  y=1..19: 興奮性 (380 - 抑制性数) + 抑制性 (HCP 12 近傍向け局所)

z=1..3 (内部 3 層):
  20×20×3 = 1,200 セル
  - 興奮性 (1,200 × 0.82 ≒ 984)
  - 抑制性 (1,200 × 0.18 ≒ 216)
  - ランダム分散

z=4 (背面):
  20×20=400 セルから 30 個を seed ベースで選択 → 出力 30
  残り 370 セルは内部処理

合計:
  入力 20
  内部 興奮 + 抑制 (約 1,950 個、 抑制比 ~18%)
  出力 30
  = 2,000 (完全充填)
```

### 2.6 抑制比 18%

M1 (2D) と同じ Markram et al. 2004 基準。 3D でも層全体で 18%。

---

## 第 3 部: 実装仕様

### 3.1 ファイル構成

```
src_phase3_3d/
├── mod.rs                  ← lib モジュール宣言
├── thermo_neuron.rs        ← position を (i32, i32, i32) に拡張
├── thermo_synapse.rs       ← src_phase2_f/ から流用 (シナプスは位置依存性なし)
├── topology3d.rs           ← HCP 12 近傍
├── axon_growth_3d.rs       ← 3D 距離での冷たい隣接探索
├── thermo_network_3d.rs    ← 3D 配置ロジック + step
└── bin/
    └── m1_3d_evaluation.rs  ← M1 3D 単体評価 (M0 から音素入力)
```

### 3.2 ThermoNeuron の変更

```rust
pub struct ThermoNeuron {
    // 既存フィールド (membrane, available_enthalpy, ...)
    // ...
    /// 3D 配置位置
    pub position: (i32, i32, i32),  // ← 2D (i32, i32) から拡張
}
```

他は ほぼ完全に流用可能 (物理プロセスは位置非依存)。

### 3.3 Topology3d

```rust
pub struct Topology3d {
    pub grid_w: i32,
    pub grid_h: i32,
    pub grid_d: i32,  // 奥行き
}

impl Topology3d {
    /// HCP 12 近傍 (z の偶奇に応じて x 軸オフセット適用)
    pub fn neighbors(&self, p: (i32, i32, i32)) -> Vec<(i32, i32, i32)> { /* ... */ }
}
```

### 3.4 ThermoNetworkConfig3d

```rust
pub struct ThermoNetworkConfig3d {
    pub grid_w: i32,
    pub grid_h: i32,
    pub grid_d: i32,
    pub n_input: usize,       // 例 20
    pub n_output: usize,      // 例 30
    pub n_excitatory: usize,  // 内部 + 出力 興奮性
    pub n_inhibitory: usize,
    pub input_layer: i32,     // z=0 (表面)
    pub output_layer: i32,    // z=grid_d-1 (背面)
    pub input_fanout: usize,
    pub delay_range: (i32, i32),
    pub seed: u64,
    pub axon_growth_interval: i32,
    pub enable_up_down: bool,
}
```

### 3.5 配置ロジック

```rust
// 概念実装
let mut neurons = Vec::with_capacity(grid_w * grid_h * grid_d);
let mut input_neurons = Vec::new();
let mut output_neurons = Vec::new();

// (1) 入力 (z=0, y=0, x=0..n_input)
for x in 0..n_input {
    let p = (x as i32, 0, 0);
    neurons.push(ThermoNeuron::input(p));
    input_neurons.push(neurons.len() - 1);
}

// (2) 出力 (z=grid_d-1 から seed ベースで 30 個選択)
let back_face: Vec<(i32, i32, i32)> =
    (0..grid_h).flat_map(|y| (0..grid_w).map(move |x| (x, y, grid_d - 1))).collect();
let mut shuffled = back_face.clone();
shuffled.shuffle(&mut rng);
let output_positions: Vec<_> = shuffled.into_iter().take(n_output).collect();
// → output_positions の各位置に excitatory ニューロンを配置、 output_neurons に登録

// (3) 内部 (残りの全セル) で 興奮/抑制 をランダム配置 (M1 2D と同じロジック)
```

### 3.6 軸索成長 3D

```rust
pub fn axon_growth_step_3d(neurons, synapses, topology3d, position_index) {
    for n in neurons.iter() {
        if n.local_entropy > GROWTH_THRESHOLD {
            let neighbors = topology3d.neighbors(n.position);  // 12 近傍
            let coldest = neighbors.iter()
                .filter_map(|&p| position_index.get(&p))
                .flat_map(|ids| ids.iter())
                .min_by_key(|&&id| neurons[id].local_entropy)
                .copied();
            // ... 新規シナプス作成、 熱伝達
        }
    }
}
```

---

## 第 4 部: 評価計画

### 4.1 一次評価 (M1 3D 単体)

| 指標 | 2D (Fork F-G1-R1 v2) | 3D (目標) |
|---|---|---|
| 固定 A-E POST selectivity | 0.795 | ?? (測定して比較) |
| within | 0.953 | ?? |
| between | 0.158 | ?? |
| Kenet 2003 t-statistic | 4.169 | ?? |
| 1 trial 計算時間 | ~0.18 sec | ~0.4 sec (4.5x) |

### 4.2 設計比較の意義

3D 化が学習性能を:
- **向上** させる場合: 「皮質の 3D 構造は学習に本質的」を実証
- **同等** の場合: 「2D で十分」(DRP コスト低い方を採用)
- **悪化** させる場合: 「3D は理論的に正しいが小規模では効果なし」(将来の大規模化で再検討)

これは設計判断の重要なデータポイント。

### 4.3 計算量見積もり

20×20×5=2000 セルの場合:
- ニューロン: 2,000
- シナプス: 想定 70,000 (M1 2D 17,000 の 4 倍)
- 1 trial: ~0.5 sec
- 10k 訓練: ~80 分

これは現実的な範囲。

---

## 第 5 部: 実装ロードマップ

| Step | 内容 | 予想時間 |
|---|---|---|
| 1 | src_phase3_3d/ ディレクトリ作成、 thermo_synapse 流用 | 5 分 |
| 2 | topology3d.rs 実装、 HCP 12 近傍 + テスト | 30 分 |
| 3 | thermo_neuron.rs の position 3D 化 | 15 分 |
| 4 | thermo_network_3d.rs (配置 + step) | 60 分 |
| 5 | axon_growth_3d.rs | 30 分 |
| 6 | bin/m1_3d_evaluation.rs (M0 流用、 評価フロー) | 45 分 |
| 7 | 小規模 (10×10×3=300) で動作確認 | 15 分 |
| 8 | 中規模 (20×20×5=2000) で 100/10k 試行評価 | 100 分 |

合計 約 5 時間相当。 段階的に進めて各 Step でコミット。

---

## 第 6 部: 設計上の未決定事項

### 6.1 HCP 近傍の正確な定義

12 近傍を整数座標で表現する公式が複数あり、 実装時に最適なものを選ぶ。

### 6.2 入力配置 (1 行 vs 分散)

ユーザー直感 (入力 20 上端 1 行) vs 生物的実態 (入力 LGN 線維も皮質内に分散) のバランス。 初期は 1 行で実装、 必要なら分散化。

### 6.3 出力分散ルール

seed ベースのランダム配置 vs 「クラスタリング」(近隣の出力ニューロンが連続) vs 「均等分散」(逐次間隔)。 初期はランダムで、 結果次第で改良。

### 6.4 M0 → M1 (3D) 接続

M0 蝸牛は 2D (20 帯域)、 M1 3D の入力 z=0 y=0 x=0..19 に直接 1:1 マッピング。 これは現状の M0 → M1 (2D) と同じ単純さで OK。

### 6.5 M2 への接続

M2 を 3D 化するか 2D に戻すかは、 M1 3D の結果を見てから判断。

---

## 第 7 部: 参考文献

- Mountcastle V.B. (1957) "Modality and topographic properties of single neurons of cat's somatic sensory cortex" *J Neurophysiol* 20:408
- Rockel A.J., Hiorns R.W., Powell T.P.S. (1980) "The basic uniformity in structure of the neocortex" *Brain* 103:221
- Buxhoeveden D.P., Casanova M.F. (2002) "The minicolumn hypothesis in neuroscience" *Brain* 125:935
- Herculano-Houzel S. (2009) "The human brain in numbers" *Front Hum Neurosci* 3:31
- Markram H. et al. (2004) "Interneurons of the neocortical inhibitory system" *Nat Rev Neurosci* 5:793

---

## 実装着手前の最終確認

- ✅ ユーザー指摘: DRP マッピングは 1 PE 時分多重で 3D 論理配置可能 → 整合
- ✅ 現状 Fork F-G1-R1 v2 (2D) は完全保持、 並行開発
- ✅ 過去の POST 0.795 / Kenet t=4.169 は 2D 状態のまま記録維持
- ✅ 3D 化で何が変わるかを実証で判断
- 🔄 本ドキュメント (M1_3D_DESIGN.md) ← 現在ここ
- ⏳ Step 1-8 の段階的実装

ユーザー確認後に Step 1-8 を順次着手。
