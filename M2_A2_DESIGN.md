# M2 二次聴覚野 (A2) 設計ドキュメント

作成日: 2026-05-25
ステータス: 設計検討中 (実装前)

## 目的

DESIGN_PHILOSOPHY.md / CONTEXT.md で示された 12 モジュール構成のうち、**M1 (一次聴覚野) の次階層** にあたる M2 (二次聴覚野、A2) を設計する。

M0 (蝸牛、PAPER §5.11) と M1 (Fork F-G1-R1 v2、PAPER §5.9-5.12) が動作完了した後の **第 3 のモジュール**。M1 の連続写像 fingerprint から「**不変性 (invariance)**」を獲得することが責務。

## 設計原理 (DESIGN_PHILOSOPHY.md と整合)

M1 と同じ 6 原理を遵守:
1. **局所性**: M1 出力ニューロン (40 個) のみを入力源とする
2. **物理性**: 物理プロセスのみ、判断機構なし
3. **決定論性**: 整数演算、確率なし
4. **整数演算**: 浮動小数禁止
5. **創発性**: トップダウン設計しない、局所機構から不変性を立ち上げる
6. **平衡としての学習**: `apply_invariance_learning()` のような関数は持たない

加えて、**タイプ 1 学習のみ**を扱う (M5 以降のタイプ 2 教示学習は持ち込まない)。

---

## 第 1 部: A2 (二次聴覚野) の生理学

### 1.1 A1 → A2 の階層的役割

| 階層 | 主な役割 | 生物学的記述 |
|---|---|---|
| **A1 (M1)** | 周波数 × 時間の時空間パターン → fingerprint | Tonotopic, 周波数選択性が鋭い (Q ~10), 時間精度数 ms |
| **A2 (M2)** | **不変性獲得**、より複雑な音響特徴 (FM 掃引、AM、和音) の検出 | Less tonotopic, 時間積分窓が長い (10-100 ms), 抽象度上昇 |
| 高次 (M3 以降) | カテゴリ、 音韻、 メロディー | 言語特化、 意味付与 |

**主要文献**:
- **Recanzone et al. (2000)** "Correlation between the activity of single auditory cortical neurons and sound-localization behavior" J Neurophysiol — A2 が音源定位に関与
- **Tian & Rauschecker (2004)** "Processing of frequency-modulated sounds in the lateral auditory belt cortex" J Neurophysiol — A2 が FM 掃引に選択的
- **Bizley & Cohen (2013)** "The what, where and how of auditory-object perception" Nat Rev Neurosci — A1/A2 の機能分離
- **King & Nelken (2009)** "Unraveling the principles of auditory cortical processing" Nat Neurosci — 階層的処理レビュー

### 1.2 「不変性」とは何か

**Quian Quiroga et al. (2005)** "Invariant visual representation by single neurons in the human brain" Nature 435:1102 — 「concept cells」の発見:
- 海馬の単一ニューロンが「Jennifer Aniston」の様々な画像 (正面、横顔、似顔絵) に同じく反応
- 視覚的特徴は異なるのに、 **「同じ人物」というカテゴリで応答する**
- これが「不変性」の極致

聴覚野での不変性:
- 同じ音素 (例 /a/) が、 話者・声の高さ・速度によらず同じ応答
- 同じメロディーが、 移調 (キー変更) によらず同じ応答
- 同じ音源が、 反響・雑音によらず同じ応答

### 1.3 A2 の主な計算

A1 の連続写像 (= fingerprint) を入力として、A2 は:
1. **時間統合**: 100ms オーダーの時間窓で fingerprint を統合
2. **次元削減**: A1 出力の冗長性を取り除く (sparse coding)
3. **カテゴリ形成**: 類似した fingerprint を同じ A2 ニューロン集合にマップ
4. **不変性獲得**: 入力の微小変動 (ジッタ、声質差) に対する出力の頑健性

これは生物では、A1 → A2 のシナプス可塑性 (STDP + 構造可塑性) の自然な帰結として起きる。

---

## 第 2 部: 本プロジェクトでの M2 設計

### 2.1 階層位置

```
[音声波形 16 kHz]
        ↓
[M0 蝸牛 (実装済)]  20 帯域フィルタ + 包絡線 + 閾値発火
        ↓ (20 channel)
[M1 A1 (実装済)]   Fork F-G1-R1 v2, 440 ニューロン, STDP+vitality+UP/DOWN
        ↓ (40 output ニューロン)
[**M2 A2 (本設計)**]  不変性獲得 + 時間統合
        ↓
[M3 カテゴリ野 (将来)]
        ↓
[M4 海馬, M5 報酬系 ...]
```

### 2.2 M2 の責務範囲

**やること**:
- M1 出力 fingerprint (40 次元、時間 bin) から、より抽象的な表現を作る
- 時間統合 (例: 300ms 窓 → 100ms 窓に圧縮)
- 入力の微小変動への頑健性 (連続性) 強化
- 自己組織的にカテゴリらしいクラスタを形成 (ラベル付けはしない)

**やらないこと**:
- 「これは音素 /a/」のようなラベル付与 (M5 以降の責務)
- 報酬・正解判定 (タイプ 2 学習、M5+)
- 言語的意味付与 (M3 以降)

### 2.3 構造案

**Option A: M1 と同じ熱力学ニューロン構成 (推奨)**:
- M2 ニューロン数: 例 100-300 (M1 の 1/2 程度、 抽象化に伴う絞り込み)
- M2 入力: M1 の 40 出力ニューロン (high fanout、 各 M2 ニューロンが 40 個全て受信)
- M2 内部: STDP + vitality + 軸索成長 (M1 と同じ機構)
- M2 出力: 例 20-40 ニューロン (さらに抽象化)
- グリッド: 20 × 20 = 400 等 (M1 と同じ DRP 原則)

**Option B: 階層的サブネット (将来案)**:
- 複数の小規模 M2 サブネット (例: 5 個の 80 ニューロンサブネット)
- 各サブネットが特定の音響特徴 (周期性、エンベロープ等) に特化
- 「特化分業」を物理プロセスで実装

**まずは Option A から始める** (M1 の経験を活かせる、設計の連続性)。

### 2.4 入力時間スケール

A1 (M1) → A2 (M2) で時間積分窓を拡大:
- M1: 0.5 ms / step、 trial 300 ms
- M2: 同じ DT_MS = 0.5 ms だが、 STDP 因果窓 を拡大 (CAUSAL_WINDOW_M2 = 320 step = 160 ms)
- これにより M2 は「音節 〜 単語」スケールの時間構造を捉える

### 2.5 不変性獲得のメカニズム (創発的)

明示的に「不変性を学習する」関数は作らない。代わりに:

1. **入力多様性**: 訓練時に同じ「意味」の刺激を多様な微小変動付きで提示
   - 例: 音素 /a/ を 速度・ピッチ・ジッタを変えて多数回
2. **STDP の汎化効果**:
   - 似た fingerprint には同じ M2 ニューロン集合が共発火
   - 共発火を STDP が強化 → 多様な入力が同じ M2 ニューロンへ収束
   - これが「不変性」として現象する
3. **構造的可塑性**:
   - 使われない経路は vitality 減衰で消失
   - 共通の「不変経路」だけが生き残る

これは M1 の動的平衡形成 (PAPER §5.9-5.12) と同じ哲学。

---

## 第 3 部: 実装仕様

### 3.1 ファイル構成

```
src_phase2_f/
  m2_a2.rs                      ← M2 構造体と物理プロセス (M1 と類似)
  bin/
    m0_m1_m2_pipeline.rs        ← M0+M1+M2 統合実験ハーネス
```

### 3.2 ThermoNetwork の再利用

M1 で実装した `ThermoNetwork` を **そのまま再利用** する設計が最も哲学に整合:
- M2 は別の `ThermoNetwork` インスタンス
- M1 の出力 40 ニューロンの発火を M2 の入力 40 ニューロンへ電流として渡す
- 各クロックで M1.step() → M2.step() を順に呼ぶ

```rust
// 概念実装
let mut m1 = ThermoNetwork::new(cfg_m1);
let mut m2 = ThermoNetwork::new(cfg_m2);

for step in 0..n_steps {
    let m1_output = m1.step(&external_input);
    // M1 の出力ニューロン発火を M2 の入力電流に変換
    let m2_input = m1_output_to_m2_input(&m1, &m1_output);
    let m2_output = m2.step(&m2_input);
}
```

**利点**: M1 のコードを 1 行も変えずに M2 を追加可能。設計の orthogonality (直交性) が高い。

### 3.3 M2 評価指標

M2 単体の評価指標 (M1 評価と同等):
- **時間 bin 化 fingerprint** (M2 出力 × 時間 bin)
- **不変性テスト**: 同じカテゴリ刺激の多様な変種に対する fingerprint 類似度 (within > between)
- **連続性テスト**: ジッタ・欠落・追加に対する応答の滑らかさ
- **内部状態レパートリ** (Kenet 2003 同様の検証)

### 3.4 既存実験との比較

| 指標 | M1 単体 | M1+M2 統合 (期待) |
|---|---|---|
| within (同パターン応答一致) | 0.953 | より高い (~0.97?) |
| between (異パターン応答類似) | 0.158 | より低い (~0.05?) |
| 連続性 (ジッタ ±5ms) | 未測定 | M2 で大幅改善 |
| カテゴリ形成 | なし | 自己組織的に立ち上がる |

---

## 第 4 部: 実装ロードマップ

### Step 1: M0+M1 のリファクタ (前準備)

ThermoNetwork のインスタンスを 2 個並行で持てるよう、 main 構造を整理:
- 現状: 単一 ThermoNetwork
- 新: M1 と M2 の 2 インスタンス、 step を順に呼ぶ

### Step 2: M2 単体構築 (ThermoNetworkConfig で M2 用設定)

M2 用の config:
- grid: 例 20×20 = 400 (M1 と同じか少し小さく)
- input: 40 (M1 出力数と一致)
- output: 20-40 (抽象化、絞り込み)
- 内部: 残り = 興奮 + 抑制 (18% 抑制比)

### Step 3: M1 → M2 配線

M1 の出力ニューロン発火イベントを M2 の入力電流ベクトルに変換:
```rust
fn m1_output_to_m2_input(m1: &ThermoNetwork, m1_fired: &[usize]) -> Vec<i32> {
    let mut m2_input = vec![0i32; M2_N_INPUT];
    for &nid in m1_fired {
        if let Some(oi) = m1.output_index_of(nid) {
            m2_input[oi] = INPUT_CURRENT;  // M1 出力 → M2 入力電流
        }
    }
    m2_input
}
```

### Step 4: 統合 bin と動作確認 (100 trial)

`bin/m0_m1_m2_pipeline.rs` を作成、 音素 5 種 (pa, ki, tu, se, mo) を入力し、M2 出力の fingerprint を測定。

### Step 5: 不変性テスト

同じ音素を多様な変種 (ピッチ ±10%、速度 ±20%、ジッタ ±5ms) で提示し、M2 内部 fp の within / between を測定。

### Step 6: 10k 訓練 + 統合評価

固定 5 音素 (現状) と多様変種版を併用して 10k 訓練、M2 出力で連続性・識別性が M1 単体より向上するか検証。

---

## 第 5 部: 未解決の設計判断

### 5.1 M1 → M2 信号変換

`m1_output_to_m2_input` で「発火イベント → 電流」の変換は単純化しすぎる可能性。代案:
- **時間平滑化**: M1 出力の発火を低域通過フィルタで平滑し、 M2 の連続入力に
- **直接シナプス**: M1 の出力ニューロン と M2 の入力ニューロンを物理的シナプスで接続 (より生物的)

Option B (直接シナプス) を採るなら、ThermoNetwork を拡張するか、新たな統合 NetworkLayered を作る必要あり。

### 5.2 M2 の STDP 時定数

A1 (M1) では CAUSAL_WINDOW=160 step (80ms)。 A2 (M2) ではより長くすべき (音節レベル):
- 候補: 320 step (160ms) or 400 step (200ms)
- ThermoSynapse 定数を M1/M2 で分離する必要あり (現状はグローバル定数)

### 5.3 多重 M2 サブネット (Option B)

将来的に複数 M2 サブネットを並行運用する場合の設計:
- 各サブネットがどの音響特徴を「専門」とするか自律的に決まる必要 (top-down で割り当てない)
- これは creative emergence の領域で、 まずは Option A (単一 M2) で動作確認

### 5.4 M2 → M3 への接続

M2 完成後、 M3 (カテゴリ野) を加えるとき:
- M3 はカテゴリ形成 = 教師なしクラスタリング相当を物理プロセスで
- M2 → M3 で更なる抽象化
- 設計は M2 と同じ哲学だが、 学習則 (Hebb? STDP?) の微調整あり得る

---

## 第 6 部: 評価方針

### 6.1 一次評価 (M2 単体機能)

- 音素 5 種の識別: M2 出力で M1 同等以上の selectivity
- 連続性: M1 出力 fingerprint にジッタを加えても M2 出力が安定
- 内部状態レパートリ (Kenet 2003): M1 同様、訓練後の自発活動が刺激応答と類似

### 6.2 二次評価 (不変性の物理実証)

- 同じ音素を **多様変種** で提示 → M2 fp の within 高、between 低
- ジッタ ±5ms、ピッチ ±10%、速度 ±20% の摂動で M2 fp が滑らかに変化 (急変しない)
- これらの「不変性」が **明示的な不変性学習** なしに立ち上がる (創発的)

### 6.3 三次評価 (M1 単体との比較)

- M1 単体での識別性能 vs M1+M2 統合での識別性能
- 期待: M2 を加えると within ↑、 between ↓、 連続性 ↑↑
- これが確認されれば「M2 は M1 を補完する」という階層設計の正当性が立証

---

## 第 7 部: 参考文献

### 二次聴覚野生理学

- Recanzone G.H. et al. (2000) "Correlation between the activity of single auditory cortical neurons and sound-localization behavior" *J Neurophysiol* 83:2723
- Tian B., Rauschecker J.P. (2004) "Processing of frequency-modulated sounds in the lateral auditory belt cortex" *J Neurophysiol* 92:2993
- Bizley J.K., Cohen Y.E. (2013) "The what, where and how of auditory-object perception" *Nat Rev Neurosci* 14:693
- King A.J., Nelken I. (2009) "Unraveling the principles of auditory cortical processing" *Nat Neurosci* 12:698

### 不変性表現

- Quian Quiroga R., Reddy L., Kreiman G., Koch C., Fried I. (2005) "Invariant visual representation by single neurons in the human brain" *Nature* 435:1102
- DiCarlo J.J., Cox D.D. (2007) "Untangling invariant object recognition" *Trends Cogn Sci* 11:333

### 階層的可塑性

- Buzsáki G. (2010) "Neural syntax: cell assemblies, synapsembles, and readers" *Neuron* 68:362
- Markram H. et al. (2011) "A history of spike-timing-dependent plasticity" *Front Synaptic Neurosci* 3:4

---

## 実装着手前の最終確認

- ✅ M1 (Fork F-G1-R1 v2) 完成、 PAPER §5.12.3a で Kenet 2003 を t=4.169 で実証済
- ✅ M0 蝸牛完成、 音素生成 OK
- ✅ 検定手法 (Welch-Satterthwaite, 空間保存シャッフル, 状態遷移行列) 整備済
- ✅ Tier 1-3 コード品質改善完了
- 🔄 M2 設計 (本ドキュメント) ← 現在ここ
- ⏳ M2 実装着手判断

ユーザー確認後に Step 1-6 の実装に進む。
