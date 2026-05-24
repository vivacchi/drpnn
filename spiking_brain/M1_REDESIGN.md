# M1 設計変更指示書

最終更新: 2026-05-22
ステータス: 新方針確定、実装フェーズへ移行

このドキュメントは、これまでの試行錯誤を踏まえて **M1 (一次聴覚野相当モジュール) の設計を確定**したものです。`CONTEXT.md` と `HANDOFF.md` の補足として、いまから何を作るかを明示します。

---

## 1. これまでの経緯と転換点

### 失敗パターンの認識

前回までの 13 試行は、以下の根本的な設計ミスから発生していました:

- **M1 に M5 (連合野・報酬系) の機能を入れようとしていた**
- 具体的には `target_a/target_b` で「正解出力ニューロン」を指定し、`apply_reward(r)` で外部報酬学習させていた
- これは教師あり学習であり、A1 (一次聴覚野) の生物学的責務から逸脱
- Neuron 23 の B-specific → shared 逆転、selectivity が pre 状態より下がる現象、初期 180 試行の悪化はすべて「外部報酬が初期の良好な構造を破壊している」ことの帰結

### 階層アーキテクチャの確立

M1 は階層的脳模倣の最初のモジュールであり、以下の責務分離が確定:

```
[M0 蝸牛] → [M1 A1 = 現在ここ] → [M2 A2] → [M3 カテゴリ野] → [M5 連合野・報酬系]
```

- **M1 の責務**: 入力時空間スパイクパターン → 安定したフィンガープリント
- **M1 がやらないこと**: 報酬学習、target 指定、行動評価、不変性学習、意味解釈

### 学習機構の位置づけ

ユーザーとの議論で確定:

> M1 の学習は「収束させる学習」であって「最適化する学習」ではない

つまり、M1 は**「外部から見て正解か」を学ぶのではなく、「自身の応答が再現性を持つように構造を安定化する」**。何のパターンがどのフィンガープリントに対応するかは任意で、それは後段 (M2 以降) が解釈する。

---

## 2. M1 の設計確定版

### 2.1 学習機構

外部報酬・target 指定は完全廃止。代わりに以下 2 つの自己組織化機構のみを実装:

#### 機構 A: 内在的可塑性 (Intrinsic Plasticity) — 破綻防止

各ニューロンが自分の発火率を追跡し、目標範囲外なら閾値を調整。

```
発火率が高すぎる (例: 直近 100 試行で 70 回超) → threshold を +1
発火率が低すぎる (例: 直近 100 試行で 30 回未満) → threshold を -1
```

これにより、ネットワークが発火爆発・全沈黙のどちらにも陥らない。

#### 機構 B: 持続的 Eligibility による構造化 — 収束促進

各シナプスの eligibility を試行をまたいで累積し、安定して正/負の方向にあるものだけが構造変化を起こす。

```
各シナプスに cumulative_sign: f32 を追加
試行末ごとに cumulative_sign += eligibility (符号と大きさ反映)
cumulative_sign が閾値以上で N 試行継続 → open
cumulative_sign が閾値以下で N 試行継続 → close
古い累積はゆっくり減衰 (例: 1 試行ごとに 0.95 倍)
```

これにより、ノイズ的な単発共起は無視され、繰り返し共起したパターンだけが構造に焼き付く。

#### オプション機構 C: ホモシナプティック競合 — 後で必要なら追加

機構 A+B で十分な selectivity が出なければ追加検討。最初は実装しない。

### 2.2 完成基準 (4 条件)

M1 が「完成」したと判定する条件。報酬や正解との一致ではなく、収束特性で判定する。

#### 基準 1: 非破綻性 (Stability)

長時間走らせても発火率が極端化しない:
- 全ニューロン平均発火率が 5-50 Hz の範囲を維持
- silent ニューロン比率が 90% を超えない
- 単一試行あたり 1000 spike を超える「爆発」が起きない

#### 基準 2: 収束性 (Convergence)

同じパターン繰り返し提示でフィンガープリントが安定:
- 訓練後期 100 試行で、同パターン提示時のフィンガープリント類似度 0.85 以上
- 訓練後期と訓練序盤を比較して、後期の方が再現性が高い

#### 基準 3: 識別性 (Discrimination)

異なるパターンが異なるフィンガープリントに収束:
- 5 パターン (A-E) で selectivity 0.4 以上
- 訓練前後で selectivity が明確に向上 (例: pre 0.2 → post 0.5)

#### 基準 4: 持続性 (Persistence)

一度収束したら安定:
- 訓練後期 200 試行を追加実行しても、selectivity が低下しない
- 過学習・振動が起きない

### 2.3 入力パターン設計 (B-ii: 時間構造あり)

ランダムタイミングではなく、明示的な時間構造を持つパターンを使う。これは Polychronization の本領を発揮させるため。

```
パターン長: 100ms
入力ニューロン数: 20
パターン A (ascending sweep):
  neuron 0 @ 0ms, neuron 1 @ 5ms, ..., neuron 19 @ 95ms
パターン B (descending sweep):
  neuron 19 @ 0ms, neuron 18 @ 5ms, ..., neuron 0 @ 95ms
パターン C (ascending fast): A の半分の時間で完了
パターン D (descending slow): B の倍の時間
パターン E (random but fixed): 固定の時間順だがランダム
```

5 パターンを各 200 試行、合計 1000 試行ランダム順序で提示。

---

## 3. 実装タスク

### 3.1 削除するもの (binary_brain.rs / binary_network.rs)

- `target_a`, `target_b` の概念全削除
- `evaluate_reward` 関数削除
- `apply_reward(r)` 関数削除 (置き換え)
- annealing 機構 (`close_threshold` の試行ごとの変動) 削除
- pulses random scaling (`magnitude = pulses / 6.0`) 削除
- 報酬 magnitude による eligibility スケーリング (`weighted_e = e_now * r.abs()`) 削除
- `delay_extra` 関連の残骸も削除可 (既に無効化済み)

### 3.2 追加するもの

#### `BinaryNeuron` に追加するフィールド

```rust
pub recent_fire_count: u32,  // 直近 window 試行での発火回数
pub threshold_min: i32,       // 内在的可塑性の下限
pub threshold_max: i32,       // 内在的可塑性の上限
```

#### `BinarySynapse` に追加するフィールド

```rust
pub cumulative_sign: f32,  // 試行をまたぐ eligibility 累積
```

#### `BinaryNetwork` に追加するメソッド

```rust
/// 試行末に呼ぶ。報酬は受け取らない。
/// 機構 A (内在的可塑性) と機構 B (持続的構造化) を適用。
pub fn apply_self_organization(&mut self) {
    // 機構 A: 内在的可塑性
    for n in &mut self.neurons {
        let rate = n.recent_fire_count as f32 / window_size as f32;
        if rate > TARGET_RATE_MAX {
            n.threshold = (n.threshold + 1).min(n.threshold_max);
        } else if rate < TARGET_RATE_MIN {
            n.threshold = (n.threshold - 1).max(n.threshold_min);
        }
        // 発火カウントは moving window で管理
    }
    
    // 機構 B: 持続的 eligibility による構造化
    for s in &mut self.synapses {
        if !s.plastic { continue; }
        s.cumulative_sign = s.cumulative_sign * CUMULATIVE_DECAY + s.eligibility;
        s.eligibility = 0.0;
        
        if s.cumulative_sign > SUSTAINED_THRESHOLD && !s.exists {
            s.exists = true;
            s.cumulative_sign = 0.0;
        } else if s.cumulative_sign < -SUSTAINED_THRESHOLD && s.exists {
            s.exists = false;
            s.cumulative_sign = 0.0;
        }
    }
}
```

#### Phase 2 訓練ループの再構成

```rust
// Phase 1: 訓練前評価 (5 パターン各 20 試行のフィンガープリント収集)

// Phase 2: 自己組織化フェーズ
let patterns = [pattern_a, pattern_b, pattern_c, pattern_d, pattern_e];
let mut rng = StdRng::seed_from_u64(42);
for trial in 0..1000 {
    let idx = rng.gen_range(0..patterns.len());
    let pat = &patterns[idx];
    present_pattern(net, pat, ..., trial_seed);
    net.apply_self_organization();  // 報酬なし
    
    if (trial + 1) % 100 == 0 {
        // 進捗ログ
    }
}

// Phase 3: 訓練後評価 (Phase 1 と同じ測定)

// Phase 4: 持続性テスト (基準 4)
// さらに 200 試行自己組織化、その後再度評価して selectivity が維持されているか確認
```

### 3.3 評価バイナリ

`binary_brain.rs` を以下に置き換えるか、新規バイナリとして `m1_evaluation.rs` を作る:

```
Phase 1: 訓練前評価 (5 パターン x 20 試行のフィンガープリント記録)
Phase 2: 1000 試行の自己組織化
Phase 3: 訓練後評価 (5 パターン x 20 試行のフィンガープリント記録)
Phase 4: 200 試行の追加自己組織化 + 評価 (持続性チェック)

出力:
- CSV: フィンガープリント全データ
- CSV: 試行ごとの統計 (発火率分布、open シナプス数、構造変化数)
- 標準出力: 4 基準の判定結果
```

### 3.4 可視化スクリプトの拡張

`visualize.py` を拡張して以下を追加:

- **発火率時系列**: 全ニューロンの平均発火率が target 範囲内に収まっているか
- **閾値分布の進化**: 内在的可塑性で threshold がどう動いたか
- **5x5 類似度マトリクス**: Phase 3 のフィンガープリントで A-E の分離度を可視化
- **持続性プロット**: Phase 3 と Phase 4 の selectivity 比較

---

## 4. 期待される結果と判定

### 期待される動作

1. **初期数十試行**: ネットワークがランダム発火、内在的可塑性が閾値を調整 (機構 A の効果)
2. **数百試行**: cumulative_sign が安定パターンを蓄積し始める (機構 B が動き始める前段階)
3. **数百〜千試行**: 構造変化が発生、各パターンに対応する経路が固定化
4. **1000 試行以降**: フィンガープリントが安定、追加試行しても selectivity 維持

### 各基準の判定

完成判定は 4 基準すべて満たすこと:

```
基準 1 (非破綻性):    [OK/NG]  発火率範囲、silent 比率、爆発の有無
基準 2 (収束性):       [OK/NG]  訓練後期の同パターン類似度 0.85+
基準 3 (識別性):       [OK/NG]  5 パターン selectivity 0.4+
基準 4 (持続性):       [OK/NG]  Phase 4 で selectivity 維持
```

### 不合格時の対応 (前回の失敗を繰り返さないために)

**前回の最大の失敗**: 数値が低かったとき、パラメータをいじって「あと一押し」を続けた。13 試行で selectivity 0.51 〜 0.58 をうろうろした結果、設計の根本問題に気づくのが遅れた。

**今回のルール**: 4 基準のどれかが不合格だった場合、まず以下を**観察**してから対処を決める:

1. 何 phase で不合格になっているか確認 (基準ごと)
2. 可視化を見て、ネットワークが**何をしているか**を理解する
3. パラメータ調整は最大 3 試行まで。それで改善しなければ設計レベルの再検討
4. 機構 C (ホモシナプティック競合) の追加は最後の手段、A+B で頑張ってからの話

「あと一押し」と感じたら立ち止まる、というのを設計ルールにする。

---

## 5. 重要な原則 (再掲)

1. **M1 は特徴抽出器であって分類器ではない**
2. **「正解」との一致を測らない、再現性と識別性を測る**
3. **外部報酬・target 指定は M1 に入れない (M5 の責務)**
4. **学習は「収束させる学習」、最適化ではない**
5. **パラメータチューニングで指標を 0.05 上げる作業は完成判定の後**
6. **「あと一押し」と感じたら立ち止まる、可視化を見る**

---

## 6. 次のステップ (M1 完成後)

M1 が 4 基準すべて合格したら:

1. **M1 の仕様書を作成** (`docs/M1_SPECIFICATION.md`) — 応答潜時、時間分解能、ノイズ耐性、容量などを定量化
2. **M0 (蝸牛模擬) の実装** — Gammatone フィルタバンクで実音声 → スパイク列
3. **M0 + M1 結合** — 実音声で M1 を再評価
4. **M2 (二次聴覚野) の設計** — M1 の出力フィンガープリント列を入力として不変表現を学ぶ

ただし、これらは M1 完成後の話。**いまは M1 を仕上げることだけに集中する**。

---

## 7. 質問・確認事項

実装中に判断に迷ったら、以下のいずれかで判断する:

1. **設計思想に照らして判断** (本書 §1, §5 を参照)
2. **生物学的根拠を確認** (A1 で起きていないことは M1 でもやらない)
3. **ユーザーに確認** (大きな設計変更を伴う場合)

特に**新機能を追加したくなった場合**は、まずそれが M1 の責務なのか M2 以降の責務なのかを判定し、M1 の責務であることが明確な場合のみ追加する。
