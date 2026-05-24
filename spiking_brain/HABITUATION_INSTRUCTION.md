# M1 改修指示: 慣化 (Habituation) 機構の導入

## 1. 背景と動機

### 現状の問題
- 機構 A (内在的可塑性) が「全ニューロンを target_rate に保とう」とする圧力を持つ
- silent ニューロンの閾値を下げ続け、下限に張り付くと「ノイズ受容体」化
- これが PERSIST 評価での selectivity 崩壊 (97.4%, 67.8%) の根本原因
- 診断 1-3 で「閾値が下限に達すると崩壊」という共通パターンを観察

### ユーザーの直感が示唆した解
「同じ刺激が繰り返し起こると慣れる。時間経過で元に戻る」
→ 神経科学的には**慣化 (habituation)** および**シナプス枯渇 (neurotransmitter depletion)** に対応
→ よく使われるシナプスが自分から疲労し、時間経過で回復する

### この機構が解決する問題
- 過活性ニューロンの**自動抑制** (機構 A の「上限制御」を肩代わり)
- パターン特異性の自然な促進 (使い回されるシナプスが疲労する)
- PERSIST 評価時の回復による安定性 (シナプスが新鮮な状態で評価可能)
- 機構 A の役割が「silent を救う」だけに簡略化される

---

## 2. 実装変更

### 2.1 BinarySynapse 構造体の拡張

```rust
pub struct BinarySynapse {
    // 既存フィールド (維持)
    pub pre: usize,
    pub post: usize,
    pub delay: usize,
    pub exists: bool,
    pub eligibility: f32,
    pub cumulative_sign: f32,
    
    // 追加: 慣化用
    pub fatigue: f32,  // 0.0 (新鮮) - 1.0 (完全疲労)
}
```

### 2.2 スパイク伝送ロジックの変更

スパイク伝送時に確率的失敗 + 疲労蓄積を入れる:

```rust
// 旧コード (擬似コード)
if syn.exists {
    deliver_spike(syn);
}

// 新コード (擬似コード)
if syn.exists {
    // 伝送確率 = (1.0 - fatigue)
    let transmission_prob = 1.0 - syn.fatigue;
    if random() < transmission_prob {
        deliver_spike(syn);
        syn.fatigue = (syn.fatigue + FATIGUE_PER_SPIKE).min(1.0);
    }
    // 伝送失敗時は fatigue は変わらない (発射しようとしたが空打ち)
}
```

定数:
```rust
const FATIGUE_PER_SPIKE: f32 = 0.05;  // 1 スパイクで 5% 疲労
```

### 2.3 試行末の回復処理

`apply_self_organization` 内、または毎ステップ内で:

```rust
// 各試行末で全シナプスの fatigue を回復
for s in &mut self.synapses {
    s.fatigue *= FATIGUE_DECAY;  // 5% 回復
}
```

定数:
```rust
const FATIGUE_DECAY: f32 = 0.95;  // 試行ごとに 5% 回復
```

### 2.4 機構 A の簡略化

機構 A は「下限のみ動作」に変更。上限制御は慣化が担う:

```rust
fn apply_intrinsic_plasticity(&mut self) {
    for n in &mut self.neurons {
        let rate = n.recent_fire_count as f32 / window_size as f32;
        
        // 上限制御は削除 (慣化が代替):
        // 旧: if rate > TARGET_MAX { threshold += 1 }
        
        // 下限制御は維持、ただし完全 silent は許容:
        if rate < TARGET_RATE_MIN && rate > 0.0 {
            n.threshold = (n.threshold - 1).max(threshold_min);
        }
        // rate == 0 のニューロンは閾値を変えない (silent 許容)
    }
}
```

### 2.5 機構 B 間欠化は維持

50 試行に 1 回の apply_structural_plasticity 呼び出しはそのまま。

---

## 3. 実験設定

### 3.1 試行数

- Phase 1: 訓練前評価 (5 パターン × 20 試行)
- Phase 2: 3000 試行の自己組織化 (機構 A + 慣化 + 機構 B 間欠化)
- Phase 3: 訓練後評価 (5 パターン × 20 試行)
- Phase 4: 持続性評価 (追加 200 試行 + 5 パターン × 20 試行)

### 3.2 追加すべき snapshot 項目

500 試行ごとに以下を記録:
- 既存項目: near_zero_ratio, sc100, fp_within, thr_mean
- **追加: fatigue_mean, fatigue_max, fatigue_dist (10 ビンのヒストグラム)**

### 3.3 期待される動作

うまく機能した場合:
- Phase 2 序盤: 全シナプスの fatigue がランダムに動く
- Phase 2 中盤: 特定のパターンに繰り返し使われるシナプスの fatigue が高めに保たれる
- Phase 2 終盤: cumulative_sign が安定し、構造変化が落ち着く
- POST 評価: パターンごとに異なる「fresh シナプス群」が反応
- PERSIST 評価: 200 試行で全 fatigue が回復、新鮮な状態で安定した反応

---

## 4. 判定基準

§5 の事前確定基準を維持:

| POST selectivity | 判定 | 次のアクション |
|---|---|---|
| > 0.55 | 大成功 | 完成判定 (基準 1-4) へ |
| 0.40 - 0.55 | 方向正しい | 機構 C (ホモシナプティック競合) を追加検討 |
| 0.32 - 0.40 | 慣化単独では不十分 | 慣化パラメータ調整 1 回のみ、その後 M2 設計検討 |
| < 0.32 | 改善なし | 慣化採用前に戻し、根本見直し |

### 重要な追加判定: PERSIST 崩壊の解消

POST selectivity の数値とは別に、以下を必ず確認:
- **PERSIST drop < 30%**: PERSIST 評価で大幅崩壊しないこと
- これが満たされない場合、selectivity 数値に関わらず慣化パラメータ調整 (FATIGUE_PER_SPIKE, FATIGUE_DECAY) を 1 回試す

PERSIST 崩壊が解消しなければ慣化機構は機能していない。

---

## 5. やってはいけないこと

1. **慣化パラメータの「あと一押し」調整**: FATIGUE_PER_SPIKE や FATIGUE_DECAY を 0.05, 0.04, 0.06 と微調整して数値を上げる作業は禁止
2. **複数の変更を同時に入れない**: 今回は慣化導入のみ。機構 A の追加修正や機構 C 導入は別フェーズ
3. **新パターンを変えない**: 評価パターン (新 A-E) は固定、比較ができなくなる
4. **試行数をさらに伸ばさない**: 3000 試行が前提。それでダメなら別の問題

---

## 6. 報告タイミング

実行完了後、以下を提示:

### 必須項目
- snapshot 履歴テーブル (500 試行ごとの nz_ratio, sc100, fp_within, thr_mean, **fatigue_mean, fatigue_max**)
- Phase 3 POST: selectivity, within, between
- Phase 4 PERSIST: selectivity, within, between, drop %
- 5×5 類似度マトリクス (POST と PERSIST 両方)
- 各パターンの total spikes (応答強度の対称性)
- 各 output ニューロンの「最大反応パターン」分類
- 全シナプスの fatigue 最終分布

### 判定アクション
- §4 の判定基準と PERSIST drop 条件に従って、次のアクションを明示

---

## 7. 補足: 慣化の時間スケール

参考値として、慣化のパラメータが持つ意味:

- FATIGUE_PER_SPIKE = 0.05 → 20 回連続発火で完全疲労
- FATIGUE_DECAY = 0.95 (試行ごと) → 約 60 試行で 5% まで減衰

パターン提示中 (100ms = 200 ステップ) では各シナプスは数回しか発火しないので、fatigue は 0.1-0.3 程度を行き来する想定。Phase 4 の 200 試行追加で完全回復する。

このタイムスケールが合っていなければ調整するが、まずは上記値で実行。
