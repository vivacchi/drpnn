# HANDOFF.md — 現状と次のタスク

最終更新: 2026-08-04 (全体アーキ検証: リレー核原理を確立、roster を relay-interleaved 化)

## アーキ検証の結論 (2026-08-04、`ARCHITECTURE_REVIEW_2026-08-04.md`)

各セクションの役割・期待機能を意図/実装/実証で突き合わせた。中心的発見:
**階層 SNN は全境界にリレー核 (時間構造処理段) を要する。**
- M0→M1 は M0.5 蝸牛神経核で解決済 (✅ 最大の成功)
- M1→M2 は M1.5 皮質中継が未挿入 → M2 collapse (⬜ 最優先)
- M2 の「長時定数」は名目のみ (causal_window だけ、delay/decay は M1 と同一)
- ki-tu 分離と M2 collapse は同一原因 (M1.5 欠落) に帰着
- roster を clean stack から relay-interleaved へ改訂 (CONTEXT.md §3 同期済)
- M0.5 ラベル衝突を解消 (M0.5 = 蝸牛神経核、両耳統合 SOC は別スロット)

## M1.5 皮質中継 調査の結論 (2026-08-04、`M1_5_CORTICAL_RELAY_DESIGN.md`、8 実験)

M1.5 を設計・実装・検証した結果、**M1→M2 の壁はリレー核 (入力前処理) では越えられない**
ことが判明。これはアーキ検証の「リレー核が全境界を救う」仮説への重要な**反証**。

判明した事実:
- `m1_output_probe`: M1 出力は **150ms の疎な単発判断バースト** (全音素同時、~20 spikes)。
  timing は音素非依存 (cosine 0.954)、identity だけが音素を運ぶ。
- 入力再符号化 5 機構 (素通し/案A 遅延/案B 同時性/coinc_spread/coinc_sustain) → **全て M2 5/20 collapse**。
  案A は可逆で無効、案B は分化を少し創る (per-pair 0.410→0.385) が M2 collapse は解けず。
- M2 診断 (causal_window 320/160/80 × 抑制 10/18/30%) → **全構成 5/20 collapse**、原因でない。
- **根本原因**: M2 の構造的可塑性 (vitality) が、M1 の単発バースト (時間変化しない疎入力) 下で
  必ず 5 出力へ刈り込む。M0.5 は M1 に 300ms 全体の時間変化リッチ入力を与えたが、M1 は
  M2 へ単発バーストしか出せない。

**⚠️ 次の 4 方向は全て稼働モジュールへの重大変更 → ユーザー合意が必須 (未着手)**:
- (a) M1 が時間展開した出力を出すよう発火動態を見直す (per-pair 最適 decay_slow=30 と衝突可能性)
- (b) M2 の vitality を疎入力でも多経路保持するよう根本見直し
- (c) trial バースト paradigm を連続ストリーム積分へ見直し
- (d) M1.5/M2 を保留し、確立した M0.5+M1 (per-pair 0.765) を論文へ結実

実装済み資産 (次セッションで再利用可): `cortical_relay.rs` (CorticalRelay 遅延 /
CoincidenceRelay 同時性+持続)、`m1_output_probe`/`relay_probe`/`m0_cn_m1_relay_m2_pipeline`
(mode: none/coinc/coinc_spread/coinc_sustain、M2 cw/抑制 上書き引数付き)。

---

## 直近の状態 (2026-05-31、 4 ラン実施)

### 本日の最大成果 (PAPER §5.15)

per-pair between (真の分化指標) で **過去最高 0.765** に到達 (R4: mo fix + decay 30 + 10K):
- **ki-tu = 0.883** ← 両 Plosive の音響本質的類似の壁を 0.96 → 0.88 で **初突破**
- 主要音素間 (pa-* + ki-se) が R1 比 Δ −0.10 ~ −0.25 で大改善
- active 20/40 (capacity 50%)、 全音素均一応答 (18-20 hits)
- 軸索 **刈り取り 0**、 24,273/24,273 全シナプス保存

### selectivity 指標の限界が露呈

| 実行 | 構成 | sel | per-pair |
|---|---|---|---|
| R1 | mo 旧, decay 3, 5K | **0.581** ← sel 最良 | 0.867 |
| R4 | mo fix, decay 30, 10K | 0.508 | **0.765** ← per-pair 最良 |

selectivity = within − between という単純差は **between 改善を罰する** (within も同時に低下するため)。 真の分化能力は per-pair (音素間 cosine 平均) で測るべき。

### 構成詳細 (R4 が最良)
- bin: `m0_cn_m1_pipeline -- 10000 3 30`
- 40 帯域 蝸牛 + 84ch 蝸牛神経核 + M1 grid 20×26 (520 neurons, 22,740 シナプス)
- speed=3 (音素 67ms、 STDP 窓内)、 decay_slow=30 (vitality 300,000 step)

### mo 音素バグの修正 (校正不整合)
- Nasal 振幅 [3000, 1500] (peak 4500) で蝸牛 firing threshold 未満 → 全帯域取りこぼし
- 修正: amps [6000, 3000] (peak 9000、 母音と同等)
- mo が 4 帯域 (ch 7, 8, 9, 14) で発火 (旧は ch 14 のみ)
- 副次効果: **mo と無関係なペアまで per-pair 大改善** (pa-ki 0.913 → 0.739) — 入力 1 音素の校正修正がネットワーク全体を再編

### 表記方針 (ユーザー要望)
- モジュール名・略号 (M1, A1, DRP, STDP 等): **そのまま**
- コード識別子 (ThermoNetwork, ThermoNeuron 等): **そのまま**
- 一般英単語 (cochlea, fingerprint 等): **カタカナ or 漢字** (蝸牛、 フィンガープリント)

### 主要モジュールの状態 (relay-interleaved roster)
- **M0 蝸牛** (src_phase2_f/cochlea.rs): ✅ **40 帯域** (旧 20→40 拡張)、 mo 修正済
- **M0.5 蝸牛神経核** [リレー核 #1] (src_phase2_f/cochlear_nucleus.rs): ✅ **84ch** (4+40+40)
- **M1 A1** (ThermoNetwork): ✅⚠️ `for_m1_cn_40` (84入力) で per-pair 0.765 (R4)、 識別性は ki-tu で頭打ち
- **M1.5 皮質中継** [リレー核 #2]: ⬜ **未実装 (最優先)** — M1 出力に蝸牛神経核相当を適用
- **M2 A2**: ⚠️ for_m2 (40→20) 実装済だが collapse (M1.5 欠落 + 時定数が名目のみ)
- **Phase 2 DRAM** (src_phase2_dram/): 物理モデル + 参照ランプ ADC 検証済 (§9-a 突破の道筋)
- **Phase 3 3D** (src_phase3_3d/): 11 実験完了、 打ち切り (PAPER §5.13)

### 設計知見 (本日累積)
- selectivity 単独指標は不十分、 per-pair between を併記すべき
- vitality 完全保存 (decay 30) で各音素が独自経路を持ち真の分化が進む
- 入力 1 音素の校正修正が網全体を再編する (散逸構造平衡点の入力統計依存性)
- ki-tu (両 Plosive) の本質的類似の壁を 0.96 → 0.88 で初突破、 完全分離は M1.5/M2 課題

## 次のタスク (優先度順、2026-08-04 M1.5 調査後)

0. **【要ユーザー判断】M1→M2 の壁への方針決定** — 上記「M1.5 調査の結論」の 4 方向 (a/b/c/d)
   から選ぶ。a〜c は稼働モジュール (M1/M2) への重大変更で合意必須。d は安全 (論文結実)。
   M1.5 は「入力再符号化では M2 collapse を解けない」ことが 8 実験で確定済み。
1. **(d 候補) M0.5+M1 成果の論文結実** — per-pair 0.765、リレー核原理 (M0.5 実証)、
   M1.5 の負の結果 (階層境界ごとに必要な処置は一様でない) を PAPER に。負の結果も価値。
2. **指標体系の整理** — §5.4 系で sel vs per-pair を再整理 (per-pair を主指標に格上げ)
3. **解像度 60/80 帯域** の収益逓減ライン
4. **DRAM 参照ランプ ADC を DramNetwork に統合** — graded spike 実現
5. **note 記事化** — 「蝸牛 解像度倍化 + mo 修正 + 指標の限界」 (5 本目)

詳細は M1_5_CORTICAL_RELAY_DESIGN.md、ARCHITECTURE_REVIEW_2026-08-04.md、
DAILY_SUMMARY_2026-05-31.md 参照。

---



詳細な設計思想は `DESIGN_PHILOSOPHY.md`、全体像は `CONTEXT.md` を参照。
このドキュメントは**いま何が起きていて、次に何をすべきか**にフォーカス。

---

## 1. プロジェクトの現在地

### 完了した作業

1. **Phase 1 (物理置換型) の実装と部分的成功**
   - 慣化機構の導入で POST selectivity 0.702 達成
   - PERSIST 崩壊が 97.4% → 15.9% に改善
   - ただし実装が判断機構を含む (設計哲学に厳密に従っていない)

2. **設計哲学の完成**
   - 6 つの設計原理を確立
   - 学習の二分類 (タイプ 1 / タイプ 2) を整理
   - 非平衡熱力学系としての SNN という統一描像に到達

3. **論文 (PAPER_DRAFT.md) の初稿作成**
   - EWD スタイル、日本語、思想と方法論の体系化
   - 公開予定 (CC BY-SA 4.0、GitHub + note を想定)

4. **ハードウェア見積もり**
   - RZ/V2H 1〜2 個でフル構成 (M0-M11) 実装可能
   - ノート PC サイズ、15-25 W、ファンレス動作
   - 量産時の複製可能性が原理的に保証される

### 次のフェーズ: Phase 2 (熱力学版) の構築

ユーザーの決定:
- 熱力学的描像を正式採用
- Phase 2 を 0 から構築し、Phase 1 と並行比較
- 概念実装で進める (整数値で熱力学量を表現)
- 軸索成長は隣接 PE の比較で実装

実装の詳細指示: **PHASE2_INSTRUCTION.md** を参照すること。

---

## 2. ファイル構成

```
spiking_brain/
├── DESIGN_PHILOSOPHY.md     ← 哲学的中核 (最重要、§11 に熱力学的描像)
├── CLAUDE.md                ← Claude Code 用クイックリファレンス
├── CONTEXT.md               ← 設計全体像
├── HANDOFF.md               ← 本ファイル
├── PAPER_DRAFT.md           ← 論文 (公開予定)
├── PHASE2_INSTRUCTION.md    ← Phase 2 実装の詳細指示 (新規、最重要)
├── M1_REDESIGN.md           ← 過去の指示書 (参考、Phase 1 用)
├── M1_COMPLETION_INSTRUCTION.md  ← 過去の指示書 (参考)
├── HABITUATION_INSTRUCTION.md    ← 過去の指示書 (参考)
├── Cargo.toml
├── src/                     ← Phase 1 (現状コード、保持)
│   ├── binary_network.rs
│   └── bin/m1_evaluation.rs
├── src_phase2/              ← Phase 2 (新規作成予定)
│   ├── thermo_neuron.rs
│   ├── thermo_synapse.rs
│   ├── thermo_network.rs
│   ├── topology.rs
│   ├── axon_growth.rs
│   └── bin/
│       ├── thermo_m1_evaluation.rs
│       └── compare_phases.rs
└── python/                  ← PyTorch + CUDA 版 (補助、未使用)
```

---

## 3. Claude Code の次の作業

### ステップ 1: 文書理解 (必須)

実装を始める前に、以下を必ず読む:

1. **DESIGN_PHILOSOPHY.md** 全体、特に §11 (非平衡熱力学系としての SNN)
2. **CONTEXT.md** (プロジェクト全体像)
3. **PHASE2_INSTRUCTION.md** (Phase 2 実装の詳細指示)
4. **本ファイル** (HANDOFF.md)

理解できたら、ユーザーに「設計哲学と Phase 2 指示書を理解しました。次の作業に進んでよいですか?」と確認すること。

### ステップ 2: 設計確認

PHASE2_INSTRUCTION.md §10 に従い、以下をユーザーと確認:

1. 本指示書の理解 (核心となる物理プロセスを自分の言葉で説明)
2. ディレクトリ構造 (src_phase2) の合意
3. 初期パラメータの妥当性
4. 軸索成長機構の理解 (これが最も新規性が高い部分)
5. 評価方法と比較計画の合意

ユーザーが「進めて」と明示的に言うまで、コード作成を始めない。

### ステップ 3: 段階的実装

合意後、以下の順で実装:

1. **ThermoNeuron** (熱力学的ニューロン) の実装と単体テスト
2. **ThermoSynapse** (熱力学的シナプス) の実装と単体テスト
3. **Topology** (物理配置と隣接関係) の実装
4. **ThermoNetwork** (ネットワーク統合) の実装
5. **axon_growth_step** (軸索成長) の実装
6. **thermo_m1_evaluation** (評価ランナー) の実装
7. 短時間実行 (100 試行程度) で動作確認
8. 3000 試行のフル実行と結果収集

各段階で結果をユーザーに報告し、確認を取ること。

### ステップ 4: 比較実験

Phase 2 が動作したら、Phase 1 と並行して評価。

- 同じパターン (新 A-E)
- 同じ試行数 (3000)
- 同じ評価指標 (5 性質)

比較項目:
- 収束性 (selectivity 推移)
- PERSIST 崩壊率
- active output 比率
- 構造変化のダイナミクス
- 計算コスト

結果を `compare_phases.rs` で集計し、可視化する。

---

## 4. やってはいけないこと (再確認)

DESIGN_PHILOSOPHY.md と CLAUDE.md にも記載されているが、特に重要な禁止事項を再掲:

1. **target/reward を導入しない**: M1 はタイプ 1 のみ
2. **判断機構を入れない**: target_rate との比較、累積閾値判定、確率的伝送など
3. **損失関数・目的関数を使わない**: 物理プロセスの結果として学習が起きる
4. **「あと一押し」モードでパラメータをいじらない**: 設計を見直す方が先
5. **複数の変更を同時に入れない**: 何が効いたか分からなくなる
6. **Phase 1 のコードを上書きしない**: src_phase2 に新規作成
7. **確率や乱数を使わない**: 初期化時を除く
8. **浮動小数点を使わない**: 整数演算のみ

---

## 5. 環境

- Windows 11 ネイティブ
- CUDA Toolkit 12.6
- NVIDIA Driver 561.17
- GPU: RTX 3060 Laptop GPU 6GB
- Rust: インストール済み、cargo build --release 動作確認済み

Renesas Hardware User's Manual (R01UH1015): まだ未取得 (認証承認待ち)。取得後、drp_cost.rs を更新予定。

---

## 6. 論文の公開予定

ユーザーの判断:
- 実証できた部分まで公開する
- 「微量にも貢献できる可能性があるなら公開」という姿勢
- AI (Claude) の協力を隠さず明示する
- CC BY-SA 4.0 ライセンス、GitHub + note を想定

公開準備の作業は別途進行 (本セッションまたは次セッションで)。Phase 2 の実装と並行して、公開用版の編集を進める。

---

## 7. 判断に迷ったとき

1. **DESIGN_PHILOSOPHY.md を読み返す** (特に 6 原理と §11)
2. **責務範囲を確認** (M1 か他のモジュールか、タイプ 1 かタイプ 2 か)
3. **物理性を確認** (判断機構が入っていないか)
4. **それでも迷えばユーザーに確認**

「どうしましょうか」ではなく、「A か B か、どちらに進みますか」という形で質問する。
