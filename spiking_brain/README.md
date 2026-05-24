# spiking_brain — Polychronous Spiking Network Prototype

言語、辞書、教師信号を一切持たない「赤ちゃん脳」プロトタイプ。
入力刺激の時間的構造そのものから、共鳴パターンとして「同一性(identity)」を獲得する。

## 設計思想

LLM とは切り離した、原理レベルの動的 NN:

- **ニューロン数は固定**、学習はシナプス経路と重みの増減のみ
- **軸索遅延 (1〜20ms)** をシナプスごとに持たせる → Polychronization が成立
- **STDP** で因果的に協調するシナプスのみ強化、それ以外は弱化
- 出力層は **発火タイムスタンプを区切りなく記録**、各スパイクは指数減衰する「残像 (Spike Lifetime)」を持つ
- 残像の重ね合わせ = フィンガープリント。同じ入力は同じフィンガープリントに収束する(共鳴)
- 言語的意味は与えない。「りんご」という音の固有性は、出力層に立ち上がる固有値(時空間モザイク)として自己組織する

## アーキテクチャ

```
[入力層 N=20]            [皮質層 N=400 (Exc 320 + Inh 80)]              [出力読出し N=40]
  音声特徴   ─dense─→   ┌─ 興奮性 ─sparse(10%) with delays(1-20ms) ─┐ ──→  出力ニューロン
  (蝸牛模擬)            │                                            │      (フィンガープリント)
                        └─ 抑制性 ─sparse fixed ─→ 興奮性 ─────────┘
                              ↑                  ↑
                              STDP on Exc-Exc only (可塑シナプス)
```

- 入力 → 皮質: 固定強駆動投射 (各入力 → 80 皮質ニューロン)
- 皮質 → 皮質: 興奮性は再帰、遅延ありで STDP の対象
- 抑制性: 局所フィードバック制御 (重み固定、短遅延)
- 出力: 興奮性ニューロンの一部を読み出すだけ(別の層は作らない、皮質の一部として読む)

## 実行

```bash
cd spiking_brain
cargo run --release
```

数秒で終わる。コンソールに以下が出る:

```
== Network ==
  total neurons     : 420
  synapses          : 17600 (plastic: 12800)
  delay range       : 1.0 - 20.0 ms

== Phase 1: pre-training fingerprints ==
== Phase 2: training (alternating A/B, STDP on) ==
== Phase 3: post-training fingerprints ==

== Results ==
  BEFORE: within_A=0.72  within_B=0.70  between=0.54  selectivity=0.18
  AFTER : within_A=0.78  within_B=0.78  between=0.48  selectivity=0.30
```

`selectivity` = 同パターン内類似度 − 異パターン間類似度。
これが学習で大きくなる ≡ 「同じものは同じ、違うものは違う」が
**ラベル教師なし** に立ち上がる。

CSV が `out/` に書き出されるので、好きな手段で可視化できる。

## ファイル構成

```
spiking_brain/
├── Cargo.toml
├── README.md
└── src/
    ├── main.rs       — 実験ランナー
    ├── neuron.rs     — Izhikevich ニューロン
    ├── network.rs    — 遅延付きシナプス、STDP、配送リングバッファ
    └── trace.rs      — 出力トレース(Spike Lifetime)とフィンガープリント
```

## 次の拡張

1. **実音声入力**: 蝸牛フィルタバンク (Gammatone bandpass × 20-40 帯域)
   で WAV → スパイク列にエンコード。`hound` クレートで WAV を読む。
2. **構造的可塑性**: 死んだシナプスを刈り取り、活発な経路の周辺に新規シナプスを生成
   (脳の刈り込み + 形成のシミュレーション)。配送リングバッファの再構築が必要。
3. **DRP マッピング**: ルネサスの動的再構成プロセッサに載せる。
   - 配送リングバッファは DRP の SRAM/Mat 領域に固定マップ
   - ニューロン更新は DRP の演算アレイ(タイル並列)に
   - シナプスデータフロー (pre → post + delay) は **タイル間の dataflow tile**として再構成
   - 構造的可塑性のたびに DRP コンフィグレーションを書き換え → 「ハードウェアそのものが脳と一緒に育つ」
4. **多パターン同時学習**: A/B 二択でなく、数十パターンの同時学習で
   フィンガープリント空間が音素空間のような多様体になるかを観察

## なぜ Izhikevich モデルか

- LIF (Leaky Integrate-and-Fire) より発火パターンが豊か (バースト、リバウンドなど)
- 計算コストが Hodgkin-Huxley の 1/10
- Polychronization の元論文 (Izhikevich 2006) と同じモデル → 結果が比較しやすい

## なぜ Rust か

- ホットループ(数百万シナプス × タイムステップ)を GC なしで高速に回す
- 所有権が明確になるので「どのスレッドがどのリングバッファに書くか」が型で決まる
  → 後で DRP/マルチコア並列化するときに移植しやすい
- 概念としての「Spike Lifetime」 と 言語機能の `&'a` ライフタイムが、
  実装上きれいに分離して書ける(後者でメモリ安全、前者で情報減衰)
