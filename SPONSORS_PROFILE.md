# GitHub Sponsors プロフィール文章 (起案)

GitHub Sponsors 開設時にコピー&編集して使う想定の文章集です。
日本語/英語の両方を用意。GitHub Sponsors は英語推奨だが、日本語フォロワー向けに併記すると良い。

---

## A. 短い自己紹介 (バイオ欄、150 字程度)

### 英語版

```
Independent researcher building a 1-bit Spiking Neural Network for
Renesas DRP (Dynamically Reconfigurable Processor). Pure integer math,
deterministic physics, no probability — an alternative to LLMs. CC BY-SA 4.0.
```

### 日本語版

```
個人研究者 / Renesas DRP 向けの 1-bit Spiking Neural Network を開発中。
整数演算のみ、確率なしの決定論的物理プロセスで「学習」を再構築する研究。
LLM とは異なる方向性の AI。CC BY-SA 4.0 で公開。
```

---

## B. プロフィール詳細 (Sponsors ページ本文、 1500-2500 字)

### 英語版

```markdown
# DRPNN — Autonomous Spiking Neural Network for Renesas DRP

## What I'm building

I'm developing **drpnn**, an autonomous, growth-capable 1-bit Spiking Neural
Network (SNN) designed to run on Renesas DRP (Dynamically Reconfigurable
Processor) — a class of edge AI hardware that performs dynamic circuit
reconfiguration at runtime.

This is a fundamentally different approach from LLMs and gradient-descent
deep learning:

- **Pure integer arithmetic** — no floating point, anywhere
- **Deterministic physical processes** — no probability, no randomness at runtime
- **Emergent learning** — no loss function, no objective, no apply_learning() call
- **Hardware-software unity** — "learning" *is* DRP reconfiguration, physically

The system is grounded in non-equilibrium thermodynamics (Prigogine 1977):
learning emerges as dissipative structure formation in a network of
thermodynamic neurons that consume enthalpy, generate entropy, and connect
via axons that grow along thermal gradients.

## Current state

- ✅ **M0 cochlea**: 20-band ERB-spaced filter bank, phoneme synthesis (16 kHz)
- ✅ **M1 (A1, primary auditory cortex)**: Fork F-G1-R1, integer deterministic,
  STDP, structural plasticity, axon growth, achieves POST selectivity 0.795
  on time-binned fingerprint evaluation
- ✅ **UP/DOWN states**: dual-attractor implementation (Ikegaya 2005)
- ✅ **Kenet 2003 internal state repertoire**: **quantitatively reproduced**
  (Welch t = 4.169, p < 0.001 against spatially-preserved shuffle null,
  72.4% state transition rate matching biological observation)
- ✅ **Real-time GPU visualization** (macroquad) of spikes, conductance, growth
- 🔄 **PAPER** under active development (CC BY-SA 4.0)
- ⏳ **Renesas RZ/V2H implementation** — upcoming

## Why it matters

The current AI landscape is dominated by gigantic models requiring enormous
energy. drpnn explores a different path:

- **Edge AI that genuinely runs at low power** (target: embedded devices)
- **AI grounded in biological neuroscience** (referencing Izhikevich 2004,
  Bi & Poo 1998, Ikegaya 2005, Kenet 2003)
- **Public good research** — everything is CC BY-SA 4.0, open code, open paper
- **An independent voice** in AI research, not driven by corporate incentives

## What sponsorship enables

This is a personal project. Your support keeps it independent:

- Development time (full-time research costs roughly $30K/year minimum)
- Hardware (Renesas evaluation kits, FPGA boards)
- Conference travel (when applicable)
- Continued open publication

I do not sell products, do not run advertising, do not lock content behind
paywalls. Sponsorship is for the research itself.

## Tiers

- **$1 Curious**: monthly progress email
- **$5 Supporter**: + early access to articles (1 week before public)
- **$25 Researcher**: + monthly Q&A (Discord or email)
- **$100 Patron**: + acknowledgment in the PAPER

## Links

- Code (GitHub): https://github.com/vivacchi/drpnn
- Articles (note): [coming soon]
- Twitter/X: [coming soon]

---

*"We're not trying to scale LLMs. We're trying to understand how brains
actually work, and build something that runs on a thumbnail of silicon."*
```

### 日本語版

```markdown
# DRPNN — Renesas DRP 向け自律成長型 SNN

## 何を作っているか

**drpnn** は、Renesas DRP (動的再構成プロセッサ) 上で動作する、自律成長型の
1-bit Spiking Neural Network (SNN) の研究プロジェクトです。

LLM や勾配降下とは根本的に異なる方向性:

- **整数演算のみ** — 浮動小数点はどこにも使わない
- **決定論的物理プロセス** — ランタイムでは確率も乱数も一切使わない
- **創発的学習** — 損失関数も目的関数も apply_learning() 関数もない
- **ハードウェアとソフトウェアの統一** — 「学習」と「DRP の動的再構成」を物理的に同一視

理論基盤は **Prigogine の非平衡熱力学** (1977 ノーベル化学賞)。学習は、
エンタルピーを消費しエントロピーを生成する熱力学的ニューロンが、熱勾配に
沿って軸索を伸ばしながら、散逸構造として動的平衡に到達するプロセスとして
実装されます。

## 現状

- ✅ **M0 蝸牛**: 20 帯域 ERB スケール フィルタバンク + 音素合成 (16 kHz)
- ✅ **M1 (A1、一次聴覚野)**: Fork F-G1-R1、整数決定論、STDP、構造的可塑性、
  軸索成長を実装。時間 bin 化評価で POST selectivity 0.795 を達成
- ✅ **UP/DOWN 状態**: 多重アトラクター物理実装 (池谷 2005)
- ✅ **Kenet 2003 内部状態レパートリ**: **定量的に再現**
  (Welch t = 4.169, p < 0.001 [空間保存シャッフル帰無分布]、
  状態遷移率 72.4% で生物観察値と整合)
- ✅ **リアルタイム GPU 可視化** (macroquad) — スパイク、conductance、成長を観察
- 🔄 **論文 (PAPER)** 鋭意執筆中 (CC BY-SA 4.0 で公開)
- ⏳ **Renesas RZ/V2H 実装** — 次の段階

## なぜこれが重要か

現在の AI 業界は巨大モデルと膨大なエネルギー消費に支配されています。drpnn は
別の道を探ります:

- **低消費電力で本当にエッジで動く AI** (組み込みデバイスを目標)
- **生物の脳に学ぶ設計** (Izhikevich 2004、Bi & Poo 1998、池谷 2005、
  Kenet 2003 などを参照)
- **公共財としての研究** — コードも論文も CC BY-SA 4.0 で完全公開
- **企業の論理に縛られない独立した研究**

## ご支援で可能になること

これは個人プロジェクトです。皆さまのご支援が独立性を支えます:

- 研究時間の確保 (フルタイム研究には年間最低 30 万円程度)
- ハードウェア (Renesas 評価キット、FPGA ボードなど)
- 学会出張 (該当する場合)
- オープンな発表の継続

私は製品を販売せず、広告を出さず、ペイウォールでコンテンツを囲い込みません。
ご支援はあくまで「研究そのもの」のためです。

## 支援階層

- **$1 Curious (好奇心)**: 月次進捗メール
- **$5 Supporter (支援者)**: + 記事の早期アクセス (公開 1 週間前)
- **$25 Researcher (研究者仲間)**: + 月次 Q&A (Discord またはメール)
- **$100 Patron (後援者)**: + 論文 (PAPER) 謝辞への掲載

## リンク

- コード (GitHub): https://github.com/vivacchi/drpnn
- 記事 (note): [準備中]
- Twitter/X: [準備中]

---

*「LLM をスケールしようとしているのではありません。脳が実際にどう動いているかを
理解し、爪の先ほどのシリコンで動くものを作ろうとしているのです。」*
```

---

## C. README に追加する Sponsors セクション (drpnn リポジトリ用)

```markdown
## 💖 Support

This project is developed as a personal research effort and published under
CC BY-SA 4.0 as a public good. If you find this work valuable, please consider
supporting its continued development:

[![GitHub Sponsors](https://img.shields.io/badge/Sponsor-GitHub-pink?style=for-the-badge&logo=github-sponsors)](https://github.com/sponsors/vivacchi)

Sponsorship enables:
- Independent research time
- Hardware (Renesas evaluation kits)
- Continued open publication

See [SPONSORS_PROFILE.md](SPONSORS_PROFILE.md) for tier details.
```

---

## D. リターン提供の運用ガイド

### Curious ($1/月): 月次進捗メール
- 毎月末に 1 通、月の主な成果 (~500 字 + 図 1 枚)
- 自動配信 (GitHub Sponsors の Welcome message + 月次のメッセージ)

### Supporter ($5/月): 記事早期アクセス
- note 記事の下書きを 1 週間早く Sponsors-only ページで公開
- 公開後にコメント・フィードバック受け付け

### Researcher ($25/月): 月次 Q&A
- 月 1 回、Discord または メールで質問受付
- 技術的な質問、研究方向の相談、参考文献の議論
- 1 回あたり 30-60 分相当の回答

### Patron ($100/月): PAPER 謝辞
- PAPER の Acknowledgments セクションに名前を記載
- 名前は本名・ハンドル・組織名のいずれか希望に応じて
- 6 ヶ月以上継続支援の場合に有効

---

## E. その他検討事項

### 受取設定
- Stripe アカウント (日本円受取可)
- 個人事業主届出は年 20 万円超の場合に検討
- 確定申告: 雑所得 (収入)

### 商用化との関係
- CC BY-SA 4.0 と Sponsors は両立可能
- 商用ライセンス販売も将来検討可 (デュアルライセンス)
- ただし「公共財」スタンスは維持

### 法務上の留意
- Sponsors は寄付・継続支援であり、商品販売ではない
- 「研究への支援」と明示することで税法上の扱いを明確化
- リターン提供 (Q&A など) はサービス提供だが少額なので税務上問題なし

---

## 起案者からのコメント

この文章はあくまで叩き台です。実際に開設する前に以下を調整してください:

1. **目標額の設定**: 月額目標を明示するか (透明性 vs プレッシャー)
2. **画像**: バナー、ロゴ、可視化スクショなど (記事 1 公開時に作る)
3. **トーン**: 「公共財」「独立研究者」「神経科学的妥当性」を強調
4. **更新**: 進捗があるたびに更新 (M2 開始時、DRP 実装時など)
