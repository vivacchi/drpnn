# M0 蝸牛 (A0 Cochlea) 設計ドキュメント

作成日: 2026-05-24
ステータス: 設計検討中 (実装前)

## 目的

DESIGN_PHILOSOPHY.md / CONTEXT.md で示された 12 モジュール構成のうち、**最下層の感覚変換モジュール M0 蝸牛** を設計する。これは「**片耳の音声波形 → 時間構造をもつスパイク列**」への決定論的変換装置であり、M1 (一次聴覚野) への入力を供給する。

M1 単体実験では「ホワイトノイズ刺激下で自律振動に収束 (入力情報を捨てる)」現象が観測された (PAPER §5.9.5-5.9.7)。これは **M1 の入力に十分な時間構造が必要** であることを示し、M0 蝸牛の必要性を浮き彫りにした。

## 階層位置 (聴覚経路)

```
[空気振動]
   ↓
[M0 蝸牛 (本実装)] ──── 単耳、ms 精度の phase locking、20 帯域分解
   ↓ (20 input neuron への電流)
[M0.5 SOC (将来)]  ──── 両耳統合、ITD/ILD 計算 (μs 精度)
                       MSO (内側上オリーブ核): ITD coincidence detection
                       LSO (外側上オリーブ核): ILD detection
   ↓
[M1 A1 (実装済み)]  ─── 時空間パターン → フィンガープリント
   ↓
[M2 A2, M3 カテゴリ ...]
```

**重要**: 両耳間時間差 (ITD ~10μs) は **MSO の責務**であって蝸牛の責務ではない。M0 は単耳のスパイク列生成に専念し、両耳統合は将来 M0.5 として実装。

## 設計原理 (DESIGN_PHILOSOPHY.md と整合)

1. **学習なし固定**: 蝸牛は構造的に固定、可塑性なし (生物の蝸牛も「再生しない」)
2. **物理プロセスのみ**: 共振・包絡線抽出・閾値超過、判断機構なし
3. **整数演算**: 浮動小数禁止、IIR フィルタも整数係数
4. **決定論性**: 同じ音波からは同じスパイク列
5. **局所性**: 各帯域フィルタは独立、隣接帯域とは情報交換しない
6. **1 ビット出力**: 各 input neuron への発火/不発火のみ

---

## 第 1 部: 蝸牛の生理学 (実装が模倣すべき対象)

### 1.1 解剖学的構造

| 部位 | 数 / 寸法 | 役割 |
|---|---|---|
| 基底膜 (Basilar Membrane) | 長さ ~35mm | 場所依存共振、tonotopy の物理基盤 |
| 内有毛細胞 (IHC) | ~3,500 個 | 機械→電気変換、聴神経への信号源 |
| 外有毛細胞 (OHC) | ~12,000 個 | Cochlear amplifier、AGC、能動運動 |
| 聴神経線維 (Type I afferent) | ~30,000 本 | IHC 1 個に対し 10-20 本 (展開) |

**重要観察**: IHC → 聴神経で **約 1 : 10 の展開** が起きている (時間情報の冗長符号化)。これは「聴覚は周波数より時間情報が決定的」ということを示唆。

### 1.2 基底膜の周波数応答

**von Békésy (1960) Nobel** 進行波理論:
- 高周波 → base 近く (蝸牛入口、stapes 寄り) で共振ピーク
- 低周波 → apex 近く (奥) で共振ピーク
- 距離 x [mm] と特性周波数 f [Hz] の関係:

```
f(x) = 165.4 × (10^(2.1 × (1-x/35)) - 0.88)    (Greenwood 1990)
```

x=0 (apex)  → f ≈ 20 Hz
x=35 (base) → f ≈ 20 kHz

つまり **35mm の物理距離が 10 オクターブの周波数範囲をカバー**、これが対数スケールの起源。

### 1.3 ERB スケール (Equivalent Rectangular Bandwidth)

**Glasberg & Moore (1990)** 心理音響学的に妥当な帯域幅:

```
ERB(f) = 24.7 × (4.37 × f / 1000 + 1)    [Hz]
```

例:
- f = 100 Hz → ERB ≈ 35.5 Hz
- f = 500 Hz → ERB ≈ 78 Hz
- f = 1000 Hz → ERB ≈ 132 Hz
- f = 4000 Hz → ERB ≈ 457 Hz

ERB 単位での周波数軸 (Cambridge ERB scale):
```
ERBs(f) = 21.4 × log₁₀(1 + 0.00437 × f)
```

これに沿って **対数間隔** で 20 個のチャンネルを並べると、生物の蝸牛と整合する周波数分解能になる。

### 1.4 内有毛細胞 (IHC) の符号化

機械→電気変換:
1. **AC 成分**: 基底膜振動に同期した受容器電位の振動 (phase locking)
2. **DC 成分**: 振動の RMS に比例した平均電位の上昇 (rate code)

聴神経発火の特性:
- **Phase locking**: < 5 kHz では波形の位相に同期、時間精度 ~10 μs
- **Volley principle**: 5 kHz 超では複数線維の交互発火
- **Rate adaptation**: 持続音で発火頻度低下 (~20 ms 時定数)
- **Spontaneous rate**: 静音時 50-100 Hz (noise floor)
- **動的レンジ**: 30-130 dB SPL (リニア圧縮ではなく対数 + AGC)

### 1.5 外有毛細胞 (OHC) — Cochlear Amplifier

**Hudspeth (2014)** prestin タンパクによる能動運動:
- 微音域 (~40 dB SPL 以下) で 100-1000 倍の利得
- 大音域では飽和 (AGC)
- 結果として **対数圧縮** が起きる (130 dB → 50 dB の神経出力)
- 周波数選択性を 1/3 oct → 1/10 oct まで鋭くする

これがないと「補聴器をつけても言葉が聞き取れない」(sensorineural hearing loss の典型症状)。

### 1.6 鋭い時間分解能の重要性

聴神経の phase locking は **波形位相に同期した ms オーダーの精度** を持つ。これは:
- 周波数情報の補完 (低周波の細かい音高判別)
- **上位の MSO で行われる ITD (両耳間時間差) 計算の素材**
- AM 変調 (エンベロープ) の追従

**注意: 音源定位 (ITD ~10μs 精度) は蝸牛の責務ではない**:
- 蝸牛 = 片耳ごとの「音 → スパイク列」変換
- ITD/ILD は **上オリーブ核 (Superior Olivary Complex, SOC)** で計算される
  - 内側上オリーブ核 (MSO): ITD 検出、coincidence detection
  - 外側上オリーブ核 (LSO): ILD 検出
- これらは将来別モジュール (M0.5 SOC) として実装、または M2 で扱う

M0 蝸牛は **単耳の音 → スパイク列変換** に専念する。phase locking は ms オーダーの時間情報保持として実装するが、μs 精度は不要。

聴覚は「周波数の精度よりも時間の精度が高い」のは事実だが、これは:
- 蝸牛: ms 精度の phase locking
- MSO: μs 精度の ITD (蝸牛由来 phase 情報の coincidence detection で実現)

の階層的役割分担で達成される。「時間 bin 化評価」の正当性は ms オーダーの phase locking と聴神経 rate code の保持で十分。

---

## 第 2 部: 人工内耳に学ぶ実装パターン

### 2.1 主要符号化戦略の比較

| 戦略 | 提唱 | 帯域数 | phase locking | 商用採用 |
|---|---|---|---|---|
| **CIS** | Wilson et al. 1991 | 6-22 | なし | 初期、現在も基本形 |
| **n-of-m / SPEAK** | Patrick et al. 1990s | 16-22 | なし、各時刻で n=8 最大のみ | Cochlear Nucleus |
| **ACE** | Cochlear Ltd. | 22 | なし | Nucleus 主力 (商用 No.1) |
| **FSP** | MED-EL 2006 | 12 | 低周波で残す | MED-EL Opus |
| **MP3000** | AB | 16 | 心理音響モデル化 | Advanced Bionics |

**主要文献**:
- Wilson B.S. et al. (1991) "Better speech recognition with cochlear implants" Nature 352:236
- Wilson B.S. & Dorman M.F. (2008) "Cochlear implants: a remarkable past and a brilliant future" Hear Res 242:3-21
- Zeng F.G. et al. (2008) "Cochlear implants: system design, integration, and evaluation" IEEE Rev Biomed Eng 1:115

### 2.2 CIS (Continuous Interleaved Sampling) 詳細

最も基本的かつ我々の設計に最適:

```
音波 (16 kHz サンプル)
  ↓
[20 帯域 帯域通過フィルタ]   ← IIR Butterworth または Gammatone
  ↓ (20 チャンネル並列)
[包絡線検出] (半波整流 + LPF) ← AC → DC 変換 (IHC の rate code 相当)
  ↓
[圧縮] log/sqrt 圧縮         ← OHC の AGC 相当
  ↓
[閾値超過で発火]              ← 聴神経の rate code
  ↓
20 input neurons へパルス
```

**特徴**:
- 各チャンネル独立処理 (= 原理 1 局所性)
- Phase locking なし (低周波音楽は弱いが音素識別には十分)
- 計算量低、整数演算化容易

### 2.3 我々の M0 に採用する変種

**「CIS + 弱い phase locking + 自発発火」** の組み合わせ:

1. **基本は CIS**: 20 帯域、包絡線検出、閾値発火
2. **低周波域 (< 1 kHz、低 8 チャンネル) で phase locking**: 波形のゼロクロスでも発火
3. **自発発火**: 静音時も各チャンネルが低頻度 (例 1/100 step) で発火 → これが M1 への「noise floor」になり、過剰 sparsification を防ぐ可能性

最後の自発発火は生物的にも実装されている (聴神経の spontaneous rate)。これは M1 ホワイトノイズ実験での「自律振動」を内側から崩す効果もありそう。

---

## 第 3 部: 整数演算での実装仕様

### 3.1 入力フォーマット

- **音声波形**: 16 kHz サンプリングレート、16 ビット PCM (i16 配列)
- **DT_MS との対応**: M1 の DT_MS = 0.5 ms → 1 step = 8 サンプル
- つまり 1 step で 8 サンプルを処理し、出力 input neuron に発火 / 不発火を決める

### 3.2 帯域フィルタ (整数 IIR)

**Gammatone フィルタの整数近似**:

```
中心周波数 fc を ERB スケールで対数間隔に 20 個並べる:
fc[0..19] = ERB_to_freq(linspace(ERB(50), ERB(4000), 20))

例 (10 個まで表示):
  fc[0] =   50 Hz   (apex 側、低周波)
  fc[1] =   83 Hz
  fc[2] =  124 Hz
  fc[3] =  175 Hz
  ...
  fc[10] =  600 Hz
  ...
  fc[19] = 4000 Hz (base 側、高周波)
```

各帯域は **2 次 IIR (biquad)** で実装:
```rust
// 整数係数 (Q1.15 固定小数点)
struct BandPass {
    b0_i: i32, b1_i: i32, b2_i: i32,  // 分子係数 × 32768
    a1_i: i32, a2_i: i32,             // 分母係数 × 32768
    x1: i32, x2: i32,                 // 過去サンプル
    y1: i32, y2: i32,                 // 過去出力
}
fn step(&mut self, x0: i32) -> i32 {
    let y0 = ((self.b0_i * x0 + self.b1_i * self.x1 + self.b2_i * self.x2
             - self.a1_i * self.y1 - self.a2_i * self.y2) >> 15)
             .clamp(i32::MIN, i32::MAX);
    self.x2 = self.x1; self.x1 = x0;
    self.y2 = self.y1; self.y1 = y0;
    y0
}
```

### 3.3 包絡線検出 (envelope)

**半波整流 + 整数 leaky integrator**:
```
env[i] += |bp_out[i]|;       // 半波整流相当 (絶対値)
env[i] -= env[i] >> SHIFT;   // 漏れ (SHIFT=4 で時定数 ~1ms)
```

### 3.4 圧縮 (AGC 相当)

人工内耳でよく使われる **対数圧縮** の整数版:
```
compressed = log2_int(env[i]) << 4    // log2 を 4 ビット精度で取る
```
または簡易な **平方根圧縮** (整数 sqrt):
```
compressed = isqrt(env[i])
```

### 3.5 発火生成

各 step (8 サンプル) ごとに各チャンネルの圧縮値を見て:
```
if compressed[i] > THRESHOLD {
    fire(input_neuron[i]);  // PULSE_WIDTH 8 step 発火
}
```

低周波チャンネル (i < 8) のみ追加で:
```
if 波形のゼロクロス点 && env > MIN_PHASE_LOCK {
    fire(input_neuron[i]);   // phase locking
}
```

### 3.6 自発発火 (spontaneous activity) — M0 蝸牛が担当 (Step 0 検証結果)

**初期検討**: 「同じ脳・同じリズム」原則 (ユーザー指摘) から、M1 input neuron の
`spontaneous_input = 0 → 2` に変更して内部ニューロンと同じリズムにする案を Step 0 で
試行 (2026-05-24)。

**Step 0 結果 (負の結果)**:
| 指標 | spont=0 (旧) | spont=2 (試行) | 変化 |
|---|---|---|---|
| PRE selectivity | 0.472 | 0.410 | -0.062 |
| **POST selectivity** | 0.497 | **0.282** | **-0.215** |
| within | 0.704 | 0.651 | -0.053 |
| **between** | 0.207 | **0.370** | **+0.163** |
| active | 10 | 10 | 同じ |

**原因分析**:
- M1 input neuron に自発発火を入れると静音時も常時 ~67 Hz で発火
- 全パターン提示時に「外部刺激由来 + 自発発火由来」が混在
- パターン特異な時間構造が常時背景活動に埋もれる
- 出力ニューロンが「パターン特異性」ではなく「背景活動への応答」を学んでしまう
- 結果として between (異パターン応答類似度) が大幅上昇、識別性低下

**生物学的にも整合する解釈**:
- 聴神経の spontaneous rate は **聴神経自体** の性質 (蝸牛 IHC 由来ではない)
- M1 input neuron は「皮質に届いた信号」を表すので、ここに自発活動を入れると皮質が
  「環境からの情報か内部ノイズか」を区別不能になる
- 「同じ原則の盲目的適用は害になる」典型例

**確定した設計**:
- M0 蝸牛が「聴神経 spontaneous rate」を含むスパイク列を生成
  - 蝸牛フィルタが音波から計算する電流に「決定論的個体差付きの背景電流」を加算
  - 各帯域の自発発火頻度を 50-100 Hz スケールで設定
- M1 input neuron の `spontaneous_input = 0` のまま維持 (受信専用トランスデューサ)
- 階層責務分離: 信号生成 = M0、信号変換 = M1 input neuron

M0 側での実装:
```rust
// cochlea.rs (Step 3 で実装):
const SPONTANEOUS_BASE_CURRENT: [i32; 20] = [/* 各帯域の自発電流 */];
// 値は決定論的個体差 (idx % N で計算)、聴神経 spontaneous rate を模倣

fn process_step(&mut self, samples: &[i16]) -> [i32; 20] {
    let mut output = [0i32; 20];
    for ch in 0..20 {
        // 帯域フィルタ + 包絡線 + 閾値判定で発火生成
        let signal = self.process_band(ch, samples);
        // 加えて自発発火 (静音時も常時 background)
        output[ch] = signal + SPONTANEOUS_BASE_CURRENT[ch];
    }
    output
}
```

---

## 第 4 部: 音素生成器 (テスト入力)

### 4.1 母音 (整数フォルマント合成)

**Klatt (1980)** 風の cascade/parallel formant synthesizer の簡易版:

```
各母音は 3 つの正弦波 (F1, F2, F3) の加算 + Hann 窓
sample(t) = A1 sin(2π F1 t) + A2 sin(2π F2 t) + A3 sin(2π F3 t)
ただし全て整数 sin テーブル (256 entry) で計算
```

| 母音 | F1 (Hz) | F2 (Hz) | F3 (Hz) | 振幅比 |
|---|---|---|---|---|
| /a/ あ | 800 | 1300 | 2700 | 1.0 : 0.7 : 0.3 |
| /i/ い | 300 | 2300 | 3000 | 1.0 : 0.5 : 0.4 |
| /u/ う | 350 | 850 | 2400 | 1.0 : 0.8 : 0.2 |
| /e/ え | 500 | 2000 | 2700 | 1.0 : 0.6 : 0.3 |
| /o/ お | 500 | 900 | 2400 | 1.0 : 0.8 : 0.3 |

時間長: 100-200 ms (持続母音)。

### 4.2 子音

| 種別 | 例 | 合成方法 |
|---|---|---|
| 破裂音 | /p/, /t/, /k/ | 短い無音 → 10-20ms の broadband burst (LFSR ノイズ) → 後続母音へのフォルマント遷移 |
| 摩擦音 | /s/, /sh/ | 100-150 ms の高周波バンドノイズ (3-8 kHz) |
| 鼻音 | /n/, /m/ | 低周波 (200-500 Hz) フォルマント + 高周波 anti-resonance |

LFSR (線形帰還シフトレジスタ) で決定論的に noise を生成:
```rust
let mut lfsr: u32 = 0xACE1;
fn next_noise(&mut self) -> i32 {
    let bit = ((lfsr >> 0) ^ (lfsr >> 2) ^ (lfsr >> 3) ^ (lfsr >> 5)) & 1;
    self.lfsr = (self.lfsr >> 1) | (bit << 15);
    (self.lfsr as i16) as i32
}
```

### 4.3 音素列 (CVCV 構造)

母音と子音を組み合わせて短い音節列:
- "pa" = /p/ → /a/
- "ki" = /k/ → /i/
- "tu" = /t/ → /u/
- ...

5 音節 (pa, ki, tu, se, mo) を「新 A-E パターン」の置き換えとして使用。

---

## 第 5 部: M0 → M1 接続

### 5.1 階層構造

```
[音声波形 i16 (16 kHz)]
    ↓
[M0 蝸牛]  20 帯域フィルタ + 包絡線 + 圧縮 + 閾値発火 + 自発発火
    ↓
[電流パルス] external_input[20]: i32 (各 step で各 input neuron への電流)
    ↓
[M1 ThermoNetwork.step(&external_input)]
    ↓
[出力ニューロン 40 個の発火パターン]
    ↓
[時間 bin 化 fingerprint で識別性評価]
```

### 5.2 タイミング整合

- 音声: 16 kHz サンプル = 62.5 μs / sample
- M1: DT_MS = 0.5 ms / step = 8 sample / step
- 1 trial = 300 ms = 4,800 samples = 600 step

### 5.3 評価指標

時間 bin 化 fingerprint (PAPER §5.9.3) をそのまま使用:
- 40 出力 × 30 bin (10ms 区切り) = 1,200 次元
- 5 音節 × 20 sample で within / between selectivity 計測

### 5.4 期待される挙動

1. **同じ音節は似た時間 bin パターン** → within 高
2. **違う音節は違う時間 bin パターン** → between 低
3. **音節の時間構造を保つ** → 母音/子音の遷移が見える
4. **発達期 (sparsification 過程) → 安定期** で selectivity が上昇

特にホワイトノイズ実験との対比で「**音素なら自律振動に収束しない**」を実証できれば、§5.9.6 の「時間スケールではなく入力統計が動的平衡点を決める」仮説の重要な実験的支持となる。

---

## 第 6 部: 実装ロードマップ

### 6.1 ファイル構成

```
src_phase2_f/
  cochlea.rs                ← M0 蝸牛本体 (フィルタ + 包絡線 + 発火生成)
  phoneme_synth.rs           ← 音素生成器 (フォルマント合成 + LFSR ノイズ)
  bin/
    m0_m1_pipeline.rs        ← M0 → M1 統合実験ハーネス
```

### 6.2 実装ステップ

1. **Step 0 (準備)**: M1 input neuron の `spontaneous_input = 0 → 2` (内部ニューロンと同じリズム)、100 試行で背景活動が増えること、ホワイトノイズ自律振動が緩和するか確認
2. **Step 1**: 整数 biquad IIR フィルタの実装 + テスト (周波数応答確認、純音 → 単一チャンネル発火)
3. **Step 2**: 包絡線検出 + 圧縮 + 閾値発火 (1 チャンネル動作確認、振幅変動で発火頻度変化)
4. **Step 3**: 20 帯域に拡張 (ホワイトノイズ音波で全帯域発火確認)
5. **Step 4**: 音素生成器 (5 母音 + 5 子音、波形 + スペクトル可視化)
6. **Step 5**: M0 + M1 統合 bin、音素単発で発火応答確認
7. **Step 6**: 10k 訓練 + 時間 bin 化評価 (M1 単体・固定パターン・音素 3 条件で selectivity 比較)

注: 当面は **単耳構造** で開発。両耳・ITD/ILD・MSO 計算は将来モジュール (M0.5) で別途。

### 6.3 検証目標

| 目標 | 達成基準 |
|---|---|
| M0 単体: 周波数選択性 | 純音 1 kHz で対応チャンネルのみ発火 |
| M0 単体: 時間応答 | 1 ms パルス音で 1 step 以内に発火 |
| M0 単体: 音素分離 | /a/ と /i/ で発火チャンネル分布が明確に違う |
| M0+M1 PRE | selectivity > 0.5 (音素はアプリオリに違う) |
| M0+M1 POST 10k | selectivity > 0.6 (学習効果あり) |
| ホワイトノイズ音波対比 | M0 経由のホワイトノイズで M1 が自律振動 **しない** (時間構造が入るから) |

---

## 第 7 部: 設計上の未解決問題

### 7.1 phase locking の必要性

低周波域での phase locking は実装複雑度を増す。なくても CIS は実用化しているが、母音識別精度に影響する可能性。

**判断**: 第 1 ラウンドは phase locking なし (CIS 風) で実装、必要なら後追い。

### 7.2 自発発火の強度 (解決済み)

§3.6 で決定: M0 では生成せず M1 input neuron の `spontaneous_input = 2` に統合。
内部ニューロンと同じリズム、「同じ脳」原則と整合。値が過剰なら Step 1 動作確認時に調整。

### 7.3 AGC の時定数

人工内耳では 100-300 ms の AGC が使われる。 整数 leaky integrator で実装するなら shift 量を調整。

### 7.4 入力 fanout (M1 側)

現状 input_fanout = 80 だが、M0 経由で意味のある時間構造が入ると、もっと小さくても十分かも。これは M1 側の調整候補。

---

## 第 8 部: 参考文献

### 蝸牛生理学

- von Békésy G. (1960) *Experiments in Hearing* McGraw-Hill (Nobel 1961)
- Greenwood D.D. (1990) "A cochlear frequency-position function for several species" J Acoust Soc Am 87:2592
- Glasberg B.R., Moore B.C.J. (1990) "Derivation of auditory filter shapes from notched-noise data" Hear Res 47:103
- Hudspeth A.J. (2014) "Integrating the active process of hair cells with cochlear function" Nat Rev Neurosci 15:600
- Carney L.H. (2018) "Supra-Threshold Hearing and Fluctuation Profiles" J Assoc Res Otolaryngol 19:331

### 蝸牛モデル

- Patterson R.D. et al. (1995) "Time-domain modeling of peripheral auditory processing" J Acoust Soc Am 98:1890
- Slaney M. (1993) "An Efficient Implementation of the Patterson-Holdsworth Auditory Filter Bank" Apple TR #35
- Lyon R. (2017) *Human and Machine Hearing* Cambridge University Press

### 人工内耳

- Wilson B.S. et al. (1991) "Better speech recognition with cochlear implants" Nature 352:236
- Wilson B.S., Dorman M.F. (2008) "Cochlear implants: a remarkable past and a brilliant future" Hear Res 242:3
- Zeng F.G. et al. (2008) "Cochlear implants: system design, integration, and evaluation" IEEE Rev Biomed Eng 1:115
- Patrick J.F., Busby P.A., Gibson P.J. (2006) "The development of the Nucleus Freedom Cochlear implant system" Trends Amplif 10:175

### 音声合成

- Klatt D.H. (1980) "Software for a cascade/parallel formant synthesizer" J Acoust Soc Am 67:971
- Stevens K.N. (2000) *Acoustic Phonetics* MIT Press

---

このドキュメントの実装着手判断は、ユーザー確認後とする。
