//! 熱力学的ニューロン Fork F G-1 (発生学的・原理厳格版、Tier 2 整理済 2026-05-25)
//!
//! G-1 の変更:
//!   - 絶対不応期廃止 (refractory_remaining/_period フィールドも削除)
//!   - 不応期は ENTHALPY_PER_SPIKE 消費 + 自然回復で emergent に発生
//!   - 発火条件: enthalpy >= ENTHALPY_PER_SPIKE (= 3)
//!   - 発火時: enthalpy -= 3
//!   - 結果: enthalpy_max=10 で 4 回バースト発火可能 (10→7→4→1)
//!           その後 2-3 step 不応期 (回復で再発火)
//!
//! UP/DOWN 状態追加 (§5.12.7-A、 sparse 入力時に有効、dense 入力では崩壊リスクあり)
//!
//! 物理プロセス (毎クロック):
//!   1. spike_trace 減衰
//!   2. 膜電位への入力 (シナプス + 自発入力 - リーク)
//!   3. エンタルピー自然回復 (上限まで)
//!   4. エントロピー自然散逸
//!   5. 発火判定: membrane >= (閾値 + entropy) かつ enthalpy >= ENTHALPY_PER_SPIKE
//!      発火時: enthalpy -= 3, entropy += K, membrane = 0, spike_trace = 160
//!
//! 慣化は明示機構ではなく、エンタルピー消費と局所エントロピー蓄積の自然な帰結。

/// G-1: 発火 1 回あたりのエンタルピー消費量
/// 3: 4 回バースト発火 + 2-3 step 不応期 (機能的不応期相当)
pub const ENTHALPY_PER_SPIKE: i32 = 3;

/// 入力ニューロンに**慣化 (local_entropy → 閾値上昇)** を持たせるか。(2026-08-27・F 案)
///
/// ## なぜ
///
/// §14.44 で実測: **入力ニューロン 0.0605 発火/step ≈ 121 Hz /
/// 皮質ニューロン 0.0028 ≈ 5.6 Hz = 比 21.2 倍。**
/// LTD は「pre 発火時に post の痕跡が生きていたら」なので、
/// **pre 側が 21 倍速いと LTD の機会だけが一方的に増える。**
///
/// **M1 には既に恒常性が入っている**: `local_entropy` が発火のたびに溜まり、
/// `effective_threshold = threshold_base + local_entropy` で閾値を上げる
/// (= 内在興奮性による発火率恒常性)。
///
/// **だが入力ニューロンだけ、それが切ってある**:
/// `entropy_per_spike: 0` **かつ** `generates_entropy: false`。**両方切らないと効かない。**
///
/// **つまり 21.2 倍の非対称は、恒常性が足りないのではなく、片側で意図的に切ってある結果である。**
/// 恒常性を新設する前に、**既にあるものを両側で効かせたらどうなるか**を測る。
///
/// 値は皮質の興奮性ニューロンと同じ 10 を使う (**新しい値の発明ではなく、同じ機構の既存値**)。
///
/// **既定 true (= F 案・2026-08-27 ユーザー承認で採用)。**
/// LTD/LTP 比 3.61 → 1.37・伝達可 5.2 倍 (§14.47)。E と組で単語弁別が初めて帰無を超え、
/// 3 シードで再現 (§14.48.7)。`DRPNN_INPUT_HABITUATION=0` で従来に戻せる。
///
/// ## 直していないこと
///
/// **M0.5 (神経核) にも `local_entropy` による周波数別の適応がある。**
/// 入力層にも慣化を入れると**二重になる**。それが良いか悪いかは測ってから。
static INPUT_HABITUATION: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(2);

/// 入力ニューロンの慣化が有効か。初回だけ環境変数を読む。**乱数は使わない。**
pub fn input_habituation_enabled() -> bool {
    use std::sync::atomic::Ordering;
    let v = INPUT_HABITUATION.load(Ordering::Relaxed);
    if v == 2 {
        let on = std::env::var("DRPNN_INPUT_HABITUATION").map(|s| s != "0").unwrap_or(true);
        INPUT_HABITUATION.store(on as u8, Ordering::Relaxed);
        return on;
    }
    v == 1
}

/// 実行時に切り替える (対照実験用)。**false で従来と厳密に同一。**
pub fn set_input_habituation(on: bool) {
    INPUT_HABITUATION.store(on as u8, std::sync::atomic::Ordering::Relaxed);
}


/// 熱力学的ニューロン
#[derive(Clone, Debug)]
pub struct ThermoNeuron {
    // ─── 動的状態 (すべて整数) ─────────────────────
    /// 膜電位 (積分された入力)
    pub membrane: i32,
    /// 利用可能エンタルピー (発火可能エネルギー、生物の神経伝達物質在庫に対応)
    pub available_enthalpy: i32,
    /// 局所エントロピー (蓄積された熱、生物の細胞内代謝産物に対応)
    pub local_entropy: i32,
    /// 最終発火クロック (互換性のため保持、内部状態比較で使用)
    pub last_spike_time: i32,
    /// B4: スパイク痕跡カウンタ。発火時に CAUSAL_WINDOW にセット、毎クロック -1。
    /// `> 0 && < CAUSAL_WINDOW` で「因果窓内に発火していた」と判定 (last_spike_time と等価)。
    /// 利点: 整数有界 (long-run でも i32 オーバーフローしない)、局所性向上、ハードウェア親和。
    pub spike_trace: i32,

    // ─── 固定パラメータ ─────────────────────
    /// 基準閾値 (実効閾値 = threshold_base + local_entropy)
    pub threshold_base: i32,
    /// エンタルピー上限
    pub enthalpy_max: i32,
    /// エンタルピーの回復速度 (per step)
    pub enthalpy_recovery_rate: i32,
    /// エントロピー散逸の**比例項**の shift (2026-08-25 追加)。
    ///
    /// 散逸 = `max(local_entropy >> entropy_decay_shift, entropy_decay_rate)`。
    ///
    /// **なぜ比例項が要るか**: 旧実装は一定レートの線形散逸だったので、
    /// 生成が散逸を上回ると**際限なく溜まる**（上限が無い）。実測で M0.5 は
    /// 同じ音節を 120 回提示すると応答が **15%** まで落ち、10 秒の無音を挟んでも
    /// 27% までしか戻らなかった（適応ではなく**累積失聴**）。
    /// 線形散逸には**時定数が存在しない**ので、平衡点も存在しない。
    ///
    /// **なぜ比例が物理的に正しいか**: DESIGN_PHILOSOPHY §11 は
    /// エントロピー散逸を「熱の自然減衰（放熱）」と定義している。
    /// **放熱は温度差に比例する**（ニュートンの冷却則）。
    /// 一定レートで放熱する物体は存在しない。`EnvelopeDetector` は既にこの形。
    ///
    /// **なぜ `max(..., rate)` が要るか**: 整数演算では
    /// `local_entropy < 2^shift` のとき比例項が 0 になり、**残留が永久に残る**。
    /// 一定レートを下限に置くことで「**使われなければ 0 まで冷める**」が保証される。
    /// 発火し続けるものだけが比例項の支配する**中間の平衡点**に落ち着く。
    ///
    /// **値の導き方（当てはめではない）**: `2^shift ≈ entropy_per_spike / entropy_decay_rate`。
    /// これは旧実装が「1 発火ぶんのエントロピーを散逸しきる時間」として
    /// 設計していた時定数そのもの（`make_octopus` の「80 を 400 step (200ms) で散逸」）。
    pub entropy_decay_shift: i32,
    /// エントロピー散逸: entropy_decay_interval step に 1 回 entropy -= 散逸量
    /// (整数演算で 1/N step の散逸速度を実現)
    pub entropy_decay_rate: i32,
    pub entropy_decay_interval: i32,
    /// 散逸用カウンタ (内部状態)
    pub entropy_decay_counter: i32,
    /// 発火 1 回で生じる局所エントロピー
    pub entropy_per_spike: i32,
    /// 抑制性か
    pub is_inhibitory: bool,
    /// 外部駆動ニューロン (入力層) は entropy 生成しない (パターン入力を壊さないため)
    pub generates_entropy: bool,

    // ─── Spontaneous activity (案 a + リーク) ─────────────
    /// 毎 step の自発入力 (個体差で決定論的に固定)。生物の Na/K ポンプ密度差に相当。
    /// 0 なら自発発火なし、大きいほど自発発火しやすい。
    pub spontaneous_input: i32,
    /// 背景活動ノイズの振幅 (**0 = 無効**・2026-08-25 追加)。
    ///
    /// `spontaneous_input` は**定数**駆動なので各ニューロンは規則的に発火する
    /// = メトロノームであってノイズではない。`idx % 4` はニューロン**間**の
    /// 個体差にすぎず、時間方向の不規則さを与えない。
    ///
    /// ここで加えるのは**時間方向の不規則さ**——大脳皮質の自発活動 (背景ノイズ) に対応し、
    /// JEPA でノイズが果たす役割 (崩壊の防止・対称性の破壊) と同じ位置にある。
    /// 膜電位への駆動なので閾値装置と違い**グラデーションが作れる**
    /// (蝸牛の FireGenerator は閾値+不応期なので 0Hz か 400Hz しか出せないと実測で判明・S9)。
    ///
    /// 原理 3「乱数を使わない (初期化時を除く)」を満たすため乱数ではなく LFSR を使う。
    pub spontaneous_jitter: i32,
    /// 背景ノイズの LFSR 状態 (ニューロン index 由来で決定論的に初期化)
    pub jitter_state: u16,
    /// 背景ノイズを入れる間隔 [step] (1 = 毎 step・既定)。
    ///
    /// 実測 (S11) で「1 step・1 ニューロン・+1 の摂動では指紋がバイト単位で不変」
    /// = M1 は単発の摂動に完全に頑健、と判明した。一方 ±1 を毎 step 全ニューロンに
    /// 入れると再現性が 0.966 → 0.373 に崩壊する。壊しているのは**累積**なので、
    /// 注入頻度を落とせる軸を用意する。`current_time % interval` で判定するので
    /// 追加の状態を持たない (決定論性を保つ)。
    pub jitter_interval: i32,
    /// 毎 step の膜電位漏れ (リーク電流)。生物の細胞膜漏電に相当。
    pub leak: i32,

    // ─── UP/DOWN 状態 (池谷 2005 §5.12.7-A 実装) ─────────────
    /// UP 状態か (true=UP脱分極、false=DOWN静止). PAPER §5.12.7-A.
    /// 自発的に振動、決定論的個体差付き周期 (確率なし)
    /// 池谷 2005「自発的に内部状態を生み出すオートポイエーシス系」の物理実装
    pub up_state: bool,
    /// 内部カウンタ (周期内の位置)
    pub up_down_counter: i32,
    /// UP 状態の長さ (個体差、step 単位)
    pub up_period: i32,
    /// DOWN 状態の長さ (個体差、step 単位)
    pub down_period: i32,
    /// UP 状態時の膜電位オフセット (脱分極相当)。0 なら UP/DOWN 無効化と等価
    pub up_offset: i32,

    // ─── 物理配置 ─────────────────────
    /// 2D グリッド上の位置 (軸索成長で参照)
    pub position: (i32, i32),

    // ─── 階層別パラメータ (M1/M2 で異なる) ─────────────────────
    /// 発火時に spike_trace に設定する値 (= 因果窓の最大長)
    /// M1: 160 step (80ms、 一次聴覚野)
    /// M2: 320 step (160ms、 二次聴覚野、 音節スケール)
    pub spike_trace_init: i32,
}

impl ThermoNeuron {
    /// 興奮性ニューロン (案A + spontaneous activity 案 a)
    /// spontaneous_input と leak のデフォルトは中央値。
    /// 個体差は ThermoNetwork::new() で配置後に index 由来で決定論的に上書きする。
    pub fn excitatory(position: (i32, i32)) -> Self {
        Self {
            membrane: 0,
            available_enthalpy: 10,
            local_entropy: 0,
            last_spike_time: i32::MIN,
            spike_trace: 0,
            threshold_base: 80,
            enthalpy_max: 10,
            enthalpy_recovery_rate: 1,
            entropy_decay_rate: 1,
            entropy_decay_interval: 50,
            // 2^shift ≈ entropy_per_spike / entropy_decay_rate = 10/1 → shift 3 (=8)
            entropy_decay_shift: 3,
            entropy_decay_counter: 0,
            entropy_per_spike: 10,
            is_inhibitory: false,
            generates_entropy: true,
            spontaneous_input: 2, // デフォルト中央値、後で個体差で上書き
            spontaneous_jitter: 0,
            jitter_state: 0xACE1,
            jitter_interval: 1,
            leak: 2,
            // UP/DOWN デフォルト OFF (up_offset=0 で既存動作と互換、ThermoNetwork で設定)
            up_state: false,
            up_down_counter: 0,
            up_period: 100,
            down_period: 100,
            up_offset: 0,
            position,
            spike_trace_init: 160,  // M1 default (M2 では 320 に書き換える)
        }
    }

    /// 抑制性ニューロン (高速制御)
    pub fn inhibitory(position: (i32, i32)) -> Self {
        Self {
            membrane: 0,
            available_enthalpy: 10,
            local_entropy: 0,
            last_spike_time: i32::MIN,
            spike_trace: 0,
            threshold_base: 40,
            enthalpy_max: 10,
            enthalpy_recovery_rate: 1,
            entropy_decay_rate: 1,
            entropy_decay_interval: 50,
            // 2^shift ≈ entropy_per_spike / entropy_decay_rate = 10/1 → shift 3 (=8)
            entropy_decay_shift: 3,
            entropy_decay_counter: 0,
            entropy_per_spike: 10,
            is_inhibitory: true,
            generates_entropy: true,
            spontaneous_input: 2,
            spontaneous_jitter: 0,
            jitter_state: 0xACE1,
            jitter_interval: 1,
            leak: 2,
            // UP/DOWN デフォルト OFF
            up_state: false,
            up_down_counter: 0,
            up_period: 100,
            down_period: 100,
            up_offset: 0,
            position,
            spike_trace_init: 160,
        }
    }

    /// 入力ニューロン (外部駆動、純粋トランスデューサ)。
    ///
    /// 設計判断履歴:
    ///   - 旧 (M1 単体): spontaneous_input=0、leak=0 で外部電流のみ
    ///   - Step 0 試行: spontaneous_input=2、leak=1 (「同じ脳・同じリズム」原則)
    ///       → POST selectivity 0.497 → 0.282 と低下、between 大幅上昇 (0.207→0.370)
    ///       → 自発発火が「外部刺激由来かノイズか」を区別不能に
    ///       → 設計違反として確定、復旧
    ///   - 現在: spontaneous_input=0 (受信専用)、自発発火は M0 蝸牛が担当する設計に
    ///
    /// 階層責務分離:
    ///   - M0 蝸牛: 聴神経 spontaneous rate を含むスパイク列生成
    ///   - M1 input neuron: M0 出力を素直に受け取るトランスデューサ
    pub fn input(position: (i32, i32)) -> Self {
        Self {
            membrane: 0,
            available_enthalpy: 10,
            local_entropy: 0,
            last_spike_time: i32::MIN,
            spike_trace: 0,
            threshold_base: 30,
            enthalpy_max: 10,
            enthalpy_recovery_rate: 10,
            entropy_decay_rate: 1,
            // **F 案 (2026-08-27)**: **3 つ目のスイッチ。**
            // 散逸は max(local_entropy >> 3, entropy_decay_rate) で、
            // interval=1 だと床が 1/step。蓄積は 10 × 発火率 0.06 = 0.6/step なので
            // **床の方が大きく、エントロピーは絶対に溜まらない。**
            // 旧コメント「入力ニューロンは entropy_per_spike=0 なので比例項は効かない」が
            // 示すとおり、interval=1 は**エントロピーが 0 である前提**でセットされていた。
            // 慣化を有効にするなら皮質と同じ 50 にしないと効かない (**既存値。発明ではない**)。
            entropy_decay_interval: if input_habituation_enabled() { 50 } else { 1 },
            entropy_decay_shift: 3,
            entropy_decay_counter: 0,
            // **F 案 (2026-08-27)**: 慣化は `entropy_per_spike` と `generates_entropy` の
            // **両方**が生きていないと効かない。値は皮質の興奮性と同じ 10。
            entropy_per_spike: if input_habituation_enabled() { 10 } else { 0 },
            is_inhibitory: false,
            generates_entropy: input_habituation_enabled(),
            // 2026-08-26: 2 → **0** (受信専用トランスデューサ)。
            //
            // このフィールドは「検証中: 仮想 M0 等価性」のまま結論が出ずに放置され、
            // さらに `ThermoNetwork::new` のガード
            // `if n.spontaneous_input == 0 && n.leak == 0 { continue; }` が
            // この値 (2, leak=1) に対して**常に偽**だったため素通りし、
            // 入力ニューロンにも `idx % 4` が配られていた
            // (= 設計書 §3.6 の決定でも §3.6.1 の復旧案でもない**第三の構成**)。
            //
            // 実測 (累計 6/6 で一律 0 が優勢):
            //   N_BANDS=40: selectivity 0.699/0.747/0.712 → 0.865/0.838/0.847
            //   N_BANDS=80: selectivity 0.400/0.312/0.426 → 0.771/0.773/0.798
            // 現状では無音でも入力層の半数が最大 135 Hz で自走し、
            // 完全無音 1 秒で M1 出力層が 616 spike 出ていた。
            //
            // 0 は :196 のドキュメントコメントとも、設計思想の
            // 「階層責務分離: 信号生成 = M0、信号変換 = M1 input neuron」とも一致する。
            spontaneous_input: 0,
            spontaneous_jitter: 0,
            jitter_state: 0xACE1,
            jitter_interval: 1,
            leak: 1,              // 過剰発火防止
            // UP/DOWN デフォルト OFF
            up_state: false,
            up_down_counter: 0,
            up_period: 100,
            down_period: 100,
            up_offset: 0,
            position,
            spike_trace_init: 160,
        }
    }

    /// 背景活動ノイズを 1 個生成 (-jitter..+jitter・決定論的 LFSR)。
    #[inline]
    pub fn next_jitter(&mut self) -> i32 {
        let bit = ((self.jitter_state >> 0)
            ^ (self.jitter_state >> 2)
            ^ (self.jitter_state >> 3)
            ^ (self.jitter_state >> 5))
            & 1;
        self.jitter_state = (self.jitter_state >> 1) | (bit << 15);
        let span = (2 * self.spontaneous_jitter + 1) as u16;
        ((self.jitter_state % span) as i32) - self.spontaneous_jitter
    }

    /// 1 クロックの物理プロセス。戻り値: 発火したか
    ///
    /// 判断機構は一切ない。物理プロセスのみで構成。
    pub fn update(&mut self, input_current: i32, current_time: i32) -> bool {
        // (0) B4: spike_trace の自然減衰 (発火痕跡が時間とともに消える)
        if self.spike_trace > 0 { self.spike_trace -= 1; }

        // (1) UP/DOWN 状態遷移 (池谷 2005 PAPER §5.12.7-A)
        //     up_offset=0 なら無効化 (既存動作と完全互換)
        //     up_offset>0 なら up_period/down_period で自発振動、UP 時のみ膜電位ブースト
        if self.up_offset > 0 {
            self.up_down_counter += 1;
            let cur_period = if self.up_state { self.up_period } else { self.down_period };
            if self.up_down_counter >= cur_period {
                self.up_state = !self.up_state;
                self.up_down_counter = 0;
            }
        }

        // G-1: 絶対不応期撤廃 (refractory_remaining/_period フィールドも削除済 2026-05-25)
        // 不応期は ENTHALPY_PER_SPIKE 消費 + 自然回復で emergent に発生

        // (2) 膜電位の物理プロセス: 入力 + 自発活動 - リーク (+ UP 状態ブースト)
        //     - 外部/シナプス入力 (input_current)
        //     - 自発入力 (Na/K ポンプ密度差に対応する個体差ある定常入力)
        //     - リーク (細胞膜漏電による自然減衰)
        //     - UP 状態時の膜電位オフセット (脱分極相当)
        self.membrane = self.membrane.saturating_add(input_current);
        self.membrane = self.membrane.saturating_add(self.spontaneous_input);
        // 背景活動ノイズ (時間方向の不規則さ)。0 のとき従来とバイト同一。
        if self.spontaneous_jitter > 0
            && (self.jitter_interval <= 1 || current_time % self.jitter_interval == 0)
        {
            let j = self.next_jitter();
            self.membrane = self.membrane.saturating_add(j);
        }
        if self.up_state {
            self.membrane = self.membrane.saturating_add(self.up_offset);
        }
        self.membrane = self.membrane.saturating_sub(self.leak);
        if self.membrane < 0 { self.membrane = 0; }

        // (3) エンタルピーの自然回復
        if self.available_enthalpy < self.enthalpy_max {
            self.available_enthalpy += self.enthalpy_recovery_rate;
            if self.available_enthalpy > self.enthalpy_max {
                self.available_enthalpy = self.enthalpy_max;
            }
        }

        // (4) エントロピーの自然散逸 (放熱)
        //     interval step に 1 回 -rate (整数演算で 1/N step の散逸速度を実現)
        //     例: rate=1, interval=50 → 50 step に 1 回 -1 (500 step で完全回復)
        self.entropy_decay_counter += 1;
        if self.entropy_decay_counter >= self.entropy_decay_interval {
            self.entropy_decay_counter = 0;
            if self.local_entropy > 0 {
                // 散逸 = 比例項 (放熱 ∝ 温度差) と 一定レート の大きい方。
                // 比例項が高エントロピー側で平衡点を作り、
                // 一定レートが低エントロピー側で「0 まで冷める」を保証する。
                let proportional = self.local_entropy >> self.entropy_decay_shift;
                self.local_entropy -= proportional.max(self.entropy_decay_rate);
                if self.local_entropy < 0 { self.local_entropy = 0; }
            }
        }

        // (5) 発火条件: 膜電位 >= (基準閾値 + 局所エントロピー) かつ enthalpy >= ENTHALPY_PER_SPIKE
        // G-1: enthalpy>0 → enthalpy>=3 に変更 (発火 1 回で 3 消費する分の保有が必要)
        let effective_threshold = self.threshold_base + self.local_entropy;
        if self.membrane >= effective_threshold && self.available_enthalpy >= ENTHALPY_PER_SPIKE {
            // 発火 (エネルギー消費 + 熱生成)
            // G-1: enthalpy -= 3 (絶対不応期相当の効果、4 回バースト発火可能)
            self.available_enthalpy -= ENTHALPY_PER_SPIKE;
            if self.generates_entropy {
                self.local_entropy += self.entropy_per_spike;
            }
            self.membrane = 0;
            // G-1: 絶対不応期なし、enthalpy 消費が不応期を emergent に作る
            self.last_spike_time = current_time;
            // B4: 発火痕跡を CAUSAL_WINDOW にセット (因果窓 step 数、thermo_synapse 定数と一致)
            self.spike_trace = self.spike_trace_init;
            true
        } else {
            false
        }
    }

    /// 状態リセット (試行間の状態クリア)
    pub fn reset_state(&mut self) {
        self.membrane = 0;
        self.last_spike_time = i32::MIN;
        self.spike_trace = 0; // B4: 試行間で痕跡もリセット
        // available_enthalpy と local_entropy は意図的に保持
        // (試行間で持ち越して、慣化と回復の動力学を生かす)
    }
}

#[cfg(test)]
mod tests {
    // ── エントロピー散逸: 比例形 (2026-08-25・ユーザー判断で既定 ON) ──

    /// **使われなければ 0 まで冷める** (ユーザーの定式化・2026-08-25)。
    ///
    /// 比例散逸だけだと整数演算で `entropy < 2^shift` のとき減算が 0 になり、
    /// **残留が永久に残る**。一定レートを下限に置いてあるので 0 に到達する。
    #[test]
    fn entropy_cools_to_zero_when_unused() {
        let mut n = super::ThermoNeuron::excitatory((0, 0));
        n.local_entropy = 10_000;
        // 入力なしで十分長く回す
        for t in 0..200_000 {
            n.update(0, t);
        }
        assert_eq!(n.local_entropy, 0,
            "使われないのにエントロピーが残っている: {}", n.local_entropy);
    }

    /// **発火し続けるものは中間の平衡点に至る** (ユーザーの定式化)。
    ///
    /// 強い入力を与え続けたとき、エントロピーが**発散せず有界**であること。
    /// 旧実装 (一定レートの線形散逸) では際限なく溜まった。
    #[test]
    fn entropy_reaches_bounded_equilibrium_when_driven() {
        let mut n = super::ThermoNeuron::excitatory((0, 0));
        let mut samples = Vec::new();
        for t in 0..60_000 {
            n.update(50, t); // 強い入力を与え続ける
            if t % 10_000 == 9_999 {
                samples.push(n.local_entropy);
            }
        }
        // 後半 3 点が互いに近い = 平衡している (単調増加なら発散)
        let tail = &samples[samples.len() - 3..];
        let mx = *tail.iter().max().unwrap();
        let mn = *tail.iter().min().unwrap();
        assert!(mx > 0, "駆動しているのにエントロピーが 0 = 適応が死んでいる");
        assert!((mx - mn) * 10 <= mx,
            "エントロピーが平衡していない (発散の疑い): {:?}", samples);
    }

    /// 散逸は比例項と一定レートの**大きい方**であること (形の固定)。
    #[test]
    fn entropy_dissipation_is_max_of_proportional_and_rate() {
        let mut n = super::ThermoNeuron::excitatory((0, 0));
        n.entropy_decay_interval = 1; // 毎 step 散逸させて観測しやすくする
        // 高エントロピー: 比例項が支配
        n.local_entropy = 8192;
        let before = n.local_entropy;
        n.update(0, 0);
        let drop_high = before - n.local_entropy;
        assert_eq!(drop_high, 8192 >> n.entropy_decay_shift,
            "高エントロピー側で比例項が支配していない");
        // 低エントロピー: 一定レートが支配
        n.local_entropy = 1;
        n.update(0, 1);
        assert_eq!(n.local_entropy, 0, "低エントロピー側で 0 に到達しない");
    }

    // ── 背景活動ノイズ (2026-08-25) ──

    /// 既定は無効で、有効時と挙動が違うこと (バイト同一性 + 経路が生きていること)。
    #[test]
    fn spontaneous_jitter_defaults_off() {
        let n = super::ThermoNeuron::excitatory((0, 0));
        assert_eq!(n.spontaneous_jitter, 0, "背景ノイズの既定は 0");

        let mut off = super::ThermoNeuron::excitatory((0, 0));
        let mut on = super::ThermoNeuron::excitatory((0, 0));
        on.spontaneous_jitter = 3;
        let (mut a, mut b) = (Vec::new(), Vec::new());
        for t in 0..500 {
            a.push(off.update(1, t));
            b.push(on.update(1, t));
        }
        assert_ne!(a, b, "背景ノイズを入れても発火列が変わらない = 経路が死んでいる");
    }

    /// 決定論的であること (原理 3: 乱数を使わない)。
    #[test]
    fn spontaneous_jitter_is_deterministic() {
        let run = || {
            let mut n = super::ThermoNeuron::excitatory((0, 0));
            n.spontaneous_jitter = 3;
            (0..1000).map(|t| n.update(1, t)).collect::<Vec<bool>>()
        };
        assert_eq!(run(), run(), "同じ条件で違う発火列が出た");
    }

    /// ノイズ値が宣言した範囲 -A..+A に収まること。
    #[test]
    fn spontaneous_jitter_stays_in_range() {
        for amp in [1i32, 2, 3, 4, 8] {
            let mut n = super::ThermoNeuron::excitatory((0, 0));
            n.spontaneous_jitter = amp;
            let (mut lo, mut hi) = (i32::MAX, i32::MIN);
            for _ in 0..5000 {
                let j = n.next_jitter();
                lo = lo.min(j);
                hi = hi.max(j);
            }
            assert!(lo >= -amp && hi <= amp, "振幅 {} で範囲外 [{}, {}]", amp, lo, hi);
            // 両側に振れること (片側に偏っていたら「ノイズ」でなくバイアス)
            assert!(lo < 0 && hi > 0, "振幅 {} で片側にしか振れない [{}, {}]", amp, lo, hi);
        }
    }

    use super::*;

    #[test]
    fn excitatory_fires_with_strong_input() {
        let mut n = ThermoNeuron::excitatory((0, 0));
        let fired = n.update(100, 0);
        assert!(fired);
        assert_eq!(n.membrane, 0);
        // G-1: enthalpy 初期 10 で enthalpy_max=10 のため (3) の回復をスキップ、
        //      (5) で -3 → 7。 回復 +1 は次 step で起こる
        assert_eq!(n.available_enthalpy, 7, "G-1: 初期 enthalpy=enthalpy_max なので回復スキップ、発火で -3");
        assert_eq!(n.local_entropy, 10);
        // G-1: refractory_remaining/_period フィールド削除済 (2026-05-25 Tier 2 修正)
        // 不応期は enthalpy 消費による emergent 機構
        // 発火痕跡は CAUSAL_WINDOW=160 にセット
        assert_eq!(n.spike_trace, 160, "B4: spike_trace に痕跡 160");
    }

    #[test]
    fn fatigue_emerges_from_entropy_accumulation() {
        let mut n = ThermoNeuron::excitatory((0, 0));
        // 連続強入力で慣化発現を確認 (entropy 蓄積 → 実効閾値上昇)
        // 案A: 1/50step の散逸なので、連続発火で entropy が大きく蓄積するはず
        let mut fire_count = 0;
        for t in 0..200 {
            if n.update(100, t) { fire_count += 1; }
        }
        assert!(fire_count > 0);
        assert!(n.local_entropy > 0);  // 慣化が物理的に発現
        // 1/50step の散逸では、200step 中 4 回しか -1 されない
        // entropy = (発火数 × 10) - 4 程度の蓄積になるはず
        assert!(n.local_entropy >= 10, "entropy should accumulate: got {}", n.local_entropy);
    }

    #[test]
    fn input_neuron_does_not_accumulate_entropy() {
        let mut n = ThermoNeuron::input((0, 0));
        n.update(100, 0);
        assert_eq!(n.local_entropy, 0);  // entropy 生成なし
    }
}
