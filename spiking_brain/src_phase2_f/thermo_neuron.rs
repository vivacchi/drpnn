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
    /// エントロピー散逸: entropy_decay_interval step に 1 回 entropy -= entropy_decay_rate
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
            entropy_decay_counter: 0,
            entropy_per_spike: 10,
            is_inhibitory: false,
            generates_entropy: true,
            spontaneous_input: 2, // デフォルト中央値、後で個体差で上書き
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
            entropy_decay_counter: 0,
            entropy_per_spike: 10,
            is_inhibitory: true,
            generates_entropy: true,
            spontaneous_input: 2,
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
            entropy_decay_interval: 1,
            entropy_decay_counter: 0,
            entropy_per_spike: 0,
            is_inhibitory: false,
            generates_entropy: false,
            spontaneous_input: 2, // 検証中: 仮想 M0 等価性 (Step 0 結果の再解釈)
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
                self.local_entropy -= self.entropy_decay_rate;
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
