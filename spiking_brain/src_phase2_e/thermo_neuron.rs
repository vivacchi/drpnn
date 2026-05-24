//! 熱力学的ニューロン (ThermoNeuron)
//!
//! 物理プロセスのみで構成 (判断機構なし、確率なし、整数演算のみ)。
//!
//! 各クロックでの自然なふるまい:
//!   1. 不応期処理
//!   2. 入力電流による膜電位上昇 (エンタルピー流入)
//!   3. エンタルピーの自然回復 (上限あり)
//!   4. 局所エントロピーの散逸 (放熱)
//!   5. 発火条件: membrane >= (threshold_base + local_entropy) かつ enthalpy > 0
//!      発火時: enthalpy -= 1, entropy += K (発熱), membrane = 0, refractory セット
//!
//! 慣化は明示機構ではなく、エンタルピー消費と局所エントロピー蓄積の自然な帰結。

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
    /// 不応期残りクロック
    pub refractory_remaining: i32,
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
    /// 不応期長
    pub refractory_period: i32,
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

    // ─── 物理配置 ─────────────────────
    /// 2D グリッド上の位置 (軸索成長で参照)
    pub position: (i32, i32),
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
            refractory_remaining: 0,
            last_spike_time: i32::MIN,
            spike_trace: 0,
            threshold_base: 80,
            enthalpy_max: 10,
            enthalpy_recovery_rate: 1,
            entropy_decay_rate: 1,
            entropy_decay_interval: 50,
            entropy_decay_counter: 0,
            entropy_per_spike: 10,
            refractory_period: 4,
            is_inhibitory: false,
            generates_entropy: true,
            spontaneous_input: 2, // デフォルト中央値、後で個体差で上書き
            leak: 2,
            position,
        }
    }

    /// 抑制性ニューロン (高速制御)
    pub fn inhibitory(position: (i32, i32)) -> Self {
        Self {
            membrane: 0,
            available_enthalpy: 10,
            local_entropy: 0,
            refractory_remaining: 0,
            last_spike_time: i32::MIN,
            spike_trace: 0,
            threshold_base: 40,
            enthalpy_max: 10,
            enthalpy_recovery_rate: 1,
            entropy_decay_rate: 1,
            entropy_decay_interval: 50,
            entropy_decay_counter: 0,
            entropy_per_spike: 10,
            refractory_period: 2,
            is_inhibitory: true,
            generates_entropy: true,
            spontaneous_input: 2,
            leak: 2,
            position,
        }
    }

    /// 入力ニューロン (外部駆動)。
    /// spontaneous_input=0、leak=0 で外部電流のみで動作する純粋トランスデューサ。
    pub fn input(position: (i32, i32)) -> Self {
        Self {
            membrane: 0,
            available_enthalpy: 10,
            local_entropy: 0,
            refractory_remaining: 0,
            last_spike_time: i32::MIN,
            spike_trace: 0,
            threshold_base: 30,
            enthalpy_max: 10,
            enthalpy_recovery_rate: 10,
            entropy_decay_rate: 1,
            entropy_decay_interval: 1,
            entropy_decay_counter: 0,
            entropy_per_spike: 0,
            refractory_period: 2,
            is_inhibitory: false,
            generates_entropy: false,
            spontaneous_input: 0, // 自発入力なし (外部パターンのみ受ける)
            leak: 0,              // リークなし (外部入力を素直に通す)
            position,
        }
    }

    /// 1 クロックの物理プロセス。戻り値: 発火したか
    ///
    /// 判断機構は一切ない。物理プロセスのみで構成。
    pub fn update(&mut self, input_current: i32, current_time: i32) -> bool {
        // (0) B4: spike_trace の自然減衰 (発火痕跡が時間とともに消える)
        if self.spike_trace > 0 { self.spike_trace -= 1; }

        // (1) 不応期処理
        if self.refractory_remaining > 0 {
            self.refractory_remaining -= 1;
            return false;
        }

        // (2) 膜電位の物理プロセス: 入力 + 自発活動 - リーク
        //     - 外部/シナプス入力 (input_current)
        //     - 自発入力 (Na/K ポンプ密度差に対応する個体差ある定常入力)
        //     - リーク (細胞膜漏電による自然減衰)
        self.membrane = self.membrane.saturating_add(input_current);
        self.membrane = self.membrane.saturating_add(self.spontaneous_input);
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

        // (5) 発火条件: 膜電位 >= (基準閾値 + 局所エントロピー) かつ enthalpy > 0
        let effective_threshold = self.threshold_base + self.local_entropy;
        if self.membrane >= effective_threshold && self.available_enthalpy > 0 {
            // 発火 (エネルギー消費 + 熱生成)
            self.available_enthalpy -= 1;
            if self.generates_entropy {
                self.local_entropy += self.entropy_per_spike;
            }
            self.membrane = 0;
            self.refractory_remaining = self.refractory_period;
            self.last_spike_time = current_time;
            // B4: 発火痕跡を CAUSAL_WINDOW にセット (因果窓 step 数、thermo_synapse 定数と一致)
            // CAUSAL_WINDOW=160 だが thermo_neuron からは見えないので定数を直接書く。
            // この値は ThermoSynapse の CAUSAL_WINDOW と必ず一致させる。
            self.spike_trace = 160;
            true
        } else {
            false
        }
    }

    /// 状態リセット (試行間の状態クリア)
    pub fn reset_state(&mut self) {
        self.membrane = 0;
        self.refractory_remaining = 0;
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
        assert_eq!(n.available_enthalpy, 10);  // 1 引かれた後 1 回復 = 10
        assert_eq!(n.local_entropy, 10);
        assert_eq!(n.refractory_remaining, 4);
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
