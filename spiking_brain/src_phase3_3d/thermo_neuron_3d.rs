//! 熱力学的ニューロン 3D 版 (Phase 3)
//!
//! src_phase2_f/thermo_neuron.rs から派生。 position を (i32, i32, i32) に拡張。
//! 物理プロセス (update, reset_state) は完全に同一 (位置非依存)。

/// G-1: 発火 1 回あたりのエンタルピー消費量
pub const ENTHALPY_PER_SPIKE: i32 = 3;

#[derive(Clone, Debug)]
pub struct ThermoNeuron3d {
    // ─── 動的状態 ───
    pub membrane: i32,
    pub available_enthalpy: i32,
    pub local_entropy: i32,
    pub last_spike_time: i32,
    pub spike_trace: i32,

    // ─── 固定パラメータ ───
    pub threshold_base: i32,
    pub enthalpy_max: i32,
    pub enthalpy_recovery_rate: i32,
    pub entropy_decay_rate: i32,
    pub entropy_decay_interval: i32,
    pub entropy_decay_counter: i32,
    pub entropy_per_spike: i32,
    pub is_inhibitory: bool,
    pub generates_entropy: bool,

    // ─── Spontaneous activity ───
    pub spontaneous_input: i32,
    pub leak: i32,

    // ─── UP/DOWN 状態 ───
    pub up_state: bool,
    pub up_down_counter: i32,
    pub up_period: i32,
    pub down_period: i32,
    pub up_offset: i32,

    // ─── 物理配置 (3D に拡張) ───
    pub position: (i32, i32, i32),
}

impl ThermoNeuron3d {
    pub fn excitatory(position: (i32, i32, i32)) -> Self {
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
            spontaneous_input: 2,
            leak: 2,
            up_state: false,
            up_down_counter: 0,
            up_period: 100,
            down_period: 100,
            up_offset: 0,
            position,
        }
    }

    pub fn inhibitory(position: (i32, i32, i32)) -> Self {
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
            up_state: false,
            up_down_counter: 0,
            up_period: 100,
            down_period: 100,
            up_offset: 0,
            position,
        }
    }

    pub fn input(position: (i32, i32, i32)) -> Self {
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
            spontaneous_input: 2,
            leak: 1,
            up_state: false,
            up_down_counter: 0,
            up_period: 100,
            down_period: 100,
            up_offset: 0,
            position,
        }
    }

    /// 1 クロック更新 (2D 版と完全に同じ物理プロセス、 位置非依存)
    pub fn update(&mut self, input_current: i32, current_time: i32) -> bool {
        if self.spike_trace > 0 { self.spike_trace -= 1; }

        if self.up_offset > 0 {
            self.up_down_counter += 1;
            let cur_period = if self.up_state { self.up_period } else { self.down_period };
            if self.up_down_counter >= cur_period {
                self.up_state = !self.up_state;
                self.up_down_counter = 0;
            }
        }

        self.membrane = self.membrane.saturating_add(input_current);
        self.membrane = self.membrane.saturating_add(self.spontaneous_input);
        if self.up_state {
            self.membrane = self.membrane.saturating_add(self.up_offset);
        }
        self.membrane = self.membrane.saturating_sub(self.leak);
        if self.membrane < 0 { self.membrane = 0; }

        if self.available_enthalpy < self.enthalpy_max {
            self.available_enthalpy += self.enthalpy_recovery_rate;
            if self.available_enthalpy > self.enthalpy_max {
                self.available_enthalpy = self.enthalpy_max;
            }
        }

        self.entropy_decay_counter += 1;
        if self.entropy_decay_counter >= self.entropy_decay_interval {
            self.entropy_decay_counter = 0;
            if self.local_entropy > 0 {
                self.local_entropy -= self.entropy_decay_rate;
                if self.local_entropy < 0 { self.local_entropy = 0; }
            }
        }

        let effective_threshold = self.threshold_base + self.local_entropy;
        if self.membrane >= effective_threshold && self.available_enthalpy >= ENTHALPY_PER_SPIKE {
            self.available_enthalpy -= ENTHALPY_PER_SPIKE;
            if self.generates_entropy {
                self.local_entropy += self.entropy_per_spike;
            }
            self.membrane = 0;
            self.last_spike_time = current_time;
            self.spike_trace = 160;
            true
        } else {
            false
        }
    }

    pub fn reset_state(&mut self) {
        self.membrane = 0;
        self.last_spike_time = i32::MIN;
        self.spike_trace = 0;
    }
}
