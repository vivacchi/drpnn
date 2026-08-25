//! M0.5 蝸牛神経核 (Cochlear Nucleus)
//!
//! 設計: M0_5_COCHLEAR_NUCLEUS_DESIGN.md
//!
//! M0 蝸牛 (有毛細胞、 20 帯域スパイク) と M1 (A1) の間に欠落していた信号分解ステージ。
//! 3 細胞型 (Octopus / Bushy / Stellate) で音を時間・周波数・オンセットに並列分解。
//!
//! 3 細胞型実装 (全て ThermoNeuron):
//!   - Octopus (4): 広帯域オンセット検出 (全帯域同時性、 entropy 適応でオンセットのみ)
//!   - Bushy (20):  周波数別 忠実中継 + 立ち上がり強調 (entropy 適応が notch を生む)
//!   - Stellate (20): 周波数別 スペクトル包絡 rate (隣接 ±1 プール、 chopper)
//!
//! 出力: 4 + N_BANDS + N_BANDS ch (N_BANDS=20 → 44、 N_BANDS=40 → 84) を M1 入力へ。

use super::thermo_neuron::ThermoNeuron;
use super::cochlea::{N_BANDS, FIRE_CURRENT};

/// Octopus 細胞数 (オンセット感度の異なる 4 細胞)
pub const N_OCTOPUS: usize = 4;
/// Bushy 細胞数 (各帯域に 1)
pub const N_BUSHY: usize = N_BANDS;
/// Stellate 細胞数 (各帯域に 1)
pub const N_STELLATE: usize = N_BANDS;
/// 蝸牛神経核の総出力チャネル数
pub const N_CN_OUTPUT: usize = N_OCTOPUS + N_BUSHY + N_STELLATE;  // 4 + N_BANDS + N_BANDS
/// 各 Octopus の同時発火閾値 (帯域数)。
///
/// **帯域数に比例させる** (2026-08-25 修正)。旧実装は `[3, 5, 8, 12]` の
/// **絶対本数**をハードコードしていたが、これは N_BANDS=40 のときに
/// 「スペクトルの 7.5% / 12.5% / 20% / 30% が同時に鳴ったら発火」を意味していた。
/// N_BANDS を変えると同じ音が鳴らす帯域数が変わるので、**閾値の意味が動く**
/// (独立監査の指摘)。割合を保つよう N_BANDS から計算する。
///   N_BANDS=40 → [3, 5, 8, 12]（従来と同一）
///   N_BANDS=80 → [6, 10, 16, 24]
pub const OCTOPUS_COINCIDENCE_TH: [i32; N_OCTOPUS] = [
    (N_BANDS as i32 * 3) / 40,
    (N_BANDS as i32 * 5) / 40,
    (N_BANDS as i32 * 8) / 40,
    (N_BANDS as i32 * 12) / 40,
];

/// Octopus 細胞用 ThermoNeuron を作る
/// coincidence_th 帯域が同時発火 (= coincidence_th × FIRE_CURRENT の入力) で発火するよう調整
fn make_octopus(coincidence_th: i32) -> ThermoNeuron {
    let mut n = ThermoNeuron::excitatory((0, 0));
    // 閾値 = 同時発火帯域数 × FIRE_CURRENT (例: 3 帯域 → 180)
    n.threshold_base = coincidence_th * FIRE_CURRENT;
    // 中程度の leak: 数 step の積分 (オンセットは数 ms かけて立ち上がる)
    n.leak = 40;
    // エンタルピー: 速い回復で連続検出を許可
    n.enthalpy_max = 10;
    n.enthalpy_recovery_rate = 10;
    // 強い entropy 適応: 一度発火したら閾値上昇 → 持続音では再発火せずオンセットのみ
    n.entropy_per_spike = 80;
    n.entropy_decay_rate = 1;
    n.entropy_decay_interval = 5;  // 80 を 400 step (200ms) で散逸
    // 自発活動なし (純粋な検出器)
    n.spontaneous_input = 0;
    n.generates_entropy = true;
    n
}

/// Bushy 細胞用 ThermoNeuron
/// 各帯域を忠実に中継 (1 cochlea スパイク=60 で発火)、 ただし entropy 適応で
/// 持続発火を抑制 → 立ち上がり (notch) を強調。 生物の Globular bushy の "primary-like with notch"。
fn make_bushy() -> ThermoNeuron {
    let mut n = ThermoNeuron::excitatory((0, 0));
    // 低閾値: 単一 cochlea スパイク (60) で発火 (忠実中継)
    n.threshold_base = 50;
    // 中 leak: 短時間の積分のみ
    n.leak = 10;
    n.enthalpy_max = 10;
    n.enthalpy_recovery_rate = 10;
    // 中程度の entropy 適応: 持続発火で閾値上昇 → 立ち上がり強調 (notch)
    n.entropy_per_spike = 40;
    n.entropy_decay_rate = 1;
    n.entropy_decay_interval = 5;  // 40 を 200 step (100ms) で散逸 → 100ms 周期で再オンセット検出
    n.spontaneous_input = 0;
    n.generates_entropy = true;
    n
}

/// Stellate 細胞用 ThermoNeuron
/// 隣接 ±1 帯域をプールし、 持続的な rate (chopper) を生成。 スペクトル包絡符号化。
/// 低 leak で積分、 弱 entropy 適応で持続発火を維持。
fn make_stellate() -> ThermoNeuron {
    let mut n = ThermoNeuron::excitatory((0, 0));
    // 中閾値: プール入力 (複数帯域) を積分して発火
    n.threshold_base = 80;
    // 低 leak: 時間方向に積分 (持続 rate を生む)
    n.leak = 5;
    n.enthalpy_max = 10;
    n.enthalpy_recovery_rate = 10;
    // 弱 entropy 適応: 持続発火を許可 (rate code)
    n.entropy_per_spike = 5;
    n.entropy_decay_rate = 1;
    n.entropy_decay_interval = 50;
    n.spontaneous_input = 0;
    n.generates_entropy = true;
    n
}

#[derive(Clone)]
pub struct CochlearNucleus {
    /// Octopus 細胞 (広帯域オンセット検出、 全帯域横断)
    pub octopus: Vec<ThermoNeuron>,
    /// Bushy 細胞 (各帯域 忠実中継 + 立ち上がり強調)
    pub bushy: Vec<ThermoNeuron>,
    /// Stellate 細胞 (各帯域 隣接プール スペクトル包絡 rate)
    pub stellate: Vec<ThermoNeuron>,
    pub current_time: i32,
}

impl CochlearNucleus {
    pub fn new() -> Self {
        let octopus: Vec<ThermoNeuron> = OCTOPUS_COINCIDENCE_TH.iter()
            .map(|&th| make_octopus(th))
            .collect();
        let bushy: Vec<ThermoNeuron> = (0..N_BUSHY).map(|_| make_bushy()).collect();
        let stellate: Vec<ThermoNeuron> = (0..N_STELLATE).map(|_| make_stellate()).collect();
        Self { octopus, bushy, stellate, current_time: 0 }
    }

    /// 1 step 処理 (3 ストリーム)
    /// cochlea_out: 蝸牛 20 帯域出力 (各 ch: 発火で FIRE_CURRENT、 他 0)
    /// 戻り値: 44 ch [octopus 4 | bushy 20 | stellate 20] (発火で FIRE_CURRENT、 他 0)
    pub fn process_step(&mut self, cochlea_out: &[i32]) -> [i32; N_CN_OUTPUT] {
        debug_assert_eq!(cochlea_out.len(), N_BANDS);
        let t = self.current_time;
        let mut out = [0i32; N_CN_OUTPUT];

        // (1) Octopus: 全帯域の総和を入力 (広帯域オンセット)
        let total_input: i32 = cochlea_out.iter().sum();
        for (k, oct) in self.octopus.iter_mut().enumerate() {
            if oct.update(total_input, t) {
                out[k] = FIRE_CURRENT;
            }
        }

        // (2) Bushy: 各帯域を 1:1 中継 (立ち上がり強調)
        for ch in 0..N_BUSHY {
            if self.bushy[ch].update(cochlea_out[ch], t) {
                out[N_OCTOPUS + ch] = FIRE_CURRENT;
            }
        }

        // (3) Stellate: 隣接 ±1 帯域プール (スペクトル包絡 rate)
        for ch in 0..N_STELLATE {
            let lo = ch.saturating_sub(1);
            let hi = (ch + 1).min(N_BANDS - 1);
            let pooled: i32 = (lo..=hi).map(|c| cochlea_out[c]).sum();
            if self.stellate[ch].update(pooled, t) {
                out[N_OCTOPUS + N_BUSHY + ch] = FIRE_CURRENT;
            }
        }

        self.current_time += 1;
        out
    }

    /// 試行間リセット (entropy は持ち越し、 Phase 2 と同じ方針)
    pub fn reset(&mut self) {
        for n in &mut self.octopus { n.reset_state(); }
        for n in &mut self.bushy { n.reset_state(); }
        for n in &mut self.stellate { n.reset_state(); }
    }
}

impl Default for CochlearNucleus {
    fn default() -> Self { Self::new() }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn octopus_fires_on_broadband_onset() {
        let mut cn = CochlearNucleus::new();
        // 12 帯域同時発火 (強いオンセット) → 全 Octopus が発火するはず
        let mut strong = [0i32; N_BANDS];
        for i in 0..12 { strong[i] = FIRE_CURRENT; }
        let out = cn.process_step(&strong);
        // out[0..4] が Octopus。 少なくとも低閾値 (th=3,5) は発火
        assert!(out[0] > 0, "octopus[0] (th=3) should fire on 12-band onset");
        assert!(out[1] > 0, "octopus[1] (th=5) should fire");
    }

    #[test]
    fn octopus_silent_on_weak_input() {
        let mut cn = CochlearNucleus::new();
        // 2 帯域のみ発火 (弱い) → 高閾値 Octopus は沈黙
        let mut weak = [0i32; N_BANDS];
        weak[0] = FIRE_CURRENT;
        weak[1] = FIRE_CURRENT;
        let out = cn.process_step(&weak);
        // th=12 の Octopus[3] は発火しない (2 帯域では足りない)
        assert_eq!(out[3], 0, "octopus[3] (th=12) should be silent on 2-band input");
    }

    #[test]
    fn bushy_relays_single_channel() {
        let mut cn = CochlearNucleus::new();
        // ch5 のみ発火 → Bushy[5] (out[N_OCTOPUS+5]) が中継発火
        let mut single = [0i32; N_BANDS];
        single[5] = FIRE_CURRENT;
        let out = cn.process_step(&single);
        assert!(out[N_OCTOPUS + 5] > 0, "bushy[5] should relay ch5 spike");
        // 他の bushy は沈黙
        assert_eq!(out[N_OCTOPUS + 0], 0, "bushy[0] should be silent");
    }

    #[test]
    fn output_length_matches_config() {
        let mut cn = CochlearNucleus::new();
        let out = cn.process_step(&[0i32; N_BANDS]);
        assert_eq!(out.len(), N_CN_OUTPUT);
        assert_eq!(N_CN_OUTPUT, N_OCTOPUS + 2 * N_BANDS);
    }

    #[test]
    fn octopus_adapts_to_sustained() {
        let mut cn = CochlearNucleus::new();
        // 持続的な広帯域入力 → 最初は発火、 entropy 適応で減少するはず
        let mut strong = [0i32; N_BANDS];
        for i in 0..12 { strong[i] = FIRE_CURRENT; }
        let mut fire_count = 0;
        for _ in 0..200 {
            let out = cn.process_step(&strong);
            if out[0] > 0 { fire_count += 1; }
        }
        // 持続 200 step でも、 適応により発火回数は制限される (全 step 発火しない)
        assert!(fire_count < 200, "octopus should adapt, not fire every step");
        assert!(fire_count > 0, "octopus should fire at least at onset");
    }
}
