//! M1.5 皮質中継 (Cortical Relay) — リレー核 #2
//!
//! 設計: M1_5_CORTICAL_RELAY_DESIGN.md
//!
//! M1 (A1) と M2 (A2) の間の欠落リレー核。probe (m1_output_probe) で判明:
//!   M1 出力は「全音素が ~150ms に同期する疎な単一バースト」で、timing は音素非依存
//!   (cosine 0.954)、identity (どのニューロンが鳴るか) だけが音素を運ぶ。
//!   → M2 の polychronization が使うタイミング差が無く collapse する。
//!
//! M1.5 の責務は蝸牛神経核 (分解) とは双対の「時相変換」:
//!   同期した空間コード (identity) を、音素ごとに異なる時間コード (到着時刻系列) へ展開する。
//!
//! 案 A (最小介入): 各 M1 出力チャネル i に固有の軸索遅延 d_i を与える 1:1 遅延リレー。
//!   「150ms に集合 S が同期発火」→「{150 + d_i : i ∈ S}」の時間シーケンスに展開。
//!   S が音素ごとに違えば到着系列も違う → M2 が分離可能。
//!
//! 物理性: 純粋な配線遅延 (判断機構なし)。DRP では「軸索遅延 = 配線経路長」に直結。
//! 決定論: 遅延は初期化時に LFSR で決定し以後固定。乱数は初期化のみ (原理 3)。
//! 整数演算: 遅延は整数 step。

use super::cochlea::FIRE_CURRENT;

/// 遅延の最小・最大 (step、dt=0.5ms なので 2→1ms, 60→30ms)
pub const RELAY_DELAY_MIN: i32 = 2;
pub const RELAY_DELAY_MAX: i32 = 60;

/// M1.5 皮質中継 (遅延多様 1:1 リレー)
#[derive(Clone)]
pub struct CorticalRelay {
    /// チャネル数 (= M1 出力数)
    n_ch: usize,
    /// 各チャネルの遅延 (step)、初期化時に決定し固定
    delays: Vec<i32>,
    /// リングバッファ長 (= max_delay + 1)
    ring_len: usize,
    /// ring[slot][ch] = その slot で発火予定か
    ring: Vec<Vec<bool>>,
    /// 現在時刻 (step)
    current_time: i32,
}

impl CorticalRelay {
    /// n_ch チャネルの遅延リレーを作る。
    /// 遅延は [RELAY_DELAY_MIN, RELAY_DELAY_MAX] に LFSR で分散 (決定論的、初期化時のみ)。
    /// seed でチャネル→遅延の割り当てを変えられる (M1 トポロジーとの相関を避けるため)。
    pub fn new(n_ch: usize, seed: u16) -> Self {
        let span = (RELAY_DELAY_MAX - RELAY_DELAY_MIN).max(1);
        // LFSR (16bit、taps 0^2^3^5、cochlea/phoneme と同系) で決定論的に遅延を割り当てる
        let mut lfsr: u16 = if seed == 0 { 0xACE1 } else { seed };
        let mut next = || {
            // Galois LFSR 1 step
            let lsb = lfsr & 1;
            lfsr >>= 1;
            if lsb != 0 { lfsr ^= 0xB400; }
            lfsr
        };
        let delays: Vec<i32> = (0..n_ch)
            .map(|_| RELAY_DELAY_MIN + (next() as i32 % (span + 1)))
            .collect();
        let max_delay = *delays.iter().max().unwrap_or(&RELAY_DELAY_MAX);
        let ring_len = (max_delay + 1) as usize;
        Self {
            n_ch,
            delays,
            ring_len,
            ring: vec![vec![false; n_ch]; ring_len],
            current_time: 0,
        }
    }

    /// 1 step 処理。
    /// m1_out: M1 出力の発火電流ベクトル (長さ n_ch、発火チャネルは >0、他 0)。
    /// 戻り値: この step に遅延到着したチャネルの発火電流ベクトル (発火で FIRE_CURRENT)。
    pub fn process_step(&mut self, m1_out: &[i32]) -> Vec<i32> {
        debug_assert_eq!(m1_out.len(), self.n_ch);
        let t = self.current_time;
        let now_slot = (t as usize) % self.ring_len;

        // (1) M1 出力の発火を、遅延先の slot に予約
        for ch in 0..self.n_ch {
            if m1_out[ch] > 0 {
                let arrive = (t + self.delays[ch]) as usize % self.ring_len;
                self.ring[arrive][ch] = true;
            }
        }

        // (2) この step に到着予定のチャネルを出力し、slot をクリア
        let mut out = vec![0i32; self.n_ch];
        for ch in 0..self.n_ch {
            if self.ring[now_slot][ch] {
                out[ch] = FIRE_CURRENT;
                self.ring[now_slot][ch] = false;
            }
        }

        self.current_time += 1;
        out
    }

    /// 試行間リセット (遅延バッファをクリア、遅延割り当ては保持)
    pub fn reset(&mut self) {
        for slot in &mut self.ring {
            for b in slot.iter_mut() { *b = false; }
        }
        // current_time は連続 (Phase 2 の他モジュールと同方針で持ち越し)
    }

    /// 遅延割り当てを参照 (診断用)
    pub fn delays(&self) -> &[i32] { &self.delays }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deterministic_delays() {
        let a = CorticalRelay::new(40, 0x1234);
        let b = CorticalRelay::new(40, 0x1234);
        assert_eq!(a.delays(), b.delays(), "同じ seed は同じ遅延 (決定論)");
    }

    #[test]
    fn delays_within_range() {
        let r = CorticalRelay::new(40, 0);
        for &d in r.delays() {
            assert!(d >= RELAY_DELAY_MIN && d <= RELAY_DELAY_MAX, "遅延が範囲内");
        }
    }

    #[test]
    fn single_spike_arrives_delayed() {
        let mut r = CorticalRelay::new(4, 0);
        let d = r.delays()[1];
        // t=0 で ch1 が発火
        let mut inp = vec![0i32; 4];
        inp[1] = FIRE_CURRENT;
        let out0 = r.process_step(&inp);
        assert_eq!(out0[1], 0, "遅延中はまだ出力されない (d>=2)");
        // d-1 step 進める (何も入れない)
        let empty = vec![0i32; 4];
        for _ in 0..(d - 1) { let o = r.process_step(&empty); assert_eq!(o[1], 0); }
        // d step 目で到着
        let arr = r.process_step(&empty);
        assert_eq!(arr[1], FIRE_CURRENT, "遅延 d 後に ch1 が到着発火");
    }

    #[test]
    fn synchronous_set_spreads_in_time() {
        // 同期発火した集合が、異なる遅延で時間展開されることを確認
        let mut r = CorticalRelay::new(8, 0x55);
        let mut inp = vec![0i32; 8];
        for ch in 0..8 { inp[ch] = FIRE_CURRENT; }  // 全 8ch 同時発火
        let _ = r.process_step(&inp);
        // 以後、各 step で到着したチャネル数を数える
        let empty = vec![0i32; 8];
        let mut arrivals_per_step = Vec::new();
        for _ in 0..(RELAY_DELAY_MAX + 2) {
            let o = r.process_step(&empty);
            let cnt = o.iter().filter(|&&v| v > 0).count();
            arrivals_per_step.push(cnt);
        }
        let total: usize = arrivals_per_step.iter().sum();
        assert_eq!(total, 8, "全 8 チャネルが到着する");
        // 全部が同じ step に到着してはいない (時間展開されている) — 遅延が分散していれば
        let max_in_one_step = *arrivals_per_step.iter().max().unwrap();
        assert!(max_in_one_step < 8, "同期集合が複数 step に展開される (時相変換)");
    }
}
