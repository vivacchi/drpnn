//! Spike Lifetime (減衰する残像) を持つ出力トレース。
//!
//! 設計の核心:
//! Rust の言語機能としてのライフタイム `&'a` ではなく、
//! 概念としての「情報のライフタイム」を出力層に持たせる。
//!
//! 各出力ニューロンの発火時刻は時系列に蓄積される。
//! 時刻 t におけるそのニューロンの「残像強度」は、
//! 過去の全発火時刻 t_k に対する指数減衰の重ね合わせ:
//!
//!     trace_i(t) = Σ_k exp(-(t - t_k) / τ)     (t_k <= t)
//!
//! N_out 次元のベクトル `(trace_0, ..., trace_{N-1})` がその瞬間の
//! 「フィンガープリント」(時空間モザイクの瞬時表現)。
//! 同じ音は同じフィンガープリント形状へ収束する(共鳴)。

use std::collections::VecDeque;

pub struct OutputTrace {
    n_outputs: usize,
    /// `spikes[i]` = 出力ニューロン i の発火時刻列 (古い順)
    spikes: Vec<VecDeque<f64>>,
    /// 残像の時定数 (ms)
    pub tau: f64,
    /// `5τ` より古い発火は誤差レベルなので刈り取る閾値
    evict_threshold: f64,
}

impl OutputTrace {
    pub fn new(n_outputs: usize, tau: f64) -> Self {
        Self {
            n_outputs,
            spikes: (0..n_outputs).map(|_| VecDeque::new()).collect(),
            tau,
            evict_threshold: 5.0 * tau,
        }
    }

    pub fn record_spike(&mut self, output_idx: usize, t: f64) {
        self.spikes[output_idx].push_back(t);
    }

    /// `t_now` 時点で `evict_threshold` を超えて古い発火を捨てる(残像が消えた)
    pub fn evict_old(&mut self, t_now: f64) {
        let cutoff = t_now - self.evict_threshold;
        for q in &mut self.spikes {
            while q.front().map_or(false, |&t| t < cutoff) {
                q.pop_front();
            }
        }
    }

    /// 時刻 `t_now` におけるフィンガープリント(残像重ね合わせベクトル)
    pub fn fingerprint(&self, t_now: f64) -> Vec<f64> {
        self.spikes
            .iter()
            .map(|q| q.iter().map(|&ts| (-((t_now - ts) / self.tau)).exp()).sum())
            .collect()
    }

    /// 時間 bin 化フィンガープリント (聴覚的評価向け、time x output の 2D を 1D に flatten)
    ///
    /// 戻り値: `n_outputs × n_bins` の長さの Vec<f64>
    ///   インデックス: `out_idx * n_bins + bin_idx`
    ///   各 bin の値: その bin 内で発火した回数 (整数値、f64 で保持)
    ///
    /// これにより:
    ///   - 「いつ発火したか」(時間順序、bin 精度) が保存される
    ///   - cosine similarity は 1D Vec として計算できる (既存ロジック流用)
    ///   - 聴覚モデルとして妥当な評価が可能 (時間構造を直接比較)
    pub fn time_binned_fingerprint(&self, t_end_ms: f64, bin_width_ms: f64) -> Vec<f64> {
        let n_bins = (t_end_ms / bin_width_ms).ceil() as usize;
        let mut fp = vec![0.0f64; self.n_outputs * n_bins];
        for (i, q) in self.spikes.iter().enumerate() {
            for &ts in q {
                if ts < 0.0 || ts >= t_end_ms { continue; }
                let b = ((ts / bin_width_ms) as usize).min(n_bins - 1);
                fp[i * n_bins + b] += 1.0;
            }
        }
        fp
    }

    /// 蓄積している発火数
    pub fn total_spikes(&self) -> usize {
        self.spikes.iter().map(|q| q.len()).sum()
    }

    pub fn clear(&mut self) {
        for q in &mut self.spikes {
            q.clear();
        }
    }

    pub fn n_outputs(&self) -> usize {
        self.n_outputs
    }

    /// 出力ニューロン i の生の発火時刻列
    pub fn raw_spikes(&self, i: usize) -> impl Iterator<Item = &f64> {
        self.spikes[i].iter()
    }
}

/// コサイン類似度
pub fn cosine_similarity(a: &[f64], b: &[f64]) -> f64 {
    let dot: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let na: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
    let nb: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();
    if na * nb < 1e-9 { 0.0 } else { dot / (na * nb) }
}
