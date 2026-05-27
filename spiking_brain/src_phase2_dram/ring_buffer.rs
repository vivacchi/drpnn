//! リングバッファ (STAGE2GAMMA V2 §1 原理 1, §2.3 領域 A)
//!
//! 物理対応:
//!   - 行アドレス = 時刻 t (mod N)
//!   - 1 行 = 全ニューロンの「将来到達する電荷」 のスナップショット
//!   - 個別セル書き込み: ring[(t+1+delay) mod N][post] += charge
//!   - 破壊読み出し: 行を読んだら 0 クリア (現在時刻が前進する物理表現)
//!
//! V2 §3.3 ケース a (自然リーク活用) に従い、 ring 内の電荷はそのまま保持。
//! 自然リークは DramCell::update() が行うので、 ring 自体は静的に蓄積。
//!
//! N (リングサイズ) は最大遅延 + 余裕で決定。 Phase 2 と同様 max_delay=40 なら N=64 で十分。

#[derive(Clone, Debug)]
pub struct RingBuffer {
    /// 行 × ニューロン の電荷マトリックス
    /// data[row_idx][neuron_idx] = その (時刻, ニューロン) ペアで蓄積される予定電荷
    data: Vec<Vec<i32>>,
    /// リングサイズ (= 履歴 step 数 + 最大遅延余裕)
    pub n_rows: usize,
    /// セル数 (= ニューロン数)
    pub n_cells: usize,
    /// 現在の読み出し行 (= 現在時刻 mod n_rows)
    pub head: usize,
}

impl RingBuffer {
    pub fn new(n_rows: usize, n_cells: usize) -> Self {
        Self {
            data: vec![vec![0i32; n_cells]; n_rows],
            n_rows,
            n_cells,
            head: 0,
        }
    }

    /// 現在行のセル i の電荷を読み出して、 即座に 0 クリア (破壊読み出し)
    /// V2 §3.1 ステップ 1 の物理表現
    pub fn read_and_clear_current(&mut self, cell: usize) -> i32 {
        let charge = self.data[self.head][cell];
        self.data[self.head][cell] = 0;
        charge
    }

    /// 現在行全体を読み出して 0 クリア (バッチ版)
    pub fn drain_current_row(&mut self) -> Vec<i32> {
        std::mem::replace(&mut self.data[self.head], vec![0i32; self.n_cells])
    }

    /// 行 (head + offset) mod n_rows のセル cell に電荷を加算
    /// V2 §3.1 ステップ 4 の物理表現 (個別セル書き込み = WL + BL)
    pub fn add_charge_at_offset(&mut self, offset: usize, cell: usize, charge: i32) {
        let target_row = (self.head + offset) % self.n_rows;
        self.data[target_row][cell] = self.data[target_row][cell].saturating_add(charge);
    }

    /// head を 1 つ進める (V2 §3.1 ステップ 6 の時刻進行)
    pub fn advance(&mut self) {
        self.head = (self.head + 1) % self.n_rows;
    }

    /// 全行リセット (試行間で完全初期化したい場合)
    pub fn reset_all(&mut self) {
        for row in &mut self.data {
            for v in row.iter_mut() { *v = 0; }
        }
        self.head = 0;
    }

    /// 統計用: 全 ring 内に蓄積されている電荷総量
    pub fn total_pending_charge(&self) -> i64 {
        self.data.iter()
            .flat_map(|row| row.iter().copied())
            .map(|c| c as i64)
            .sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ring_new_zero_state() {
        let r = RingBuffer::new(8, 4);
        assert_eq!(r.head, 0);
        assert_eq!(r.total_pending_charge(), 0);
    }

    #[test]
    fn write_and_read_current() {
        let mut r = RingBuffer::new(8, 4);
        r.add_charge_at_offset(0, 2, 50);  // 現在行のセル 2 に +50
        let c = r.read_and_clear_current(2);
        assert_eq!(c, 50);
        // 破壊後は 0
        let c2 = r.read_and_clear_current(2);
        assert_eq!(c2, 0);
    }

    #[test]
    fn write_to_future_row() {
        let mut r = RingBuffer::new(8, 4);
        r.add_charge_at_offset(3, 1, 100);  // 3 step 先のセル 1 に +100
        // 現在は 0
        assert_eq!(r.read_and_clear_current(1), 0);
        // 3 step 進める
        for _ in 0..3 { r.advance(); }
        assert_eq!(r.head, 3);
        assert_eq!(r.read_and_clear_current(1), 100);
    }

    #[test]
    fn ring_wraps_around() {
        let mut r = RingBuffer::new(4, 2);
        // 5 step 先 (= ring 1 周 + 1) に書き込み
        r.add_charge_at_offset(5, 0, 30);  // (0+5) % 4 = 1 行目
        for _ in 0..1 { r.advance(); }
        assert_eq!(r.read_and_clear_current(0), 30);
    }

    #[test]
    fn add_multiple_inputs_accumulates() {
        let mut r = RingBuffer::new(8, 4);
        r.add_charge_at_offset(2, 1, 30);
        r.add_charge_at_offset(2, 1, 40);
        r.add_charge_at_offset(2, 1, 50);
        for _ in 0..2 { r.advance(); }
        assert_eq!(r.read_and_clear_current(1), 120);
    }

    #[test]
    fn drain_current_row_returns_and_clears() {
        let mut r = RingBuffer::new(4, 3);
        r.add_charge_at_offset(0, 0, 10);
        r.add_charge_at_offset(0, 1, 20);
        r.add_charge_at_offset(0, 2, 30);
        let drained = r.drain_current_row();
        assert_eq!(drained, vec![10, 20, 30]);
        // 排出後は全部 0
        let empty = r.drain_current_row();
        assert_eq!(empty, vec![0, 0, 0]);
    }
}
