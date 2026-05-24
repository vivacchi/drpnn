//! 物理配置と隣接性
//!
//! ニューロンを 2D グリッドに配置する。DRP の物理 PE 配置に対応。
//! 軸索成長は隣接 PE のみに限定 (原理 1: 局所性 を保証)。
//!
//! 境界効果は許容 (トーラス化しない)。DRP 物理配置は周期境界条件を持たないため。

/// 2D グリッド上のニューロン物理配置と 4 近傍隣接
#[derive(Clone, Debug)]
pub struct Topology {
    pub grid_width: i32,
    pub grid_height: i32,
}

impl Topology {
    pub fn new(grid_width: i32, grid_height: i32) -> Self {
        Self { grid_width, grid_height }
    }

    /// 4 近傍 (上下左右) の有効座標を返す。境界外は除外。
    pub fn neighbors(&self, position: (i32, i32)) -> Vec<(i32, i32)> {
        let (x, y) = position;
        let mut result = Vec::with_capacity(4);
        for (dx, dy) in &[(-1, 0), (1, 0), (0, -1), (0, 1)] {
            let nx = x + dx;
            let ny = y + dy;
            if nx >= 0 && nx < self.grid_width && ny >= 0 && ny < self.grid_height {
                result.push((nx, ny));
            }
        }
        result
    }

    /// 位置が有効か (グリッド内か)
    pub fn is_valid(&self, position: (i32, i32)) -> bool {
        let (x, y) = position;
        x >= 0 && x < self.grid_width && y >= 0 && y < self.grid_height
    }

    /// 線形インデックス (デバッグ・整列用)。同位置に複数ニューロンを置く場合は別管理。
    pub fn linear_idx(&self, position: (i32, i32)) -> Option<usize> {
        if !self.is_valid(position) { return None; }
        Some((position.1 * self.grid_width + position.0) as usize)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn neighbors_interior() {
        let t = Topology::new(20, 20);
        let nbs = t.neighbors((10, 10));
        assert_eq!(nbs.len(), 4);
    }

    #[test]
    fn neighbors_corner() {
        let t = Topology::new(20, 20);
        let nbs = t.neighbors((0, 0));
        assert_eq!(nbs.len(), 2);
        assert!(nbs.contains(&(1, 0)));
        assert!(nbs.contains(&(0, 1)));
    }

    #[test]
    fn neighbors_edge() {
        let t = Topology::new(20, 20);
        let nbs = t.neighbors((0, 5));
        assert_eq!(nbs.len(), 3);
    }
}
