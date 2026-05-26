//! 3D 物理配置 (擬似 HCP) と近傍判定
//!
//! 擬似 HCP (六方最密充填) を整数 grid 上で表現:
//!   - z 偶数層: 同層 6 近傍 (六角形)
//!   - z 奇数層: 半セル分ずらした六角形 (x, y は同じ整数だが、 近傍計算でオフセット)
//!   - 上下層: 各 3 近傍
//!   - 合計 12 等距離近傍 (生物の細胞最密配置と整合)

#[derive(Clone, Debug)]
pub struct Topology3d {
    pub grid_w: i32,  // x 軸 (W)
    pub grid_h: i32,  // y 軸 (H)
    pub grid_d: i32,  // z 軸 (D、 奥行き)
}

impl Topology3d {
    pub fn new(grid_w: i32, grid_h: i32, grid_d: i32) -> Self {
        Self { grid_w, grid_h, grid_d }
    }

    /// 擬似 HCP 12 近傍を返す。 境界外は除外。
    ///
    /// z 偶数層: 同層は (-1,0)(+1,0)(0,-1)(0,+1)(-1,-1)(+1,-1) の 6 近傍 (六角)
    /// z 奇数層: 同層は (-1,0)(+1,0)(0,-1)(0,+1)(-1,+1)(+1,+1) の 6 近傍 (反転六角)
    /// 上下層: 偶奇に応じた 3 近傍ずつ
    pub fn neighbors(&self, position: (i32, i32, i32)) -> Vec<(i32, i32, i32)> {
        let (x, y, z) = position;
        let z_even = z % 2 == 0;
        let mut result = Vec::with_capacity(12);

        // 同層 6 近傍 (六角形)
        let same_layer: [(i32, i32); 6] = if z_even {
            [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (1, -1)]
        } else {
            [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, 1), (1, 1)]
        };
        for (dx, dy) in &same_layer {
            result.push((x + dx, y + dy, z));
        }

        // 上下層 各 3 近傍
        let updown: [(i32, i32); 3] = if z_even {
            [(0, 0), (-1, 0), (0, -1)]
        } else {
            [(0, 0), (1, 0), (0, 1)]
        };
        for &dz in &[-1, 1] {
            for (dx, dy) in &updown {
                result.push((x + dx, y + dy, z + dz));
            }
        }

        // 境界外除外
        result.into_iter()
            .filter(|&(nx, ny, nz)| {
                nx >= 0 && nx < self.grid_w
                    && ny >= 0 && ny < self.grid_h
                    && nz >= 0 && nz < self.grid_d
            })
            .collect()
    }

    pub fn is_valid(&self, position: (i32, i32, i32)) -> bool {
        let (x, y, z) = position;
        x >= 0 && x < self.grid_w
            && y >= 0 && y < self.grid_h
            && z >= 0 && z < self.grid_d
    }

    pub fn linear_idx(&self, position: (i32, i32, i32)) -> Option<usize> {
        if !self.is_valid(position) { return None; }
        let (x, y, z) = position;
        Some((z * self.grid_w * self.grid_h + y * self.grid_w + x) as usize)
    }

    pub fn total_cells(&self) -> usize {
        (self.grid_w * self.grid_h * self.grid_d) as usize
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn neighbors_interior_even_z() {
        let t = Topology3d::new(5, 6, 22);
        let nbs = t.neighbors((2, 3, 10));  // z=10 偶数
        // 内部点: 12 近傍が全部返る
        assert_eq!(nbs.len(), 12);
        // 同層 6 個
        assert!(nbs.contains(&(1, 3, 10)));
        assert!(nbs.contains(&(3, 3, 10)));
        assert!(nbs.contains(&(2, 2, 10)));
        assert!(nbs.contains(&(2, 4, 10)));
        assert!(nbs.contains(&(1, 2, 10)));
        assert!(nbs.contains(&(3, 2, 10)));
        // 上下層 各 3 個
        assert!(nbs.contains(&(2, 3, 9)));
        assert!(nbs.contains(&(2, 3, 11)));
    }

    #[test]
    fn neighbors_interior_odd_z() {
        let t = Topology3d::new(5, 6, 22);
        let nbs = t.neighbors((2, 3, 11));  // z=11 奇数
        assert_eq!(nbs.len(), 12);
        // 同層 6 個 (反転六角)
        assert!(nbs.contains(&(2, 4, 11)));  // (0, 1) 同層 (偶奇共通)
        assert!(nbs.contains(&(1, 4, 11)));  // (-1, 1) 奇数特有
        assert!(nbs.contains(&(3, 4, 11)));  // (1, 1) 奇数特有
    }

    #[test]
    fn neighbors_corner() {
        let t = Topology3d::new(5, 6, 22);
        let nbs = t.neighbors((0, 0, 0));  // 角
        // 角は 近傍が大幅減 (境界外がほぼ)
        assert!(nbs.len() < 12);
        assert!(nbs.len() > 0);
    }

    #[test]
    fn neighbors_top_layer() {
        let t = Topology3d::new(5, 6, 22);
        let nbs = t.neighbors((2, 3, 21));  // 上面 z=21
        // 上 (z=22) は境界外、 下 (z=20) は OK
        for nb in &nbs {
            assert_ne!(nb.2, 22, "z=22 should be out of bounds");
        }
        // 12 - 3 (上層除外) = 9 近傍
        assert!(nbs.len() <= 9);
    }

    #[test]
    fn linear_idx_unique() {
        let t = Topology3d::new(5, 6, 22);
        let mut seen = std::collections::HashSet::new();
        for z in 0..22 {
            for y in 0..6 {
                for x in 0..5 {
                    let idx = t.linear_idx((x, y, z)).unwrap();
                    assert!(seen.insert(idx), "duplicate idx at ({},{},{})", x, y, z);
                }
            }
        }
        assert_eq!(seen.len(), 5 * 6 * 22);
        assert_eq!(seen.len(), t.total_cells());
    }
}
