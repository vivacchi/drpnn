//! 犯人は M0 か M1 か — 情報は失われているのか、読めていないだけなのか (2026-08-26)
//!
//! ## 問い
//!
//! §14.6 で、声の高さを変えると かなの同定が 7.1%、音量を変えると 0.7% に落ちた。
//! だがこれは **素のコサイン 1-NN という一つの弱い読み出し** での値でしかない。
//!
//! - 情報が **失われている** なら、M1 をどう作っても直らない。
//! - 情報は **在るが読めていない** だけなら、M1 の設計問題。
//!
//! ## 実測前監査で潰した欠陥 (2026-08-26・初版を一度も実行せずに 8 レンズで監査)
//!
//! 6 件。**うち 2 件は判定を正反対の方向に狂わせるもの**だった。
//!
//! 1. **判定軸が固定 seed だった**。wave_of は毎回 LfsrNoise を巻き戻し、子音は f0 に
//!    依存しないので、同じかなの 4 条件も、同じ行のかな同士も、先頭 30ms がバイト同一。
//!    線形プローブはこの完全不変な部分空間を教師ありで狙い撃ちできるので、
//!    **音響でなく同一雑音標本の再照合で解けてしまう** → 「M0 は無実」に誤判定。
//!    kana_identify で一度潰した欠陥を持ち込み直していた。
//!    → **主軸を utterance_seed (全条件で異なる seed) にし、判定はそこだけで行う。**
//! 2. **リッジが強すぎて帰無が chance でなく 0% だった**。λ = mean(diag(XtX)) は
//!    バイアス列の対角 (= n-1 = 183) を桁で潰し、切片が死ぬ。その極限で
//!    score_c ∝ 訓練内クラス件数 になり、LOO では真クラスだけ 1 件少ないので
//!    **構造的に真クラス以外が選ばれる**。しかもシャッフル対照も同じ理由で同時に 0 に
//!    落ちるので **この故障を検出できない** → 「M0 が犯人」に誤判定。
//!    → **fold 内で X と Y を中心化し、バイアス列を廃止**。λ はグリッドの最大で判定。
//!    → **無情報ベースライン (乱数特徴) を必ず印字**し、chance 付近に来ることを確認する。
//! 3. **線形プローブのチャンスに 1-NN の値を流用していた**。正しくは 1/46 = 2.174%。
//! 4. **健全性の印字が丸ごと落ちていた** (無音・重複ベクトル・同点棄却・特異 fold)。
//!    solve が None のとき continue するので、**黙って不正解に計上**されていた。
//! 5. **6 セル印字されるのにどれで判定するかが未宣言だった**。実測後に都合のよい
//!    セルを選べる。→ 統合規則を下に固定する。
//! 6. **時間を畳んだ計器で無限定に「M0 が犯人」と断定していた**。240 step を 1 本に
//!    畳むと、子音 (step 0-59) が母音 (60-239) に 3 倍の長さで薄められる。
//!    落ちているのは子音 (§14.6.5) なので、**計器の構造だけで null が出る確率が高い**。
//!    → 全ての判定文を「この読み出しでは」に限定し、**2 窓の時間分解**を足す。
//!       M0 の有罪は **時間分解した特徴でも null のときにだけ** 言う。
//!
//! ## 測り方 — 読み出しの梯子
//!
//! 1. **素のコサイン 1-NN** (同点棄却) — 現行。§14.6 の再現。
//! 2. **中心化 / チャネル標準化 / 白色化** — 教師なし。**ただし transductive**
//!    (held を含む全標本から平均・分散・共分散を推定している)。判定には使わない。
//! 3. **線形プローブ** (リッジ回帰・fold 内中心化・leave-one-out) — 教師あり。上界。
//! 4. **置換検定** — ラベルを条件ごとに独立に 99 回置換し、p 値を出す。
//!
//! ## これは計測器であって系ではない
//!
//! 3 は教師あり、全体は浮動小数点。**M1 に入れるものではない。**
//! M1 はタイプ 1 のみ (target 指定・正解判定は責務外) で整数演算。
//! ここでやっているのは「情報が在るか」を測ることだけ。
//!
//! ## 判定規則 (実測前に固定・以後動かさない)
//!
//! **判定軸は 主軸 (話者の言い直し) のみ。** F0 単独軸・レベル軸・雑音軸は
//! 診断として印字するが **判定に入れない**。
//!
//! **判定は 主軸 × {M0, M0.5} × {時間平均, 2窓} の 4 セル。** 統合規則:
//!
//! - M0.5 のいずれかのセルで p < 0.01 → **情報は M1 に届いている**
//! - M0.5 が全滅 (全て p > 0.2) だが M0 のいずれかで p < 0.01 → **M0.5 が落としている**
//! - M0 も M0.5 も全滅 → **この読み出しでは情報が見つからない**
//!   (時刻に残る可能性・非線形に符号化されている可能性は否定できない)
//! - それ以外 → **中間**。どちらにも倒さず、追加測定を 1 つ名指しして終える。
//!
//! λ はグリッド {0.01, 1, 100} の **最大** で判定し、置換側も同じグリッドの最大を取る
//! (存在の上界を測る問いなので固定グリッド上の max は正当。対照が同じ max を取るので
//! 勝つまで調整にはならない)。**このグリッドは実測前に固定する。**
//!
//! ## 予測 (結果を見る前に固定)
//!
//! - **予測 1**: 線形プローブは置換対照を有意に上回る (p < 0.01)。
//! - **予測 2**: 教師なしの正規化だけでも素のコサインより上がる。
//! - **予測 3**: レベル軸は F0 軸より回復しにくい (どちらも診断・判定外)。
//! - **数値**: 主軸 × M0.5 × 時間平均 で **30〜60%**。
//!   根拠 = 母音が素のコサインでも 81.5% 生きている。子音が線形で上がるとして。
//!   **この推論は §14.6.4 で 5 倍外したのと同じ形**である。自覚した上で置く。
//!   (初版の「F0 軸で 60-85%」は固定 seed 軸に対する予測だった。
//!    その軸は判定から外したので、予測も主軸に置き直した。)
//!
//! ## 第 1 回実測で見つかった自分のバグ (2026-08-26)
//!
//! - **合格側の棄却域が空だった。** N_PERM=99 だと置換 p 値の床は 1/(99+1) = **ちょうど 0.01**。
//!   ゲートが `p < 0.01` だったので、**「情報は M1 に届いている」の枝は原理的に発火しない**。
//!   0 回中 0 回が実測値に届かなくても p = 0.010 で「中間」に落ちる。
//!   §14.6 で潰したはずの「棄却域が空」を、**今度は合格側でやった**。
//!   → N_PERM = 999 (床 0.001) に上げる。**これはコード自身が事前に
//!      「追加測定: 置換回数を 999 に上げて p を締めること」と宣言していた手当てである。**
//! - **レベル軸を時間平均でしか測っていなかった。** 主軸で 2 窓にすると
//!   どの読み出しも約 2 倍になったので、時間平均だけでレベル軸を「情報が無い」と
//!   結論するのは早い。→ 2 窓を全軸で測る (診断軸なので判定規則は変えない)。
//!
//! ## 第 1 回実測で分かった、この計器についての事実
//!
//! **線形プローブは上界ではない。** 参照点 (雑音軸) で 1-NN 97.8% に対し線形プローブ 76.6%。
//! つまり線形プローブが低いことは情報の不在を意味しない。
//! ヘッダの「情報が在るかの上界」という表現は**誤りだった**。
//!
//! CLI: where_is_the_information

use spiking_brain::phase2_f::cochlea::{Cochlea, N_BANDS, SAMPLES_PER_STEP};
use spiking_brain::phase2_f::cochlear_nucleus::{CochlearNucleus, N_CN_OUTPUT};
use spiking_brain::phase2_f::kana::{moras_from_kana, synth_utterance};
use spiking_brain::phase2_f::phoneme_synth::LfsrNoise;

const KANA: &[&str] = &[
    "あ","い","う","え","お","か","き","く","け","こ","さ","し","す","せ","そ",
    "た","ち","つ","て","と","な","に","ぬ","ね","の","は","ひ","ふ","へ","ほ",
    "ま","み","む","め","も","や","ゆ","よ","ら","り","る","れ","ろ","わ","を","ん",
];

const F0S: [f64; 4] = [100.0, 130.0, 160.0, 200.0];
const LEVELS: [(i32, i32); 3] = [(1, 1), (1, 2), (1, 4)];
const SEEDS: [u16; 4] = [0xACE1, 0x1234, 0x7FFF, 0x0BAD];

/// リッジのグリッド。**実測前に固定**。判定はこの上の最大で行う。
const RIDGE_GRID: [f64; 3] = [0.01, 1.0, 100.0];
/// 置換回数。p 値の床 = 1/(N+1) なので p<0.01 を言うには 99 以上要る。
const N_PERM: usize = 999;
const PERM_SEED: u64 = 0x5EED_1234_ABCD_0001;
/// 子音区間の終わり (CONSONANT_MS=30ms・16kHz・SAMPLES_PER_STEP=8 → 60 step)
const CONSONANT_STEPS: usize = 60;

/// 主軸の seed。(かな, 変種) ごとに全条件で異なる。
/// かな単位で固定すると**かなごとの指紋**ができて指紋照合で解けてしまう。
fn utterance_seed(kana_idx: usize, variant: usize) -> u16 {
    ((kana_idx as u16).wrapping_mul(97).wrapping_add(variant as u16).wrapping_mul(2851)) | 1
}

fn wave_of(text: &str, f0: f64, seed: u16, gain_num: i32, gain_den: i32) -> Vec<i32> {
    let mut noise = LfsrNoise::new(seed);
    let (moras, skipped) = moras_from_kana(text);
    assert_eq!(skipped, 0, "未対応のかな: {}", text);
    let w = synth_utterance(&moras, f0, &mut noise);
    if gain_num == gain_den { w } else { w.iter().map(|&s| s * gain_num / gain_den).collect() }
}

/// M0 / M0.5 のスパイクを、時間平均 (1窓) と 2窓 (子音区間/母音区間) で返す。
fn features(wave: &[i32], use_cn: bool) -> (Vec<f64>, Vec<f64>) {
    let c = if use_cn { N_CN_OUTPUT } else { N_BANDS };
    let mut co = Cochlea::new();
    let mut cn = CochlearNucleus::new();
    let mut flat = vec![0f64; c];
    let mut win = vec![0f64; 2 * c];
    for (step, chunk) in wave.chunks(SAMPLES_PER_STEP).enumerate() {
        if chunk.len() < SAMPLES_PER_STEP { break; }
        let m0 = co.process_step(chunk);
        let w = if step < CONSONANT_STEPS { 0 } else { 1 };
        if use_cn {
            for (i, &v) in cn.process_step(&m0).iter().enumerate() {
                if v != 0 { flat[i] += 1.0; win[w * c + i] += 1.0; }
            }
        } else {
            for (i, &v) in m0.iter().enumerate() {
                if v != 0 { flat[i] += 1.0; win[w * c + i] += 1.0; }
            }
        }
    }
    (flat, win)
}

// ------------------------------------------------------------------ 線形代数

/// n×n の逆行列 (ガウス・ジョルダン・部分ピボット)。相対閾値で特異判定。
fn invert(a0: &[f64], n: usize) -> Option<Vec<f64>> {
    let scale = a0.iter().fold(0f64, |m, v| m.max(v.abs()));
    if scale == 0.0 { return None; }
    let mut a = a0.to_vec();
    let mut inv = vec![0f64; n * n];
    for i in 0..n { inv[i * n + i] = 1.0; }
    for col in 0..n {
        let mut piv = col;
        for r in (col + 1)..n {
            if a[r * n + col].abs() > a[piv * n + col].abs() { piv = r; }
        }
        if a[piv * n + col].abs() < 1e-12 * scale { return None; }
        if piv != col {
            for c in 0..n { a.swap(col * n + c, piv * n + c); inv.swap(col * n + c, piv * n + c); }
        }
        let d = a[col * n + col];
        for c in 0..n { a[col * n + c] /= d; inv[col * n + c] /= d; }
        for r in 0..n {
            if r == col { continue; }
            let f = a[r * n + col];
            if f == 0.0 { continue; }
            for c in 0..n {
                a[r * n + c] -= f * a[col * n + c];
                inv[r * n + c] -= f * inv[col * n + c];
            }
        }
    }
    Some(inv)
}

/// leave-one-out の各 fold について、スコア分解ベクトル s を返す。
///
/// リッジ回帰 W = (Xc'Xc + λI)^-1 Xc' Yc の予測は
///   score_c = z' W[:,c] = Σ_i s_i · Yc[i,c]
/// と分解できる。s = Xc (A^-1 z) は **ラベルに依存しない** ので、
/// 実ラベルでも置換ラベルでも同じ s を使い回せる。
/// これで 99 回の置換が実質タダになる。
///
/// X も Y も **fold 内で中心化**する (訓練側の平均だけを使うので漏れは無い)。
/// バイアス列は置かない (中心化が切片の役目を果たす)。
fn fold_scores(x: &[Vec<f64>], ridge_mul: f64) -> (Vec<Vec<f64>>, usize) {
    let n = x.len();
    let p = x[0].len();
    let mut out = vec![vec![0f64; n]; n];
    let mut singular = 0usize;
    for held in 0..n {
        let nt = n - 1;
        let mut mean = vec![0f64; p];
        for i in 0..n {
            if i == held { continue; }
            for k in 0..p { mean[k] += x[i][k] / nt as f64; }
        }
        let xc: Vec<Vec<f64>> = (0..n)
            .map(|i| if i == held { vec![0f64; p] }
                 else { (0..p).map(|k| x[i][k] - mean[k]).collect() })
            .collect();
        let mut a = vec![0f64; p * p];
        for i in 0..n {
            if i == held { continue; }
            for r in 0..p {
                let v = xc[i][r];
                if v == 0.0 { continue; }
                for c in 0..p { a[r * p + c] += v * xc[i][c]; }
            }
        }
        let tr: f64 = (0..p).map(|k| a[k * p + k]).sum::<f64>() / p as f64;
        let lambda = ridge_mul * tr.max(1e-9);
        for k in 0..p { a[k * p + k] += lambda; }
        let ainv = match invert(&a, p) { Some(m) => m, None => { singular += 1; continue; } };
        let z: Vec<f64> = (0..p).map(|k| x[held][k] - mean[k]).collect();
        let u: Vec<f64> = (0..p)
            .map(|r| (0..p).map(|c| ainv[r * p + c] * z[c]).sum())
            .collect();
        for i in 0..n {
            if i == held { continue; }
            out[held][i] = (0..p).map(|k| u[k] * xc[i][k]).sum();
        }
    }
    (out, singular)
}

/// s から、あるラベル付けでの LOO 正答率を出す。
/// Y も fold 内で中心化する: score_c = Σ_{i∈c} s_i − (n_c/nt) Σ_i s_i
fn accuracy_from(s: &[Vec<f64>], y: &[usize], n_class: usize) -> f64 {
    let n = s.len();
    let mut hit = 0usize;
    for held in 0..n {
        let nt = (n - 1) as f64;
        let mut per = vec![0f64; n_class];
        let mut cnt = vec![0f64; n_class];
        let mut total = 0f64;
        for i in 0..n {
            if i == held { continue; }
            per[y[i]] += s[held][i];
            cnt[y[i]] += 1.0;
            total += s[held][i];
        }
        let mut best = (f64::NEG_INFINITY, usize::MAX);
        for c in 0..n_class {
            let sc = per[c] - cnt[c] / nt * total;
            if sc > best.0 { best = (sc, c); }
        }
        if best.1 == y[held] { hit += 1; }
    }
    hit as f64 / n as f64 * 100.0
}

fn cosine(a: &[f64], b: &[f64]) -> f64 {
    let dot: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let na: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
    let nb: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();
    if na == 0.0 || nb == 0.0 { 0.0 } else { dot / (na * nb) }
}

/// 同点棄却つき 1-NN (§14.6 と同じ規則)。返り値 = (正答率, 判定不能数)
fn nn_accuracy(x: &[Vec<f64>], y: &[usize], sim: &dyn Fn(&[f64], &[f64]) -> f64) -> (f64, usize) {
    let n = x.len();
    let mut hit = 0usize;
    let mut undec = 0usize;
    for i in 0..n {
        let mut best = f64::NEG_INFINITY;
        for j in 0..n {
            if j == i { continue; }
            let s = sim(&x[i], &x[j]);
            if s > best { best = s; }
        }
        let tied: Vec<usize> = (0..n).filter(|&j| j != i && sim(&x[i], &x[j]) == best)
            .map(|j| y[j]).collect();
        if tied.iter().all(|&t| t == tied[0]) {
            if tied[0] == y[i] { hit += 1; }
        } else { undec += 1; }
    }
    (hit as f64 / n as f64 * 100.0, undec)
}

fn center(x: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let (n, p) = (x.len(), x[0].len());
    let mut m = vec![0f64; p];
    for v in x { for k in 0..p { m[k] += v[k] / n as f64; } }
    x.iter().map(|v| (0..p).map(|k| v[k] - m[k]).collect()).collect()
}

fn zscore(x: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let (n, p) = (x.len(), x[0].len());
    let mut m = vec![0f64; p];
    for v in x { for k in 0..p { m[k] += v[k] / n as f64; } }
    let mut sd = vec![0f64; p];
    for v in x { for k in 0..p { sd[k] += (v[k] - m[k]).powi(2) / n as f64; } }
    for k in 0..p { sd[k] = sd[k].sqrt().max(1e-9); }
    x.iter().map(|v| (0..p).map(|k| (v[k] - m[k]) / sd[k]).collect()).collect()
}

/// 白色化 (共分散の Cholesky で写す)。写した空間でのユークリッド距離 =
/// 元空間のマハラノビス距離。**transductive** (全標本から共分散を推定)。
fn whiten(x: &[Vec<f64>]) -> Option<Vec<Vec<f64>>> {
    let xc = center(x);
    let (n, p) = (xc.len(), xc[0].len());
    let mut cov = vec![0f64; p * p];
    for v in &xc {
        for a in 0..p {
            if v[a] == 0.0 { continue; }
            for b in 0..p { cov[a * p + b] += v[a] * v[b] / n as f64; }
        }
    }
    let tr: f64 = (0..p).map(|k| cov[k * p + k]).sum::<f64>() / p as f64;
    for k in 0..p { cov[k * p + k] += 0.1 * tr.max(1e-9); }
    let mut l = vec![0f64; p * p];
    for i in 0..p {
        for j in 0..=i {
            let mut s = cov[i * p + j];
            for k in 0..j { s -= l[i * p + k] * l[j * p + k]; }
            if i == j {
                if s <= 0.0 { return None; }
                l[i * p + j] = s.sqrt();
            } else { l[i * p + j] = s / l[j * p + j]; }
        }
    }
    Some(xc.iter().map(|v| {
        let mut z = vec![0f64; p];
        for i in 0..p {
            let mut s = v[i];
            for k in 0..i { s -= l[i * p + k] * z[k]; }
            z[i] = s / l[i * p + i];
        }
        z
    }).collect())
}

fn shuffled(y: &[usize], seed: u64) -> Vec<usize> {
    let mut out = y.to_vec();
    let mut s = seed;
    for i in (1..out.len()).rev() {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let j = ((s >> 33) as usize) % (i + 1);
        out.swap(i, j);
    }
    out
}

struct Cell {
    probe: f64,
    perm_max: f64,
    p_value: f64,
    singular: usize,
}

/// 線形プローブ + 置換検定。λ グリッドの最大で判定する。
fn probe_with_permutation(x: &[Vec<f64>], y: &[usize], n_class: usize) -> Cell {
    let mut real_by_lambda = vec![f64::NEG_INFINITY; RIDGE_GRID.len()];
    let mut perm_by_lambda = vec![vec![0f64; N_PERM]; RIDGE_GRID.len()];
    let mut singular = 0usize;
    for (li, &lam) in RIDGE_GRID.iter().enumerate() {
        let (s, sing) = fold_scores(x, lam);
        singular += sing;
        real_by_lambda[li] = accuracy_from(&s, y, n_class);
        for r in 0..N_PERM {
            let ys = shuffled(y, PERM_SEED.wrapping_add(r as u64 * 0x9E37_79B9_7F4A_7C15));
            perm_by_lambda[li][r] = accuracy_from(&s, &ys, n_class);
        }
    }
    let real = real_by_lambda.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let perms: Vec<f64> = (0..N_PERM)
        .map(|r| (0..RIDGE_GRID.len()).map(|li| perm_by_lambda[li][r])
             .fold(f64::NEG_INFINITY, f64::max))
        .collect();
    let ge = perms.iter().filter(|&&v| v >= real).count();
    Cell {
        probe: real,
        perm_max: perms.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
        p_value: (1 + ge) as f64 / (N_PERM + 1) as f64,
        singular,
    }
}

struct Axis { name: &'static str, judged: bool, waves: Vec<(usize, Vec<i32>)>, variants: usize }

fn build_axes() -> Vec<Axis> {
    let (mut main, mut f0, mut lv, mut sd) = (vec![], vec![], vec![], vec![]);
    for (k, &kana) in KANA.iter().enumerate() {
        for (v, &f) in F0S.iter().enumerate() {
            main.push((k, wave_of(kana, f, utterance_seed(k, v), 1, 1)));
            f0.push((k, wave_of(kana, f, SEEDS[0], 1, 1)));
        }
        for &(gn, gd) in LEVELS.iter() { lv.push((k, wave_of(kana, F0S[0], SEEDS[0], gn, gd))); }
        for &s in SEEDS.iter() { sd.push((k, wave_of(kana, F0S[0], s, 1, 1))); }
    }
    vec![
        Axis { name: "話者の言い直し (F0+雑音実現が全条件で異なる) [判定軸]", judged: true, waves: main, variants: F0S.len() },
        Axis { name: "F0 のみ (雑音実現固定・指紋照合込み) [判定対象外]", judged: false, waves: f0, variants: F0S.len() },
        Axis { name: "レベルのみ (0/-6/-12 dB) [判定対象外]", judged: false, waves: lv, variants: LEVELS.len() },
        Axis { name: "雑音実現のみ (参照点) [判定対象外]", judged: false, waves: sd, variants: SEEDS.len() },
    ]
}

fn health(x: &[Vec<f64>]) -> (usize, usize) {
    let n = x.len();
    let silent = x.iter().filter(|v| v.iter().all(|&z| z == 0.0)).count();
    let twin = (0..n).filter(|&i| (0..n).any(|j| j != i && x[j] == x[i])).count();
    (silent, twin)
}

fn run_cell(stage: &str, x: &[Vec<f64>], y: &[usize], n_variants: usize) -> Cell {
    let n = x.len();
    let (silent, twin) = health(x);
    let ch_nn = (n_variants - 1) as f64 / (n - 1) as f64 * 100.0;
    let ch_probe = 100.0 / KANA.len() as f64;
    let (raw, un_raw) = nn_accuracy(x, y, &|a, b| cosine(a, b));
    let (cen, _) = nn_accuracy(&center(x), y, &|a, b| cosine(a, b));
    let (zs, _) = nn_accuracy(&zscore(x), y, &|a, b| cosine(a, b));
    let wh = match whiten(x) {
        Some(w) => nn_accuracy(&w, y, &|a, b| -a.iter().zip(b.iter()).map(|(p, q)| (p - q).powi(2)).sum::<f64>()).0,
        None => f64::NAN,
    };
    let cell = probe_with_permutation(x, y, KANA.len());
    println!("  {:<16} n={:>3} p次元={:>3} | 無音{:>2} 重複{:>3} 同点棄却{:>3} 特異fold{:>3}",
             stage, n, x[0].len(), silent, twin, un_raw, cell.singular);
    println!("  {:<16} 1-NN: 素 {:>5.1}% 中心化 {:>5.1}% 標準化 {:>5.1}% 白色化 {:>5.1}% (transductive・チャンス {:.2}%)",
             "", raw, cen, zs, wh, ch_nn);
    println!("  {:<16} 線形プローブ **{:>5.1}%** vs 置換最大 {:>5.1}% ・ **p = {:.3}** (チャンス {:.2}%)",
             "", cell.probe, cell.perm_max, cell.p_value, ch_probe);
    cell
}

fn main() {
    println!("=== 犯人は M0 か M1 か — 情報は失われているのか、読めていないだけなのか ===");
    println!();
    println!("【計測器であって系ではない】線形プローブは教師あり・浮動小数点。M1 に入れるものではない。");
    println!("【この計器の非対称】時間を畳む/2窓に切る読み出しなので、null が出ても");
    println!("  『時刻に情報が無い』ことは示せない。**M0 を無罪にする方には使えるが有罪にする方には弱い。**");
    println!();
    println!("判定軸 = 主軸のみ。λ グリッド {:?} の最大で判定・置換も同じ最大。置換 {} 回。",
             RIDGE_GRID, N_PERM);
    println!("統合規則 (実測前に固定):");
    println!("  M0.5 のどれかで p<0.01            -> 情報は M1 に届いている");
    println!("  M0.5 全滅(全て p>0.2) & M0 で p<0.01 -> M0.5 が落としている");
    println!("  M0 も M0.5 も全滅                  -> この読み出しでは情報が見つからない");
    println!("  それ以外                           -> 中間 (どちらにも倒さない)");

    // --- 無情報ベースライン: 乱数特徴でプローブがチャンス付近に来るか ---
    println!();
    println!("--- 無情報ベースライン (乱数特徴・情報ゼロ) ---");
    let y_ref: Vec<usize> = (0..KANA.len()).flat_map(|k| std::iter::repeat(k).take(4)).collect();
    let mut st = 0x1234_5678_9ABC_DEF0u64;
    let rnd: Vec<Vec<f64>> = (0..y_ref.len()).map(|_| (0..84).map(|_| {
        st = st.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((st >> 33) as f64 / (1u64 << 31) as f64) * 50.0
    }).collect()).collect();
    let base = probe_with_permutation(&rnd, &y_ref, KANA.len());
    println!("  線形プローブ {:.1}% ・ 置換最大 {:.1}% ・ p = {:.3} (チャンス {:.2}%)",
             base.probe, base.perm_max, base.p_value, 100.0 / KANA.len() as f64);
    println!("  ここが チャンス付近 (約 2.2%) に来ていなければ、推定器が壊れている。");

    let axes = build_axes();
    let mut judged: Vec<(&str, Cell)> = Vec::new();
    for axis in axes.iter() {
        println!();
        println!("--- 軸: {} ({} 条件) ---", axis.name, axis.waves.len());
        let y: Vec<usize> = axis.waves.iter().map(|(k, _)| *k).collect();
        for &(use_cn, name) in [(false, "M0"), (true, "M0.5")].iter() {
            let feats: Vec<(Vec<f64>, Vec<f64>)> =
                axis.waves.iter().map(|(_, w)| features(w, use_cn)).collect();
            let flat: Vec<Vec<f64>> = feats.iter().map(|(f, _)| f.clone()).collect();
            let c1 = run_cell(&format!("{} 時間平均", name), &flat, &y, axis.variants);
            let win: Vec<Vec<f64>> = feats.iter().map(|(_, w)| w.clone()).collect();
            let c2 = run_cell(&format!("{} 2窓", name), &win, &y, axis.variants);
            if axis.judged {
                judged.push((if use_cn { "M0.5 時間平均" } else { "M0 時間平均" }, c1));
                judged.push((if use_cn { "M0.5 2窓" } else { "M0 2窓" }, c2));
            }
        }
    }

    println!();
    println!("=== 判定 (主軸のみ・規則は実測前に固定) ===");
    for (n, c) in judged.iter() {
        println!("  {:<16} 線形 {:>5.1}% ・ p = {:.3}", n, c.probe, c.p_value);
    }
    let sig = |k: &str| judged.iter().any(|(n, c)| n.starts_with(k) && c.p_value < 0.01);
    let dead = |k: &str| judged.iter().filter(|(n, _)| n.starts_with(k)).all(|(_, c)| c.p_value > 0.2);
    println!();
    if sig("M0.5") {
        println!("  -> **情報は M1 に届いている。** M0/M0.5 は情報を消していない。");
        println!("     ただし示したのは『線形に読める成分が有意に残る』ことだけで、");
        println!("     M1 が原理の内側で同じことをできるかは別問題。");
    } else if dead("M0.5") && sig("M0") {
        println!("  -> **M0.5 が落としている。** M0 には在るが M0.5 で消えている。");
    } else if dead("M0.5") && dead("M0") {
        println!("  -> **この読み出しでは情報が見つからない。**");
        println!("     時刻に残る可能性・非線形に符号化されている可能性は否定できない。");
        println!("     追加測定: 時間分解を細かくする (2窓 -> 8窓) を先に 1 つやること。");
    } else {
        println!("  -> **中間。どちらにも倒さない。**");
        println!("     追加測定: 置換回数を 999 に上げて p を締めること。");
    }
}
