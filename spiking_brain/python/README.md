# Python + CUDA 版 1ビット SNN

Rust 版 (`binary_brain`) と同じアルゴリズムを PyTorch + CUDA で並列実行。
RTX 3060 Laptop 6GB を想定。

## セットアップ (Windows + CUDA 12.6)

```powershell
# 仮想環境推奨
python -m venv venv
.\venv\Scripts\Activate.ps1

# PyTorch (CUDA 12.4 ホイールが 12.6 ドライバで動く)
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install numpy

# 動作確認
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

## 実行

### 標準実験 (Rust 版と同じ規模、報酬学習)

```powershell
python brain_torch.py
```

400 cortex ニューロン、報酬変調 R-STDP 200 試行、訓練前後で出力選択性を測る。
3060 なら 5-10 秒で終わる (Rust 版の 30 秒台に対して大幅短縮)。

### スケールアップ

```powershell
python brain_torch.py --neurons 10000 --fanout 60
python brain_torch.py --neurons 100000 --fanout 100
```

### スケーリングテスト (推奨、最初に走らせる)

```powershell
python scale_test.py
```

ネットワーク規模を 400 → 1,000 → 3,000 → ... → 300,000 と振って、
3060 6GB で実際にどこまで載るか、リアルタイム比を測る。
出力例:

```
   neurons     synapses    build       VRAM      trial   realtime   status
------------------------------------------------------------------------
       400        17600    0.32s      18.4M      1.5ms    200.00x      OK
      1000        44000    0.45s      28.1M      2.1ms    142.86x      OK
      3000       132000    0.81s      62.4M      4.5ms     66.67x      OK
     10000       440000    1.95s     189.2M     11.2ms     26.79x      OK
     30000      1320000    5.41s     564.8M     32.5ms      9.23x      OK
    100000      4400000   17.83s    1872.4M    104.7ms      2.87x      OK
    300000     13200000   53.40s    5612.7M    302.1ms      0.99x      OK
```

(数字は目安、実際は環境による)

### CPU と比較

```powershell
python scale_test.py --device cpu
```

3060 の GPU 効率がどれくらい効いているか直接比較できる。

## ファイル

- `brain_torch.py` — メイン実装 (BinaryBrain クラス、実験ランナー)
- `scale_test.py` — スケーリングベンチマーク

## アーキテクチャの再確認

- ニューロン: 1ビット LIF カウンタ (Rust 版と同等)
- シナプス: 結線あり/なし (1ビット)、遅延、eligibility、可塑フラグ
- 学習: R-STDP eligibility trace + 報酬での構造書き換え
- 出力: タイムスタンプ列の指数減衰重ね合わせフィンガープリント

## Rust 版との対応

| 概念 | Rust | Python |
|---|---|---|
| ネットワーク | `BinaryNetwork` | `BinaryBrain` |
| シナプス (構造体) | `BinarySynapse` | テンソル群 (SoA) |
| シミュレーション 1 ステップ | `step()` | `step()` |
| 報酬適用 | `apply_reward(r)` | `apply_reward(r)` |
| パターン提示 | `present_pulse_pattern` | `present_pattern` |
| フィンガープリント | `fingerprint_from_log` | `fingerprint` |

データ構造はほぼ 1 対 1 対応、Rust 版で動くものは Python 版でも同じ結果になる
(乱数シードが違うので数値そのものは違うが、選択性・収束パターンは同じ)。

## デバッグ

CUDA エラーが出る場合:

```powershell
# 同期実行に切り替えてエラー場所を特定
$env:CUDA_LAUNCH_BLOCKING="1"
python brain_torch.py
```

VRAM 不足:

```powershell
# 規模を小さく
python brain_torch.py --neurons 1000
```
