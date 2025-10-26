# LeRobot Webカメラステレオビジョン - 使い方ガイド

2台のWebカメラを使用したステレオビジョンシステムの使用方法

---

## 📋 必要なもの

### ハードウェア
- 物理Webカメラ ×2台
- USB 3.0ハブ（推奨）
- 三脚またはカメラマウント ×2台
- チェッカーボード（キャリブレーション用、A4サイズ推奨）

### ソフトウェア
- Python 3.10+
- LeRobot環境（既にセットアップ済み）
- OpenCV（インストール済み）

---

## 🚀 クイックスタート

### 1. ステレオキャプチャの実行

```bash
cd /Users/teradakousuke/Developer/lerobot
conda activate lerobot
python examples/stereo/final_stereo_setup.py
```

**操作方法:**
- ESCキー: 終了
- プログラムは自動的にフレームをキャプチャして表示

**出力:**
- ウィンドウに左右のカメラ映像を表示
- サンプル画像を `outputs/stereo_final_optimal/` に保存

---

## ⚙️ カメラ設定

### 現在の構成

本プロジェクトで使用しているカメラ:
- **Camera #0** (左): 1280x720 @ 30fps
- **Camera #1** (右): 1280x720 @ 30fps

### カメラインデックスの確認

カメラが変わった場合、インデックスを再確認：

```bash
lerobot-find-cameras opencv
```

または、macOSで物理カメラを確認：

```bash
system_profiler SPCameraDataType
```

### 設定の変更

`final_stereo_setup.py`の設定部分を編集：

```python
camera_configs = {
    "left": OpenCVCameraConfig(
        index_or_path=0,  # ← カメラインデックス
        fps=30,           # ← FPS
        width=1280,       # ← 幅
        height=720,       # ← 高さ
        color_mode=ColorMode.BGR,
    ),
    "right": OpenCVCameraConfig(
        index_or_path=1,  # ← カメラインデックス
        fps=30,
        width=1280,
        height=720,
        color_mode=ColorMode.BGR,
    ),
}
```

---

## 🔧 トラブルシューティング

### カメラが認識されない

**確認事項:**
1. USBケーブルが正しく接続されているか
2. カメラに電力が供給されているか（USBハブ使用時）
3. 他のアプリケーションがカメラを使用していないか

**対処法:**
```bash
# カメラを使用しているプロセスを確認
lsof | grep -i camera

# 見つかった場合、該当アプリを終了
```

### "read failed" エラー

**原因:**
- Camera #0は初回読み取りに失敗する特性がある
- プログラムは自動的にリトライする

**メッセージ:**
```
⚠️  Initial read failed (expected for Camera #0), retrying...
```

これは**正常な動作**です。2回目の読み取りで成功します。

### FPSが低い（14fps程度）

**原因:**
- 2台のカメラを同時に読み取る処理オーバーヘッド
- 1280x720の解像度

**対処法（必要な場合）:**

1. **解像度を下げる:**
```python
width=640,
height=480,
```

2. **別スレッドでカメラを読み取る**（上級者向け）

### 色が赤みがかっている

**原因:**
- カメラのホワイトバランス設定
- 照明環境

**影響:**
- ステレオマッチング（深度推定）には**影響なし**
- 色は輝度ベースの処理なので問題なく動作

**対処法（必要な場合）:**
- より良い照明環境を用意
- 白/グレーカードでホワイトバランスを調整
- ソフトウェアで色補正（後処理）

### OBSやiPhoneカメラが混在している

**対処法:**

1. **OBS仮想カメラを無効化:**
   - OBSアプリを終了
   - または OBS > Tools > Virtual Camera > Stop

2. **iPhoneカメラを無効化:**
   - システム設定 > 一般 > AirDropとHandoff
   - "Continuity Camera" をオフ

---

## 📐 カメラの配置

### 推奨配置

```
    ← 6-10cm →
   Left      Right
   Camera    Camera
     |         |
     ↓         ↓
   [作業空間]
   [  Robot   ]
```

**ポイント:**
- ベースライン距離: 6-10cm（深度精度に影響）
- 両カメラの光軸を平行に
- 作業空間全体が両方の視野に収まる高さ
- しっかり固定（振動や移動を防ぐ）

---

## 🎯 次のステップ

### 1. ステレオキャリブレーション

カメラ間の幾何学的関係を計算：

```bash
# キャリブレーション実行
python examples/stereo/02_calibrate_stereo.py \
    --left-camera 0 \
    --right-camera 1 \
    --num-images 20 \
    --output configs/camera/stereo_calibration.yaml
```

**必要なもの:**
- チェッカーボードパターン（9x6マス、各25mm推奨）
- 20枚程度の異なる角度からの画像

**操作方法:**
- SPACEキー: パターン検出時に画像をキャプチャ
- ESCキー: キャプチャ終了してキャリブレーション実行

### 2. 深度推定

キャリブレーション完了後、深度マップを生成：

```bash
# リアルタイム深度可視化
python examples/stereo/03_test_depth.py \
    --calibration configs/camera/stereo_calibration.yaml \
    --left-camera 0 \
    --right-camera 1 \
    --method sgbm
```

**操作方法:**
- ESCキー: 終了
- Sキー: 現在のフレームと深度マップを保存
- SPACEキー: 表示モード切替（カラー深度/視差）

---

## 📊 技術仕様

### カメラ仕様

| 項目 | Camera #0 (左) | Camera #1 (右) |
|------|---------------|---------------|
| ネイティブ解像度 | 1280x960 | 1920x1080 |
| 使用解像度 | 1280x720 | 1280x720 |
| FPS設定値 | 30fps | 30fps |
| 実測FPS | ~14fps（両カメラ合計） | ~14fps（両カメラ合計） |
| 初回読み取り | 失敗する（自動リトライ） | 成功 |

### システム性能

- **同期精度**: ソフトウェア同期（±数ms）
- **処理遅延**: フレーム読み取り + 表示で約70ms
- **深度範囲**: 0.3m〜3m程度（キャリブレーション後に決定）

---

## 📝 開発メモ

### LeRobot標準パターン

このプロジェクトはLeRobotの標準パターンに準拠：

```python
# dict[str, CameraConfig] で複数カメラを管理
camera_configs = {
    "left": OpenCVCameraConfig(...),
    "right": OpenCVCameraConfig(...),
}

# 各カメラを独立して初期化
cameras = {}
for name, config in camera_configs.items():
    cameras[name] = OpenCVCamera(config)
    cameras[name].connect()

# フレーム取得
frames = {}
for name, camera in cameras.items():
    frames[name] = camera.read()
```

このパターンは、LeRobotのデータセット記録やロボット制御と互換性があります。

---

## 🔗 関連ドキュメント

- [STEREO_VISION_GUIDE.md](../../docs/STEREO_VISION_GUIDE.md) - 実装詳細ガイド
- [STEREO_IMPLEMENTATION_REPORT.md](../../STEREO_IMPLEMENTATION_REPORT.md) - プロジェクト報告書
- [LeRobot公式ドキュメント](https://huggingface.co/docs/lerobot)

---

## 💬 サポート

問題が発生した場合:
1. このREADMEのトラブルシューティングセクションを確認
2. プロジェクトのIssueを検索
3. 新しいIssueを作成

---

**最終更新**: 2025年10月26日
