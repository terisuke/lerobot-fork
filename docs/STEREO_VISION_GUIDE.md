# LeRobot Webカメラステレオビジョン実装ガイド

**最終更新**: 2025年10月26日
**ステータス**: ✅ Phase 1完了（基本キャプチャ）

## 概要

LeRobotでは**複数カメラは標準機能**として既にサポートされています。
ステレオビジョンのために専用のクラスを作る必要は**ありません**。

**実装完了項目:**
- ✅ 2台の物理Webカメラによるステレオキャプチャ
- ✅ 1280x720 @ 30fps設定での安定動作
- ✅ LeRobot標準パターンに準拠
- ⏳ ステレオキャリブレーション（次フェーズ）
- ⏳ 深度推定（次フェーズ）

## 正しい実装方法

### 1. カメラ設定の定義

LeRobotの標準パターン：`dict[str, CameraConfig]`

```python
from lerobot.cameras.opencv import OpenCVCameraConfig

camera_configs = {
    "left": OpenCVCameraConfig(
        index_or_path=1,
        fps=30,
        width=1920,
        height=1080,
    ),
    "right": OpenCVCameraConfig(
        index_or_path=4,
        fps=30,
        width=1920,
        height=1080,
    ),
}
```

### 2. カメラの初期化と使用

```python
from lerobot.cameras.opencv import OpenCVCamera

cameras = {}
for name, config in camera_configs.items():
    cameras[name] = OpenCVCamera(config)
    cameras[name].connect(warmup=True)

# フレーム取得
frames = {}
for name, camera in cameras.items():
    frames[name] = camera.read()

# frames["left"] と frames["right"] が取得できる
```

### 3. ロボット設定への統合

ロボット設定クラスに組み込む場合：

```python
from dataclasses import dataclass, field
from lerobot.cameras import CameraConfig
from lerobot.robots.config import RobotConfig

def stereo_cameras_config() -> dict[str, CameraConfig]:
    return {
        "left": OpenCVCameraConfig(
            index_or_path=1,
            fps=30,
            width=1920,
            height=1080,
        ),
        "right": OpenCVCameraConfig(
            index_or_path=4,
            fps=30,
            width=1920,
            height=1080,
        ),
    }

@RobotConfig.register_subclass("my_robot")
@dataclass
class MyRobotConfig(RobotConfig):
    port: str = "/dev/ttyUSB0"
    cameras: dict[str, CameraConfig] = field(default_factory=stereo_cameras_config)
```

## カメラインデックスの特定

### macOSでのカメラ識別

```bash
# LeRobotコマンドで検出
lerobot-find-cameras opencv

# システム情報で物理カメラ確認
system_profiler SPCameraDataType
```

### 推奨カメラペア（本プロジェクト）

検出結果：
- Camera #0: 物理Webカメラ1 (1280x720 @ 30fps) - **左カメラ推奨**
- Camera #1: 物理Webカメラ2 (1280x720 @ 30fps) - **右カメラ推奨**
- Camera #2: MacBook Pro内蔵カメラ - **非推奨**
- Camera #3: OBS仮想カメラ - **非推奨**
- Camera #4: iPhone連携カメラ - **非推奨**

**推奨ペア: Camera #0 (左) + Camera #1 (右)**

注: Camera #0は初回読み取りに失敗する特性がありますが、リトライで正常動作します。

## 深度推定の実装

深度推定は**後処理**として実装します（LeRobot標準にはない機能）。

### アプローチ

1. **データ収集時**: 左右のカメラから独立してフレームを取得
2. **キャリブレーション**: オフラインでステレオキャリブレーション実行
3. **深度推定**: 後処理または推論時にステレオマッチング適用

### ユーティリティモジュール構成

```
lerobot/utils/stereo/
├── __init__.py
├── calibration.py      # ステレオキャリブレーション
└── depth_estimation.py # 深度マップ計算
```

## 実装例

### 基本的なステレオキャプチャ

```bash
python examples/stereo/final_stereo_setup.py
```

### ステレオキャリブレーション

```bash
python examples/stereo/02_calibrate_stereo.py \
    --left-camera 0 \
    --right-camera 1 \
    --num-images 20 \
    --output configs/camera/stereo_calibration.yaml
```

### 深度推定の使用

```bash
# リアルタイム深度可視化
python examples/stereo/03_test_depth.py \
    --calibration configs/camera/stereo_calibration.yaml \
    --left-camera 0 \
    --right-camera 1 \
    --method sgbm
```

プログラムから使用する場合:

```python
from lerobot.utils.stereo import compute_depth_map, load_stereo_calibration

# カメラから左右フレーム取得
left_frame = cameras["left"].read()
right_frame = cameras["right"].read()

# キャリブレーションデータ読み込み
calib_data = load_stereo_calibration("configs/camera/stereo_calibration.yaml")

# 深度マップ計算
depth_map = compute_depth_map(left_frame, right_frame, calib_data)
```

## 注意事項

### カメラ同期

2台のカメラを完全に同期させるのは難しい：
- ハードウェアトリガーなしでは±数msのずれが発生
- 静的なシーンや低速動作では影響少ない
- 高速動作にはハードウェア同期が必要

### 解像度とFPS

- 両カメラは**同じ解像度・FPS**を推奨
- 1920x1080 @ 30fps が一般的
- 640x480に下げると処理は速いが精度低下

### ベースライン距離

カメラ間の距離（ベースライン）:
- **6-10cm**: 机上作業に適切
- 近すぎる: 遠距離の精度低下
- 遠すぎる: 近距離の視差不足

## トラブルシューティング

### カメラが検出されない

```bash
# カメラインデックス再確認
lerobot-find-cameras opencv

# 権限確認（macOS）
# システム設定 > プライバシーとセキュリティ > カメラ
```

### FPS/解像度エラー

```
RuntimeError: OpenCVCamera(X) failed to set fps=30 (actual_fps=60.0)
```

→ カメラのネイティブ設定を使用してください

```python
# カメラ#3は60fps固定
OpenCVCameraConfig(index_or_path=3, fps=60, ...)  # OK
OpenCVCameraConfig(index_or_path=3, fps=30, ...)  # NG
```

### 2台のカメラでFPSが異なる

異なるFPSのカメラペアは同期困難です。
同じFPSのカメラを選択してください。

## まとめ

✅ **正しいアプローチ:**
- 既存の`dict[str, CameraConfig]`を使用
- `OpenCVCamera`をそのまま利用
- 深度推定は後処理ユーティリティとして実装

❌ **不要な実装:**
- 専用`StereoCamera`クラス
- カメララッパーの独自実装
- LeRobot標準からの逸脱

LeRobotの設計思想に従うことで、メンテナンス性とコミュニティとの互換性が保たれます。
