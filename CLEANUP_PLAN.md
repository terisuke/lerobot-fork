# ステレオビジョン実装 - クリーンアップ計画

## 削除するファイル（間違ったアプローチ/中間テスト）

### ❌ 不要な実装（間違ったアプローチ）
```
src/lerobot/cameras/stereo/
├── __init__.py
├── camera_stereo.py
└── configuration_stereo.py
```
理由: LeRobotは既にdict[str, CameraConfig]で複数カメラをサポート。専用クラスは不要。

### ❌ 中間テストスクリプト（15個）
```
examples/stereo/
├── 01_basic_capture.py
├── adjust_camera_colors.py
├── correct_stereo_setup.py
├── detailed_camera_check.py
├── identify_physical_webcams.py
├── identify_webcams.py
├── probe_camera0_capabilities.py
├── quick_camera_test.py
├── stereo_cam0_cam1.py
├── stereo_mixed_fps.py
├── test_camera_1_and_2.py
├── test_camera0_fps.py
├── test_camera1_720p.py
├── test_color_mode.py
└── test_webcam_pair.py
```
理由: 調査・デバッグ用の一時スクリプト。最終実装に不要。

### ❌ 古いドキュメント
```
docs/WEB_CAMERA_STEREO_IMPLEMENTATION.md
examples/stereo/DISABLE_IPHONE_CAMERA.md
```
理由: 初期計画書。内容を統合して削除。

---

## ✅ 保持するファイル

### 最終実装
```
examples/stereo/
└── final_stereo_setup.py  # ✅ 最終動作スクリプト
```

### ドキュメント
```
docs/
└── STEREO_VISION_GUIDE.md  # ✅ 実装ガイド（更新）
```

---

## 📝 新規作成するファイル

### 1. マネージャー向けレポート
```
STEREO_IMPLEMENTATION_REPORT.md
```
内容:
- プロジェクトサマリー
- 実装結果
- 技術的な発見
- 次のステップ

### 2. examples/stereo/README.md
```
examples/stereo/README.md
```
内容:
- 使い方
- セットアップ手順
- トラブルシューティング

---

## 実行コマンド

```bash
# 1. 不要な実装を削除
rm -rf src/lerobot/cameras/stereo/

# 2. 中間テストスクリプトを削除
cd examples/stereo
rm 01_basic_capture.py \
   adjust_camera_colors.py \
   correct_stereo_setup.py \
   detailed_camera_check.py \
   identify_physical_webcams.py \
   identify_webcams.py \
   probe_camera0_capabilities.py \
   quick_camera_test.py \
   stereo_cam0_cam1.py \
   stereo_mixed_fps.py \
   test_camera_1_and_2.py \
   test_camera0_fps.py \
   test_camera1_720p.py \
   test_color_mode.py \
   test_webcam_pair.py \
   DISABLE_IPHONE_CAMERA.md

# 3. 古いドキュメントを削除
cd ../../docs
rm WEB_CAMERA_STEREO_IMPLEMENTATION.md
```
