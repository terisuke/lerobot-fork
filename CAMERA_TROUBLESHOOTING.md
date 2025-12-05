# SO101評価時のカメラ問題と解決策

## 問題の状況

ロボットアームが動かない原因は、**カメラ0が1280x960の解像度でフレームを読み取れない**ことです。

### 確認結果

- ✅ **ロボット接続**: 正常
- ✅ **カメラ1**: 正常（640x480で読み取り可能）
- ❌ **カメラ0**: 1280x960でフレーム読み取り失敗

## 解決策

### 解決策1: カメラ解像度を下げる（推奨）

モデルは1280x960を期待していますが、カメラがサポートしていない場合は、640x480に設定して自動リサイズに任せます：

```bash
lerobot-record \
    --robot.type=so101_follower \
    --robot.port=/dev/tty.wchusbserial5AB90691861 \
    --robot.id=my_awesome_follower_arm \
    --robot.cameras='{"front": {"type": "opencv", "index_or_path": 0, "width": 640, "height": 480, "fps": 15}, "side": {"type": "opencv", "index_or_path": 1, "width": 640, "height": 480, "fps": 15}}' \
    --display_data=false \
    --dataset.repo_id=Terisuke/eval_so101_20k \
    --dataset.num_episodes=1 \
    --dataset.episode_time_s=30 \
    --dataset.single_task="Pat the head of the person in front of you" \
    --policy.path=./outputs/so101-20k-final/020000/pretrained_model
```

### 解決策2: カメラを物理的に再接続

1. USBケーブルを抜く
2. 数秒待つ
3. 再度接続
4. `lerobot-find-cameras opencv`でカメラIDを再確認

### 解決策3: 他のアプリケーションを閉じる

macOSの場合：

- Zoom、Skype、Photo Boothなど、カメラを使用しているアプリをすべて閉じる
- システム環境設定 > セキュリティとプライバシー > カメラで権限を確認

### 解決策4: カメラ0の代わりにカメラ1を使用

カメラ1が正常に動作している場合、両方のカメラをカメラ1に設定することも可能です（ただし、モデルの性能に影響する可能性があります）。

## 現在の評価コマンド実行状況

評価コマンドをバックグラウンドで実行中です。ログは `evaluation_output.log` に保存されます。

ログを確認するには：

```bash
tail -f evaluation_output.log
```

## ロボットアームが動かない場合の確認事項

1. **カメラ接続**: カメラが正常に接続されているか
2. **ポリシー読み込み**: ポリシーが正常に読み込まれているか
3. **アクション生成**: ポリシーがアクションを生成しているか（ログで確認）
4. **ロボット通信**: ロボットへのアクション送信が成功しているか

## デバッグ方法

ポリシーがアクションを生成しているか確認するには、ログの以下の行を探してください：

```
Policy action at X.XXs: ...
```

この行が表示されていれば、ポリシーはアクションを生成しています。表示されない場合は、カメラの問題で観測が取得できていない可能性が高いです。
