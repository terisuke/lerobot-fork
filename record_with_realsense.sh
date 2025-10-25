#!/bin/bash

# Intel RealSense d435とSO101ロボットアームを使用したデータ収集スクリプト
# 使用方法: ./record_with_realsense.sh

# 環境変数の設定
export HF_USER="teradakousuke"
export ROBOT_PORT="/dev/tty.wchusbserial5AB90691861"
export TELEOP_PORT="/dev/tty.wchusbserial5AB90684961"

# LeRobot環境のアクティベート
source ~/opt/anaconda3/envs/lerobot/bin/activate

echo "=== Intel RealSense d435 + SO101 データ収集セッション ==="
echo "Hugging Face User: $HF_USER"
echo "Robot Port: $ROBOT_PORT"
echo "Teleop Port: $TELEOP_PORT"
echo ""

# RealSenseカメラの検出
echo "RealSenseカメラを検出中..."
lerobot-find-cameras realsense

if [ $? -ne 0 ]; then
    echo "警告: RealSenseカメラが検出されませんでした。sudo権限で再試行します..."
    sudo lerobot-find-cameras realsense
fi

echo ""
echo "データ収集を開始します..."
echo "注意: RealSenseカメラが接続されていることを確認してください。"
echo ""

# データ収集コマンドの実行
lerobot-record \
    --robot.type=so101_follower \
    --robot.port=${ROBOT_PORT} \
    --robot.id=my_awesome_follower_arm \
    --teleop.type=so101_leader \
    --teleop.port=${TELEOP_PORT} \
    --teleop.id=my_awesome_leader_arm \
    --robot.cameras="{ front: {type: intelrealsense, width: 640, height: 480, fps: 30, use_depth: true} }" \
    --dataset.repo_id=${HF_USER}/so101-dynamic-grasp \
    --dataset.num_episodes=50 \
    --dataset.single_task="Pick up objects and place them in designated locations" \
    --display_data=true \
    --play_sounds=true

echo ""
echo "データ収集が完了しました。"
echo "データセット: ${HF_USER}/so101-dynamic-grasp"
