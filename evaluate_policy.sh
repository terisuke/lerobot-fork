#!/bin/bash

# 学習済みポリシーの評価スクリプト
# 使用方法: ./evaluate_policy.sh

# 環境変数の設定
export HF_USER="teradakousuke"
export ROBOT_PORT="/dev/tty.wchusbserial5AB90691861"

# LeRobot環境のアクティベート
source ~/opt/anaconda3/envs/lerobot/bin/activate

echo "=== 学習済みポリシーの評価 ==="
echo "Hugging Face User: $HF_USER"
echo "Robot Port: $ROBOT_PORT"
echo ""

# ポリシーの確認
echo "学習済みポリシーを確認中..."
echo "ポリシー: ${HF_USER}/so101-dynamic-grasp-policy"

# 評価コマンドの実行
echo ""
echo "ポリシー評価を開始します..."
echo "注意: RealSenseカメラが接続されていることを確認してください。"
echo ""

lerobot-record \
    --robot.type=so101_follower \
    --robot.port=${ROBOT_PORT} \
    --robot.id=my_awesome_follower_arm \
    --robot.cameras="{ front: {type: intelrealsense, width: 640, height: 480, fps: 30, use_depth: true} }" \
    --dataset.repo_id=${HF_USER}/eval-so101-dynamic-grasp-policy \
    --dataset.num_episodes=20 \
    --dataset.single_task="Evaluate learned policy for dynamic grasping" \
    --policy.path=${HF_USER}/so101-dynamic-grasp-policy \
    --display_data=true \
    --play_sounds=true

echo ""
echo "評価が完了しました。"
echo "評価データセット: ${HF_USER}/eval-so101-dynamic-grasp-policy"
