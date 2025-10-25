#!/bin/bash

# 3D認識型ポリシーのトレーニングスクリプト
# 使用方法: ./train_3d_policy.sh

# 環境変数の設定
export HF_USER="teradakousuke"

# LeRobot環境のアクティベート
source ~/opt/anaconda3/envs/lerobot/bin/activate

echo "=== 3D認識型ポリシーのトレーニング ==="
echo "Hugging Face User: $HF_USER"
echo ""

# データセットの確認
echo "データセットを確認中..."
lerobot-dataset-viz --repo-id ${HF_USER}/so101-dynamic-grasp --episode-index 0

if [ $? -ne 0 ]; then
    echo "エラー: データセットが見つかりません。先にデータ収集を実行してください。"
    exit 1
fi

echo ""
echo "3D認識型ポリシーのトレーニングを開始します..."
echo "注意: このプロセスには時間がかかる場合があります。"
echo ""

# トレーニングコマンドの実行
lerobot-train \
    --dataset.repo_id=${HF_USER}/so101-dynamic-grasp \
    --policy.type=diffusion_policy_3d \
    --policy.device=mps \
    --policy.horizon=16 \
    --policy.n_action_steps=8 \
    --policy.vision_backbone=resnet18 \
    --policy.crop_shape="[128, 128]" \
    --policy.use_separate_rgb_encoder_per_camera=false \
    --policy.down_dims="[256, 512, 1024]" \
    --policy.kernel_size=5 \
    --policy.n_groups=8 \
    --policy.diffusion_step_embed_dim=128 \
    --policy.use_film_scale_modulation=true \
    --policy.noise_scheduler_type=DDPM \
    --policy.num_train_timesteps=1000 \
    --policy.beta_schedule=linear \
    --policy.beta_start=0.0001 \
    --policy.beta_end=0.02 \
    --policy.prediction_type=epsilon \
    --policy.clip_sample=true \
    --policy.clip_sample_range=1.0 \
    --policy.num_inference_steps=10 \
    --policy.do_mask_loss_for_padding=true \
    --policy.optimizer_lr=0.0001 \
    --policy.optimizer_betas="[0.9, 0.999]" \
    --policy.optimizer_eps=1e-8 \
    --policy.optimizer_weight_decay=0.01 \
    --policy.scheduler_name=cosine \
    --policy.scheduler_warmup_steps=100 \
    --wandb.enable=true \
    --policy.repo_id=${HF_USER}/so101-dynamic-grasp-policy \
    --job_name=train-3d-diffusion-on-so101

echo ""
echo "トレーニングが完了しました。"
echo "ポリシー: ${HF_USER}/so101-dynamic-grasp-policy"
