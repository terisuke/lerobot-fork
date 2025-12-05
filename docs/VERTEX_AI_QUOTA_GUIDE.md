# Vertex AI GPU クォータガイド

## クォータの確認

Vertex AIでGPUを使用したトレーニングを実行するには、適切なクォータが必要です。

### 現在のクォータ状況

```
Vertex AI API
Custom model training Nvidia T4 GPUs per region
割り当て: 6
リージョン: asia-northeast1
```

### クォータの確認方法

```bash
# 現在のクォータを確認
gcloud compute project-info describe --project=lerobot-480101 \
  --format="table(quotas.metric,quotas.limit,quotas.usage)"

# Vertex AIのクォータを確認
gcloud alpha services quota list \
  --service=aiplatform.googleapis.com \
  --consumer=projects/lerobot-480101 \
  --filter="metric:aiplatform.googleapis.com/nvidia_t4_gpus_per_region"
```

## GPUトレーニングの実行可能性

### クォータ要件

- **要求**: `NVIDIA_TESLA_T4` x 1
- **利用可能なクォータ**: 6 (asia-northeast1)
- **結論**: ✅ **GPUトレーニングは実行可能です**

### 重要な注意点

1. **リージョンの一致**:
   - クォータは `asia-northeast1` リージョンに設定されています
   - トレーニングジョブも同じリージョンで実行する必要があります
   - スクリプトに `--region="asia-northeast1"` を指定してください

2. **その他の必要なクォータ**:
   - CPUクォータ（n1-standard-8など）
   - メモリクォータ
   - ネットワーククォータ

3. **同時実行**:
   - クォータが6なので、理論的には最大6つのT4 GPUジョブを同時実行可能
   - ただし、他のリソース（CPU、メモリ）のクォータも確認が必要

## 正しいコマンド例

### asia-northeast1リージョンで実行

```bash
./scripts/vertex_ai_train.sh \
  --job-name="lerobot-training-$(date +%Y%m%d-%H%M%S)" \
  --dataset-path="gs://lerobot-datasets-480101/your-dataset" \
  --output-path="gs://lerobot-models-480101/outputs" \
  --config="configs/vertex_ai_example.yaml" \
  --machine-type="n1-standard-8" \
  --accelerator-type="NVIDIA_TESLA_T4" \
  --accelerator-count=1 \
  --region="asia-northeast1"
```

### クォータ不足の場合の対処

もしクォータ不足エラーが発生した場合：

1. **現在の使用状況を確認**:

   ```bash
   gcloud ai custom-jobs list --region=asia-northeast1
   ```

2. **他のジョブをキャンセル**:

   ```bash
   gcloud ai custom-jobs cancel JOB_ID --region=asia-northeast1
   ```

3. **クォータの増加をリクエスト**:
   - Google Cloud Consoleでクォータ増加をリクエスト
   - または、別のリージョン（us-central1など）で実行

## リージョン別のクォータ

### asia-northeast1（東京）

- **T4 GPU**: 6
- **推奨**: 日本のユーザーには低レイテンシー

### us-central1（アイオワ）

- **デフォルトリージョン**
- **推奨**: グローバルな使用、より多くのリソースが利用可能

### リージョンの変更

スクリプトに `--region` オプションを追加しました：

```bash
# asia-northeast1で実行
./scripts/vertex_ai_train.sh \
  --region="asia-northeast1" \
  ...

# us-central1で実行（デフォルト）
./scripts/vertex_ai_train.sh \
  --region="us-central1" \
  ...
```

## トラブルシューティング

### エラー: "Quota exceeded"

**原因**: リージョンのクォータが不足している

**解決策**:

1. 別のリージョンで実行（クォータがある場合）
2. 既存のジョブをキャンセル
3. クォータの増加をリクエスト

### エラー: "Resource not found in region"

**原因**: リージョンが正しく指定されていない、またはリソースが存在しない

**解決策**:

1. `--region` オプションで正しいリージョンを指定
2. データセットやバケットが同じリージョンにあるか確認

### エラー: "Accelerator type not available"

**原因**: 指定したリージョンでGPUタイプが利用できない

**解決策**:

1. 利用可能なGPUタイプを確認:
   ```bash
   gcloud compute accelerator-types list --filter="zone:asia-northeast1-*"
   ```
2. 別のGPUタイプを試す（V100、A100など）
3. 別のリージョンで実行

## 参考資料

- [Vertex AI クォータ](https://cloud.google.com/vertex-ai/docs/general/quotas)
- [GPU リージョン別の利用可能性](https://cloud.google.com/compute/docs/gpus/gpu-regions-zones)
- [クォータの増加リクエスト](https://cloud.google.com/docs/quota#request_increase)
