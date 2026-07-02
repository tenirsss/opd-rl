#复现serl的原逻辑
#!/usr/bin/env bash
set -euo pipefail

# SERL README default reproduction:
# Qwen2.5-7B-Instruct + ALFWorld + immediate_feedback + response.

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_DIR=${REPO_DIR:-$(cd -- "${SCRIPT_DIR}/../.." && pwd)}
MODEL_ROOT=${MODEL_ROOT:-$REPO_DIR/models}
DATA_DIR=${DATA_DIR:-$REPO_DIR/data}
cd "$REPO_DIR"

export ALFWORLD_DATA=${ALFWORLD_DATA:-$DATA_DIR}

OUTPUT_ROOT=${OUTPUT_ROOT:-$REPO_DIR/outputs/serl_alfworld_7b}
mkdir -p "$OUTPUT_ROOT/logs"
LOG_FILE="$OUTPUT_ROOT/logs/run_$(date +%Y%m%d_%H%M%S).log"
echo "日志写入: $LOG_FILE"

MODEL_PATH=${MODEL_PATH:-$MODEL_ROOT/Qwen2.5-7B-Instruct} \
TRAIN_FILE=${TRAIN_FILE:-$DATA_DIR/text/train.parquet} \
VAL_FILE=${VAL_FILE:-$DATA_DIR/text/test.parquet} \
OUTPUT_ROOT="$OUTPUT_ROOT" \
SAMPLING_MODE=immediate_feedback \
TRAJECTORY_FORMAT=response \
PPO_MICRO_BATCH_SIZE_PER_GPU=8 \
ROLLOUT_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU=8 \
TENSOR_MODEL_PARALLEL_SIZE=2 \
N_GPUS_PER_NODE=4 \
SAVE_FREQ=150 \
TOTAL_EPOCHS=150 \
LOGGER="['console','tensorboard']" \
bash recipe/serl/run_alfworld_node2.sh 2>&1 | tee "$LOG_FILE"
