#!/usr/bin/env bash
set -euo pipefail

# SERL method with an OPD-style fixed external teacher:
# student = Qwen2.5-1.5B-Instruct,
# teacher = GRPO-trained Qwen2.5-7B-Instruct on ALFWorld from HuggingFace/local HF dir.

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_DIR=${REPO_DIR:-$(cd -- "${SCRIPT_DIR}/../.." && pwd)}
MODEL_ROOT=${MODEL_ROOT:-$REPO_DIR/models}
cd "$REPO_DIR"

OUTPUT_ROOT=${OUTPUT_ROOT:-$REPO_DIR/outputs/serl_opdstyle_1p5b_teacher_grpo7b_hf}
STUDENT_PATH=${STUDENT_PATH:-$MODEL_ROOT/Qwen2.5-1.5B-Instruct}
TEACHER_PATH=${TEACHER_PATH:-$MODEL_ROOT/grpo-qwen2.5-7b-alfworld-global-step-150}

mkdir -p "$OUTPUT_ROOT/logs"
LOG_FILE="$OUTPUT_ROOT/logs/run_$(date +%Y%m%d_%H%M%S).log"
echo "日志写入: $LOG_FILE"

if [ ! -f "$STUDENT_PATH/config.json" ]; then
  echo "找不到 student: $STUDENT_PATH" | tee -a "$LOG_FILE"
  exit 1
fi

if [ ! -f "$TEACHER_PATH/config.json" ]; then
  echo "找不到 teacher: $TEACHER_PATH" | tee -a "$LOG_FILE"
  echo "先下载: hf download siaosiao/grpo-qwen2.5-7b-alfworld-global-step-150 --local-dir $TEACHER_PATH" | tee -a "$LOG_FILE"
  echo "或用 TEACHER_PATH=/path/to/model 覆盖后重跑。" | tee -a "$LOG_FILE"
  exit 1
fi

MODEL_PATH="$STUDENT_PATH" \
OUTPUT_ROOT="$OUTPUT_ROOT" \
SAMPLING_MODE=immediate_feedback \
TRAJECTORY_FORMAT=response \
PPO_MICRO_BATCH_SIZE_PER_GPU=32 \
ROLLOUT_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU=32 \
TENSOR_MODEL_PARALLEL_SIZE=2 \
N_GPUS_PER_NODE=8 \
SAVE_FREQ=30 \
TOTAL_EPOCHS=150 \
LOGGER=${LOGGER:-"['console']"} \
bash recipe/serl/run_alfworld_node2.sh \
  actor_rollout_ref.model.opd_teacher_path="$TEACHER_PATH" \
  actor_rollout_ref.actor.serl.sync_teacher=False \
  "$@" 2>&1 | tee "$LOG_FILE"
