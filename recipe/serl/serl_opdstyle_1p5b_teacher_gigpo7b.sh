# 保留serl的逻辑，将opsd换成opd，只是换teacher：teacher 换成 GiGPO/RL 过的 7B：student = Qwen2.5-1.5B-Instruct，teacher = GiGPO-Qwen2.5-7B-Instruct-ALFWorld
#!/usr/bin/env bash
set -euo pipefail

# SERL method with an OPD-style fixed external teacher:
# student = Qwen2.5-1.5B-Instruct, teacher = GiGPO-Qwen2.5-7B-Instruct-ALFWorld.

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_DIR=${REPO_DIR:-$(cd -- "${SCRIPT_DIR}/../.." && pwd)}
MODEL_ROOT=${MODEL_ROOT:-$REPO_DIR/models}
cd "$REPO_DIR"

OUTPUT_ROOT=${OUTPUT_ROOT:-$REPO_DIR/outputs/serl_opdstyle_1p5b_teacher_gigpo7b}
STUDENT_PATH=${STUDENT_PATH:-$MODEL_ROOT/Qwen2.5-1.5B-Instruct}
TEACHER_PATH=${TEACHER_PATH:-$MODEL_ROOT/GiGPO-Qwen2.5-7B-Instruct-ALFWorld}

mkdir -p "$OUTPUT_ROOT/logs"
LOG_FILE="$OUTPUT_ROOT/logs/run_$(date +%Y%m%d_%H%M%S).log"
echo "日志写入: $LOG_FILE"

if [ ! -f "$TEACHER_PATH/config.json" ]; then
  echo "找不到 teacher: $TEACHER_PATH" | tee -a "$LOG_FILE"
  echo "如果 GiGPO 目录名被截断, 用 TEACHER_PATH=/path/to/model 覆盖后重跑。" | tee -a "$LOG_FILE"
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
