# GiGPO的 7B teacher 蒸 1.5B instruct student。

#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_DIR=${REPO_DIR:-$(cd -- "${SCRIPT_DIR}/../.." && pwd)}
MODEL_ROOT=${MODEL_ROOT:-$REPO_DIR/models}
DATA_DIR=${DATA_DIR:-$REPO_DIR/data}
cd "$REPO_DIR"

ENGINE=${ENGINE:-vllm}
if [ $# -gt 0 ] && [[ "$1" != *=* ]]; then
  ENGINE="$1"
  shift
fi
STUDENT_PATH=${STUDENT_PATH:-$MODEL_ROOT/Qwen2.5-1.5B-Instruct}
TEACHER_PATH=${TEACHER_PATH:-$MODEL_ROOT/GiGPO-Qwen2.5-7B-Instruct-ALFWorld}
OUTPUT_ROOT=${OUTPUT_ROOT:-$REPO_DIR/outputs/opd_only_gigpo7b_distill_1p5b}
OPD_COEF=${OPD_COEF:-1.0}

export ALFWORLD_DATA=${ALFWORLD_DATA:-$DATA_DIR}
export VLLM_ATTENTION_BACKEND=${VLLM_ATTENTION_BACKEND:-FLASH_ATTN}
export HF_HUB_ENABLE_HF_TRANSFER=${HF_HUB_ENABLE_HF_TRANSFER:-0}

mkdir -p "$OUTPUT_ROOT/logs"
LOG_FILE="$OUTPUT_ROOT/logs/run_$(date +%Y%m%d_%H%M%S).log"
echo "日志写入: $LOG_FILE"

if [ ! -f "$TEACHER_PATH/config.json" ]; then
  echo "找不到默认 teacher: $TEACHER_PATH" | tee -a "$LOG_FILE"
  echo "如果你的 GiGPO 目录名被截断, 用 TEACHER_PATH=/path/to/model 覆盖后重跑。" | tee -a "$LOG_FILE"
  exit 1
fi

test -f "$STUDENT_PATH/config.json"
test -d "$ALFWORLD_DATA/json_2.1.1/train"
test -d "$ALFWORLD_DATA/json_2.1.1/valid_seen"
grep -q "opd_teacher_path" "$REPO_DIR/verl/trainer/config/ppo_trainer.yaml"
grep -q "opd_only" "$REPO_DIR/verl/workers/actor/dp_actor.py"

train_data_size=${TRAIN_BATCH_SIZE:-16}
val_data_size=${VAL_BATCH_SIZE:-128}
group_size=${GROUP_SIZE:-8}
num_cpus_per_env_worker=${NUM_CPUS_PER_ENV_WORKER:-0.1}

if [ ! -f "$DATA_DIR/text/train.parquet" ] || [ ! -f "$DATA_DIR/text/test.parquet" ]; then
  python3 -m examples.data_preprocess.prepare \
    --mode text \
    --local_dir "$DATA_DIR" \
    --train_data_size "$train_data_size" \
    --val_data_size "$val_data_size" 2>&1 | tee -a "$LOG_FILE"
else
  echo "parquet 已存在, 跳过 prepare: $DATA_DIR/text/" | tee -a "$LOG_FILE"
fi

python3 -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  data.train_files="$DATA_DIR/text/train.parquet" \
  data.val_files="$DATA_DIR/text/test.parquet" \
  data.train_batch_size="$train_data_size" \
  data.val_batch_size="$val_data_size" \
  data.max_prompt_length=2048 \
  data.max_response_length=512 \
  data.filter_overlong_prompts=True \
  data.truncation=error \
  data.return_raw_chat=True \
  actor_rollout_ref.model.path="$STUDENT_PATH" \
  actor_rollout_ref.model.opd_teacher_path="$TEACHER_PATH" \
  actor_rollout_ref.actor.optim.lr=${LR:-1e-6} \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE:-256} \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${PPO_MICRO_BATCH_SIZE_PER_GPU:-32} \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef="$OPD_COEF" \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.actor.opd_only=True \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.fsdp_config.param_offload=False \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${ROLLOUT_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-32} \
  actor_rollout_ref.rollout.tensor_model_parallel_size=${TENSOR_MODEL_PARALLEL_SIZE:-2} \
  actor_rollout_ref.rollout.name="$ENGINE" \
  actor_rollout_ref.rollout.gpu_memory_utilization=${GPU_MEMORY_UTILIZATION:-0.6} \
  actor_rollout_ref.rollout.enable_chunked_prefill=False \
  actor_rollout_ref.rollout.enforce_eager=False \
  actor_rollout_ref.rollout.free_cache_engine=False \
  actor_rollout_ref.rollout.val_kwargs.temperature=${VAL_TEMPERATURE:-0.4} \
  actor_rollout_ref.rollout.val_kwargs.do_sample=True \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${REF_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-32} \
  actor_rollout_ref.ref.fsdp_config.param_offload=False \
  actor_rollout_ref.actor.use_invalid_action_penalty=True \
  actor_rollout_ref.actor.invalid_action_penalty_coef=${INVALID_ACTION_PENALTY_COEF:-0.1} \
  algorithm.use_kl_in_reward=False \
  env.env_name=alfworld/AlfredTWEnv \
  env.seed=${SEED:-0} \
  env.max_steps=${MAX_STEPS:-50} \
  env.rollout.n="$group_size" \
  env.resources_per_worker.num_cpus="$num_cpus_per_env_worker" \
  trainer.critic_warmup=0 \
  trainer.logger=${LOGGER:-['console']} \
  trainer.project_name=verl_agent_alfworld \
  trainer.experiment_name=opd_only_gigpo7b_distill_1p5b \
  trainer.n_gpus_per_node=${N_GPUS_PER_NODE:-4} \
  trainer.nnodes=${NNODES:-1} \
  trainer.save_freq=${SAVE_FREQ:-30} \
  trainer.default_local_dir="$OUTPUT_ROOT/checkpoints" \
  trainer.resume_mode=auto \
  trainer.test_freq=${TEST_FREQ:-5} \
  trainer.total_epochs=${TOTAL_EPOCHS:-150} \
  trainer.val_before_train=${VAL_BEFORE_TRAIN:-False} "$@" 2>&1 | tee -a "$LOG_FILE"
