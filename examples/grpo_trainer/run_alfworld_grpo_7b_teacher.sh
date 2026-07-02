#跑一个grpo的alfworld 7b teacher
#!/usr/bin/env bash
set -x
set -o pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_DIR=${REPO_DIR:-$(cd -- "${SCRIPT_DIR}/../.." && pwd)}
MODEL_ROOT=${MODEL_ROOT:-$REPO_DIR/models}
DATA_DIR=${DATA_DIR:-$REPO_DIR/data}
CKPT_ROOT=${CKPT_ROOT:-$REPO_DIR/ckpts}
MODEL_PATH=${MODEL_PATH:-$MODEL_ROOT/Qwen2.5-7B-Instruct}
cd "$REPO_DIR"

ENGINE=${ENGINE:-vllm}
if [ $# -gt 0 ] && [[ "$1" != *=* ]]; then
    ENGINE="$1"
    shift
fi

export WANDB_MODE=offline
export WANDB_DIR=${WANDB_DIR:-"$(cd "$(dirname "$0")" && pwd)/logs/wandb"}
mkdir -p "$WANDB_DIR"

export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
export HF_HUB_ENABLE_HF_TRANSFER=0

# DATA_DIR is for parquet and, by default, the ALFWorld game cache.
export ALFWORLD_DATA=${ALFWORLD_DATA:-$DATA_DIR}

LOG_DIR=${LOG_DIR:-"$(cd "$(dirname "$0")" && pwd)/logs/alfworld_grpo_7b_teacher"}
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/run_$(date +%Y%m%d_%H%M%S).log"
echo "日志写入: $LOG_FILE"
echo "MODEL_PATH=$MODEL_PATH" | tee "$LOG_FILE"
echo "DATA_DIR=$DATA_DIR" | tee -a "$LOG_FILE"
echo "ALFWORLD_DATA=$ALFWORLD_DATA" | tee -a "$LOG_FILE"

test -f "$MODEL_PATH/config.json"
test -d "$ALFWORLD_DATA/json_2.1.1/train"
test -d "$ALFWORLD_DATA/json_2.1.1/valid_seen"

num_cpus_per_env_worker=0.1
train_data_size=16
val_data_size=128
group_size=8

python3 -m examples.data_preprocess.prepare \
    --mode 'text' \
    --local_dir "$DATA_DIR" \
    --train_data_size $train_data_size \
    --val_data_size $val_data_size 2>&1 | tee -a "$LOG_FILE"

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$DATA_DIR/text/train.parquet \
    data.val_files=$DATA_DIR/text/test.parquet \
    data.train_batch_size=$train_data_size \
    data.val_batch_size=$val_data_size \
    data.max_prompt_length=2048 \
    data.max_response_length=512 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=$ENGINE \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.4 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.use_invalid_action_penalty=True \
    actor_rollout_ref.actor.invalid_action_penalty_coef=0.1 \
    algorithm.use_kl_in_reward=False \
    algorithm.gamma=0.95 \
    env.env_name=alfworld/AlfredTWEnv \
    env.seed=0 \
    env.max_steps=50 \
    env.rollout.n=$group_size \
    env.resources_per_worker.num_cpus=$num_cpus_per_env_worker \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='verl_agent_alfworld' \
    trainer.experiment_name='grpo_teacher_qwen2.5_7b_alfworld' \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=150 \
    trainer.default_local_dir=$CKPT_ROOT/grpo_teacher_qwen2.5_7b_alfworld \
    trainer.resume_mode=auto \
    trainer.test_freq=5 \
    trainer.total_epochs=150 \
    trainer.val_before_train=True $@ 2>&1 | tee -a "$LOG_FILE"
