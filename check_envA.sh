#!/usr/bin/env bash
# 节点环境体检 —— 对照 setup_envA.sh 逐项验证(core 包 / flash-attn / ALFWorld 数据 / 模型 / OPD 代码 / 环境能否 reset)
# 用法: 先 source ~/venvs/verl-agent/bin/activate, 再 bash check_envA.sh
set +e

REPO_DIR=${REPO_DIR:-$(pwd)}
M_DIR=${MODEL_ROOT:-$REPO_DIR/models}
DATA_DIR=${DATA_DIR:-$REPO_DIR/data}
ALFWORLD_DATA=${ALFWORLD_DATA:-$DATA_DIR}

echo "========== 1) 核心包 + 版本(对应 setup [2][3][4][5]) =========="
python - <<'PY'
import importlib, os
for m in ["torch","flash_attn","vllm","datasets","transformers","verl","gymnasium","alfworld"]:
    try:
        mod=importlib.import_module(m); print(f"  OK  {m:13}{getattr(mod,'__version__','')}")
    except Exception as e:
        print(f"  XX  {m:13}{type(e).__name__}: {e}")
import torch; print("  cuda_available:", torch.cuda.is_available(), "| torch", torch.__version__)
import verl; print("  verl_path:", os.path.dirname(verl.__file__))
PY

echo "========== 2) flash_attn 真能用(catch undefined symbol / ABI 不匹配) =========="
python -c "import flash_attn, flash_attn_2_cuda; print('  flash_attn 底层 OK', flash_attn.__version__)" 2>&1 | tail -2

echo "========== 3) ALFWorld 数据(对应 setup [5]) =========="
echo "  DATA_DIR=$DATA_DIR"
echo "  ALFWORLD_DATA=$ALFWORLD_DATA"
echo -n "  train 游戏数: ";       ls "$ALFWORLD_DATA"/json_2.1.1/train       2>/dev/null | wc -l
echo -n "  valid_seen 游戏数: ";  ls "$ALFWORLD_DATA"/json_2.1.1/valid_seen  2>/dev/null | wc -l
echo -n "  valid_unseen 游戏数: ";ls "$ALFWORLD_DATA"/json_2.1.1/valid_unseen 2>/dev/null | wc -l
[ -f "$ALFWORLD_DATA/logic/alfred.pddl" ] && echo "  OK  PDDL/grammar" || echo "  XX  缺 logic/alfred.pddl"
[ -f "$DATA_DIR/text/train.parquet" ] && echo "  OK  train.parquet" || echo "  XX  缺 $DATA_DIR/text/train.parquet"
[ -f "$DATA_DIR/text/test.parquet" ] && echo "  OK  test.parquet" || echo "  XX  缺 $DATA_DIR/text/test.parquet"

echo "========== 4) 模型(student + teacher) =========="
for M in "$M_DIR/Qwen2.5-1.5B-Instruct" "$M_DIR/Qwen2.5-7B-Instruct"; do
  [ -f "$M/config.json" ] && echo "  OK  $M" || echo "  XX  缺 $M"
done

echo "========== 5) OPD 代码在"运行的"verl 里(3 处) =========="
VERL=$(python -c "import verl,os;print(os.path.dirname(verl.__file__))")
grep -q opd_teacher_path "$VERL/trainer/config/ppo_trainer.yaml"  && echo "  OK  opd_teacher_path (yaml)"      || echo "  XX  yaml 缺 opd_teacher_path"
grep -q opd_teacher_path "$VERL/workers/fsdp_workers.py"          && echo "  OK  ref->teacher (fsdp_workers)" || echo "  XX  fsdp_workers 缺"
grep -q opd_only          "$VERL/workers/actor/dp_actor.py"       && echo "  OK  opd_only (dp_actor)"          || echo "  XX  dp_actor 缺 opd_only"

echo ""
echo "========== 6) [可选/最强] ALFWorld 环境端到端 reset =========="
echo "  注意: 这步会起一个 Ray + 占一点资源; 若正有训练在跑, 先别跑这步(会抢资源)。"
echo "  要跑就手动执行(在 repo 根目录):"
echo "    cd $VERL/.. && python -c \"from agent_system.environments.env_package.alfworld import build_alfworld_envs as B; print(type(B('agent_system/environments/env_package/alfworld/configs/config_tw.yaml',seed=1,env_num=1,group_n=1,is_train=False,env_kwargs={'eval_dataset':'eval_in_distribution'},resources_per_worker={'num_cpus':0.05,'num_gpus':0.0}).reset()))\""
echo "  打印出 <class 'tuple'> = 环境+数据端到端 OK。"

echo ""
echo "判定: 1~5 全 OK = 环境/数据没问题。哪条 XX 就是那块缺。"
