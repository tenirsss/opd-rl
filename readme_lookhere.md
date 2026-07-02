# 复现说明

本仓库在 GitHub 上对应 `tenirsss/opd-rl`，代码基于 `verl-agent`，增加了 SERL/OPD 相关复现实验脚本。

## 1. 配环境

1.1在 repo 根目录执行：

```bash
git clone https://github.com/tenirsss/opd-rl.git
cd opd-rl
bash setup_envA.sh
source ~/venvs/verl-agent/bin/activate
```



1.2配完后可以检查：

```bash
source ~/venvs/verl-agent/bin/activate
bash check_envA.sh
```

1.3 数据默认都放在 repo 根目录的 `data/` 下：

```text
data/json_2.1.1/          # ALFWorld 游戏数据
data/logic/               # ALFWorld PDDL/grammar
data/detectors/           # ALFWorld detector 文件
data/text/train.parquet   # verl dataloader 用的 prompt parquet
data/text/test.parquet
```

如果已经安装/激活环境，但 ALFWorld 或 parquet 数据不对，可以跑只准备数据的脚本，不重配完整环境：

```bash
bash scripts/prepare_alfworld_data.sh
```

默认等价于：

```bash
DATA_DIR=/path/to/opd-rl/data \
ALFWORLD_DATA=/path/to/opd-rl/data \
bash scripts/prepare_alfworld_data.sh
```

如果不要用 repo 内默认数据目录，改用指定的位置准备/检查数据：

```bash
DATA_DIR=/path/to/data \
ALFWORLD_DATA=/path/to/data \
bash scripts/prepare_alfworld_data.sh
```


## 2. 下载模型

基础需要三个模型，第三个 TODO 还需要一个 GRPO 过的 7B teacher：

```text
models/Qwen2.5-1.5B-Instruct
models/Qwen2.5-7B-Instruct
models/GiGPO-Qwen2.5-7B-Instruct-ALFWorld #需要下载，是gigpo后的7b模型，成功率90
models/grpo-qwen2.5-7b-alfworld-global-step-150 # 纯 GRPO 后的 Qwen2.5-7B teacher
```


手动下载命令：

```bash
huggingface-cli download Qwen/Qwen2.5-1.5B-Instruct \
  --local-dir models/Qwen2.5-1.5B-Instruct

huggingface-cli download Qwen/Qwen2.5-7B-Instruct \
  --local-dir models/Qwen2.5-7B-Instruct

huggingface-cli download langfeng01/GiGPO-Qwen2.5-7B-Instruct-ALFWorld \
  --local-dir models/GiGPO-Qwen2.5-7B-Instruct-ALFWorld

hf download siaosiao/grpo-qwen2.5-7b-alfworld-global-step-150 \
  --local-dir models/grpo-qwen2.5-7b-alfworld-global-step-150
```

如果模型放在别的位置：

```bash
MODEL_ROOT=/path/to/models bash scripts/download_models.sh
```

`scripts/download_models.sh` 默认也会下载 GRPO teacher；如果要换成别的 repo：

```bash
GRPO_TEACHER_REPO=other-user/grpo-qwen2.5-7b-alfworld-global-step-150 \
bash scripts/download_models.sh
```

如果不想下载 GRPO teacher：

```bash
DOWNLOAD_GRPO_TEACHER=0 bash scripts/download_models.sh
```

## 3. TODO 实验

1. 验证 SERL-style OPD，teacher 使用 GiGPO 训练过的 7B instruct。

   启动命令：

   ```bash
   bash recipe/serl/serl_opdstyle_1p5b_teacher_gigpo7b.sh
   ```

   换路径：

   ```bash
   STUDENT_PATH=/path/to/Qwen2.5-1.5B-Instruct \
   TEACHER_PATH=/path/to/GiGPO-Qwen2.5-7B-Instruct-ALFWorld \
   bash recipe/serl/serl_opdstyle_1p5b_teacher_gigpo7b.sh
   ```

   默认配置：

   ```text
   student = models/Qwen2.5-1.5B-Instruct
   teacher = models/GiGPO-Qwen2.5-7B-Instruct-ALFWorld
   output  = outputs/serl_opdstyle_1p5b_teacher_gigpo7b
   ```

   跑完后的日志和结果位置：

   ```text
   outputs/serl_opdstyle_1p5b_teacher_gigpo7b/logs/
   outputs/serl_opdstyle_1p5b_teacher_gigpo7b/checkpoints/
   outputs/serl_opdstyle_1p5b_teacher_gigpo7b/checkpoints/final_eval_metrics.json
   outputs/serl_opdstyle_1p5b_teacher_gigpo7b/rollout/
   ```

2. 验证 SERL-style OPD，teacher 使用原始 Qwen2.5 7B instruct。

   启动命令：

   ```bash
   bash recipe/serl/serl_opdstyle_1p5b_teacher_qwen7b.sh
   ```


   换路径：

   ```bash
   STUDENT_PATH=/path/to/Qwen2.5-1.5B-Instruct \
   TEACHER_PATH=/path/to/Qwen2.5-7B-Instruct \
   bash recipe/serl/serl_opdstyle_1p5b_teacher_qwen7b.sh
   ```
   
   默认配置：

   ```text
   student = models/Qwen2.5-1.5B-Instruct
   teacher = models/Qwen2.5-7B-Instruct
   output  = outputs/serl_opdstyle_1p5b_teacher_qwen7b
   ```

   跑完后的日志和结果位置：

   ```text
   outputs/serl_opdstyle_1p5b_teacher_qwen7b/logs/
   outputs/serl_opdstyle_1p5b_teacher_qwen7b/checkpoints/
   outputs/serl_opdstyle_1p5b_teacher_qwen7b/checkpoints/final_eval_metrics.json
   outputs/serl_opdstyle_1p5b_teacher_qwen7b/rollout/
   ```

3. 验证 SERL-style OPD，teacher 使用纯 GRPO 训练过的 Qwen2.5 7B instruct。

   启动命令：

   ```bash
   bash recipe/serl/serl_opdstyle_1p5b_teacher_grpo7b_hf.sh
   ```

   如果还没下载 teacher：

   ```bash
   hf download siaosiao/grpo-qwen2.5-7b-alfworld-global-step-150 \
     --local-dir models/grpo-qwen2.5-7b-alfworld-global-step-150
   ```

   换路径：

   ```bash
   STUDENT_PATH=/path/to/Qwen2.5-1.5B-Instruct \
   TEACHER_PATH=/path/to/grpo-qwen2.5-7b-alfworld-global-step-150 \
   bash recipe/serl/serl_opdstyle_1p5b_teacher_grpo7b_hf.sh
   ```

   默认配置：

   ```text
   student = models/Qwen2.5-1.5B-Instruct
   teacher = models/grpo-qwen2.5-7b-alfworld-global-step-150
   output  = outputs/serl_opdstyle_1p5b_teacher_grpo7b_hf
   ```

   跑完后的日志和结果位置：

   ```text
   outputs/serl_opdstyle_1p5b_teacher_grpo7b_hf/logs/
   outputs/serl_opdstyle_1p5b_teacher_grpo7b_hf/checkpoints/
   outputs/serl_opdstyle_1p5b_teacher_grpo7b_hf/checkpoints/final_eval_metrics.json
   outputs/serl_opdstyle_1p5b_teacher_grpo7b_hf/rollout/
   ```
