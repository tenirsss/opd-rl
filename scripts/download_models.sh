#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_DIR=${REPO_DIR:-$(cd -- "${SCRIPT_DIR}/.." && pwd)}
MODEL_ROOT=${MODEL_ROOT:-$REPO_DIR/models}

export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
mkdir -p "$MODEL_ROOT"

download_model() {
  local repo_id="$1"
  local local_name="$2"
  local local_dir="$MODEL_ROOT/$local_name"

  if [[ -f "$local_dir/config.json" ]]; then
    echo "skip existing: $local_dir"
    return
  fi

  echo "download $repo_id -> $local_dir"
  huggingface-cli download "$repo_id" --local-dir "$local_dir"
}

download_model Qwen/Qwen2.5-1.5B-Instruct Qwen2.5-1.5B-Instruct
download_model Qwen/Qwen2.5-7B-Instruct Qwen2.5-7B-Instruct

if [[ "${DOWNLOAD_GIGPO_TEACHER:-1}" == "1" ]]; then
  download_model langfeng01/GiGPO-Qwen2.5-7B-Instruct-ALFWorld GiGPO-Qwen2.5-7B-Instruct-ALFWorld
fi

GRPO_TEACHER_REPO=${GRPO_TEACHER_REPO:-siaosiao/grpo-qwen2.5-7b-alfworld-global-step-150}
if [[ "${DOWNLOAD_GRPO_TEACHER:-1}" == "1" ]]; then
  download_model "$GRPO_TEACHER_REPO" grpo-qwen2.5-7b-alfworld-global-step-150
fi

echo "models ready under $MODEL_ROOT"
