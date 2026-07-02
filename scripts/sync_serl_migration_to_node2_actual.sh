#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
SRC=${SRC:-$(cd -- "${SCRIPT_DIR}/.." && pwd)}
DST=${DST:-$SRC}

if [[ "$DST" == "$SRC" ]]; then
  echo "SRC and DST are both $SRC; set DST=/path/to/target-repo to sync elsewhere."
  exit 0
fi

copy_file() {
  local rel="$1"
  mkdir -p "$(dirname "$DST/$rel")"
  cp "$SRC/$rel" "$DST/$rel"
}

copy_dir() {
  local rel="$1"
  mkdir -p "$(dirname "$DST/$rel")"
  rm -rf "$DST/$rel"
  cp -a "$SRC/$rel" "$DST/$rel"
}

copy_dir recipe/serl
copy_dir recipe/gigpo
copy_dir judge_utils
copy_file recipe/serl/run_alfworld_node2.sh
copy_file agent_system/multi_turn_rollout/rollout_loop.py
copy_file verl/trainer/ppo/core_algos.py
copy_file verl/trainer/ppo/ray_trainer.py
copy_file verl/workers/actor/dp_actor.py
copy_file verl/workers/fsdp_workers.py
copy_file verl/trainer/config/ppo_trainer.yaml

echo "SERL migration synced to $DST"
