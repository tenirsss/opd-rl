#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_DIR=${REPO_DIR:-$(cd -- "${SCRIPT_DIR}/.." && pwd)}
DATA_DIR=${DATA_DIR:-$REPO_DIR/data}

export ALFWORLD_DATA=${ALFWORLD_DATA:-$DATA_DIR}

if [[ ! -d "$ALFWORLD_DATA/json_2.1.1/train" ]]; then
  if ! command -v alfworld-download >/dev/null 2>&1; then
    echo "alfworld-download not found. Activate the project environment first, or run setup_envA.sh." >&2
    exit 127
  fi
  echo "download ALFWorld data to $ALFWORLD_DATA"
  alfworld-download -f
else
  echo "skip existing ALFWorld data: $ALFWORLD_DATA"
fi

test -d "$ALFWORLD_DATA/json_2.1.1/train"
test -d "$ALFWORLD_DATA/json_2.1.1/valid_seen"
test -d "$ALFWORLD_DATA/json_2.1.1/valid_unseen"
test -f "$ALFWORLD_DATA/logic/alfred.pddl"
test -f "$ALFWORLD_DATA/logic/alfred.twl2"

if [[ -f "$DATA_DIR/text/train.parquet" && -f "$DATA_DIR/text/test.parquet" ]]; then
  echo "skip existing parquet: $DATA_DIR/text"
else
  python3 -m examples.data_preprocess.prepare \
    --mode text \
    --local_dir "$DATA_DIR" \
    --train_data_size "${TRAIN_DATA_SIZE:-16}" \
    --val_data_size "${VAL_DATA_SIZE:-128}"
fi

test -f "$DATA_DIR/text/train.parquet"
test -f "$DATA_DIR/text/test.parquet"

echo "ALFWorld and parquet data ready"
