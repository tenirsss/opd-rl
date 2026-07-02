#!/usr/bin/env bash
# ============================================================
# Env A - main verl-agent environment (Py3.12)
# verl + vllm0.11 + flash-attn2.8.3 + tensorboard + ALFWorld + parquet data
# Usage: bash setup_envA.sh
# ============================================================
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="${VENV:-$HOME/venvs/verl-agent}"
DATA_DIR="${DATA_DIR:-$REPO/data}"
ALFWORLD_DATA="${ALFWORLD_DATA:-$DATA_DIR}"
FA_WHEEL="${FA_WHEEL:-https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3%2Bcu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl}"

echo ">>> [1/7] create and activate venv: $VENV (Py3.12)"
uv venv "$VENV" --python 3.12
# shellcheck disable=SC1091
source "$VENV/bin/activate"

echo ">>> [2/7] install vllm==0.11.0 (brings torch 2.8.0+cu128)"
cd "$REPO"
uv pip install vllm==0.11.0

echo ">>> [3/7] install flash-attn prebuilt wheel"
uv pip install "$FA_WHEEL"
python -c "import flash_attn; print('flash-attn OK:', flash_attn.__version__)"

echo ">>> [4/7] install verl-agent and logging deps"
uv pip install -e .
uv pip install tensorboard

echo ">>> [5/7] ALFWorld"
uv pip install gymnasium==0.29.1 stable-baselines3==2.6.0 alfworld
mkdir -p "$ALFWORLD_DATA"
export ALFWORLD_DATA
alfworld-download -f

if grep -q '^export ALFWORLD_DATA=' "$VENV/bin/activate" 2>/dev/null; then
  sed -i "s#^export ALFWORLD_DATA=.*#export ALFWORLD_DATA=\"$ALFWORLD_DATA\"#" "$VENV/bin/activate"
else
  echo "export ALFWORLD_DATA=\"$ALFWORLD_DATA\"" >> "$VENV/bin/activate"
fi

echo "    check ALFWorld data: $ALFWORLD_DATA"
test -d "$ALFWORLD_DATA/json_2.1.1/train"
test -d "$ALFWORLD_DATA/json_2.1.1/valid_seen"
test -d "$ALFWORLD_DATA/json_2.1.1/valid_unseen"
test -f "$ALFWORLD_DATA/logic/alfred.pddl"
test -f "$ALFWORLD_DATA/logic/alfred.twl2"
python -c "import alfworld; print('alfworld OK')"

echo ">>> [6/7] verl-agent text parquet"
if [[ -f "$DATA_DIR/text/train.parquet" && -f "$DATA_DIR/text/test.parquet" ]]; then
  echo "    parquet exists, skip: $DATA_DIR/text"
else
  python3 -m examples.data_preprocess.prepare \
    --mode text \
    --local_dir "$DATA_DIR" \
    --train_data_size "${TRAIN_DATA_SIZE:-16}" \
    --val_data_size "${VAL_DATA_SIZE:-128}"
fi
test -f "$DATA_DIR/text/train.parquet"
test -f "$DATA_DIR/text/test.parquet"

echo ">>> [7/7] Sokoban + Gym Cards"
uv pip install "gym==0.26.2" gym_sokoban==0.0.6 matplotlib
uv pip install -e ./agent_system/environments/env_package/gym_cards/gym-cards/

MPL=$(python -c "import matplotlib,os;print(os.path.join(os.path.dirname(matplotlib.__file__),'mpl-data/fonts/ttf'))")
mkdir -p /usr/share/fonts/dejavu
cp "$MPL/DejaVuSans.ttf" "$MPL/DejaVuSans-Bold.ttf" /usr/share/fonts/dejavu/ 2>/dev/null \
  && echo "    DejaVu fonts are ready (/usr/share/fonts/dejavu)" \
  || echo "    !! font copy failed; gym_cards rendering may need DejaVu fonts manually"

echo ""
echo ">>> Env A done. Activate later with: source $VENV/bin/activate"
