#!/usr/bin/env bash

CONFIG=$1
GPUS=$2
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29500}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

# Keep distributed workers on the activated Conda interpreter.  On shared
# servers, plain `python` can otherwise import a different user-site PyTorch.
if [ -z "${PYTHON_BIN:-}" ]; then
    if [ -n "${CONDA_PREFIX:-}" ] && [ -x "${CONDA_PREFIX}/bin/python" ]; then
        PYTHON_BIN="${CONDA_PREFIX}/bin/python"
    else
        PYTHON_BIN="${PYTHON:-python}"
    fi
fi
export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"
echo "[dist_train] python: ${PYTHON_BIN}"

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
"${PYTHON_BIN}" -m torch.distributed.run \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --nproc_per_node=$GPUS \
    --master_port=$PORT \
    $(dirname "$0")/train.py \
    $CONFIG \
    --launcher pytorch ${@:3}
