#!/bin/bash

set -ex

export RUN_NAME="${RUN_NAME:-diffrun101}"
export GIN_CONFIG="${GIN_CONFIG:-configs/diffrun101.gin}"
export BUCKET="${BUCKET:-dota-euw4a}"
export DATASET_FILES="${DATASET_FILES:-datasets/octo2k/octo2k-0*}"
export MODEL_DIR="${MODEL_DIR:-runs}"
export TPU_NAME="${TPU_NAME:-tpu-v3-8-euw4a-200}"

exec bash scripts/run_diffrun100.sh
