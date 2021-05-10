#!/usr/bin/env bash

set -x
NGPUS="8"
PY_ARGS="--cfg_file tools/cfgs/kitti_models/pvcae_rcnn.yaml"

python -m torch.distributed.launch --nproc_per_node=${NGPUS} tools/train.py --launcher pytorch ${PY_ARGS}

