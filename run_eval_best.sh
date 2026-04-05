#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: bash run_eval_best.sh /path/to/data [python_bin]" >&2
  exit 1
fi

DATA_ROOT="$1"
PYTHON_BIN="${2:-python}"

exec "$PYTHON_BIN" test.py \
  --config_file configs/BallShow/vit_transreid_stride.yml \
  MODEL.PRETRAIN_PATH pretrained/jx_vit_base_p16_224-80ecf9dd.pth \
  DATASETS.ROOT_DIR "$DATA_ROOT" \
  OUTPUT_DIR outputs/s2ft_stride256_sie3_ls_lr4e3_re05 \
  TEST.WEIGHT outputs/s2ft_stride256_sie3_ls_lr4e3_re05/transformer_120.pth \
  MODEL.SIE_CAMERA True \
  MODEL.SIE_COE 3.0 \
  MODEL.IF_LABELSMOOTH on \
  INPUT.RE_PROB 0.5 \
  SOLVER.IMS_PER_BATCH 64 \
  TEST.IMS_PER_BATCH 256
