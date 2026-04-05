# TransReID BallShow Clean Package

This package contains the cleaned code and best-performing run for `s2ft_stride256_sie3_ls_lr4e3_re05`.

Best confirmed result:
- `mAP`: `91.5`
- `Rank-1`: `93.6`
- Source run: `stage2 / s2ft_stride256_sie3_ls_lr4e3_re05`

What is included:
- Merged code fixes needed to run BallShow on the real dataset layout
- Minimal training/evaluation code paths
- `vit_transreid_stride.yml`
- ImageNet pretrained ViT weight
- Best checkpoint and evaluation logs

What is intentionally removed:
- Dataset files
- Other sweep runs and unrelated logs
- Extra launch scripts used only for large-scale tuning

Important merged fixes already included:
- BallShow dataset root resolves both `BallShow` and `Occluded_Duke`
- Multi-digit camera IDs are parsed correctly
- Global camera-id remapping fixes SIE camera embedding overflow
- `CUDA_VISIBLE_DEVICES` is respected during train/test launch
- `finetune` checkpoint loading is supported

Package layout:
- `configs/BallShow/vit_transreid_stride.yml`: base config
- `outputs/s2ft_stride256_sie3_ls_lr4e3_re05/transformer_120.pth`: best checkpoint
- `outputs/s2ft_stride256_sie3_ls_lr4e3_re05/*.log`: train/eval logs
- `pretrained/jx_vit_base_p16_224-80ecf9dd.pth`: ImageNet pretrained weight

Dataset is not bundled. Point `DATASETS.ROOT_DIR` to your BallShow data root.

Evaluation example:

```bash
bash run_eval_best.sh /path/to/data
```

Equivalent command:

```bash
python test.py \
  --config_file configs/BallShow/vit_transreid_stride.yml \
  MODEL.PRETRAIN_PATH pretrained/jx_vit_base_p16_224-80ecf9dd.pth \
  DATASETS.ROOT_DIR /path/to/data \
  OUTPUT_DIR outputs/s2ft_stride256_sie3_ls_lr4e3_re05 \
  TEST.WEIGHT outputs/s2ft_stride256_sie3_ls_lr4e3_re05/transformer_120.pth \
  MODEL.SIE_CAMERA True \
  MODEL.SIE_COE 3.0 \
  MODEL.IF_LABELSMOOTH on \
  INPUT.RE_PROB 0.5 \
  SOLVER.IMS_PER_BATCH 64 \
  TEST.IMS_PER_BATCH 256
```
