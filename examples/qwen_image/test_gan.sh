#!/usr/bin/env bash
export qwen_path=path2model/Qwen-Image

EXP_DATE="20251030-035453"
ALIGN_METHOD="wavelet" # adain no

CUDA_VISIBLE_DEVICES="0" python examples/qwen_image/test_sr.py \
  --input_path inputdir_up_to_you \
  --output_path outputdir_up_to_you \
  --trained_ckpt experiments/qwen_one_step_gan/${EXP_DATE}/checkpoints/net_gen_iter_10001.pth \
  --scale 4.0 \
  --cfg 1.0 \
  --align_method ${ALIGN_METHOD}
