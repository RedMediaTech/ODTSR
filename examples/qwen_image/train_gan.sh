export qwen_path=path2model/Qwen-Image
export wan_path=path2model/Wan2.1-T2V-1.3B

LS_DIR_TXT="xxxxxxx/datasets/lsdir/lsdir_range1_0_3_range2_1_84991.txt"
FFHQ_TXT="xxxxxxx/datasets/ffhq_pre_1w/ffhq_pre1w_range1_0_3_range2_0_9999.txt"

accelerate launch --num_processes=8 --gpu_ids="0,1,2,3,4,5,6,7" --main_process_port 29300 examples/qwen_image/train_gan.py \
  --mmaigc_dataset_yml "./examples/qwen_image/configs/b32_g002_r1_5_dynamic_lnnp05.yaml" \
  --deg_file_path "./examples/qwen_image/deg_pisa.yaml" \
  --dataset_txt_paths "$LS_DIR_TXT $FFHQ_TXT" \
  --null_text_ratio 0.0001 \
  --task train
