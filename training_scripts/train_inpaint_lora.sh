#export MODEL_NAME="runwayml/stable-diffusion-inpainting"
#export MODEL_NAME="runwayml/stable-diffusion-v1-5-inpainting"
#export MODEL_NAME="stabilityai/stable-diffusion-2-inpainting"
#export MODEL_NAME="botp/stable-diffusion-v1-5-inpainting"
#export MODEL_NAME="/home/dsi/hbai/code/runwayml_ckpt/models--runwayml--stable-diffusion-v1-5/snapshots/1d0c4ebf6ff58a5caecab40fa1406526bca4b5b9"
#export MODEL_NAME="booksforcharlie/stable-diffusion-inpainting"
export MODEL_NAME="/home/dsi/hbai/code/runwayml_ckpt/models--runwayml--stable-diffusion-inpainting/snapshots/51388a731f57604945fddd703ecb5c50e8e7b49d"

export INSTANCE_DIR="/home/dsi/hbai/code/UniversalFakeDetect-main/data/diffusion_datasets/ldm_200_cfg/1_fake_thick_random_fine-tune"

export OUTPUT_DIR="./exps/lora_ldm200cfg_0.5k_thick_random_runwayml"

lora_pti \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --train_text_encoder \
  --train_inpainting \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=2 \
  --gradient_checkpointing \
  --scale_lr \
  --learning_rate_unet=2e-4 \
  --learning_rate_text=1e-6 \
  --learning_rate_ti=5e-4 \
  --color_jitter \
  --lr_scheduler="linear" \
  --lr_warmup_steps=0 \
  --lr_scheduler_lora="constant" \
  --lr_warmup_steps_lora=100 \
  --placeholder_tokens="<s1>|<s2>" \
  --placeholder_token_at_data="<krk>|<s1><s2>" \
  --save_steps=500 \
  --max_train_steps_ti=3000 \
  --max_train_steps_tuning=3000 \
  --perform_inversion=True \
  --clip_ti_decay \
  --weight_decay_ti=0.000 \
  --weight_decay_lora=0.000 \
  --device="cuda:1" \
  --lora_rank=8 \
  --use_face_segmentation_condition \
  --lora_dropout_p=0.1 \
  --lora_scale=2.0 \
  --cached_latents=False\
  --cross_attention_dim=8
