export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export INSTANCE_DIR="./data/data_disney"
export OUTPUT_DIR="./exps/output_dsn"

/home/dsi/hbai/anaconda3/envs/inpaint/bin/lora_pti
--pretrained_model_name_or_path=$MODEL_NAME
--instance_data_dir=$INSTANCE_DIR
--output_dir=$OUTPUT_DIR
--train_text_encoder
--resolution=512
--train_batch_size=1
--gradient_accumulation_steps=4
--scale_lr
--learning_rate_unet=1e-4
--learning_rate_text=1e-5
--learning_rate_ti=5e-4
--color_jitter
--lr_scheduler="linear"
--lr_warmup_steps=0
--placeholder_tokens="|"
--use_template="style"
--save_steps=100
--max_train_steps_ti=1000
--max_train_steps_tuning=1000
--perform_inversion=True
--clip_ti_decay
--weight_decay_ti=0.000
--weight_decay_lora=0.001
--continue_inversion
--continue_inversion_lr=1e-4
--device="cuda:0"
--lora_rank=1 \

