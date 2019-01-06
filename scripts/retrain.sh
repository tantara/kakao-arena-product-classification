#!/bin/bash

export PYTHONPATH=$PYTHONPATH:`pwd`
GPU_ID=${1}
EXP=${2}
OUTPUT_DIR=/data/output
EXP_DIR=/data/output/${EXP}
BASE_EXP=${3}
BASE_EPOCH=${4}
PRETRAINED=/data/output/${BASE_EXP}/model.ckpt-${BASE_EPOCH}

CUDA_VISIBLE_DEVICES=${GPU_ID} python train.py \
  --tokenizer okt_sub \
  --output_dir ${OUTPUT_DIR} \
  --exp_dir ${EXP_DIR} \
  --total_chunk 20 \
  --train_chunk 20 \
  --val_chunk 1 \
  --pretrained ${PRETRAINED} \
  --is_quant=true \
  --weight_bits 8 \
  --activation_bits 8 \
  --batch_size 1024 \
  --vocab_size 150000 \
  --max_len 20 \
  --postfix "-150000-max20" \
  --zero_based=true \
  --model mlp \
  --bmsd=true \
  --use_b=false \
  --use_m=true \
  --fc_layers 4096 \
  --fc_img_feat 0 \
  --embd_size 256 \
  --embd_avg_type 'divide_by_valid' \
  --embd_init_type uniform \
  --embd_dropout_rate 0.5 \
  --fc_dropout_rate 0.5 \
  --noise_type adv \
  --unigram_adv_noise 0.5 \
  --img_adv_noise 5.0 \
  --optimizer adam \
  --base_lr 0.0005 \
  --decay_steps 1000 \
  --decay_rate 0.995 \
  --staircase=true \
  --momentum 0.9 \
  --epochs 30 \
  --val_per_epoch 2 \
