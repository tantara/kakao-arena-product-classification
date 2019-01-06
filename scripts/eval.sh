#!/bin/bash

export PYTHONPATH=$PYTHONPATH:`pwd`

GPU_ID=${1}
EXP=${2}
OUTPUT_DIR=/data/output
EXP_DIR=/data/output/${EXP}
VAL_EPOCH=${3}

CUDA_VISIBLE_DEVICES=${GPU_ID} python eval.py \
  --tokenizer okt_sub \
  --output_dir ${OUTPUT_DIR} \
  --exp_dir ${EXP_DIR} \
  --val_epoch ${VAL_EPOCH} \
  --total_chunk 20 \
  --val_chunk 1 \
  --is_quant=false \
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
  --embd_avg_type 'divide_by_valid'
