#!/bin/bash

export PYTHONPATH=$PYTHONPATH:`pwd`

GPU_ID=${1}
EXP=${2}
OUTPUT_DIR=/data/output
EXP_DIR=/data/output/${EXP}
VAL_EPOCH=${3}
DATA_SPLIT=${4}
if [ "${DATA_SPLIT}" == "dev" ]; then
  TOTAL_CHUNK=1
fi
if [ "${DATA_SPLIT}" == "test" ]; then
  TOTAL_CHUNK=1
fi
EVAL_TYPE="acc_first"
#EVAL_TYPE="label_first"
#EVAL_TYPE="label_only"

CUDA_VISIBLE_DEVICES=${GPU_ID} python inference.py \
  --tokenizer okt_sub \
  --output_dir ${OUTPUT_DIR} \
  --exp_dir ${EXP_DIR} \
  --data_split ${DATA_SPLIT} \
  --total_chunk ${TOTAL_CHUNK} \
  --val_epoch ${VAL_EPOCH} \
  --is_quant=false \
  --vocab_size 150000 \
  --max_len 20 \
  --postfix "-150000-max20" \
  --zero_based=true \
  --model mlp \
  --bmsd=true \
  --bmsd_eval_type ${EVAL_TYPE} \
  --use_b=false \
  --use_m=true \
  --fc_layers 4096 \
  --fc_img_feat 0 \
  --embd_size 256 \
  --embd_avg_type 'divide_by_valid'

zip -r ${EXP_DIR}/`date +%Y-%m-%d`-${EXP}-${DATA_SPLIT}-from-${VAL_EPOCH}-${EVAL_TYPE}.zip ${EXP_DIR}/submission.${DATA_SPLIT}-from-${VAL_EPOCH}-${EVAL_TYPE}.tsv
