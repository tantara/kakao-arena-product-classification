#!/bin/bash

SPLIT=${1} # train, dev, test
TAGGER=okt # mecab, whitespace
DATA_ROOT=/data
# |- train.chunks.*
# |- dev.chunks.*
# `- test.chunks.*
OUTPUT_ROOT=/data/output_${TAGGER}_sub
# |- train.*.tfrecords
# |- dev.*.tfrecords
# `- test.*.tfrecords

# preprocess train set
if [ "${SPLIT}" == "train" ]; then
  python generate_token.py --split train \
    --num_chunk 9 \
    --tagger $TAGGER \
    --data_root $DATA_ROOT \
    --output_root $OUTPUT_ROOT/tmp

  python split_and_shuffle.py --split train \
    --num_input_chunk 9 \
    --num_output_chunk 20 \
    --input_root $OUTPUT_ROOT/tmp \
    --output_root $OUTPUT_ROOT/tmp

  python build_tfrecord.py --split train \
    --num_chunk 20 \
    --input_root $OUTPUT_ROOT/tmp \
    --output_root $OUTPUT_ROOT

  rm -rf $OUTPUT_ROOT/tmp
fi

# preprocess dev set
if [ "${SPLIT}" == "dev" ]; then
  python generate_token.py --split dev \
    --num_chunk 1 \
    --tagger $TAGGER \
    --data_root=$DATA_ROOT \
    --output_root=$OUTPUT_ROOT/tmp

  python split_and_shuffle.py --split dev \
    --num_input_chunk 1 \
    --num_output_chunk 1 \
    --shuffle false \
    --input_root=$OUTPUT_ROOT/tmp \
    --output_root=$OUTPUT_ROOT/tmp

  python build_tfrecord.py --split dev \
    --num_chunk 1 \
    --shuffle false \
    --input_root=$OUTPUT_ROOT/tmp \
    --output_root=$OUTPUT_ROOT

  rm -rf $OUTPUT_ROOT/tmp
fi

# preprocess test set
if [ "${SPLIT}" == "test" ]; then
  python generate_token.py --split test \
    --num_chunk 2 \
    --tagger $TAGGER \
    --data_root=$DATA_ROOT \
    --output_root=$OUTPUT_ROOT/tmp

  python split_and_shuffle.py --split test \
    --num_input_chunk 2 \
    --num_output_chunk 1 \
    --shuffle false \
    --input_root=$OUTPUT_ROOT/tmp \
    --output_root=$OUTPUT_ROOT/tmp

  python build_tfrecord.py --split test \
    --num_chunk 1 \
    --shuffle false \
    --input_root=$OUTPUT_ROOT/tmp \
    --output_root=$OUTPUT_ROOT

  rm -rf $OUTPUT_ROOT/tmp
fi
