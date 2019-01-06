#!/usr/bin/env bash

BASE_PATH=`pwd`
DATA_PATH=/path # FIXME
SSH_PORT=8706
JUPYTER_PORT=8707
TFBOARD_PORT=8708
PRJ_NAME=kakao-arena18-tf-v1

nvidia-docker run -it -d \
  -p $SSH_PORT:22 \
  -p $JUPYTER_PORT:8888 \
  -p $TFBOARD_PORT:8008 \
  -v $BASE_PATH:/base \
  -v $DATA_PATH:/data \
  --shm-size 16G \
  --name $PRJ_NAME tantara/kakao-arena18-tf
