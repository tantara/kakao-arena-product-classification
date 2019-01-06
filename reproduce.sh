#!/bin/bash

# generate train dataset
cd data
bash convert.sh train

# train a model
cd ..
bash scripts/train.sh 0 interim
bash scripts/retrain.sh 0 final interim 70

# generate dev, test dataset
cd data
bash convert.sh dev
bash convert.sh test

# evaluation
cd ..
bash scripts/inference.sh 0 final 12 dev
bash scripts/inference.sh 0 final 12 test
