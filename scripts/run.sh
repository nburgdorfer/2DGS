#!/bin/bash

DATASET=$1
MODEL=$2
DEVICES=$3
CONFIG_PATH=configs/${MODEL}/${DATASET}/

CUDA_VISIBLE_DEVICES=${DEVICES} python -W ignore -u run.py \
											--config_path $CONFIG_PATH \
											--dataset $DATASET \
											--model $MODEL
