#!/bin/bash

SCANS=( \
        "scan001" \
        "scan004" \
        "scan009" \
        "scan010" \
        "scan011" \
        "scan012" \
        "scan013" \
        "scan015" \
        "scan023" \
        "scan024" \
        "scan029" \
        "scan032" \
        "scan033" \
        "scan034" \
        "scan048" \
        "scan049" \
        "scan062" \
        "scan075" \
        "scan077" \
        "scan110" \
        "scan114" \
        "scan118" \
    )

### Run with sparse COLMAP points and dense MVSNet points
for SCAN in ${SCANS[@]}
do
    CUDA_VISIBLE_DEVICE=$1 python train.py -s /mnt/Drive1/Results/DTU/MVSNet/${SCAN} -m /mnt/Drive1/Results/DTU/2DGS/sparse/${SCAN} --input_ply_file ${SCAN}_sparse.ply
    CUDA_VISIBLE_DEVICE=$1 python render.py -s /mnt/Drive1/Results/DTU/MVSNet/${SCAN} -m /mnt/Drive1/Results/DTU/2DGS/sparse/${SCAN}

    CUDA_VISIBLE_DEVICE=$1 python train.py -s /mnt/Drive1/Results/DTU/MVSNet/${SCAN} -m /mnt/Drive1/Results/DTU/2DGS/dense/${SCAN} --input_ply_file ${SCAN}.ply
    CUDA_VISIBLE_DEVICE=$1 python render.py -s /mnt/Drive1/Results/DTU/MVSNet/${SCAN} -m /mnt/Drive1/Results/DTU/2DGS/dense/${SCAN}
done
