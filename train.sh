#!/bin/bash

CUDA_VISIBLE_DEVICES='0' python3 main.py \
--dataset 'AVEC' \
--workers 8 \
--epochs 20 \
--batch-size 4 \
--lr 1e-3 \
--weight-decay 1e-4 \
--print-freq 10 \
--milestones 25 \
--temporal-layers 1 \
--img-size 160 \
--exper-name FINAL_224 \
