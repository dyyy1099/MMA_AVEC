#!/bin/bash

CUDA_VISIBLE_DEVICES='0' python3 test_model.py \
--model-path /root/autodl-tmp/AVEC/log/AVEC-2508061843-FINAL_224-log/checkpoint/best_model.pth \
--test-annotation ./annotation/test_labels.txt \
--batch-size 8 \
--workers 8 \
--img-size 224 \
--temporal-layers 1
