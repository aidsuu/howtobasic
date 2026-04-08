#!/usr/bin/env bash

python src/train_baseline.py --epochs 50 --batch-size 16 --lr 1e-3 --input-mode magnitude_minus_freespace --num-workers 8 --torch-num-threads 16 --device cpu --early-stopping-patience 8 --min-delta 1e-4
python src/train_baseline.py --epochs 50 --batch-size 8 --lr 1e-3 --input-mode real_imag --num-workers 8 --torch-num-threads 16 --device cpu --early-stopping-patience 8 --min-delta 1e-4
