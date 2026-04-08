# Final Results

## Project goal
The goal of this project is to build a stable deep learning baseline for GPR inversion on the MERL-GPR dataset and predict the dielectric permittivity map `eps` from simulated frequency-domain field observations.

## Dataset summary
- Main dataset file: `data/MERLGPR/MERLGPR/data/v1/data_frequency.h5`
- Number of samples: `400`
- Each sample contains:
  - `eps`: shape `(63, 63)`, dtype `float64`
  - `f_field`: shape `(50, 63, 63)`, dtype `complex128`
- Background file: `data/MERLGPR/MERLGPR/data/v1/data_frequency_freespace.h5`
- Official split from `data/merl_index.csv`:
  - train: `280`
  - val: `60`
  - test: `60`

## Input mode summary
- `magnitude`
  - best validation loss: `0.457512`
  - test MSE: `0.488006`
  - test MAE: `0.392868`
  - test RMSE: `0.698574`
- `magnitude_minus_freespace`
  - best validation loss: `0.394874`
  - test MSE: `0.494521`
  - test MAE: `0.410393`
  - test RMSE: `0.703222`
- `real_imag`
  - best validation loss: `0.152178`
  - test MSE: `0.184736`
  - test MAE: `0.203690`
  - test RMSE: `0.429810`

## Main results table

| Input mode | Best epoch | Best val loss | Test MSE | Test MAE | Test RMSE |
| --- | ---: | ---: | ---: | ---: | ---: |
| magnitude | 48 | 0.457512 | 0.488006 | 0.392868 | 0.698574 |
| magnitude_minus_freespace | 50 | 0.394874 | 0.494521 | 0.410393 | 0.703222 |
| real_imag | 50 | 0.152178 | 0.184736 | 0.203690 | 0.429810 |

## Why `real_imag` was selected
`real_imag` was selected because it is the best configuration on every important metric among the tested modes. It achieves the lowest validation loss, the lowest test MSE, the lowest test MAE, and the lowest test RMSE. In contrast, background subtraction slightly improved validation loss over plain magnitude, but it did not improve the final test metrics.

## Final training configuration
- Environment: `.venv`
- Device: `cpu`
- Input mode: `real_imag`
- Epochs: `50`
- Batch size: `8`
- Learning rate: `1e-3`
- Num workers: `8`
- PyTorch threads: `16`
- Early stopping patience: `8`
- Min delta: `1e-4`

Official training command:

```bash
.venv/bin/python src/train_baseline.py --epochs 50 --batch-size 8 --lr 1e-3 --input-mode real_imag --num-workers 8 --torch-num-threads 16 --device cpu --early-stopping-patience 8 --min-delta 1e-4
```

## Final metrics
- Best epoch: `50`
- Best validation loss: `0.152178`
- Test MSE: `0.184736`
- Test MAE: `0.203690`
- Test RMSE: `0.429810`
- Early stopping triggered: `False`

## Current baseline limitations
- The model is still a small CNN encoder-decoder baseline and not a stronger inversion architecture
- Only three input representations were tested
- The pipeline is CPU-only, so experimentation is slower than a GPU workflow
- The current evaluation focuses on scalar regression metrics and qualitative plots, not uncertainty or physics-aware constraints
