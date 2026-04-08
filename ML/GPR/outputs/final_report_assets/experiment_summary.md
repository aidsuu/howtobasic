# Experiment Summary

## Comparison of the three input modes
- `magnitude`: best `val_loss=0.457512`, `test_mse=0.488006`, `test_mae=0.392868`, `test_rmse=0.698574`
- `magnitude_minus_freespace`: best `val_loss=0.394874`, `test_mse=0.494521`, `test_mae=0.410393`, `test_rmse=0.703222`
- `real_imag`: best `val_loss=0.152178`, `test_mse=0.184736`, `test_mae=0.203690`, `test_rmse=0.429810`

## Best mode
- Based on `test_rmse`, the best mode is `real_imag`
- Based on `test_mae`, the best mode is `real_imag`

## Did background subtraction help?
- No on the final test metrics
- Compared with `magnitude`, `magnitude_minus_freespace` achieved a lower best validation loss, but its final `test_rmse` and `test_mae` were slightly worse

## Is `real_imag` better than `magnitude`?
- Yes
- `real_imag` is substantially better than `magnitude` on `best_val_loss`, `test_mse`, `test_mae`, and `test_rmse`

## Final recommended configuration
- The final recommended input mode for this project is `real_imag`
- Recommended command:

```bash
.venv/bin/python src/train_baseline.py --epochs 50 --batch-size 8 --lr 1e-3 --input-mode real_imag --num-workers 8 --torch-num-threads 16 --device cpu --early-stopping-patience 8 --min-delta 1e-4
.venv/bin/python src/evaluate.py --input-mode real_imag --device cpu
.venv/bin/python src/plot_history.py
```
