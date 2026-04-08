# Environment Notes

## Environment lama
- Path: `ML/.venv`
- Status: tetap utuh, tidak dihapus, dipakai untuk training/evaluasi CPU
- Torch: `2.11.0+cu130`
- Observasi: `torch.cuda.is_available()` bernilai `False`
- Catatan: driver NVIDIA yang terpasang terlalu lama untuk build ini, jadi GPU tidak bisa dipakai

## Environment baru
- Path: `ML/.venv-cu`
- Status: sudah dihapus setelah diputuskan proyek resmi berjalan CPU-only
- Python: `3.13.5`
- Dependency umum berhasil diinstal

## Versi Torch yang dicoba di `.venv-cu`
- Attempt a: `torch==2.5.1`, `torchvision==0.20.1`, `torchaudio==2.5.1`, index `cu124`
- Hasil: gagal install karena `torchvision==0.20.1` tidak tersedia pada index tersebut untuk environment ini
- Attempt b: `torch==2.4.1`, `torchvision==0.19.1`, `torchaudio==2.4.1`, index `cu124`
- Hasil: gagal install karena `torch==2.4.1` tidak tersedia pada index tersebut untuk environment ini
- Attempt c: `torch==2.3.1`, `torchvision==0.18.1`, `torchaudio==2.3.1`, index `cu121`
- Hasil: gagal install karena `torch==2.3.1` tidak tersedia pada index tersebut untuk environment ini

## Mana yang berhasil
- Jalur GPU: tidak berhasil
- Jalur CPU: berhasil dan stabil memakai `ML/.venv`

## Apakah GPU akhirnya bisa dipakai?
- Tidak

## Status resmi proyek
- Proyek resmi berjalan full CPU
- Environment default proyek adalah `ML/.venv`

## Command final yang direkomendasikan
```bash
.venv/bin/python src/train_baseline.py --epochs 50 --batch-size 16 --lr 1e-3 --input-mode magnitude --num-workers 8 --torch-num-threads 16 --device cpu --early-stopping-patience 8 --min-delta 1e-4
.venv/bin/python src/evaluate.py --input-mode magnitude --device cpu
.venv/bin/python src/plot_history.py
```

## Catatan tambahan
- Di sandbox Codex, `num_workers=8` butuh eksekusi di luar sandbox karena multiprocessing socket diblokir.
- Di shell normal mesin ini, training CPU penuh berhasil selesai.
- Detail probe CUDA disimpan di `logs/torch_cuda_probe.txt`.
