# Final TFIM 1D Results

## Benchmark utama

- Exact diagonalization dipakai untuk `N = 4, 6, 10` sebagai baseline referensi.
- Product-state exact-vector dipakai hanya untuk `N` kecil sebagai baseline mean-field sederhana.
- MPS/DMRG dipakai untuk `N = 10, 20, 40` agar skala sistem bisa diperbesar.

## Kapan product-state gagal

- Pada overlap `N = 10`, gap energi terbesar product-state terhadap exact muncul di sekitar `h/J = 1.8` dengan selisih `1.24188`.
- Ini konsisten dengan ekspektasi bahwa product-state paling lemah saat korelasi dan entanglement meningkat di sekitar crossover kritis.

## Kapan MPS cocok dengan ED

- Pada ukuran overlap `N = 10`, MPS mereproduksi energi exact dengan deviasi maksimum `8.56469e-5` di seluruh sweep utama.
- Untuk `N = 4` dan `N = 6`, benchmark sebelumnya juga menunjukkan MPS praktis menumpuk di atas exact sampai toleransi numerik.

## Pola near-critical

- Pada sweep rapat, magnitudo turunan `dMx/d(h/J)` untuk `N = 40` mencapai puncak dekat `h/J = 0.95`, menandai crossover tercepat.
- `Mz` turun dan `Mx` naik saat medan transversal diperbesar, sesuai transisi dari dominasi interaksi Ising ke polarisasi arah-x.

## Korelasi

- Profil korelasi `zz`, `xx`, dan connected `zz` tersedia untuk `h/J = 0.6, 1.0, 1.4` dan `N = 20, 40`.
- Dekat `h/J = 1`, peluruhan korelasi menjadi lebih lambat dibanding jauh dari crossover, terutama pada ukuran sistem lebih besar.

## Entanglement entropy

- Entropy bipartisi tengah MPS ditambahkan sebagai indikator tambahan entanglement.
- Pada `N = 40`, entropy tengah maksimum dalam sweep utama muncul dekat `h/J = 1.0` dengan nilai `0.505246`.
