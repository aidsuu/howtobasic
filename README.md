# Qjulia

Baseline exact diagonalization untuk 1D transverse-field Ising model (TFIM) di Julia.

Model yang diimplementasikan:

```math
H = -J \sum_{i=1}^{N-1} \sigma^z_i \sigma^z_{i+1} - h \sum_{i=1}^{N} \sigma^x_i
```

Tujuan baseline ini adalah menyediakan fondasi yang bersih untuk tahap berikutnya:

- validasi energi ground state dengan exact diagonalization
- observables sederhana (`Mz`, `Mx`)
- lapisan variational dengan product-state dan entangled-product-state ansatz
- backend MPS/DMRG berbasis ITensors.jl untuk sistem 1D yang lebih besar
- struktur kode yang mudah diperluas ke variational ansatz dan optimizer numerik

## Struktur File

- `Project.toml`: metadata proyek Julia
- `src/Qjulia.jl`: modul utama untuk operator spin, Hamiltonian TFIM, exact diagonalization, variational exact-vector, dan backend MPS/DMRG
- `examples/tfim_example.jl`: benchmark exact diagonalization vs product-state vs MPS untuk `N=4, 6, 10`, serta MPS-only untuk `N=20`
- `examples/tfim_sweep.jl`: pipeline sweep `h/J`, ekspor CSV, dan plot observables
- `examples/tfim_correlations.jl`: analisis profil korelasi terhadap jarak `r`
- `examples/final_results.jl`: run final terkurasi untuk seluruh output proyek
- `test/runtests.jl`: test baseline exact dan lapisan variational

## Dependency

Proyek ini memakai dependency berikut:

- `LinearAlgebra`: stdlib Julia untuk `eigen`, `Hermitian`, dan `kron`
- `Printf`: stdlib Julia untuk format output pada script contoh
- `Optim.jl`: optimisasi skalar dan multi-parameter untuk baseline variational
- `ITensors.jl`: backend tensor inti
- `ITensorMPS.jl`: representasi MPO/MPS dan solver DMRG untuk sistem 1D

Karena baseline ini memakai exact diagonalization dense, pendekatan ini cocok untuk sistem kecil dan benchmark awal. Dimensi Hilbert ruang spin-1/2 bertumbuh sebagai `2^N`, sehingga matriks Hamiltonian dense berukuran `2^N × 2^N` cepat menjadi terlalu besar. Misalnya `N=10` masih nyaman, tetapi `N=20` sudah berarti dimensi `1,048,576`, yang tidak lagi praktis untuk diagonalization dense biasa.

Saat beralih ke variational ansatz, modul saat ini bisa dipakai sebagai:

- referensi ground-state energy exact
- pembanding observables
- sumber Hamiltonian untuk evaluasi ekspektasi energi

## Variational Layer

Lapisan variational yang ditambahkan di atas baseline exact memakai ansatz product state homogen:

```math
|\Phi(\theta)\rangle = \bigotimes_{i=1}^{N} |\phi(\theta)\rangle, \qquad
|\phi(\theta)\rangle = \cos(\theta/2)|0\rangle + \sin(\theta/2)|1\rangle
```

Dengan ansatz ini, energi trial didefinisikan sebagai:

```math
E_{\mathrm{var}}(\theta) = \langle \Phi(\theta) | H | \Phi(\theta) \rangle
```

Nilai minimum `E_var` tidak boleh lebih rendah dari energi ground state exact, sesuai prinsip variational. Walaupun ansatz product state ini sangat sederhana dan belum menangkap entanglement, ia memberi baseline yang baik sebelum beralih ke ansatz yang lebih kaya.

Ansatz product-state kemudian digeneralisasi menjadi parameter per-site:

```math
|\Phi(\theta_s, \phi_s)\rangle = \bigotimes_{i=1}^{N} |\phi(\theta_i, \phi_i)\rangle
```

Untuk membawa entanglement paling sederhana tanpa berpindah ke MPS, proyek ini juga menambahkan entangler nearest-neighbor:

```math
U_{ZZ}(\gamma_i) = \exp(-i \gamma_i \sigma^z_i \sigma^z_{i+1})
```

Sehingga ansatz terentang yang dipakai adalah:

```math
|\psi(\theta_s, \gamma_s)\rangle = \left[\prod_{i=1}^{N-1} U_{ZZ}(\gamma_i)\right] |\Phi(\theta_s, 0)\rangle
```

Ansatz ini masih bekerja langsung pada exact Hilbert-space vector, jadi sederhana untuk eksperimen awal. Keterbatasannya tetap jelas: ukuran state tumbuh sebagai `2^N`, sehingga tidak skalabel untuk sistem besar. Product-state sendiri juga tidak bisa menangkap entanglement, sehingga biasanya memberi energi lebih buruk daripada ansatz terentang.

## MPS / DMRG Layer

Untuk melampaui keterbatasan exact diagonalization, proyek ini juga menambahkan representasi Hamiltonian dalam bentuk MPO dan solver ground state berbasis MPS/DMRG melalui `ITensors.jl` dan `ITensorMPS.jl`.

Intuisi singkatnya:

- MPS merepresentasikan state 1D secara terkompresi, bukan sebagai vektor penuh berdimensi `2^N`
- MPO merepresentasikan Hamiltonian lokal 1D secara efisien
- DMRG mengoptimalkan MPS melalui sweep lokal dan biasanya sangat efektif untuk ground state rantai 1D

Dalam implementasi ini, operator ITensors `"Sz"` dan `"Sx"` adalah operator spin-1/2, jadi Hamiltonian TFIM Pauli

```math
H = -J \sum_i \sigma^z_i \sigma^z_{i+1} - h \sum_i \sigma^x_i
```

diterjemahkan ke MPO dengan faktor:

```math
\sigma^z = 2 S^z,\qquad \sigma^x = 2 S^x
```

sehingga term MPO memakai `-4J Sz_i Sz_{i+1}` dan `-2h Sx_i`.

## Physics Analysis Pipeline

Untuk analisis fisika, modul sekarang menyediakan:

- `sweep_h_over_J(method, Ns, h_over_J_values; ...)` untuk menjalankan benchmark pada banyak titik parameter
- `add_finite_difference_derivative(rows; ...)` untuk estimator turunan numerik sederhana
- observables dasar `E0`, `e0 = E0/N`, `Mz`, dan `Mx`
- korelasi dua titik `correlation_zz(state, r)` dan `correlation_xx(state, r)`
- korelasi terkoneksi `connected_correlation_zz(state, r)`
- profil korelasi `correlation_profile_zz`, `correlation_profile_xx`, dan `connected_correlation_profile_zz`

Korelasi dihitung sebagai rata-rata translasi sederhana di bulk:

```math
C_{zz}(r) = \frac{1}{N-r} \sum_{i=1}^{N-r} \langle \sigma^z_i \sigma^z_{i+r} \rangle
```

dan analog untuk `xx`. Untuk `zz`, versi terkoneksi didefinisikan sebagai:

```math
C^{\mathrm{conn}}_{zz}(r) =
\frac{1}{N-r} \sum_{i=1}^{N-r}
\left(
\langle \sigma^z_i \sigma^z_{i+r} \rangle
- \langle \sigma^z_i \rangle \langle \sigma^z_{i+r} \rangle
\right)
```

Pipeline sweep ini memudahkan studi crossover feromagnetik-ke-paramagnetik sebagai fungsi `h/J`, dan API-nya sengaja dijaga generik agar nanti mudah diperluas ke sweep parameter, finite-size scaling, atau ekstraksi panjang korelasi.

Untuk studi finite-size crossover di sekitar titik kritis TFIM 1D, script sweep juga mendukung mode rapat di sekitar `h/J = 1`, sehingga perubahan `e0`, `Mz`, `Mx`, dan estimator turunannya bisa dibandingkan untuk beberapa ukuran sistem.

## Cara Menjalankan

Jika environment Julia lokal sudah normal:

```bash
julia --project=. examples/tfim_example.jl
```

Script benchmark akan menghitung:

- ground-state energy exact
- energi product-state hasil optimisasi multi-parameter
- energi ground state MPS hasil DMRG
- magnetisasi rata-rata `Mz`
- magnetisasi rata-rata `Mx`
- selisih energi terhadap exact dan peningkatan MPS atas product-state

untuk:

- `N = 4`
- `N = 6`
- `N = 10`

serta benchmark tambahan:

- `N = 20` dengan jalur MPS saja

Jalankan benchmark dengan:

```bash
julia --project=. examples/tfim_example.jl
```

Untuk sweep analisis fisika dan plot:

```bash
julia --project=. examples/tfim_sweep.jl
```

Script ini menjalankan:

- exact diagonalization untuk `N = 4, 6, 10`
- product-state exact-vector untuk `N = 4, 6, 10`
- MPS/DMRG untuk `N = 10, 20, 40`
- sweep `h/J` pada rentang `0.2:0.2:2.0`

Hasil disimpan ke folder `outputs/tfim_sweep/` dalam bentuk:

- `exact_results.csv`
- `product_results.csv`
- `mps_results.csv`
- `combined_results.csv`
- `energy_per_site_vs_h_over_J.png`
- `mz_vs_h_over_J.png`
- `mx_vs_h_over_J.png`

Plot memakai styling yang lebih tegas dan kontras agar kurva lintas ukuran sistem tetap jelas saat dibandingkan.

Untuk verifikasi cepat tanpa menjalankan sweep penuh:

```bash
julia --project=. examples/tfim_sweep.jl --quick
```

Untuk sweep rapat di sekitar crossover:

```bash
julia --project=. examples/tfim_sweep.jl --critical
```

Mode ini menjalankan sweep `h/J = 0.8:0.05:1.2` dengan:

- exact dan product-state pada `N = 10`
- MPS pada `N = 10, 20, 40`

Lalu script menyimpan juga estimator turunan numerik seperti `de0_dh_over_J`, `dMz_dh_over_J`, dan `dMx_dh_over_J` ke `combined_results.csv`.

Untuk analisis profil korelasi terhadap jarak:

```bash
julia --project=. examples/tfim_correlations.jl
```

Script ini memilih beberapa nilai `h/J` seperti `0.6`, `1.0`, `1.4`, menghitung profil korelasi MPS untuk `N = 20` dan `N = 40`, lalu menyimpan:

- `correlation_profiles.csv`
- `correlation_zz_vs_r.png`
- `correlation_xx_vs_r.png`
- `connected_correlation_zz_vs_r.png`

Untuk mode cepat:

```bash
julia --project=. examples/tfim_correlations.jl --quick
```

## Recommended Final Runs

Untuk menghasilkan paket hasil akhir proyek yang paling informatif tanpa menjalankan semua eksperimen eksploratif terpisah, jalankan:

```bash
julia --project=. examples/final_results.jl
```

Script ini menghasilkan seluruh output utama ke folder `outputs/final_results/`, termasuk:

- `benchmark_main.csv`
- `benchmark_critical.csv`
- `correlation_profiles.csv`
- `entropy_mid_mps.csv`
- `energy_per_site_vs_h_over_J.png`
- `mz_vs_h_over_J.png`
- `mx_vs_h_over_J.png`
- `derivative_near_critical.png`
- `correlation_zz_vs_r.png`
- `correlation_xx_vs_r.png`
- `connected_correlation_zz_vs_r.png`
- `entropy_mid_vs_h_over_J.png`
- `summary.md`

Konfigurasi final yang dipakai:

- ED untuk `N = 4, 6, 10`
- product-state exact-vector untuk `N = 4, 6, 10`
- MPS untuk `N = 10, 20, 40`
- sweep utama `h/J = 0.2:0.2:2.0`
- sweep rapat near-critical `h/J = 0.8:0.05:1.2`
- profil korelasi pada `h/J = 0.6, 1.0, 1.4`

Jika hanya ingin memeriksa komponen tertentu:

```bash
julia --project=. examples/tfim_sweep.jl
julia --project=. examples/tfim_sweep.jl --critical
julia --project=. examples/tfim_correlations.jl
```

## Catatan Desain

Beberapa keputusan desain agar mudah dikembangkan:

- parameter model dikemas dalam `TFIMParameters`
- operator lokal `sigma_x(site, N)` dan `sigma_z(site, N)` dipisah dari konstruksi Hamiltonian
- fungsi `normalize_state`, `expectation_value`, dan `energy_expectation` tidak terikat pada ground state saja, sehingga bisa dipakai ulang untuk state trial variational
- `single_spin_state`, `product_state_ansatz`, `entangled_product_state_ansatz`, dan objective energinya dipisah agar mudah diganti backend atau parameterisasi
- operator `sigma_zz`, `zz_entangler`, dan `apply_zz_entangler` memisahkan logika gerbang dari definisi ansatz
- jalur optimisasi multi-parameter dibungkus dalam helper sendiri, sehingga nanti mudah diganti ke MPS/ITensors atau optimizer lain
- `tfim_mpo` dan `mps_ground_state` memisahkan backend tensor-network dari backend exact-vector, sehingga nanti mudah diperluas ke sweep parameter `h/J`, finite-size scaling, atau studi phase crossover
- `sweep_h_over_J` dan `save_sweep_results_csv` memisahkan analisis data dari solver, sehingga observables dan benchmark dapat dipakai ulang tanpa mengubah backend
- API korelasi skalar dan API correlation profile dipisah, sehingga mudah dipakai ulang baik untuk plotting maupun fitting panjang korelasi
- `tfim_hamiltonian` dikembalikan sebagai `Hermitian` agar aman dan jelas untuk diagonalization

## Menjalankan Test

```bash
julia --project=. test/runtests.jl
```
