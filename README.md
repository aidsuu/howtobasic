# Qjulia

Baseline Julia project for the one-dimensional transverse-field Ising model (TFIM),

```math
H = -J \sum_{i=1}^{N-1} \sigma^z_i \sigma^z_{i+1} - h \sum_{i=1}^{N} \sigma^x_i
```

The project started from dense exact diagonalization and was extended with:

- basic observables (`Mz`, `Mx`)
- exact-vector variational baselines
- an MPS/DMRG solver based on `ITensors.jl`
- parameter sweeps, correlation analysis, and final benchmark scripts

The code is organized so that the exact, variational, and MPS paths can be compared directly on the same TFIM Hamiltonian.

## Project Structure

- `Project.toml`: Julia project metadata and dependencies
- `src/Qjulia.jl`: main module with operators, Hamiltonians, exact diagonalization, variational states, MPS/DMRG, sweeps, and correlations
- `examples/tfim_example.jl`: small-system benchmark comparing exact diagonalization, product-state variational states, and MPS
- `examples/tfim_sweep.jl`: parameter sweep over `h/J`, CSV export, and observable plots
- `examples/tfim_correlations.jl`: correlation profiles as a function of distance
- `examples/final_results.jl`: curated final benchmark run for the full project output
- `test/runtests.jl`: unit tests for exact, variational, MPS, and correlation routines
- `outputs/`: generated CSV files, plots, and summary notes

## Dependencies

The project uses:

- `LinearAlgebra`: dense linear algebra, `Hermitian`, `eigen`, `kron`
- `Printf`: formatted output in example scripts
- `Optim.jl`: scalar and multi-parameter optimization for variational baselines
- `ITensors.jl`: tensor backend
- `ITensorMPS.jl`: MPS/MPO data structures and DMRG

## Exact Diagonalization Baseline

For a spin-1/2 chain of length `N`, the Hilbert-space dimension is `2^N`. The dense TFIM Hamiltonian therefore has size `2^N x 2^N`, so exact diagonalization is useful only for small systems. In practice this project uses exact diagonalization mainly for:

- reference ground-state energies
- reference observables
- validation of variational and MPS results

`N = 10` is still practical for dense diagonalization. By `N = 20`, the Hilbert-space dimension is already `1,048,576`, which is too large for ordinary dense methods in this workflow.

## Variational Layer

The first variational baseline is a uniform product state,

```math
|\Phi(\theta)\rangle = \bigotimes_{i=1}^{N} |\phi(\theta)\rangle, \qquad
|\phi(\theta)\rangle = \cos(\theta/2)|0\rangle + \sin(\theta/2)|1\rangle
```

with trial energy

```math
E_{\mathrm{var}}(\theta) = \langle \Phi(\theta) | H | \Phi(\theta) \rangle.
```

This state is intentionally simple. It does not contain entanglement, so it is a useful mean-field-like baseline but not a competitive ansatz near the TFIM crossover region.

The project then generalizes this to site-dependent single-spin parameters,

```math
|\Phi(\theta_s, \phi_s)\rangle = \bigotimes_{i=1}^{N} |\phi(\theta_i, \phi_i)\rangle,
```

and adds a nearest-neighbor entangler

```math
U_{ZZ}(\gamma_i) = \exp(-i \gamma_i \sigma^z_i \sigma^z_{i+1}),
```

which gives the entangled exact-vector ansatz

```math
|\psi(\theta_s, \gamma_s)\rangle =
\left[\prod_{i=1}^{N-1} U_{ZZ}(\gamma_i)\right] |\Phi(\theta_s, 0)\rangle.
```

These ansätze still live in the full Hilbert-space vector representation, so they are easy to test against exact diagonalization but do not scale to large `N`.

## MPS / DMRG Layer

To go beyond dense exact diagonalization, the project also includes an MPS/DMRG path using `ITensors.jl` and `ITensorMPS.jl`.

The idea is standard:

- an MPS represents a 1D quantum state in compressed form instead of a full `2^N` vector
- an MPO represents the local Hamiltonian in a compatible compressed form
- DMRG optimizes the MPS variationally through local sweeps

In the implementation, ITensors spin-1/2 operators satisfy

```math
\sigma^z = 2 S^z, \qquad \sigma^x = 2 S^x,
```

so the TFIM Hamiltonian

```math
H = -J \sum_i \sigma^z_i \sigma^z_{i+1} - h \sum_i \sigma^x_i
```

is written as an MPO with coefficients `-4J Sz_i Sz_{i+1}` and `-2h Sx_i`.

For small chains, the MPS energy can be checked directly against exact diagonalization. For larger chains, it becomes the main solver.

## Physics Analysis Pipeline

The analysis routines currently include:

- `sweep_h_over_J(method, Ns, h_over_J_values; ...)`
- `add_finite_difference_derivative(rows; ...)`
- `correlation_zz(state, r)`
- `correlation_xx(state, r)`
- `connected_correlation_zz(state, r)`
- `correlation_profile_zz(state, max_r)`
- `correlation_profile_xx(state, max_r)`
- `connected_correlation_profile_zz(state, max_r)`

For both exact vectors and MPS states, two-point correlations are reported with a simple translation average over all pairs separated by distance `r`:

```math
C_{zz}(r) = \frac{1}{N-r} \sum_{i=1}^{N-r} \langle \sigma^z_i \sigma^z_{i+r} \rangle,
```

and similarly for `xx`. The connected `zz` correlation is

```math
C^{\mathrm{conn}}_{zz}(r) =
\frac{1}{N-r} \sum_{i=1}^{N-r}
\left(
\langle \sigma^z_i \sigma^z_{i+r} \rangle
- \langle \sigma^z_i \rangle \langle \sigma^z_{i+r} \rangle
\right).
```

These routines are meant for simple finite-size crossover studies rather than a full finite-size scaling analysis.

## Running the Examples

Run the small benchmark:

```bash
julia --project=. examples/tfim_example.jl
```

This compares:

- exact diagonalization for `N = 4, 6, 10`
- product-state variational states for `N = 4, 6, 10`
- MPS/DMRG for `N = 4, 6, 10`
- MPS only for a larger system

Run the main sweep:

```bash
julia --project=. examples/tfim_sweep.jl
```

This produces:

- `exact_results.csv`
- `product_results.csv`
- `mps_results.csv`
- `combined_results.csv`
- `energy_per_site_vs_h_over_J.png`
- `mz_vs_h_over_J.png`
- `mx_vs_h_over_J.png`

Useful options:

```bash
julia --project=. examples/tfim_sweep.jl --quick
julia --project=. examples/tfim_sweep.jl --critical
```

The `--critical` mode uses a denser grid near `h/J = 1` and adds finite-difference estimates such as `de0_dh_over_J`, `dMz_dh_over_J`, and `dMx_dh_over_J`.

Run the correlation analysis:

```bash
julia --project=. examples/tfim_correlations.jl
```

This computes correlation profiles for selected values of `h/J`, stores them in CSV form, and writes the corresponding plots.

Useful option:

```bash
julia --project=. examples/tfim_correlations.jl --quick
```

## Recommended Final Runs

For the final project output, run:

```bash
julia --project=. examples/final_results.jl
```

This writes the main deliverables to `outputs/final_results/`, including:

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

The final benchmark uses:

- exact diagonalization for `N = 4, 6, 10`
- product-state baseline for `N = 4, 6, 10`
- MPS for `N = 10, 20, 40`
- a main sweep `h/J = 0.2:0.2:2.0`
- a denser near-critical sweep `h/J = 0.8:0.05:1.2`
- correlation profiles at `h/J = 0.6, 1.0, 1.4`

If only part of the workflow is needed:

```bash
julia --project=. examples/tfim_sweep.jl
julia --project=. examples/tfim_sweep.jl --critical
julia --project=. examples/tfim_correlations.jl
```

## Design Notes

The code keeps the main pieces separate on purpose:

- `TFIMParameters` stores model parameters in one place
- local operators are separate from Hamiltonian construction
- `normalize_state`, `expectation_value`, and `energy_expectation` are reusable for any trial state, not only the ground state
- the variational state builders and energy objectives are separated from the optimizers
- the exact-vector and MPS backends are separated, so the same analysis code can be reused across both
- sweep and CSV export routines are separate from the solvers
- scalar correlation functions and full correlation profiles are exposed as separate APIs
- `tfim_hamiltonian` returns a `Hermitian` matrix explicitly for clarity in dense diagonalization

## Running the Tests

```bash
julia --project=. test/runtests.jl
```
