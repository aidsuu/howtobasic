# Final TFIM 1D Results

## Main Benchmark

- Exact diagonalization is used for `N = 4, 6, 10` as the reference baseline.
- The exact-vector product-state ansatz is used only for small `N` as a simple mean-field-like baseline.
- MPS/DMRG is used for `N = 10, 20, 40` so that larger systems can be treated.

## Where The Product State Fails

- On the overlap system `N = 10`, the largest product-state energy gap relative to exact diagonalization appears near `h/J = 1.8`, with a difference of `1.24188`.
- This is consistent with the expected limitation of a product state when correlations and entanglement become stronger near the crossover region.

## Where MPS Matches Exact Diagonalization

- On the overlap system `N = 10`, the MPS energy reproduces the exact result with a maximum deviation of `8.58675e-5` across the main sweep.
- For `N = 4` and `N = 6`, earlier benchmarks also showed that the MPS result is effectively identical to exact diagonalization within numerical tolerance.

## Near-Critical Pattern

- In the dense sweep, the magnitude of `dMx/d(h/J)` for `N = 40` reaches its maximum near `h/J = 0.95`, which marks the sharpest crossover.
- `Mz` decreases and `Mx` increases as the transverse field grows, consistent with the shift from Ising-dominated order toward x-polarization.

## Correlations

- The `zz`, `xx`, and connected `zz` correlation profiles are stored for `h/J = 0.6, 1.0, 1.4` and `N = 20, 40`.
- Near `h/J = 1`, the correlation decay becomes slower than it is far from the crossover, especially for the larger systems.

## Entanglement Entropy

- The MPS mid-chain bipartite entropy is included as an additional entanglement indicator.
- For `N = 40`, the largest mid-chain entropy in the main sweep appears near `h/J = 1.0`, with value `0.505242`.
