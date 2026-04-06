using Pkg
using Printf

Pkg.activate(joinpath(@__DIR__, ".."))
using Qjulia

function print_full_benchmark(params::TFIMParameters)
    hamiltonian = tfim_hamiltonian(params)
    exact_energy, exact_state = ground_state(params)
    exact_mz = magnetization_z(exact_state, params.N)
    exact_mx = magnetization_x(exact_state, params.N)

    product_thetas, product_phis, product_energy, _ = optimize_product_state(hamiltonian, params.N; iterations = 1_500)
    product_state = product_state_ansatz(product_thetas, product_phis)
    product_mz = magnetization_z(product_state, params.N)
    product_mx = magnetization_x(product_state, params.N)

    mps_result = mps_ground_state(params)
    mps_mz = magnetization_z(mps_result.state)
    mps_mx = magnetization_x(mps_result.state)

    println("========================================")
    println("TFIM 1D benchmark: ED vs product-state vs MPS")
    @printf("N = %d, J = %.3f, h = %.3f\n", params.N, params.J, params.h)
    @printf("Hilbert dimension = %d\n", 2^params.N)
    @printf("Exact ground-state energy = %.10f\n", exact_energy)
    @printf("Exact Mz = %.10f\n", exact_mz)
    @printf("Exact Mx = %.10f\n", exact_mx)
    @printf("Product-state energy = %.10f\n", product_energy)
    @printf("Product-state Mz = %.10f\n", product_mz)
    @printf("Product-state Mx = %.10f\n", product_mx)
    @printf("Product-state gap = %.10f\n", product_energy - exact_energy)
    @printf("MPS energy = %.10f\n", mps_result.energy)
    @printf("MPS Mz = %.10f\n", mps_mz)
    @printf("MPS Mx = %.10f\n", mps_mx)
    @printf("MPS gap = %.10f\n", mps_result.energy - exact_energy)
    @printf("MPS improvement over product = %.10f\n", product_energy - mps_result.energy)
    println()
end

function print_mps_only(params::TFIMParameters)
    mps_result = mps_ground_state(params; maxdim = [10, 20, 50, 100, 200, 300, 400, 600])
    mps_mz = magnetization_z(mps_result.state)
    mps_mx = magnetization_x(mps_result.state)

    println("========================================")
    println("TFIM 1D benchmark: MPS-only large system")
    @printf("N = %d, J = %.3f, h = %.3f\n", params.N, params.J, params.h)
    @printf("MPS energy = %.10f\n", mps_result.energy)
    @printf("MPS Mz = %.10f\n", mps_mz)
    @printf("MPS Mx = %.10f\n", mps_mx)
    println()
end

for N in (4, 6, 10)
    print_full_benchmark(TFIMParameters(N = N, J = 1.0, h = 1.0))
end

print_mps_only(TFIMParameters(N = 20, J = 1.0, h = 1.0))
