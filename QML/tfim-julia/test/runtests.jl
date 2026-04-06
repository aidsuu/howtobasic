using LinearAlgebra
using Test

using Qjulia

@testset "TFIM baseline" begin
    params = TFIMParameters(N = 4, J = 1.0, h = 1.0)
    H = tfim_hamiltonian(params)
    energy, state = ground_state(params)
    normalized_state = normalize_state(3.0 .* state)

    @test size(H) == (16, 16)
    @test isapprox(norm(state), 1.0; atol = 1e-10)
    @test isapprox(norm(normalized_state), 1.0; atol = 1e-10)
    @test isapprox(expectation_value(normalized_state, I), 1.0; atol = 1e-10)
    @test isapprox(expectation_value(Matrix(H), state), energy; atol = 1e-10)
    @test -1.0 <= magnetization_z(state, params.N) <= 1.0
    @test -1.0 <= magnetization_x(state, params.N) <= 1.0
    @test isapprox(correlation_zz(state, 0), 1.0; atol = 1e-10)
    @test isfinite(correlation_zz(state, 1))
    @test isfinite(correlation_xx(state, 1))
    @test isfinite(connected_correlation_zz(state, 1))
    zz_profile = correlation_profile_zz(state, 2)
    xx_profile = correlation_profile_xx(state, 2)
    connected_zz_profile = connected_correlation_profile_zz(state, 2)
    @test length(zz_profile) == 3
    @test length(xx_profile) == 3
    @test length(connected_zz_profile) == 3
    @test all(isfinite, zz_profile)
    @test all(isfinite, xx_profile)
    @test all(isfinite, connected_zz_profile)
    @test isapprox(connected_zz_profile[1], connected_correlation_zz(state, 0); atol = 1e-10)
end

@testset "Variational product-state ansatz" begin
    @test isapprox(norm(single_spin_state(0.3)), 1.0; atol = 1e-10)
    @test isapprox(norm(single_spin_state(0.3, 0.7)), 1.0; atol = 1e-10)

    for N in (4, 6)
        params = TFIMParameters(N = N, J = 1.0, h = 1.0)
        H = tfim_hamiltonian(params)
        exact_energy, _ = ground_state(params)

        thetas = fill(0.7, N)
        phis = zeros(N)
        trial_state = product_state_ansatz(thetas, phis)
        @test isapprox(norm(trial_state), 1.0; atol = 1e-10)
        @test isapprox(expectation_value(trial_state, I), 1.0; atol = 1e-10)

        product_thetas, product_phis, product_energy, _ = optimize_product_state(H, N; iterations = 1_500)
        product_opt_state = product_state_ansatz(product_thetas, product_phis)
        @test isapprox(norm(product_opt_state), 1.0; atol = 1e-10)
        @test product_energy + 1e-8 >= exact_energy

        gammas = fill(0.2, N - 1)
        entangled_state = entangled_product_state_ansatz(thetas, gammas; phis = phis)
        @test isapprox(norm(entangled_state), 1.0; atol = 1e-10)

        entangled_thetas, entangled_phis, entangled_gammas, entangled_energy, _ =
            optimize_entangled_product_state(H, N; iterations = 2_500)
        entangled_opt_state = entangled_product_state_ansatz(entangled_thetas, entangled_gammas; phis = entangled_phis)
        @test isapprox(norm(entangled_opt_state), 1.0; atol = 1e-10)
        @test entangled_energy + 1e-8 >= exact_energy
        @test entangled_energy <= product_energy + 1e-6
    end
end

@testset "MPS TFIM solver" begin
    for N in (4, 6)
        params = TFIMParameters(N = N, J = 1.0, h = 1.0)
        hamiltonian = tfim_hamiltonian(params)
        exact_energy, _ = ground_state(params)
        _, _, product_energy, _ = optimize_product_state(hamiltonian, N; iterations = 1_500)

        mps_result = mps_ground_state(params; maxdim = [10, 20, 50, 100, 100, 200, 200, 400])
        @test isfinite(mps_result.energy)
        @test -1.0 <= magnetization_z(mps_result.state) <= 1.0
        @test -1.0 <= magnetization_x(mps_result.state) <= 1.0
        @test isapprox(correlation_zz(mps_result.state, 0), 1.0; atol = 1e-8)
        @test isfinite(correlation_zz(mps_result.state, 1))
        @test isfinite(correlation_xx(mps_result.state, 1))
        @test isfinite(connected_correlation_zz(mps_result.state, 1))
        zz_profile = correlation_profile_zz(mps_result.state, 2)
        xx_profile = correlation_profile_xx(mps_result.state, 2)
        connected_zz_profile = connected_correlation_profile_zz(mps_result.state, 2)
        @test length(zz_profile) == 3
        @test length(xx_profile) == 3
        @test length(connected_zz_profile) == 3
        @test all(isfinite, zz_profile)
        @test all(isfinite, xx_profile)
        @test all(isfinite, connected_zz_profile)
        @test isapprox(connected_zz_profile[1], connected_correlation_zz(mps_result.state, 0); atol = 1e-8)
        @test mps_result.energy + 1e-6 >= exact_energy
        @test mps_result.energy <= product_energy + 1e-5
    end
end
