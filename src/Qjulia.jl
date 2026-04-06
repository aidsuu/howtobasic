module Qjulia

using ITensors
using ITensorMPS
using LinearAlgebra
using Optim

export TFIMParameters,
       normalize_state,
       ground_state,
       pauli_x,
       pauli_z,
       sigma_x,
       sigma_z,
       sigma_zz,
       tfim_hamiltonian,
       tfim_sites,
       tfim_mpo,
       mps_ground_state,
       single_spin_state,
       product_state_ansatz,
       entangled_product_state_ansatz,
       zz_entangler,
       apply_zz_entangler,
       magnetization_x,
       magnetization_z,
       correlation_zz,
       correlation_xx,
       connected_correlation_zz,
       correlation_profile_zz,
       correlation_profile_xx,
       connected_correlation_profile_zz,
       expectation_value,
       energy_expectation,
       variational_energy,
       entangled_variational_energy,
       sweep_h_over_J,
       add_finite_difference_derivative,
       save_sweep_results_csv,
       optimize_product_state_theta,
       optimize_product_state,
       optimize_entangled_product_state

"""
    TFIMParameters

Parameter container for the 1D transverse-field Ising model:

`H = -J Σ σᶻᵢσᶻᵢ₊₁ - h Σ σˣᵢ`
"""
Base.@kwdef struct TFIMParameters
    N::Int
    J::Float64 = 1.0
    h::Float64 = 1.0
end

const pauli_x = ComplexF64[0 1; 1 0]
const pauli_z = ComplexF64[1 0; 0 -1]
const identity_2 = ComplexF64[1 0; 0 1]

"""
    sigma_x(site, N)

`σˣ` operator acting on `site` for a spin chain of length `N`.
"""
sigma_x(site::Int, N::Int) = local_operator(pauli_x, site, N)

"""
    sigma_z(site, N)

`σᶻ` operator acting on `site` for a spin chain of length `N`.
"""
sigma_z(site::Int, N::Int) = local_operator(pauli_z, site, N)

"""
    sigma_zz(site, N)

Nearest-neighbor two-site operator `σᶻᵢ σᶻᵢ₊₁` on `(site, site + 1)`.
"""
function sigma_zz(site::Int, N::Int)
    site < N || throw(ArgumentError("site must satisfy site <= N - 1 for a nearest-neighbor ZZ operator"))
    return two_site_operator(pauli_z, site, pauli_z, site + 1, N)
end

"""
    tfim_hamiltonian(params)

Dense Hamiltonian for the 1D TFIM with open boundary conditions.
"""
function tfim_hamiltonian(params::TFIMParameters)
    validate_parameters(params)

    N = params.N
    dim = 2^N
    H = zeros(ComplexF64, dim, dim)

    for site in 1:(N - 1)
        H .-= params.J .* two_site_operator(pauli_z, site, pauli_z, site + 1, N)
    end

    for site in 1:N
        H .-= params.h .* sigma_x(site, N)
    end

    return Hermitian(H)
end

"""
    tfim_sites(N)

Construct ITensors site indices for spin-1/2.
"""
tfim_sites(N::Int) = siteinds("S=1/2", N)

"""
    tfim_mpo(params)

Construct the 1D TFIM Hamiltonian as an ITensors MPO.

Note: the ITensors spin-1/2 operators `"Sz"` and `"Sx"` are
equal to `σ/2`, so the Pauli-form coefficients become
`σzσz = 4 Sz Sz` and `σx = 2 Sx`.
"""
function tfim_mpo(params::TFIMParameters)
    validate_parameters(params)

    sites = tfim_sites(params.N)
    os = OpSum()

    for site in 1:(params.N - 1)
        os += -4.0 * params.J, "Sz", site, "Sz", site + 1
    end

    for site in 1:params.N
        os += -2.0 * params.h, "Sx", site
    end

    return MPO(os, sites), sites
end

"""
    mps_ground_state(params; nsweeps = 8, maxdim = [10, 20, 50, 100, 100, 200, 200, 400], cutoff = 1e-10, linkdims = 2)

Compute the TFIM ground state with ITensors DMRG/MPS.
Returns the named tuple `(energy, state, hamiltonian, sites)`.
"""
function mps_ground_state(
    params::TFIMParameters;
    nsweeps::Int = 8,
    maxdim::AbstractVector = [10, 20, 50, 100, 100, 200, 200, 400],
    cutoff::Real = 1e-10,
    linkdims::Int = 2,
    outputlevel::Int = 0,
)
    hamiltonian, sites = tfim_mpo(params)
    psi0 = random_mps(sites; linkdims = linkdims)
    energy, psi = dmrg(
        hamiltonian,
        psi0;
        nsweeps = nsweeps,
        maxdim = collect(maxdim),
        cutoff = [cutoff],
        outputlevel = outputlevel,
    )
    return (energy = real(energy), state = psi, hamiltonian = hamiltonian, sites = sites)
end

"""
    ground_state(params)

Compute the ground-state energy and ground-state vector of the TFIM Hamiltonian.
"""
function ground_state(params::TFIMParameters)
    eigenpairs = eigen(tfim_hamiltonian(params))
    index = argmin(eigenpairs.values)
    energy = real(eigenpairs.values[index])
    state = eigenpairs.vectors[:, index]
    return energy, state
end

"""
    normalize_state(state)

Normalize a state vector to `|ψ⟩ / ||ψ||`.
"""
function normalize_state(state::AbstractVector)
    state_norm = norm(state)
    iszero(state_norm) && throw(ArgumentError("state must not have zero norm"))
    return state / state_norm
end

"""
    expectation_value(state, operator)

Expectation value `⟨ψ|O|ψ⟩` for a normalized state.
"""
function expectation_value(state::AbstractVector, operator::AbstractMatrix)
    normalized_state = normalize_state(state)
    return real(dot(normalized_state, operator * normalized_state))
end

"""
    expectation_value(operator, state)

Compatibility method for the legacy argument order.
"""
function expectation_value(operator::AbstractMatrix, state::AbstractVector)
    return expectation_value(state, operator)
end

"""
    energy_expectation(state, hamiltonian)

Energy expectation value `⟨ψ|H|ψ⟩`.
"""
function energy_expectation(state::AbstractVector, hamiltonian::AbstractMatrix)
    return expectation_value(state, hamiltonian)
end

"""
    single_spin_state(θ, ϕ = 0)

Local state
`|φ(θ, ϕ)⟩ = cos(θ/2)|0⟩ + exp(iϕ) sin(θ/2)|1⟩`.
"""
function single_spin_state(theta::Real, phi::Real = 0.0)
    return normalize_state(ComplexF64[cos(theta / 2), cis(phi) * sin(theta / 2)])
end

"""
    product_state_ansatz(N, θ)

Uniform product-state ansatz for `N` spins built from the local state `|φ(θ)⟩`.
"""
function product_state_ansatz(N::Int, theta::Real)
    N >= 1 || throw(ArgumentError("N must be >= 1"))

    local_state = single_spin_state(theta)
    state = local_state

    for _ in 2:N
        state = kron(state, local_state)
    end

    return normalize_state(state)
end

"""
    product_state_ansatz(thetas, phis = zeros(length(thetas)))

Product-state ansatz with site-dependent local parameters.
"""
function product_state_ansatz(thetas::AbstractVector, phis::AbstractVector = zeros(length(thetas)))
    length(thetas) == length(phis) || throw(ArgumentError("thetas and phis must have the same length"))
    !isempty(thetas) || throw(ArgumentError("the number of sites must be >= 1"))

    state = single_spin_state(thetas[1], phis[1])
    for site in 2:length(thetas)
        state = kron(state, single_spin_state(thetas[site], phis[site]))
    end

    return normalize_state(state)
end

"""
    zz_entangler(site, N, γ)

Nearest-neighbor entangler
`U_ZZ(γ) = exp(-im * γ * σᶻᵢσᶻᵢ₊₁)`.
"""
function zz_entangler(site::Int, N::Int, gamma::Real)
    zz = sigma_zz(site, N)
    identity_matrix = Matrix{ComplexF64}(I, size(zz, 1), size(zz, 2))
    return cos(gamma) .* identity_matrix .- im * sin(gamma) .* zz
end

"""
    apply_zz_entangler(state, site, N, γ)

Apply the nearest-neighbor `ZZ` entangler to a state.
"""
function apply_zz_entangler(state::AbstractVector, site::Int, N::Int, gamma::Real)
    return normalize_state(zz_entangler(site, N, gamma) * state)
end

"""
    entangled_product_state_ansatz(thetas, gammas; phis = zeros(length(thetas)))

Simple entangled ansatz:
`|ψ(θs, γs)⟩ = [∏ᵢ U_ZZ(γᵢ)] |Φ(θs, ϕs)⟩`
with `|Φ⟩` a site-dependent product state.
"""
function entangled_product_state_ansatz(
    thetas::AbstractVector,
    gammas::AbstractVector;
    phis::AbstractVector = zeros(length(thetas)),
)
    N = length(thetas)
    length(gammas) == N - 1 || throw(ArgumentError("gammas must have length N - 1"))

    state = product_state_ansatz(thetas, phis)
    for site in 1:(N - 1)
        state = apply_zz_entangler(state, site, N, gammas[site])
    end

    return normalize_state(state)
end

"""
    variational_energy(θ, H, N)

Energy objective for the single-parameter product-state ansatz.
"""
function variational_energy(theta::Real, hamiltonian::AbstractMatrix, N::Int)
    trial_state = product_state_ansatz(N, theta)
    return energy_expectation(trial_state, hamiltonian)
end

"""
    variational_energy(thetas, hamiltonian; phis = zeros(length(thetas)))

Variational energy for the site-dependent product-state ansatz.
"""
function variational_energy(
    thetas::AbstractVector,
    hamiltonian::AbstractMatrix;
    phis::AbstractVector = zeros(length(thetas)),
)
    trial_state = product_state_ansatz(thetas, phis)
    return energy_expectation(trial_state, hamiltonian)
end

"""
    entangled_variational_energy(thetas, gammas, hamiltonian; phis = zeros(length(thetas)))

Variational energy for the `ZZ`-entangled product-state ansatz.
"""
function entangled_variational_energy(
    thetas::AbstractVector,
    gammas::AbstractVector,
    hamiltonian::AbstractMatrix;
    phis::AbstractVector = zeros(length(thetas)),
)
    trial_state = entangled_product_state_ansatz(thetas, gammas; phis = phis)
    return energy_expectation(trial_state, hamiltonian)
end

"""
    optimize_product_state_theta(hamiltonian, N; lower = -2π, upper = 2π)

Optimize the single product-state parameter `θ` with `Optim.jl`.
Returns `(theta_opt, energy_opt, result)`.
"""
function optimize_product_state_theta(
    hamiltonian::AbstractMatrix,
    N::Int;
    lower::Real = -2pi,
    upper::Real = 2pi,
)
    objective(theta) = variational_energy(theta, hamiltonian, N)
    result = Optim.optimize(objective, lower, upper, Optim.Brent())
    theta_opt = Optim.minimizer(result)
    energy_opt = Optim.minimum(result)
    return theta_opt, energy_opt, result
end

"""
    optimize_product_state(hamiltonian, N; initial_thetas, initial_phis)

Multi-parameter optimization for the site-dependent product-state ansatz.
Returns `(thetas_opt, phis_opt, energy_opt, result)`.
"""
function optimize_product_state(
    hamiltonian::AbstractMatrix,
    N::Int;
    initial_thetas::AbstractVector = fill(pi / 2, N),
    initial_phis::AbstractVector = zeros(N),
    iterations::Int = 2_000,
)
    validate_site_parameter_lengths(N, initial_thetas, initial_phis)

    initial_params = vcat(collect(float.(initial_thetas)), collect(float.(initial_phis)))
    initial_energy = variational_energy(initial_thetas, hamiltonian; phis = initial_phis)
    objective(params) = variational_energy(view(params, 1:N), hamiltonian; phis = view(params, (N + 1):(2N)))

    result = Optim.optimize(objective, initial_params, Optim.NelderMead(), Optim.Options(iterations = iterations))
    best_params, best_energy = best_result_or_initial(result, initial_params, initial_energy)

    thetas_opt = collect(best_params[1:N])
    phis_opt = collect(best_params[(N + 1):(2N)])
    return thetas_opt, phis_opt, best_energy, result
end

"""
    optimize_entangled_product_state(hamiltonian, N; initial_thetas, initial_phis, initial_gammas)

Multi-parameter optimization for the `ZZ`-entangled ansatz.
Returns `(thetas_opt, phis_opt, gammas_opt, energy_opt, result)`.
"""
function optimize_entangled_product_state(
    hamiltonian::AbstractMatrix,
    N::Int;
    initial_thetas::AbstractVector = fill(pi / 2, N),
    initial_phis::AbstractVector = zeros(N),
    initial_gammas::AbstractVector = zeros(N - 1),
    iterations::Int = 4_000,
)
    validate_site_parameter_lengths(N, initial_thetas, initial_phis)
    length(initial_gammas) == N - 1 || throw(ArgumentError("initial_gammas must have length N - 1"))

    product_thetas, product_phis, product_energy, _ = optimize_product_state(
        hamiltonian,
        N;
        initial_thetas = initial_thetas,
        initial_phis = initial_phis,
        iterations = iterations,
    )

    initial_params = vcat(product_thetas, product_phis, collect(float.(initial_gammas)))
    initial_energy = entangled_variational_energy(product_thetas, initial_gammas, hamiltonian; phis = product_phis)
    if product_energy < initial_energy
        initial_params = vcat(product_thetas, product_phis, zeros(N - 1))
        initial_energy = product_energy
    end

    function objective(params)
        thetas = view(params, 1:N)
        phis = view(params, (N + 1):(2N))
        gammas = view(params, (2N + 1):(3N - 1))
        return entangled_variational_energy(thetas, gammas, hamiltonian; phis = phis)
    end

    result = Optim.optimize(objective, initial_params, Optim.NelderMead(), Optim.Options(iterations = iterations))
    best_params, best_energy = best_result_or_initial(result, initial_params, initial_energy)

    thetas_opt = collect(best_params[1:N])
    phis_opt = collect(best_params[(N + 1):(2N)])
    gammas_opt = collect(best_params[(2N + 1):(3N - 1)])
    return thetas_opt, phis_opt, gammas_opt, best_energy, result
end

function expectation_value(state::AbstractVector, operator::Hermitian)
    return expectation_value(state, Matrix(operator))
end

function energy_expectation(state::AbstractVector, hamiltonian::Hermitian)
    return energy_expectation(state, Matrix(hamiltonian))
end

function variational_energy(theta::Real, hamiltonian::Hermitian, N::Int)
    return variational_energy(theta, Matrix(hamiltonian), N)
end

function variational_energy(
    thetas::AbstractVector,
    hamiltonian::Hermitian;
    phis::AbstractVector = zeros(length(thetas)),
)
    return variational_energy(thetas, Matrix(hamiltonian); phis = phis)
end

function entangled_variational_energy(
    thetas::AbstractVector,
    gammas::AbstractVector,
    hamiltonian::Hermitian;
    phis::AbstractVector = zeros(length(thetas)),
)
    return entangled_variational_energy(thetas, gammas, Matrix(hamiltonian); phis = phis)
end

function optimize_product_state_theta(
    hamiltonian::Hermitian,
    N::Int;
    lower::Real = -2pi,
    upper::Real = 2pi,
)
    return optimize_product_state_theta(Matrix(hamiltonian), N; lower = lower, upper = upper)
end

function optimize_product_state(
    hamiltonian::Hermitian,
    N::Int;
    initial_thetas::AbstractVector = fill(pi / 2, N),
    initial_phis::AbstractVector = zeros(N),
    iterations::Int = 2_000,
)
    return optimize_product_state(
        Matrix(hamiltonian),
        N;
        initial_thetas = initial_thetas,
        initial_phis = initial_phis,
        iterations = iterations,
    )
end

function optimize_entangled_product_state(
    hamiltonian::Hermitian,
    N::Int;
    initial_thetas::AbstractVector = fill(pi / 2, N),
    initial_phis::AbstractVector = zeros(N),
    initial_gammas::AbstractVector = zeros(N - 1),
    iterations::Int = 4_000,
)
    return optimize_entangled_product_state(
        Matrix(hamiltonian),
        N;
        initial_thetas = initial_thetas,
        initial_phis = initial_phis,
        initial_gammas = initial_gammas,
        iterations = iterations,
    )
end

function expectation_value(state::AbstractVector, operator::UniformScaling)
    identity_matrix = Matrix(operator, length(state), length(state))
    return expectation_value(state, identity_matrix)
end

function expectation_value(operator::UniformScaling, state::AbstractVector)
    return expectation_value(state, operator)
end

function energy_expectation(state::AbstractVector, hamiltonian::UniformScaling)
    return expectation_value(state, hamiltonian)
end

function energy_expectation(hamiltonian::AbstractMatrix, state::AbstractVector)
    return energy_expectation(state, hamiltonian)
end

function energy_expectation(hamiltonian::Hermitian, state::AbstractVector)
    return energy_expectation(state, hamiltonian)
end

function energy_expectation(hamiltonian::UniformScaling, state::AbstractVector)
    return energy_expectation(state, hamiltonian)
end

"""
    correlation_zz(state, r)

Two-point `zz` correlation with a simple translation average,
`(1/(N-r)) Σᵢ ⟨σᶻᵢ σᶻᵢ₊ᵣ⟩`.
"""
function correlation_zz(state::AbstractVector, r::Int)
    return pauli_correlation_average(state, pauli_z, r)
end

"""
    correlation_xx(state, r)

Two-point `xx` correlation with a simple translation average,
`(1/(N-r)) Σᵢ ⟨σˣᵢ σˣᵢ₊ᵣ⟩`.
"""
function correlation_xx(state::AbstractVector, r::Int)
    return pauli_correlation_average(state, pauli_x, r)
end

"""
    connected_correlation_zz(state, r)

Connected `zz` correlation,
`(1/(N-r)) Σᵢ [⟨σᶻᵢ σᶻᵢ₊ᵣ⟩ - ⟨σᶻᵢ⟩⟨σᶻᵢ₊ᵣ⟩]`.
"""
function connected_correlation_zz(state::AbstractVector, r::Int)
    N = infer_num_sites(state)
    validate_correlation_distance(r, N)

    if r == 0
        return correlation_zz(state, 0) - mean_site_expectation_product(state, pauli_z, 0, N)
    end

    values = Float64[]
    for site in 1:(N - r)
        corr = expectation_value(state, two_site_operator(pauli_z, site, pauli_z, site + r, N))
        local_product = expectation_value(state, sigma_z(site, N)) * expectation_value(state, sigma_z(site + r, N))
        push!(values, corr - local_product)
    end
    return sum(values) / length(values)
end

"""
    correlation_zz(psi, r)

Translation-averaged `σᶻσᶻ` two-point correlation for an MPS.
"""
function correlation_zz(psi::MPS, r::Int)
    return mps_pauli_correlation_average(psi, "Sz", "Sz", 4.0, r)
end

"""
    correlation_xx(psi, r)

Translation-averaged `σˣσˣ` two-point correlation for an MPS.
"""
function correlation_xx(psi::MPS, r::Int)
    return mps_pauli_correlation_average(psi, "Sx", "Sx", 4.0, r)
end

"""
    connected_correlation_zz(psi, r)

Translation-averaged connected `σᶻσᶻ` correlation for an MPS.
"""
function connected_correlation_zz(psi::MPS, r::Int)
    N = length(psi)
    validate_correlation_distance(r, N)

    corr_matrix = correlation_matrix(psi, "Sz", "Sz")
    local_sz = expect(psi, "Sz")
    values = Float64[]

    for site in 1:(N - r)
        corr = 4 * real(corr_matrix[site, site + r])
        local_product = 4 * real(local_sz[site] * local_sz[site + r])
        push!(values, corr - local_product)
    end

    return sum(values) / length(values)
end

"""
    correlation_profile_zz(state, max_r)

`zz` correlation profile for `r = 0:max_r`.
"""
function correlation_profile_zz(state, max_r::Int)
    return [correlation_zz(state, r) for r in correlation_distances(state, max_r)]
end

"""
    correlation_profile_xx(state, max_r)

`xx` correlation profile for `r = 0:max_r`.
"""
function correlation_profile_xx(state, max_r::Int)
    return [correlation_xx(state, r) for r in correlation_distances(state, max_r)]
end

"""
    connected_correlation_profile_zz(state, max_r)

Connected `zz` correlation profile for `r = 0:max_r`.
"""
function connected_correlation_profile_zz(state, max_r::Int)
    return [connected_correlation_zz(state, r) for r in correlation_distances(state, max_r)]
end

"""
    sweep_h_over_J(method, Ns, h_over_J_values; J = 1.0, product_iterations = 1500, mps_kwargs...)

Run a TFIM observable sweep over several `h/J` values and system sizes `N`.
`method` must be one of `:exact`, `:product`, or `:mps`.
Returns a vector of `NamedTuple`s.
"""
function sweep_h_over_J(
    method::Symbol,
    Ns::AbstractVector{<:Integer},
    h_over_J_values::AbstractVector{<:Real};
    J::Real = 1.0,
    product_iterations::Int = 1_500,
    mps_kwargs...,
)
    rows = NamedTuple[]

    for N in Ns
        for h_over_J in h_over_J_values
            params = TFIMParameters(N = Int(N), J = float(J), h = float(J * h_over_J))
            result = solve_tfim_point(method, params; product_iterations = product_iterations, mps_kwargs...)
            push!(rows, (
                method = String(method),
                N = params.N,
                J = params.J,
                h = params.h,
                h_over_J = params.h / params.J,
                E0 = result.energy,
                e0 = result.energy / params.N,
                Mz = result.mz,
                Mx = result.mx,
            ))
        end
    end

    return rows
end

"""
    add_finite_difference_derivative(rows; field = :e0, xfield = :h_over_J, groupfields = (:method, :N), output_name = nothing)

Add a finite-difference derivative estimate to each sweep row.
Endpoints use one-sided differences and interior points use central differences.
"""
function add_finite_difference_derivative(
    rows::AbstractVector;
    field::Symbol = :e0,
    xfield::Symbol = :h_over_J,
    groupfields::Tuple = (:method, :N),
    output_name::Union{Nothing, Symbol} = nothing,
)
    output_name = isnothing(output_name) ? Symbol("d$(field)_d$(xfield)") : output_name
    grouped = Dict{NTuple{length(groupfields), Any}, Vector{NamedTuple}}()
    for row in rows
        key = ntuple(index -> getproperty(row, groupfields[index]), length(groupfields))
        push!(get!(grouped, key, NamedTuple[]), row)
    end

    enriched_rows = NamedTuple[]
    for key in sort(collect(keys(grouped)))
        group = grouped[key]
        sort!(group, by = row -> getproperty(row, xfield))
        derivatives = finite_difference_estimate(group, field, xfield)
        for (row, derivative) in zip(group, derivatives)
            push!(enriched_rows, merge(row, NamedTuple{(output_name,)}((derivative,))))
        end
    end

    return enriched_rows
end

"""
    save_sweep_results_csv(path, rows)

Write sweep results to a simple CSV file.
"""
function save_sweep_results_csv(path::AbstractString, rows::AbstractVector)
    isempty(rows) && throw(ArgumentError("rows must not be empty"))

    directory = dirname(path)
    isempty(directory) || mkpath(directory)

    keys_order = ordered_csv_keys(rows)

    open(path, "w") do io
        println(io, join(string.(keys_order), ","))
        for row in rows
            values = [row[key] for key in keys_order]
            println(io, join(csv_escape.(values), ","))
        end
    end

    return path
end

"""
    magnetization_z(state, N)

Average magnetization `Mz = (1/N) Σ ⟨σᶻᵢ⟩`.
"""
function magnetization_z(state::AbstractVector, N::Int)
    return sum(expectation_value(state, sigma_z(site, N)) for site in 1:N) / N
end

"""
    magnetization_z(psi::MPS)

Average `Mz` for an MPS in the Pauli-operator convention.
"""
function magnetization_z(psi::MPS)
    values = expect(psi, "Sz")
    return 2 * real(sum(values)) / length(values)
end

"""
    magnetization_x(state, N)

Average magnetization `Mx = (1/N) Σ ⟨σˣᵢ⟩`.
"""
function magnetization_x(state::AbstractVector, N::Int)
    return sum(expectation_value(state, sigma_x(site, N)) for site in 1:N) / N
end

"""
    magnetization_x(psi::MPS)

Average `Mx` for an MPS in the Pauli-operator convention.
"""
function magnetization_x(psi::MPS)
    values = expect(psi, "Sx")
    return 2 * real(sum(values)) / length(values)
end

function validate_parameters(params::TFIMParameters)
    params.N >= 1 || throw(ArgumentError("N must be >= 1"))
    return nothing
end

function validate_site_parameter_lengths(N::Int, thetas::AbstractVector, phis::AbstractVector)
    length(thetas) == N || throw(ArgumentError("thetas must have length N"))
    length(phis) == N || throw(ArgumentError("phis must have length N"))
    return nothing
end

function best_result_or_initial(result, initial_params::AbstractVector, initial_energy::Real)
    optimized_energy = Optim.minimum(result)
    if optimized_energy <= initial_energy
        return Optim.minimizer(result), optimized_energy
    end
    return initial_params, initial_energy
end

function infer_num_sites(state::AbstractVector)
    state_length = length(state)
    N = round(Int, log2(state_length))
    2^N == state_length || throw(ArgumentError("state length must be a power of two"))
    return N
end

function validate_correlation_distance(r::Int, N::Int)
    0 <= r < N || throw(ArgumentError("correlation distance r must satisfy 0 <= r < N"))
    return nothing
end

function correlation_distances(state, max_r::Int)
    N = num_sites_for_correlations(state)
    upper = min(max_r, N - 1)
    upper >= 0 || throw(ArgumentError("max_r must be >= 0"))
    return 0:upper
end

function pauli_correlation_average(state::AbstractVector, single_site_pauli::AbstractMatrix, r::Int)
    N = infer_num_sites(state)
    validate_correlation_distance(r, N)

    if r == 0
        return 1.0
    end

    values = Float64[]
    for site in 1:(N - r)
        operator = two_site_operator(single_site_pauli, site, single_site_pauli, site + r, N)
        push!(values, expectation_value(state, operator))
    end
    return sum(values) / length(values)
end

function mean_site_expectation_product(state::AbstractVector, single_site_pauli::AbstractMatrix, r::Int, N::Int)
    values = Float64[]
    for site in 1:(N - r)
        local_i = expectation_value(state, local_operator(single_site_pauli, site, N))
        local_j = expectation_value(state, local_operator(single_site_pauli, site + r, N))
        push!(values, local_i * local_j)
    end
    return sum(values) / length(values)
end

function mps_pauli_correlation_average(
    psi::MPS,
    operator_name_1::AbstractString,
    operator_name_2::AbstractString,
    pauli_factor::Real,
    r::Int,
)
    N = length(psi)
    validate_correlation_distance(r, N)

    corr_matrix = correlation_matrix(psi, operator_name_1, operator_name_2)
    values = [pauli_factor * real(corr_matrix[site, site + r]) for site in 1:(N - r)]
    return sum(values) / length(values)
end

num_sites_for_correlations(state::AbstractVector) = infer_num_sites(state)
num_sites_for_correlations(psi::MPS) = length(psi)

function finite_difference_estimate(group::AbstractVector, field::Symbol, xfield::Symbol)
    n = length(group)
    n >= 2 || throw(ArgumentError("each group needs at least two points for a numerical derivative"))

    derivatives = zeros(Float64, n)
    xs = [Float64(getproperty(row, xfield)) for row in group]
    ys = [Float64(getproperty(row, field)) for row in group]

    derivatives[1] = (ys[2] - ys[1]) / (xs[2] - xs[1])
    for index in 2:(n - 1)
        derivatives[index] = (ys[index + 1] - ys[index - 1]) / (xs[index + 1] - xs[index - 1])
    end
    derivatives[n] = (ys[n] - ys[n - 1]) / (xs[n] - xs[n - 1])

    return derivatives
end

function ordered_csv_keys(rows::AbstractVector)
    preferred = [:method, :N, :J, :h, :h_over_J, :E0, :e0, :Mz, :Mx, :de0_dh_over_J, :dMz_dh_over_J, :dMx_dh_over_J]
    present = Set{Symbol}()
    for row in rows
        union!(present, propertynames(row))
    end

    ordered = Symbol[symbol for symbol in preferred if symbol in present]
    for symbol in sort!(collect(present))
        symbol in ordered || push!(ordered, symbol)
    end
    return Tuple(ordered)
end

function solve_tfim_point(
    method::Symbol,
    params::TFIMParameters;
    product_iterations::Int = 1_500,
    mps_kwargs...,
)
    if method == :exact
        energy, state = ground_state(params)
        return (energy = energy, mz = magnetization_z(state, params.N), mx = magnetization_x(state, params.N))
    elseif method == :product
        hamiltonian = tfim_hamiltonian(params)
        thetas, phis, energy, _ = optimize_product_state(hamiltonian, params.N; iterations = product_iterations)
        state = product_state_ansatz(thetas, phis)
        return (energy = energy, mz = magnetization_z(state, params.N), mx = magnetization_x(state, params.N))
    elseif method == :mps
        result = mps_ground_state(params; mps_kwargs...)
        return (energy = result.energy, mz = magnetization_z(result.state), mx = magnetization_x(result.state))
    end

    throw(ArgumentError("method must be one of :exact, :product, or :mps"))
end

csv_escape(value) = value isa AbstractString ? value : string(value)

function local_operator(single_site_operator::AbstractMatrix, site::Int, N::Int)
    1 <= site <= N || throw(ArgumentError("site must be between 1 and N"))

    operator_factors = Matrix{ComplexF64}[]
    sizehint!(operator_factors, N)

    for current_site in 1:N
        push!(operator_factors, current_site == site ? single_site_operator : identity_2)
    end

    return kron_all(operator_factors)
end

function two_site_operator(
    first_operator::AbstractMatrix,
    first_site::Int,
    second_operator::AbstractMatrix,
    second_site::Int,
    N::Int,
)
    1 <= first_site <= N || throw(ArgumentError("first_site must be between 1 and N"))
    1 <= second_site <= N || throw(ArgumentError("second_site must be between 1 and N"))
    first_site != second_site || throw(ArgumentError("sites must be distinct"))

    operator_factors = Matrix{ComplexF64}[]
    sizehint!(operator_factors, N)

    for current_site in 1:N
        if current_site == first_site
            push!(operator_factors, first_operator)
        elseif current_site == second_site
            push!(operator_factors, second_operator)
        else
            push!(operator_factors, identity_2)
        end
    end

    return kron_all(operator_factors)
end

function kron_all(operators::Vector{Matrix{ComplexF64}})
    result = operators[1]
    for operator in @view operators[2:end]
        result = kron(result, operator)
    end
    return result
end

end
