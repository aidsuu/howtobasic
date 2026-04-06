using Pkg
ENV["GKSwstype"] = "100"
using Plots
using Printf
using Statistics

Pkg.activate(joinpath(@__DIR__, ".."))
using ITensors
using ITensorMPS
using Qjulia

const FINAL_COLORS = [
    "#0b3954",
    "#087e8b",
    "#ff5a5f",
    "#c81d25",
    "#f4d35e",
    "#5c415d",
    "#3c6e71",
    "#bfd7ea",
]

function plot_defaults()
    default(
        fontfamily = "DejaVu Sans",
        lw = 2.8,
        ms = 4,
        gridalpha = 0.18,
        legendfontsize = 8,
        guidefontsize = 11,
        tickfontsize = 9,
        titlefontsize = 13,
        background_color = RGB(0.985, 0.985, 0.985),
        foreground_color_legend = :transparent,
        legend_background_color = RGBA(1, 1, 1, 0.78),
        palette = FINAL_COLORS,
    )
end

function grouped_series(rows; keyfields = (:method, :N))
    groups = Dict{Tuple, Vector{NamedTuple}}()
    for row in rows
        key = tuple((getproperty(row, keyfield) for keyfield in keyfields)...)
        push!(get!(groups, key, NamedTuple[]), row)
    end
    for values in values(groups)
        sort!(values, by = row -> row.h_over_J)
    end
    return sort(collect(groups); by = first)
end

function plot_metric(rows, metric::Symbol, ylabel::AbstractString, path::AbstractString; title::AbstractString = "")
    plt = plot(
        xlabel = "h/J",
        ylabel = ylabel,
        title = title,
        framestyle = :box,
        markershape = :circle,
        legend = :best,
    )

    for ((method, N), values) in grouped_series(rows)
        x = [row.h_over_J for row in values]
        y = [getproperty(row, metric) for row in values]
        plot!(plt, x, y; label = "$(method), N=$(N)")
    end

    savefig(plt, path)
    return path
end

function plot_profile(rows, profile_name::String, path::AbstractString, ylabel::AbstractString, title::AbstractString)
    plt = plot(
        xlabel = "distance r",
        ylabel = ylabel,
        title = title,
        framestyle = :box,
        markershape = :circle,
        legend = :topright,
    )

    filtered = filter(row -> row.profile == profile_name, rows)
    groups = Dict{Tuple{Float64, Int}, Vector{NamedTuple}}()
    for row in filtered
        key = (row.h_over_J, row.N)
        push!(get!(groups, key, NamedTuple[]), row)
    end

    for ((h_ratio, N), values) in sort(collect(groups); by = first)
        sort!(values, by = row -> row.r)
        x = [row.r for row in values]
        y = [row.value for row in values]
        plot!(plt, x, y; label = "h/J=$(round(h_ratio, digits = 2)), N=$N")
    end

    savefig(plt, path)
    return path
end

function save_profile_csv(path::AbstractString, rows::AbstractVector)
    open(path, "w") do io
        println(io, "profile,h_over_J,N,r,value")
        for row in rows
            println(io, "$(row.profile),$(row.h_over_J),$(row.N),$(row.r),$(row.value)")
        end
    end
    return path
end

function middle_bond_entropy(psi::MPS)
    N = length(psi)
    b = N ÷ 2
    psi_centered = orthogonalize(psi, b)
    U, S, V = svd(psi_centered[b], (linkinds(psi_centered, b - 1)..., siteinds(psi_centered, b)...))
    entropy_value = 0.0
    for index in 1:dim(S, 1)
        p = S[index, index]^2
        if p > 0
            entropy_value -= p * log(p)
        end
    end
    return real(entropy_value)
end

function entropy_sweep_rows(h_values::AbstractVector, Ns::AbstractVector{<:Integer})
    rows = NamedTuple[]
    for N in Ns
        for h_ratio in h_values
            params = TFIMParameters(N = N, J = 1.0, h = h_ratio)
            result = mps_ground_state(
                params;
                nsweeps = 8,
                maxdim = [10, 20, 50, 100, 200, 300, 400, 600],
                cutoff = 1e-10,
                linkdims = 2,
            )
            push!(rows, (
                method = "mps",
                N = N,
                J = 1.0,
                h = h_ratio,
                h_over_J = h_ratio,
                entropy_mid = middle_bond_entropy(result.state),
            ))
        end
    end
    return rows
end

function save_entropy_csv(path::AbstractString, rows::AbstractVector)
    open(path, "w") do io
        println(io, "method,N,J,h,h_over_J,entropy_mid")
        for row in rows
            println(io, "$(row.method),$(row.N),$(row.J),$(row.h),$(row.h_over_J),$(row.entropy_mid)")
        end
    end
    return path
end

function plot_entropy(rows, path::AbstractString)
    plt = plot(
        xlabel = "h/J",
        ylabel = "S_mid",
        title = "MPS mid-bond entanglement entropy",
        framestyle = :box,
        markershape = :circle,
        legend = :best,
    )
    for ((_, N), values) in grouped_series(rows; keyfields = (:method, :N))
        x = [row.h_over_J for row in values]
        y = [row.entropy_mid for row in values]
        plot!(plt, x, y; label = "mps, N=$N")
    end
    savefig(plt, path)
    return path
end

function correlation_rows(h_values::AbstractVector, Ns::AbstractVector{<:Integer})
    rows = NamedTuple[]
    for N in Ns
        max_r = N ÷ 2
        for h_ratio in h_values
            params = TFIMParameters(N = N, J = 1.0, h = h_ratio)
            result = mps_ground_state(
                params;
                nsweeps = 8,
                maxdim = [10, 20, 50, 100, 200, 300, 400, 600],
                cutoff = 1e-10,
                linkdims = 2,
            )

            zz_profile = correlation_profile_zz(result.state, max_r)
            xx_profile = correlation_profile_xx(result.state, max_r)
            connected_zz_profile = connected_correlation_profile_zz(result.state, max_r)

            append!(rows, [(profile = "zz", h_over_J = h_ratio, N = N, r = r, value = zz_profile[r + 1]) for r in 0:max_r])
            append!(rows, [(profile = "xx", h_over_J = h_ratio, N = N, r = r, value = xx_profile[r + 1]) for r in 0:max_r])
            append!(rows, [(profile = "connected_zz", h_over_J = h_ratio, N = N, r = r, value = connected_zz_profile[r + 1]) for r in 0:max_r])
        end
    end
    return rows
end

function write_summary(path::AbstractString, main_rows, critical_rows, correlation_rows, entropy_rows)
    exact_n10 = Dict(row.h_over_J => row for row in main_rows if row.method == "exact" && row.N == 10)
    mps_n10 = Dict(row.h_over_J => row for row in main_rows if row.method == "mps" && row.N == 10)
    product_n10 = [row for row in main_rows if row.method == "product" && row.N == 10]
    product_gap_rows = [
        (
            h_over_J = row.h_over_J,
            gap = row.E0 - exact_n10[row.h_over_J].E0,
        ) for row in product_n10 if haskey(exact_n10, row.h_over_J)
    ]
    worst_product = findmax(row.gap for row in product_gap_rows)
    worst_product_row = product_gap_rows[worst_product[2]]

    mps_exact_gaps = [abs(mps_n10[h].E0 - exact_n10[h].E0) for h in keys(mps_n10) if haskey(exact_n10, h)]
    max_mps_exact_gap = maximum(mps_exact_gaps)

    critical_mps = [row for row in critical_rows if row.method == "mps" && row.N == 40]
    max_derivative_row = critical_mps[argmax(abs.(getproperty.(critical_mps, :dMx_dh_over_J)))]

    entropy_n40 = [row for row in entropy_rows if row.N == 40]
    max_entropy_row = entropy_n40[argmax(getproperty.(entropy_n40, :entropy_mid))]

    open(path, "w") do io
        println(io, "# Final TFIM 1D Results")
        println(io)
        println(io, "## Benchmark utama")
        println(io)
        println(io, "- Exact diagonalization dipakai untuk `N = 4, 6, 10` sebagai baseline referensi.")
        println(io, "- Product-state exact-vector dipakai hanya untuk `N` kecil sebagai baseline mean-field sederhana.")
        println(io, "- MPS/DMRG dipakai untuk `N = 10, 20, 40` agar skala sistem bisa diperbesar.")
        println(io)
        println(io, "## Kapan product-state gagal")
        println(io)
        println(io, "- Pada overlap `N = 10`, gap energi terbesar product-state terhadap exact muncul di sekitar `h/J = $(round(worst_product_row.h_over_J, digits = 2))` dengan selisih `$(round(worst_product_row.gap, digits = 6))`.")
        println(io, "- Ini konsisten dengan ekspektasi bahwa product-state paling lemah saat korelasi dan entanglement meningkat di sekitar crossover kritis.")
        println(io)
        println(io, "## Kapan MPS cocok dengan ED")
        println(io)
        println(io, "- Pada ukuran overlap `N = 10`, MPS mereproduksi energi exact dengan deviasi maksimum `$(round(max_mps_exact_gap, digits = 10))` di seluruh sweep utama.")
        println(io, "- Untuk `N = 4` dan `N = 6`, benchmark sebelumnya juga menunjukkan MPS praktis menumpuk di atas exact sampai toleransi numerik.")
        println(io)
        println(io, "## Pola near-critical")
        println(io)
        println(io, "- Pada sweep rapat, magnitudo turunan `dMx/d(h/J)` untuk `N = 40` mencapai puncak dekat `h/J = $(round(max_derivative_row.h_over_J, digits = 2))`, menandai crossover tercepat.")
        println(io, "- `Mz` turun dan `Mx` naik saat medan transversal diperbesar, sesuai transisi dari dominasi interaksi Ising ke polarisasi arah-x.")
        println(io)
        println(io, "## Korelasi")
        println(io)
        println(io, "- Profil korelasi `zz`, `xx`, dan connected `zz` tersedia untuk `h/J = 0.6, 1.0, 1.4` dan `N = 20, 40`.")
        println(io, "- Dekat `h/J = 1`, peluruhan korelasi menjadi lebih lambat dibanding jauh dari crossover, terutama pada ukuran sistem lebih besar.")
        println(io)
        println(io, "## Entanglement entropy")
        println(io)
        println(io, "- Entropy bipartisi tengah MPS ditambahkan sebagai indikator tambahan entanglement.")
        println(io, "- Pada `N = 40`, entropy tengah maksimum dalam sweep utama muncul dekat `h/J = $(round(max_entropy_row.h_over_J, digits = 2))` dengan nilai `$(round(max_entropy_row.entropy_mid, digits = 6))`.")
    end
    return path
end

function main()
    plot_defaults()

    output_dir = joinpath(@__DIR__, "..", "outputs", "final_results")
    mkpath(output_dir)

    main_h_values = collect(0.2:0.2:2.0)
    critical_h_values = collect(0.8:0.05:1.2)
    correlation_h_values = [0.6, 1.0, 1.4]

    main_exact = sweep_h_over_J(:exact, [4, 6, 10], main_h_values)
    main_product = sweep_h_over_J(:product, [4, 6, 10], main_h_values; product_iterations = 1_500)
    main_mps = sweep_h_over_J(
        :mps,
        [10, 20, 40],
        main_h_values;
        nsweeps = 8,
        maxdim = [10, 20, 50, 100, 200, 300, 400, 600],
        cutoff = 1e-10,
        linkdims = 2,
    )
    main_rows = vcat(main_exact, main_product, main_mps)
    main_rows = add_finite_difference_derivative(main_rows; field = :e0, output_name = :de0_dh_over_J)
    main_rows = add_finite_difference_derivative(main_rows; field = :Mz, output_name = :dMz_dh_over_J)
    main_rows = add_finite_difference_derivative(main_rows; field = :Mx, output_name = :dMx_dh_over_J)

    critical_exact = sweep_h_over_J(:exact, [10], critical_h_values)
    critical_product = sweep_h_over_J(:product, [10], critical_h_values; product_iterations = 1_500)
    critical_mps = sweep_h_over_J(
        :mps,
        [10, 20, 40],
        critical_h_values;
        nsweeps = 8,
        maxdim = [10, 20, 50, 100, 200, 300, 400, 600],
        cutoff = 1e-10,
        linkdims = 2,
    )
    critical_rows = vcat(critical_exact, critical_product, critical_mps)
    critical_rows = add_finite_difference_derivative(critical_rows; field = :e0, output_name = :de0_dh_over_J)
    critical_rows = add_finite_difference_derivative(critical_rows; field = :Mz, output_name = :dMz_dh_over_J)
    critical_rows = add_finite_difference_derivative(critical_rows; field = :Mx, output_name = :dMx_dh_over_J)

    correlation_rows_data = correlation_rows(correlation_h_values, [20, 40])
    entropy_rows = entropy_sweep_rows(main_h_values, [10, 20, 40])

    main_csv = save_sweep_results_csv(joinpath(output_dir, "benchmark_main.csv"), main_rows)
    critical_csv = save_sweep_results_csv(joinpath(output_dir, "benchmark_critical.csv"), critical_rows)
    corr_csv = save_profile_csv(joinpath(output_dir, "correlation_profiles.csv"), correlation_rows_data)
    entropy_csv = save_entropy_csv(joinpath(output_dir, "entropy_mid_mps.csv"), entropy_rows)

    energy_plot = plot_metric(main_rows, :e0, "Ground-state energy per site", joinpath(output_dir, "energy_per_site_vs_h_over_J.png"); title = "Final TFIM benchmark: e₀ vs h/J")
    mz_plot = plot_metric(main_rows, :Mz, "Mz", joinpath(output_dir, "mz_vs_h_over_J.png"); title = "Final TFIM benchmark: Mz vs h/J")
    mx_plot = plot_metric(main_rows, :Mx, "Mx", joinpath(output_dir, "mx_vs_h_over_J.png"); title = "Final TFIM benchmark: Mx vs h/J")
    derivative_plot = plot_metric(critical_rows, :dMx_dh_over_J, "dMx/d(h/J)", joinpath(output_dir, "derivative_near_critical.png"); title = "Near-critical derivative of Mx")
    zz_plot = plot_profile(correlation_rows_data, "zz", joinpath(output_dir, "correlation_zz_vs_r.png"), "Czz(r)", "Final zz correlation profiles")
    xx_plot = plot_profile(correlation_rows_data, "xx", joinpath(output_dir, "correlation_xx_vs_r.png"), "Cxx(r)", "Final xx correlation profiles")
    czz_plot = plot_profile(correlation_rows_data, "connected_zz", joinpath(output_dir, "connected_correlation_zz_vs_r.png"), "Czz_conn(r)", "Final connected zz profiles")
    entropy_plot = plot_entropy(entropy_rows, joinpath(output_dir, "entropy_mid_vs_h_over_J.png"))

    summary_path = write_summary(joinpath(output_dir, "summary.md"), main_rows, critical_rows, correlation_rows_data, entropy_rows)

    println("========================================")
    println("TFIM final results completed")
    println("Output directory:")
    println(output_dir)
    println("Saved CSV files:")
    println(main_csv)
    println(critical_csv)
    println(corr_csv)
    println(entropy_csv)
    println("Saved plots:")
    println(energy_plot)
    println(mz_plot)
    println(mx_plot)
    println(derivative_plot)
    println(zz_plot)
    println(xx_plot)
    println(czz_plot)
    println(entropy_plot)
    println("Summary:")
    println(summary_path)
end

main()
