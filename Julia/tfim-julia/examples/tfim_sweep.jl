using Pkg
ENV["GKSwstype"] = "100"
using Plots
using Printf

Pkg.activate(joinpath(@__DIR__, ".."))
using Qjulia

const SERIES_COLORS = [
    "#0b3954",
    "#087e8b",
    "#bfd7ea",
    "#ff5a5f",
    "#c81d25",
    "#f4d35e",
    "#5c415d",
    "#3c6e71",
]

function plot_defaults()
    default(
        fontfamily = "DejaVu Sans",
        lw = 2.5,
        ms = 4,
        gridalpha = 0.2,
        legendfontsize = 8,
        guidefontsize = 11,
        tickfontsize = 9,
        titlefontsize = 13,
        background_color = RGB(0.985, 0.985, 0.985),
        foreground_color_legend = :transparent,
        legend_background_color = RGBA(1, 1, 1, 0.75),
        palette = SERIES_COLORS,
    )
end

function grouped_series(rows)
    groups = Dict{Tuple{String, Int}, Vector{NamedTuple}}()
    for row in rows
        key = (row.method, row.N)
        push!(get!(groups, key, NamedTuple[]), row)
    end

    for values in values(groups)
        sort!(values, by = row -> row.h_over_J)
    end

    return sort(collect(groups); by = item -> (item[1][1], item[1][2]))
end

function plot_metric(rows, metric::Symbol, ylabel::AbstractString, path::AbstractString; title::AbstractString = "")
    plt = plot(
        xlabel = "h/J",
        ylabel = ylabel,
        title = title,
        markershape = :circle,
        legend = :best,
        framestyle = :box,
    )

    for ((method, N), values) in grouped_series(rows)
        x = [row.h_over_J for row in values]
        y = [getproperty(row, metric) for row in values]
        plot!(plt, x, y; label = "$(method), N=$(N)")
    end

    savefig(plt, path)
    return path
end

function run_sweep(; quick_mode::Bool = false, critical_mode::Bool = false)
    if critical_mode
        h_values = quick_mode ? collect(0.8:0.1:1.2) : collect(0.8:0.05:1.2)
        exact_sizes = quick_mode ? [10] : [10]
        product_sizes = [10]
        mps_sizes = quick_mode ? [10, 20] : [10, 20, 40]
        output_dir = joinpath(@__DIR__, "..", "outputs", quick_mode ? "tfim_critical_quick" : "tfim_critical")
    else
        h_values = quick_mode ? collect(0.2:0.6:2.0) : collect(0.2:0.2:2.0)
        exact_sizes = quick_mode ? [4, 6] : [4, 6, 10]
        product_sizes = [4, 6, 10]
        mps_sizes = quick_mode ? [10, 20] : [10, 20, 40]
        output_dir = joinpath(@__DIR__, "..", "outputs", quick_mode ? "tfim_sweep_quick" : "tfim_sweep")
    end

    mkpath(output_dir)

    exact_rows = sweep_h_over_J(:exact, exact_sizes, h_values)
    product_rows = sweep_h_over_J(:product, product_sizes, h_values; product_iterations = quick_mode ? 500 : 1_500)
    mps_rows = sweep_h_over_J(
        :mps,
        mps_sizes,
        h_values;
        nsweeps = quick_mode ? 5 : 8,
        maxdim = quick_mode ? [10, 20, 40, 80, 120] : [10, 20, 50, 100, 200, 300, 400, 600],
        cutoff = 1e-10,
        linkdims = 2,
    )

    all_rows = vcat(exact_rows, product_rows, mps_rows)
    enriched_rows = add_finite_difference_derivative(all_rows; field = :e0, output_name = :de0_dh_over_J)
    enriched_rows = add_finite_difference_derivative(enriched_rows; field = :Mz, output_name = :dMz_dh_over_J)
    enriched_rows = add_finite_difference_derivative(enriched_rows; field = :Mx, output_name = :dMx_dh_over_J)

    exact_csv = save_sweep_results_csv(joinpath(output_dir, "exact_results.csv"), exact_rows)
    product_csv = save_sweep_results_csv(joinpath(output_dir, "product_results.csv"), product_rows)
    mps_csv = save_sweep_results_csv(joinpath(output_dir, "mps_results.csv"), mps_rows)
    combined_csv = save_sweep_results_csv(joinpath(output_dir, "combined_results.csv"), enriched_rows)

    prefix = critical_mode ? "Critical crossover" : "Wide sweep"
    energy_plot = plot_metric(enriched_rows, :e0, "Ground-state energy per site", joinpath(output_dir, "energy_per_site_vs_h_over_J.png"); title = "$prefix: e₀ vs h/J")
    mz_plot = plot_metric(enriched_rows, :Mz, "Mz", joinpath(output_dir, "mz_vs_h_over_J.png"); title = "$prefix: Mz vs h/J")
    mx_plot = plot_metric(enriched_rows, :Mx, "Mx", joinpath(output_dir, "mx_vs_h_over_J.png"); title = "$prefix: Mx vs h/J")
    de_plot = plot_metric(enriched_rows, :de0_dh_over_J, "d(e₀)/d(h/J)", joinpath(output_dir, "de0_dh_over_J.png"); title = "$prefix: derivative of e₀")

    println("========================================")
    println(critical_mode ? "TFIM critical sweep completed" : "TFIM sweep completed")
    println("Output directory:")
    println(output_dir)
    println("Saved CSV files:")
    println(exact_csv)
    println(product_csv)
    println(mps_csv)
    println(combined_csv)
    println("Saved plots:")
    println(energy_plot)
    println(mz_plot)
    println(mx_plot)
    println(de_plot)
    @printf("Total data points = %d\n", length(enriched_rows))
end

function main()
    plot_defaults()
    quick_mode = "--quick" in ARGS
    critical_mode = "--critical" in ARGS
    run_sweep(; quick_mode = quick_mode, critical_mode = critical_mode)
end

main()
