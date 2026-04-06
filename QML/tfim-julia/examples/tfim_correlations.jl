using Pkg
ENV["GKSwstype"] = "100"
using Plots
using Printf

Pkg.activate(joinpath(@__DIR__, ".."))
using Qjulia

const CORRELATION_COLORS = ["#0b3954", "#ff5a5f", "#087e8b", "#f4d35e", "#5c415d", "#c81d25"]

function plot_defaults()
    default(
        fontfamily = "DejaVu Sans",
        lw = 2.6,
        ms = 4,
        gridalpha = 0.2,
        legendfontsize = 8,
        guidefontsize = 11,
        tickfontsize = 9,
        titlefontsize = 13,
        background_color = RGB(0.985, 0.985, 0.985),
        palette = CORRELATION_COLORS,
    )
end

function rows_from_profile(profile_name::String, h_over_J::Real, N::Int, values::AbstractVector)
    return [
        (
            profile = profile_name,
            h_over_J = Float64(h_over_J),
            N = N,
            r = r,
            value = Float64(values[r + 1]),
        ) for r in 0:(length(values) - 1)
    ]
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

function plot_profile(rows, profile_name::String, path::AbstractString, ylabel::AbstractString, title::AbstractString)
    plt = plot(
        xlabel = "distance r",
        ylabel = ylabel,
        title = title,
        markershape = :circle,
        framestyle = :box,
        legend = :topright,
    )

    filtered = filter(row -> row.profile == profile_name, rows)
    groups = Dict{Tuple{Float64, Int}, Vector{NamedTuple}}()
    for row in filtered
        key = (row.h_over_J, row.N)
        push!(get!(groups, key, NamedTuple[]), row)
    end

    for ((h_ratio, N), values) in sort(collect(groups); by = item -> (item[1][1], item[1][2]))
        sort!(values, by = row -> row.r)
        x = [row.r for row in values]
        y = [row.value for row in values]
        plot!(plt, x, y; label = "h/J=$(round(h_ratio, digits=2)), N=$N")
    end

    savefig(plt, path)
    return path
end

function main()
    plot_defaults()

    quick_mode = "--quick" in ARGS
    h_values = quick_mode ? [0.6, 1.0] : [0.6, 1.0, 1.4]
    sizes = quick_mode ? [20] : [20, 40]
    output_dir = joinpath(@__DIR__, "..", "outputs", quick_mode ? "tfim_correlations_quick" : "tfim_correlations")
    mkpath(output_dir)

    rows = NamedTuple[]
    for N in sizes
        max_r = N ÷ 2
        for h_ratio in h_values
            params = TFIMParameters(N = N, J = 1.0, h = h_ratio)
            result = mps_ground_state(
                params;
                nsweeps = quick_mode ? 5 : 8,
                maxdim = quick_mode ? [10, 20, 40, 80, 120] : [10, 20, 50, 100, 200, 300, 400, 600],
                cutoff = 1e-10,
                linkdims = 2,
            )

            append!(rows, rows_from_profile("zz", h_ratio, N, correlation_profile_zz(result.state, max_r)))
            append!(rows, rows_from_profile("xx", h_ratio, N, correlation_profile_xx(result.state, max_r)))
            append!(rows, rows_from_profile("connected_zz", h_ratio, N, connected_correlation_profile_zz(result.state, max_r)))
        end
    end

    csv_path = save_profile_csv(joinpath(output_dir, "correlation_profiles.csv"), rows)
    zz_plot = plot_profile(rows, "zz", joinpath(output_dir, "correlation_zz_vs_r.png"), "Czz(r)", "TFIM zz correlations")
    xx_plot = plot_profile(rows, "xx", joinpath(output_dir, "correlation_xx_vs_r.png"), "Cxx(r)", "TFIM xx correlations")
    czz_plot = plot_profile(rows, "connected_zz", joinpath(output_dir, "connected_correlation_zz_vs_r.png"), "Czz_conn(r)", "TFIM connected zz correlations")

    println("========================================")
    println("TFIM correlation analysis completed")
    println("Output directory:")
    println(output_dir)
    println("Saved CSV file:")
    println(csv_path)
    println("Saved plots:")
    println(zz_plot)
    println(xx_plot)
    println(czz_plot)
    @printf("Total correlation points = %d\n", length(rows))
end

main()
