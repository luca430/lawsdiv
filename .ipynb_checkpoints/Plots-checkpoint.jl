module Plots

using Statistics, StatsBase, SpecialFunctions
using CairoMakie, FHist

function combine_AFD_histograms(AFDs; nrows=2, ncols=4, fig_size=(1200, 600), data_label=nothing, savepath=nothing)
    fig = Figure(size=fig_size, fontsize = 18)
    colors = Makie.wong_colors()

    for (i, afd) in enumerate(AFDs)
        # Set figure
        row = div(i - 1, ncols) + 1
        col = mod(i - 1, ncols) + 1
        ax = Axis(fig[row, col], xlabel="log(abundances)", ylabel="pdf",
                  yscale=log10, limits=((-3, 3), (1e-6, 1.0)), title=afd["env"])

        # Extract params
        centers, yy = afd["hist"]
        β = afd["params"]["β"]
        μ_x = afd["params"]["μ_x"]
        σ_x = afd["params"]["σ_x"]
        μ = afd["hparams"]["μ"]
        σ = afd["hparams"]["σ"]
        env = afd["env"]
        
        # Theoretical Gamma distribution
        xarr = -3.0:0.05:2
        g_gamma = [10^(β * x - μ_x / σ_x^2 * exp(x) - loggamma(β) + β * log(μ_x / σ_x^2))
                   for x in (xarr .* sqrt(2 * σ^2) .+ μ)]

        # Plot
        lineplot = lines!(ax, xarr, g_gamma, color=:black, linewidth=1.5)
        scatterplot = scatter!(ax, centers, yy, color=colors[mod1(i, length(colors))], markersize=15)
        legend_entries = [lineplot, scatterplot]
        # Legend(fig[row, col], ax, tellwidth=false, halign=:right, valign=:top)
        Legend(fig[row, col], legend_entries, ["Gamma(β = $(round(β, digits=2)))", data_label];
            tellwidth = false,
            patchsize = (8, 8),
            labelsize = 18,
            framevisible = false,
            halign=:left,
            valign=:top)
    end

    if !isnothing(savepath)
        save(savepath, fig; px_per_unit=2)
    end

    return fig
end

function combine_Taylor_histograms(TAYLORs; nrows=2, ncols=4, fig_size=(1200, 600), data_label=nothing, savepath=nothing)
    fig = Figure(size=fig_size, fontsize = 18)
    colors = Makie.wong_colors()

    for (i, taylor) in enumerate(TAYLORs)
        # Set figure
        row = div(i - 1, ncols) + 1
        col = mod(i - 1, ncols) + 1
        ax = Axis(fig[row, col], xlabel = "log(μ)", ylabel = "log(σ²)", title=taylor["env"])

        # Extract params
        centers, yy = taylor["hist"]
        p_fit = [taylor["params"]["α"], taylor["params"]["q"]]
        env = taylor["env"]
        
        # Theoretical curve
        func(x, p) = p[1] .* x .+ p[2]
        xarr = minimum(centers):0.01:maximum(centers)
        fitted_y = func(xarr, p_fit)

        # Plot
        label = p_fit[2] >= 0 ?
            "y = $(round(p_fit[1], digits=2))x + $(round(p_fit[2], digits=2))" :
            "y = $(round(p_fit[1], digits=2))x - $(abs(round(p_fit[2], digits=2)))"
        lineplot = lines!(ax, xarr, fitted_y, color=:black, linewidth=1.5)
        scatterplot = scatter!(ax, centers, yy, color=colors[mod1(i, length(colors))], markersize=15)
        legend_entries = [lineplot, scatterplot]
        Legend(fig[row, col], legend_entries, [label, data_label];
            tellwidth = false,
            patchsize = (8, 8),
            labelsize = 18,
            framevisible = false,
            halign=:left,
            valign=:top)
    end

    if !isnothing(savepath)
        save(savepath, fig; px_per_unit=2)
    end

    return fig
end

function combine_MAD_histograms(MADs; nrows=2, ncols=4, fig_size=(1200, 600), data_label=nothing, savepath=nothing)
    fig = Figure(size=fig_size, fontsize = 18)
    colors = Makie.wong_colors()

    for (i, mad) in enumerate(MADs)
        # Set figure
        row = div(i - 1, ncols) + 1
        col = mod(i - 1, ncols) + 1
        ax = Axis(fig[row, col], xlabel="log(abundances)", ylabel="pdf",
                  yscale=log10, limits=((-3, 3), (5e-6, 5.0)), title=mad["env"])

        # Extract params
        centers, yy = mad["hist"]
        μ_c = mad["hparams"]["μ"]
        σ_c = mad["hparams"]["σ"]
        c = mad["cutoff"]
        env = mad["env"]
        
        # Theoretical Lognormal distribution
        xarr = -3.0:0.05:3.0
        lognorm = [10^(-x^2) for x in xarr]

        # Plot
        lineplot = lines!(ax, xarr, lognorm, color=:black, linewidth=1.5)
        scatterplot = scatter!(ax, centers, yy, color=colors[mod1(i, length(colors))], markersize=15)
        x_cutoff = (log(c) - μ_c) / sqrt(2 * σ_c^2)

        # Add cutoff
        vlines!(ax, [x_cutoff]; color=:black, linestyle=:dash, linewidth=1.5)
        text!(ax, "cutoff";
            position = (x_cutoff - 0.3, 1e-5),
            rotation = π/2,
            align = (:left, :center),
            color = :black,
            fontsize = 16)

        # Add legend
        legend_entries = [lineplot, scatterplot]
        Legend(fig[row, col], legend_entries, ["Lognormal", data_label];
            tellwidth = false,
            patchsize = (8, 8),
            labelsize = 18,
            framevisible = false,
            halign=:left,
            valign=:top)
    end

    if !isnothing(savepath)
        save(savepath, fig; px_per_unit=2)
    end

    return fig
end

function combine_autocorr_plots(AutoCorrs; nrows=2, ncols=4, fig_size=(1200, 600), savepath=nothing)
    fig = Figure(size=fig_size, fontsize=18)
    colors = Makie.wong_colors()

    for (i, corr) in enumerate(AutoCorrs)
        row = div(i - 1, ncols) + 1
        col = mod(i - 1, ncols) + 1

        ax = Axis(fig[row, col],
                  xlabel = "lag",
                  ylabel = "autocorrelation",
                  limits = ((-1, corr["max_lag"]), (-0.1, 1.0)),
                  title = corr["env"])

        # Plot all individual series in gray
        n_series = size(corr["corrs"], 2)
        lags = 0:corr["max_lag"]

        contour_lines = []
        for j in 1:n_series
            trace = vec(corr["corrs"][:, j])
            line = lines!(ax, lags, trace; color = :lightgray, linewidth = 0.5)
            push!(contour_lines, line)
        end

        # Add horizontal line at y = 0.0
        hlines!(ax, [0.0]; color=:black, linestyle=:dash, linewidth=1.5)

        # Plot mean autocorrelation
        mean_line = lines!(ax, lags, vec(corr["mean_corrs"]); color=colors[mod1(i, length(colors))], linewidth = 2)
    end

    if !isnothing(savepath)
        save(savepath, fig; px_per_unit = 2)
    end

    return fig
end

function combine_PSD_plots(PSDs; nrows=2, ncols=4, fig_size=(1200, 600), savepath=nothing)
    fig = Figure(size=fig_size, fontsize=18)
    colors = Makie.wong_colors()

    for (i, psd) in enumerate(PSDs)
        row = div(i - 1, ncols) + 1
        col = mod(i - 1, ncols) + 1

        ax = Axis(fig[row, col],
                  xlabel = "log₁₀(frequency)",
                  ylabel = "log₁₀(PSD)",
                  title = psd["env"])

        freqs, spectrum = psd["PSD"]
        log_f = log10.(freqs)
        log_S = log10.(spectrum)

        psd_line = lines!(ax, log_f, log_S; color=colors[mod1(i, length(colors))], linewidth=2)

        slope = psd["params"]["slope"][1]
        std_slope = psd["params"]["slope"][2]
        intercept = psd["params"]["intercept"][1]
        if !isnothing(psd["frange"])
            log_f_range = psd["frange"][1]:psd["frange"][2]
            fit_line = 10 .^ (intercept .+ slope .* log_f_range)
            fit_line = lines!(ax, log_f_range, log10.(fit_line); color=:black, linestyle=:dash, linewidth=2)
        else
            fit_line = 10 .^ (intercept .+ slope .* log_f)
            fit_line = lines!(ax, log_f, log10.(fit_line); color=:black, linestyle=:dash, linewidth=2)
        end

        Legend(fig[row, col], [psd_line, fit_line],
               [nothing, "α = $(round(slope, digits=2)) ± $(round(std_slope, digits=2))"],
               tellwidth = false,
               patchsize=(8, 8),
               labelsize=18,
               framevisible=false,
               halign=:left,
               valign=:bottom)
    end

    if !isnothing(savepath)
        save(savepath, fig; px_per_unit=2)
    end

    return fig
end

function combine_crossCorr_plots(CrossCorrs; nrows=2, ncols=4, fig_size=(1200, 600), Δb=0.01, savepath=nothing)

    fig = Figure(size=fig_size, fontsize=18)

    for (i, cdict) in enumerate(CrossCorrs)
        row = div(i - 1, ncols) + 1
        col = mod(i - 1, ncols) + 1

        ax = Axis(fig[row, col],
          xlabel = "correlation",
          ylabel = "pdf",
          yscale = log10,
          title = cdict["env"])

        xlims!(ax, -1.0, 1.0)

        lags = cdict["lags"]
        corrs = cdict["cross_corrs"]

        for (j, mat) in enumerate(corrs)
            vals = filter(!isnan, vec([mat[i, j] for i in 1:size(mat, 1), j in 1:size(mat, 2) if i > j]))  # use only lower triangle, no duplicates
            if !isempty(vals)
                edges = -1:Δb:1
                fh = FHist.Hist1D(vals, binedges=edges) |> FHist.normalize
                centers, counts = bincenters(fh), bincounts(fh)
                scatter!(ax, centers, counts; label="lag = $(lags[j])", markersize=15)
                ylims!(ax, minimum(counts[counts .> 0.0]), 1e1)
            end
        end

        Legend(fig[row, col], ax;
                tellwidth = false,
                framevisible = false,
                labelsize = 18,
                patchsize = (8, 8),
                halign=:left,
                valign=:top
            )
    end

    if !isnothing(savepath)
        save(savepath, fig; px_per_unit=2)
    end

    return fig
end

end # End module