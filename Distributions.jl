module PlotDistributions

using Statistics, StatsBase
using FHist
using Plots, Measures
using Distributions, SpecialFunctions, LsqFit

function make_AFD(data; Δb=0.05, plot_fig=false, save_plot=false, plot_name="AFD.png", plot_title="AFD", data_label="data", xrange=(-5, 5), min_y_range=1e-3)

    S = size(data)[2]
    non_zero_data = [data[:,i][data[:,i] .> 0.0] for i in 1:S]
    rescaled_data = [(x .- mean(x)) ./ std(x) for x in non_zero_data]
    flatten_data = vcat(rescaled_data...)
    log_data = log.(flatten_data[flatten_data .> 0.0])
    μ_x = mean(flatten_data[flatten_data .> 0.0])
    σ_x = std(flatten_data[flatten_data .> 0.0])
    β = μ_x^2 / σ_x^2

    bmin = round(minimum(log_data))
    bmax = round(maximum(log_data))
    fh = FHist.Hist1D(log_data, binedges=bmin:Δb:bmax)
    
    # Renormalize the histogram and shift the centers
    μ, σ = mean(fh), std(fh)
    centers = bincenters(fh)
    centers .-= μ
    centers ./= sqrt(2 * σ^2)
    norm_counts = bincounts(fh) ./ (integral(fh) * Δb)
    yy = [10^log(norm_counts[norm_counts.>0.0][i]) for i in eachindex(norm_counts[norm_counts.>0.0])]
    centers = centers[norm_counts.>0.0]
    
    xarr = -3.0:0.05:2
    g_gamma = [10^(β*x - μ_x/σ_x^2*exp(x) - loggamma(β) + β*log(μ_x/σ_x^2)) for x in (xarr .* sqrt(2 * σ^2) .+ μ)] # Grilli's Gamma distribution


    fig = plot(yscale=:log10, xlabel="log(abundances)", ylabel="pdf", title=plot_title, yrange=(min_y_range, 1), xrange=xrange, legend=:topleft)
    if plot_fig
        plot!(fig, xarr, g_gamma, color=:black, label="Gamma(β = $(round(β, digits=3)))")
        scatter!(fig, centers, yy, color=:red, label=data_label)
    end

    if save_plot
        savefig(plot_name)
    end

    return Dict("hist" => [centers, yy], "hparams" => Dict("μ" => μ, "σ" => σ), "params" => Dict("β" => β, "μ_x" => μ_x, "σ_x" => σ_x), "fig" => fig)

end

function make_Taylor(data; Δb=0.05, plot_fig=false, save_plot=false, plot_name="Taylor.png", plot_title="Taylor's law", data_label="data")

    S = size(data)[2]
    non_zero_data = [data[:,i][data[:,i] .> 0.0] for i in 1:S]
    
    mean_data = [mean(x) for x in non_zero_data]
    var_data = [var(x) for x in non_zero_data]

    bmin = floor(minimum(log.(mean_data)))
    bmax = ceil(maximum(log.(mean_data)))
    binedges = bmin:Δb:bmax
    centers = 0.5 .* (binedges[2:end] .+ binedges[1:end-1])
    yy = [mean(log.(var_data[(log.(mean_data) .>= binedges[i]) .& (log.(mean_data) .< binedges[i+1])])) for i in 1:length(binedges)-1]
    centers = centers[isfinite.(yy)]
    yy = yy[isfinite.(yy)]

    # Fit data to find Taylor's exponent
    func(x, p) = p[1] .* x .+ p[2]
    p0 = [2.0, 0.0]  # Initial guess for the parameters
    fit = curve_fit(func, centers, yy, p0)
    p_fit = fit.param

    xmin = centers[1] * (1 + 2 / 100)
    xmax = centers[end] * (1 - 2 / 100)
    xarr = floor(xmin):0.05:ceil(xmax)
    fitted_y = func(xarr, p_fit)

    fig = plot(xlabel="log(means)", ylabel="log(variances)", title=plot_title, xrange=(xmin, xmax), legend=:topleft)
    if plot_fig
        plot!(fig, xarr, fitted_y, color=:black, label="y = $(round(p_fit[1], digits=2))x + $(round(p_fit[2], digits=2))")
        scatter!(fig, centers, yy, color=:red, label=data_label)
    end

    if save_plot
        savefig(plot_name)
    end

    return Dict("hist" => [binedges, yy], "params" => Dict("α" => p_fit[1], "q" => p_fit[2]), "fig" => fig)

end

function make_MAD(data; Δb=0.05, plot_fig=false, save_plot=false, plot_name="MAD.png", plot_title="MAD", data_label="data", xrange=(-3,3), min_y_range=1e-7)

    S = size(data)[2]
    non_zero_data = [data[:,i][data[:,i] .> 0.0] for i in 1:S]
    log_data = [mean(log.(x)) for x in non_zero_data]
    
    bmin = floor(minimum(log_data))
    bmax = ceil(maximum(log_data))
    fh = FHist.Hist1D(log_data, binedges=bmin:Δb:bmax)
    
    # Renormalize the histogram and shift the centers
    μ, σ = mean(fh), std(fh)
    centers = bincenters(fh)
    centers .-= μ
    centers ./= sqrt(2 * σ^2)
    norm_counts = bincounts(fh) ./ (integral(fh) * Δb)
    yy = [10^log(norm_counts[norm_counts.>0.0][i] * sqrt(2 * π * σ^2)) for i in eachindex(norm_counts[norm_counts.>0.0])]
    centers = centers[norm_counts.>0.0]
    
    xarr = -3.0:0.05:3.0
    lognorm = [10^(-x^2) for x in xarr] # Gaussian distribution

    fig = plot(yscale=:log10, xlabel="log(abundances)", ylabel="pdf", title=plot_title, yrange=(min_y_range, 5), xrange=xrange, legend=:topleft)
    if plot_fig
        plot!(fig, xarr, lognorm, color=:black, label="Lognormal")
        scatter!(fig, centers, yy, color=:red, label=data_label)
    end

    if save_plot
        savefig(plot_name)
    end

    return Dict("hist" => fh, "fig" => fig)

end

end # end module










        
