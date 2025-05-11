module PlotDistributions

using Statistics, StatsBase
using FHist
using Plots
using Distributions, SpecialFunctions, LsqFit

function make_AFD(data; Δb=0.05, plot_fig=false, save_plot=false, plot_name="AFD.png", plot_title="AFD")

    log_data = log.(data)
    μ_x = mean(data)
    σ_x = std(data)
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
    
    xarr = -3.0:0.05:3.0
    g_gamma = [10^(β*x - μ_x/σ_x^2*exp(x) - loggamma(β) + β*log(μ_x/σ_x^2)) for x in (xarr .* sqrt(2 * σ^2) .+ μ)] # Grilli's Gamma distribution


    fig = plot(yscale=:log10, xlabel="log(abundances)", ylabel="pdf", title=plot_title, yrange=(minimum(yy), maximum(yy) + 5), xrange=(-3, 3))
    if plot_fig
        plot!(fig, xarr, g_gamma, color=:black, label="Gamma(β = $(round(β, digits=3)))")
        scatter!(fig, centers, yy, color=:red, label="simulation")
    end

    if save_plot
        savefig(plot_name)
    end

    return Dict("hist" => fh, "params" => Dict("β" => β, "μ_x" => μ_x, "σ_x" => σ_x), "fig" => fig)

end

function make_Taylor(data; Δb=0.05, plot_fig=false, save_plot=false, plot_name="Taylor.png", plot_title="Taylor's law")

    S = length(data[1,:])
    mean_data = [mean(data[:,i]) for i in 1:S]
    var_data = [var(data[:,i]) for i in 1:S]

    bmin = minimum(log.(mean_data))
    bmax = maximum(log.(mean_data))
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

    fig = plot(xlabel="log(means)", ylabel="log(variances)", title=plot_title, xrange=(xmin, xmax))
    if plot_fig
        plot!(fig, xarr, fitted_y, color=:black, label="y = $(round(p_fit[1], digits=2))x + $(round(p_fit[2], digits=2))")
        scatter!(fig, centers, yy, color=:red, label="simulation")
    end

    if save_plot
        savefig(plot_name)
    end

    return Dict("hist" => [binedges, yy], "params" => Dict("α" => p_fit[1], "q" => p_fit[2]), "fig" => fig)

end

function make_MAD(data; Δb=0.05, plot_fig=false, save_plot=false, plot_name="MAD.png", plot_title="MAD")

    S = length(data[1,:])
    n = length(data[:,1])
    log_data = [mean(log.(data[:,i][data[:,i] .> 0.0])) for i in 1:S]
    
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

    fig = plot(yscale=:log10, xlabel="log(abundances)", ylabel="pdf", title=plot_title, yrange=(minimum(yy), 5))
    if plot_fig
        plot!(fig, xarr, lognorm, color=:black, label="Lognormal")
        scatter!(fig, centers, yy, color=:red, label="simulation")
    end

    if save_plot
        savefig(plot_name)
    end

    return Dict("hist" => fh, "fig" => fig)

end

function AFD(model, p;
        y0 = 1.0, temporal=true, ensemble=true, skip=1,
        Δb=0.05, plot_fig=true, save_plot=false, plot_name="AFD.png", kwargs...)
    
    S, Δt, n = p
    S, n = Int64(S), Int64(n)
    y = model(S, y0 .* ones(S), Δt, n; kwargs...)
    y = y[1:skip:end, :] # Skip entries
    y ./= sum(y, dims=2) # Normalize entries

    e_afd, t_afd = Dict(), Dict()
    if ensemble
        e_data = y[end,:]
        e_data = e_data[e_data .> 0.0]
        e_afd = make_AFD(e_data; Δb=Δb, plot_fig=plot_fig, save_plot=false, plot_name=plot_name, plot_title="ensemble AFD")
    end

    if temporal
        t_data = y[:,1]
        t_data = t_data[t_data .> 0.0]
        t_afd = make_AFD(t_data; Δb=Δb, plot_fig=plot_fig, save_plot=false, plot_name=plot_name, plot_title="temporal AFD")
    end

    if plot_fig
        if (ensemble) & (temporal)
            combined = plot(e_afd["fig"], t_afd["fig"], layout = (1, 2))
        elseif (ensemble) & (!temporal)
            combined = plot(e_afd["fig"])
        elseif (!ensemble) & (temporal)
            combined = plot(t_afd["fig"])
        end

        if save_plot
            savefig(plot_name)
        end
    end
    
    return Dict("e_afd" => e_afd, "t_afd" => t_afd, "fig" => combined)
end

function Taylor(model, p;
        y0 = 1.0, temporal=true, ensemble=true, skip=1, n_ens=100, S_reduce=100, n_reduce=5,
        Δb=0.05, plot_fig=true, save_plot=false, plot_name="Taylor.png", kwargs...)

    e_taylor, t_taylor = Dict(), Dict()
    if temporal
        S, Δt, n = p
        S, n = Int64(S), Int64(n)
        y = model(S, y0 .* rand(S), Δt, n; kwargs...)
        y = y[1:skip:end, :] # Skip entries
        y ./= sum(y, dims=2) # Normalize entries
    
        t_taylor = make_Taylor(y; Δb=Δb, plot_fig=plot_fig, save_plot=false, plot_name=plot_name, plot_title="Temporal Taylor's law")
    end

    if ensemble
        S, Δt, n = p
        S, n = Int64(floor(S / S_reduce)), Int64(floor(n / n_reduce))
        state = rand(S)
        initial_state = vcat([rand(S) for _ in 1:n_ens]...)
        y = model(n_ens * S, y0 .* initial_state, Δt, n; kwargs...)
        y = y[1:skip:end, :] # Skip entries
        y ./= sum(y, dims=2) # Normalize entries

        data = reshape(vcat([y[end, (j - 1)*S + 1:j*S] for j in 1:n_ens]...), S, n_ens)'
    
        e_taylor = make_Taylor(data; Δb=Δb, plot_fig=plot_fig, save_plot=false, plot_name=plot_name, plot_title="Ensemble Taylor's law")
    end

    if plot_fig
        if (ensemble) & (temporal)
            combined = plot(e_taylor["fig"], t_taylor["fig"], layout = (1, 2))
        elseif (ensemble) & (!temporal)
            combined = plot(e_taylor["fig"])
        elseif (!ensemble) & (temporal)
            combined = plot(t_taylor["fig"])
        end

        if save_plot
            savefig(plot_name)
        end
    end
    
    return Dict("e_taylor" => e_taylor, "t_taylor" => t_taylor, "fig" => combined)
end

function MAD(model, p;
        y0 = 1.0, temporal=true, ensemble=true, skip=1, n_ens=100, S_reduce=100, n_reduce=5,
        Δb=0.05, plot_fig=true, save_plot=false, plot_name="MAD.png", kwargs...)

    e_MAD, t_MAD = Dict(), Dict()
    if temporal
        S, Δt, n = p
        S, n = Int64(S), Int64(n)
        y = model(S, y0 .* rand(S), Δt, n; kwargs...)
        y = y[1:skip:end, :] # Skip entries
        y ./= sum(y, dims=2) # Normalize entries
    
        t_MAD = make_MAD(y; Δb=Δb, plot_fig=plot_fig, save_plot=false, plot_name=plot_name, plot_title="Temporal MAD")
    end

    if ensemble
        S, Δt, n = p
        S, n = Int64(floor(S / S_reduce)), Int64(floor(n / n_reduce))
        state = rand(S)
        initial_state = vcat([rand(S) for _ in 1:n_ens]...)
        y = model(n_ens * S, y0 .* initial_state, Δt, n; kwargs...)
        y = y[1:skip:end, :] # Skip entries
        y ./= sum(y, dims=2) # Normalize entries

        data = reshape(vcat([y[end, (j - 1)*S + 1:j*S] for j in 1:n_ens]...), S, n_ens)'
    
        e_MAD = make_MAD(data; Δb=Δb, plot_fig=plot_fig, save_plot=false, plot_name=plot_name, plot_title="Ensemble MAD")
    end

    if plot_fig
        if (ensemble) & (temporal)
            combined = plot(e_MAD["fig"], t_MAD["fig"], layout = (1, 2))
        elseif (ensemble) & (!temporal)
            combined = plot(e_MAD["fig"])
        elseif (!ensemble) & (temporal)
            combined = plot(t_MAD["fig"])
        end

        if save_plot
            savefig(plot_name)
        end
    end
    
    return Dict("e_mad" => e_MAD, "t_mad" => t_MAD, "fig" => combined)
end

end # end module










        
