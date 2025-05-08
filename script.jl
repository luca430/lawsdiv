module PlotDistributions

using Statistics, StatsBase
using FHist
using Plots
using Distributions, SpecialFunctions

function make_AFD(data; Δb=0.05, plot_fig=false, save_plot=false, plot_name="AFD.png", plot_title="AFD")

    log_data = log.(data)
    μ_x = mean(data)
    σ_x = std(data)
    β = μ_x^2 / σ_x^2

    bmin = round(minimum(log_data), RoundFromZero)
    bmax = round(maximum(log_data), RoundFromZero)
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


    fig = plot(yscale=:log10, xlabel="log(abundances)", ylabel="pdf", title=plot_title)
    if plot_fig
        plot!(fig, xarr, g_gamma, color=:black, label="Gamma")
        scatter!(fig, centers, yy, color=:red, label="simulation")
    end

    if save_plot
        savefig(plot_name)
    end

    return Dict("hist" => fh, "params" => Dict("β" => β, "μ_x" => μ_x, "σ_x" => σ_x), "fig" => fig)

end

function make_MAD(data; Δb=0.05, plot_fig=false, save_plot=false, plot_name="MAD.png", plot_title="MAD")

    log_data = [mean(log.(data[:,i][data[:,i] .> 0.0])) for i in 1:length(data[1,:])]
    
    bmin = round(minimum(log_data), RoundFromZero)
    bmax = round(maximum(log_data), RoundFromZero)
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
    lognorm = [10^(-x^2 - log(2 * π * σ^2) / 2) for x in xarr] # Gaussian distribution

    fig = plot(yscale=:log10, xlabel="log(abundances)", ylabel="pdf", title=plot_title)
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
        Δb=0.05, plot_fig=true, save_plot=false, plot_name="AFD.png")
    
    S, Δt, n = p
    S, n = Int64(S), Int64(n)
    y = model(S, y0 .* ones(S), Δt, n)
    y = y[1:skip:end, :] # Skip entries
    y ./= sum(y, dims=2) # Normalize entries

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

function MAD(model, p; y0=1, skip=1,
        Δb=0.05, plot_fig=true, save_plot=false, plot_name="MAD.png")
    
    S, Δt, n = p
    S, n = Int64(S), Int64(n)
    y = model(S, y0 .* rand(S), Δt, n)
    y = y[1:skip:end, :] # Skip entries
    y ./= sum(y, dims=2) # Normalize entries

    mad = make_MAD(y; Δb=Δb, plot_fig=plot_fig, save_plot=false, plot_name=plot_name, plot_title="MAD")

    if plot_fig
        fig = plot(mad["fig"])
        if save_plot
            savefig(plot_name)
        end
    end
    
    return Dict("mad" => mad, "fig" => fig)
end

end # end module










        
