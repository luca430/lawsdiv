module PlotDistributions

using Statistics, StatsBase
using NFFT
using FHist
using Plots, Measures
using Distributions, SpecialFunctions, LsqFit
using DataFrames, DataFramesMeta, GLM, Chain

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

function make_lagCorr(data; missing_thresh=0, max_lag=Int64(floor(size(data,1) / 2)), make_log=false, plot_fig=false, save_plot=false, plot_name="autocorrelation.png", plot_title="autocorrelation", data_label="data")

    mean_corrs, corrs_mat = compute_lagged_autocorrelations(data, max_lag, make_log=true, missing_thresh=missing_thresh)
    n_series = size(corrs_mat, 2)
    
    fig = plot(xlabel="lag", ylabel="autocorrelation", title=plot_title)
    if plot_fig
        for i in 1:n_series
            plot!(fig, corrs_mat[:, i], label=nothing, color="lightgrey", lw=0.2)
        end
        plot!(fig, mean_corrs, color="red", lw=2, label=data_label)
    end

    if save_plot
        savefig(plot_name)
    end

    return Dict("corrs" => corrs_mat, "mean_corrs" => mean_corrs, "fig" => fig)
end

function make_PSD(data; Δt=1, missing_thresh=0, make_log=false, freq_range=nothing, plot_fig=false, save_plot=false, plot_name="PSD.png", plot_title="Power Spectrum Density", data_label="data")
    mat = preprocess_matrix(data, make_log=make_log)
    if size(mat,1) % 2 != 0
        mat = mat[2:end,:]
    end
    
    N = size(mat, 1)
    N_species = size(data, 2)
    fs = 1 / Δt
    Nf = fs / 2 # Since sample rate is 1 day
    frequencies = (-Int(floor(N/2)):Int(floor(N/2)) - 1) * fs / N # Frequency domain
        
    mean_S = zeros(N) # Initialize array
    otu_count = 0 # Needed for normalization
    for i in 1:N_species
        # Compute non-uniform FFT only for a 'sufficient' number of samples
        if count(ismissing.(mat[:,i])) < missing_thresh
            otu_count += 1
            
            x = mat[:,i][.!ismissing.(mat[:,i])]
            x .-= mean(x) # Detrend signal to avoid peak at zero frequency and have comparable signals
            
            t_indices = findall(!ismissing, mat[:,i])
            t_normalized = (t_indices .- minimum(t_indices)) ./ N .- 0.5  # The algorithm work for t ∈ [-0.5, 0.5)
            
            p_nfft = NFFT.plan_nfft(t_normalized, N, reltol=1e-9)
            fhat = adjoint(p_nfft) * x
        
            # Compute normalized power spectrum density (periodogram)
            S = abs2.(fhat) .* (Δt / N)
            mean_S .+= S # Mean PSD: this step should give a more accurate result for the PSD supposing that all trajectories are equivalent
        end
    end
        
    # Take only positive frequencies
    positive = frequencies .> 0
    frequencies = frequencies[positive]
    mean_S = mean_S[positive] ./ otu_count

    log_f = log10.(frequencies)
    log_S = log10.(mean_S)

    if !isnothing(freq_range)
        mask = (log_f .>= freq_range[1]) .& (log_f .<= freq_range[2])
        log_f = log_f[mask]
        log_S = log_S[mask]
    end
    
    # Put into a DataFrame and fit linear model: log_S ~ log_k
    plot_df = DataFrame(log_f=log_f, log_S=log_S)
    model = lm(@formula(log_S ~ log_f), plot_df)
    
    # Extract the slope and intercept
    coeffs = coef(model)
    slope = coeffs[2]
    intercept = coeffs[1]

    fig = plot(xlabel="log₁₀(frequency)", ylabel="log₁₀(power)", legend=:bottomleft, title=plot_title)
    if plot_fig
        plot!(fig, log_f, log_S, label=data_label, color="black")
        plot!(fig, log_f, predict(model), label="Fit: slope = $(round(slope, digits=2))", lw=2, color="red")
    end

    if save_plot
        savefig(plot_name)
    end

    return Dict("PSD" => [frequencies, mean_S], "params" => Dict("slope" => slope, "intercept" => intercept), "fig" => fig)
    
    return 
end

### HELPER FUNCTIONS
# -- Custom autocorrelation function that skips missing entries at each lag --
function autocor_skipmissing(x::Vector{Union{Missing, Float64}}, lag::Int)
    n = length(x)

    if lag == 0
        vals = collect(skipmissing(x))
        return length(vals) > 1 ? cor(vals, vals) : missing
    end

    x1 = x[1:n - lag]
    x2 = x[1 + lag:n]

    valid_pairs = [(x1[i], x2[i]) for i in 1:length(x1) if !ismissing(x1[i]) && !ismissing(x2[i])]

    if length(valid_pairs) < 2
        return missing
    end

    a = first.(valid_pairs)
    b = last.(valid_pairs)

    return cor(a, b)
end

# -- Replace 0.0 with `missing` in the data matrix --
function preprocess_matrix(matrix_data::Matrix{Float64}; make_log::Bool=false)
    mat = Matrix{Union{Missing, Float64}}(matrix_data)
    mat[mat .== 0.0] .= missing
    if make_log
        mat = passmissing(log).(mat)
    end
    return mat
end

# -- Main autocorrelation loop --
function compute_lagged_autocorrelations(matrix_data::Matrix{Float64}, max_lag::Int64; make_log::Bool=false, missing_thresh::Int64=0)
    mat = preprocess_matrix(matrix_data, make_log=make_log)

    corrs = []

    for i in 1:size(mat, 2)
        x = mat[:, i]
        if count(ismissing.(x)) < missing_thresh
            c = []
            for lag in 0:max_lag
                r = autocor_skipmissing(x, lag)
                push!(c, r)
            end
    
            push!(corrs, c)
        end
    end

    corrs_mat = hcat(corrs...)

    # Mean correlation per lag (row-wise), skipping missing
    mean_corrs = mapslices(x -> mean(skipmissing(x)), corrs_mat; dims=2)

    return mean_corrs, corrs_mat
end


end # end module










        
