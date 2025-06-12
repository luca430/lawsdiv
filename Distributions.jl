module MakeDistributions

using Statistics, StatsBase
using NFFT, NLsolve
using FHist
using Distributions, SpecialFunctions, LsqFit
using DataFrames, DataFramesMeta, GLM, Chain

function make_AFD(data; c=exp(-15), missing_thresh=size(data, 1), Δb=0.05, env=nothing)

    mat = preprocess_matrix(data, make_log=false)
    mask = map(col -> count(ismissing, col) <= missing_thresh, eachcol(mat))
    filtered_mat = data[:, mask]

    # Remove species that are never present
    mask = .!map(col -> count(ismissing, col) == size(filtered_mat, 1), eachcol(filtered_mat))
    filtered_mat = filtered_mat[:, mask]

    S = size(filtered_mat, 2)
    non_zero_data = [filtered_mat[:, i][filtered_mat[:, i] .> 0.0] for i in 1:S]
    betas = [mean(x)^2 / var(x) for x in non_zero_data]
    betas = filter(!isnan, betas)
    β = mean(betas)
    
    log_non_zero_data = [log.(filtered_mat[:, i][filtered_mat[:, i] .> 0.0]) for i in 1:S]
    rescaled_data = [(x .- mean(x)) ./ std(x) for x in log_non_zero_data]
    log_data = filter(!isnan, vcat(rescaled_data...))

    bmin = round(minimum(log_data))
    bmax = round(maximum(log_data))
    fh = FHist.Hist1D(log_data, binedges=bmin:Δb:bmax)

    μ = mean(fh)
    σ = std(fh)
    centers = bincenters(fh)
    centers .-= μ
    centers ./= sqrt(2 * σ^2)
    norm_counts = bincounts(fh) ./ (integral(fh) * Δb)

    # Filter non-zero counts
    valid = norm_counts .> 0.0
    yy = 10 .^ log.(norm_counts[valid])
    centers = centers[valid]

    return Dict(
        "hist" => [centers, yy],
        "hparams" => Dict("μ" => μ, "σ" => σ),
        "params" => Dict("β" => β),
        "env" => env
    )
end

function make_Taylor(data; c=exp(-15), missing_thresh=size(data, 1), Δb=0.05, env=nothing)

    data[data .< c] .= 0.0
    mat = preprocess_matrix(data, make_log=false)
    mask = map(col -> count(ismissing, col) <= missing_thresh, eachcol(mat))
    filtered_mat = mat[:, mask]

    # Remove species that are never present
    mask = .!map(col -> count(ismissing, col) == size(filtered_mat, 1), eachcol(filtered_mat))
    filtered_mat = filtered_mat[:, mask]

    mean_data = [mean(skipmissing(x)) for x in eachcol(filtered_mat)]
    var_data = [var(skipmissing(x)) for x in eachcol(filtered_mat)]

    log_mean = log.(mean_data)
    log_var = log.(var_data )

    bmin = minimum(log_mean)
    bmax = maximum(log_mean)
    binedges = bmin:Δb:bmax
    centers = 0.5 .* (binedges[2:end] .+ binedges[1:end-1])

    yy = [mean(log_var[(log_mean .>= binedges[i]) .& (log_mean .< binedges[i+1])]) for i in 1:length(binedges)-1]
    centers = centers[isfinite.(yy)]
    yy = yy[isfinite.(yy)]

    # Fit linear model y = αx + q
    func(x, p) = p[1] .* x .+ p[2]
    p0 = [2.0, 0.0]
    fit = curve_fit(func, centers, yy, p0)
    p_fit = fit.param

    return Dict(
        "hist" => [centers, yy],
        "params" => Dict("α" => p_fit[1], "q" => p_fit[2]),
        "env" => env
    )
end

function make_MAD(data; c=exp(-15), missing_thresh=size(data, 1), Δb=0.05, env=nothing)

    mat = preprocess_matrix(data, make_log=false)
    mask = map(col -> count(ismissing, col) <= missing_thresh, eachcol(mat))
    filtered_mat = mat[:, mask]

    # Remove species that are never present
    mask = .!map(col -> count(ismissing, col) == size(filtered_mat, 1), eachcol(filtered_mat))
    filtered_mat = filtered_mat[:, mask]

    means = [mean(skipmissing(x)) for x in eachcol(filtered_mat)]
    log_data = [log(x) for x in means if x > c]

    bmin = floor(minimum(log_data))
    bmax = ceil(maximum(log_data))
    fh = FHist.Hist1D(log_data, binedges=bmin:Δb:bmax)

    m1 = mean(log_data)
    m2 = mean(log_data .^ 2)

    if c != 0.0
        μ, σ = compute_MAD_params(m1, m2, c)
    else
        μ, σ = mean(fh), std(fh)
    end

    centers = bincenters(fh)
    centers .-= μ
    centers ./= sqrt(2 * σ^2)

    norm_counts = bincounts(fh) ./ (integral(fh) * Δb)
    valid = norm_counts .> 0.0
    erfc_arg = (log(c) - μ) / sqrt(2 * σ^2)
    yy = 10 .^ log.((norm_counts[valid]) ./ sqrt(2 / (π * σ^2)) .* erfc(erfc_arg))
    centers = centers[valid]

    return Dict(
        "hist" => [centers, yy],
        "cutoff" => c,
        "hparams" => Dict("μ" => μ, "σ" => σ),
        "env" => env
        )
end

function make_lagCorr(data;
                      missing_thresh = size(data, 1),
                      max_lag = Int(floor(size(data, 1) / 2)),
                      make_log = false, env=nothing)

    mean_corrs, corrs_mat = compute_lagged_autocorrelations(data, max_lag;
                                                             make_log = make_log,
                                                             missing_thresh = missing_thresh)

    return Dict("corrs" => corrs_mat, "mean_corrs" => mean_corrs, "max_lag" => max_lag, "env" => env)
end

function make_lagCrossCorr(data; Δb = 0.01, missing_thresh = size(data, 1), lags = [0], make_log = false, env = nothing)

    corrs = []

    for lag in lags
        cmat = compute_lagged_crosscorrelations(data, lag; make_log = make_log, missing_thresh = missing_thresh)
        push!(corrs, cmat)
    end

    return Dict("cross_corrs" => corrs, "lags" => lags, "env" => env)
end

function make_PSD(data; Δt=1, missing_thresh=size(data, 1), make_log=false, freq_range=nothing, env=nothing)
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
        if count(ismissing.(mat[:,i])) <= missing_thresh
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
        log_f_fit = log_f[mask]
        log_S_fit = log_S[mask]
    else
        log_f_fit = log_f
        log_S_fit = log_S
    end
    
    # Put into a DataFrame and fit linear model: log_S ~ log_k
    plot_df = DataFrame(log_f=log_f_fit, log_S=log_S_fit)
    model = lm(@formula(log_S ~ log_f), plot_df)
    
    # Extract the slope and intercept
    coeffs = coef(model)
    slope = coeffs[2]
    intercept = coeffs[1]

    # standard errors
    se_vec = stderror(model)
    se_intercept = se_vec[1]
    se_slope = se_vec[2]

    return Dict("PSD" => [frequencies, mean_S], "params" => Dict("slope" => (slope, se_slope), "intercept" => (intercept, se_intercept)), "frange" => freq_range, "env" => env)
end

### HELPER FUNCTIONS
# -- Replace 0.0 with `missing` in the data matrix --
function preprocess_matrix(matrix_data::Matrix{Float64}; make_log::Bool=false)
    mat = Matrix{Union{Missing, Float64}}(matrix_data)
    mat[mat .== 0.0] .= missing
    if make_log
        mat = passmissing(log).(mat)
    end
    return mat
end

# -- Custom autocorrelation function that skips missing entries at each lag --
function autocor_skipmissing(x::Union{Vector{Float64}, Vector{Union{Missing, Float64}}}, lag::Int)
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

function compute_lagged_autocorrelations(matrix_data::Matrix{Float64}, max_lag::Int64; make_log::Bool=false, missing_thresh::Int64=size(matrix_data, 1))
    mat = preprocess_matrix(matrix_data, make_log=make_log)

    corrs = Vector{Vector{Union{Missing, Float64}}}()

    for i in 1:size(mat, 2)
        x = mat[:, i]
        if count(ismissing.(x)) <= missing_thresh
            c = Vector{Union{Missing, Float64}}()
            for lag in 0:max_lag
                r = autocor_skipmissing(x, lag)
                push!(c, r)
            end
            push!(corrs, c)
        end
    end

    # Sanity check: all vectors same length
    @assert all(length(c) == max_lag + 1 for c in corrs) "Autocorr vectors are not uniform in length"

    # Proper matrix creation
    corrs_mat = hcat(corrs...)

    # Mean across columns (i.e. for each lag)
    mean_corrs = mapslices(x -> mean(skipmissing(x)), corrs_mat; dims=2)

    return mean_corrs, corrs_mat
end

# -- Custom cross correlation function that skips missing entries at each lag --
function crosscor_skipmissing(x::Union{Vector{Float64}, Vector{Union{Missing, Float64}}}, 
                              y::Union{Vector{Float64}, Vector{Union{Missing, Float64}}}, 
                              lag::Int64)
    n = min(length(x), length(y))

    if lag == 0
        vals = [(x[i], y[i]) for i in 1:n if !ismissing(x[i]) && !ismissing(y[i])]
        if length(vals) < 2
            return missing
        end
        a = first.(vals)
        b = last.(vals)
        return cor(a, b)
    elseif lag > 0
        x1 = x[1:n - lag]
        y1 = y[1 + lag:n]
    else
        x1 = x[1 - lag:n]
        y1 = y[1:n + lag]
    end

    valid_pairs = [(x1[i], y1[i]) for i in 1:length(x1) if !ismissing(x1[i]) && !ismissing(y1[i])]

    if length(valid_pairs) < 2
        return missing
    end

    a = first.(valid_pairs)
    b = last.(valid_pairs)

    return cor(a, b)
end

function compute_lagged_crosscorrelations(matrix_data::Matrix{Float64}, lag::Int; make_log::Bool=false, missing_thresh::Int64=size(matrix_data, 1))
    mat = preprocess_matrix(matrix_data, make_log=make_log)
    mask = map(col -> count(ismissing, col) <= missing_thresh, eachcol(mat))
    filtered_mat = mat[:, mask]
    S = size(filtered_mat, 2)
    
    cross_corr = fill(NaN, S, S)  # Use NaN for undefined entries

    for i in 1:S
        for j in 1:i
            if i != j
                x = filtered_mat[:, i]
                y = filtered_mat[:, j]
                val = crosscor_skipmissing(x, y, lag)
                cross_corr[i, j] = ismissing(val) ? NaN : val
                cross_corr[j, i] = ismissing(val) ? NaN : val
            end
        end
    end

    return cross_corr
end

function compute_MAD_params(m1, m2, c)
    function make_system(m1, m2, c)
        return function F!(F, x)
            F[1] = x[1] - m1 + sqrt(2/π) * x[2] * exp(-(log(c) - x[1])^2 / (2 * x[2]^2)) / erfc((log(c) - x[1]) / sqrt(2 * x[2]^2))
            F[2] = x[2]^2 + m1*x[1] + log(c)*m1 - x[1]*log(c) - m2
        end
    end
    
    # Create the system with specific parameters
    f! = make_system(m1, m2, c)
    
    # Initial guess
    initial_x = [-15.0, 2.0]
    
    # Solve the system
    result = nlsolve(f!, initial_x)
    
    # Extract solution
    solution = result.zero

    return solution[1], solution[2]
end


end # end module










        
