using DataFrames, DataFramesMeta, GLM
using Statistics, StatsBase
using NFFT
using CSV
using FHist
using Plots
using Random, Distributions, SpecialFunctions, LsqFit
using DifferentialEquations

Random.seed!(1234)

# Set parameters
N_species = 5000
y0 = 100.0 .* ones(N_species)
Δt = 0.1
n = 5000
p = 0.9

# Simulate  dynamics
y = exp_growth(N_species, y0, Δt, n; σ=1.0, p=p)

# Sample at larger timesteps
skip = 1
y = y[1:skip:end, :]

# Normalize entries
y ./= sum(y, dims=2)

# Save results into a DataFrame
df = DataFrame(
    time = repeat(1:n, inner=N_species),
    variable = repeat(1:N_species, outer=n),
    value = vec(permutedims(y)),
    log_value = vec(permutedims(log10.(y)))
)

df_filtered = df[df.value .> 0.0, :];