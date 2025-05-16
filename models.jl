module GenerativeModels

using Random, Distributions, SpecialFunctions
using LinearAlgebra, SparseArrays
using DifferentialEquations, DiffEqCallbacks

function lotka_volterra(S, y0, Δt, n; r=1.0, K=ones(S), σ=1.0, ε=1e-6, skip=1)
    # Track which species are extinct
    extinct = falses(S)

    # Deterministic part
    function logistic!(du, u, p, t)
        r, K, σ, extinct = p
        @inbounds for i in 1:S
            if extinct[i]
                du[i] = 0.0
            else
                du[i] = r * u[i] * (1.0 - u[i] / K[i])
            end
        end
    end

    # Stochastic part (multiplicative noise)
    function diffusion!(du, u, p, t)
        σ = p[3]
        extinct = p[4]
        @inbounds for i in 1:S
            du[i] = extinct[i] ? 0.0 : σ * u[i]
        end
    end

    # Extinction condition: any new u[i] < ε for non-extinct species
    function condition(u, t, integrator)
        any((u .< ε) .& .!integrator.p[4])
    end

    # Apply extinction: zero out and flag extinct species
    function apply_extinction!(integrator)
        u = integrator.u
        extinct = integrator.p[4]
        @inbounds for i in 1:S
            if u[i] < ε
                u[i] = 0.0
                extinct[i] = true
            elseif u[i] < 0.0
                u[i] = 0.0  # Clamp negative values too, just in case
            end
        end
    end

    # Combine extinction with positivity enforcement
    extinction_cb = DiscreteCallback(condition, apply_extinction!)
    projection_cb = PositiveDomain()
    cb = CallbackSet(extinction_cb, projection_cb)

    # Problem setup
    tspan = (0.0, n * Δt)
    p = (r, K, σ, extinct)
    prob = SDEProblem(logistic!, diffusion!, y0, tspan, p)

    # Solve
    sol = solve(prob, SRIW1(); callback=cb, saveat=skip * Δt)

    return Matrix(reduce(hcat, sol.u)')
end

function lotka_volterra(S, y0, Δt, n; r=1.0, A=I(S), σ=1.0, ε=1e-6, skip=1)
    # Track which species are extinct
    extinct = falses(S)

    # Deterministic part
    function logistic!(du, u, p, t)
        r, A, σ, extinct = p
        @inbounds for i in 1:S
            if extinct[i]
                du[i] = 0.0
            else
                du[i] = r * u[i] * (1.0 - dot(view(A, i, :), u))
            end
        end
    end

    # Stochastic part (multiplicative noise)
    function diffusion!(du, u, p, t)
        σ = p[3]
        extinct = p[4]
        @inbounds for i in 1:S
            du[i] = extinct[i] ? 0.0 : σ * u[i]
        end
    end

    # Extinction condition: any new u[i] < ε for non-extinct species
    function condition(u, t, integrator)
        any((u .< ε) .& .!integrator.p[4])
    end

    # Apply extinction: zero out and flag extinct species
    function apply_extinction!(integrator)
        u = integrator.u
        extinct = integrator.p[4]
        @inbounds for i in 1:S
            if u[i] < ε
                u[i] = 0.0
                extinct[i] = true
            elseif u[i] < 0.0
                u[i] = 0.0  # Clamp negative values too, just in case
            end
        end
    end

    # Combine extinction with positivity enforcement
    extinction_cb = DiscreteCallback(condition, apply_extinction!)
    projection_cb = PositiveDomain()
    cb = CallbackSet(extinction_cb, projection_cb)

    # Problem setup
    tspan = (0.0, n * Δt)
    p = (r, A, σ, extinct)
    prob = SDEProblem(logistic!, diffusion!, y0, tspan, p)

    # Solve
    sol = solve(prob, SRIW1(); callback=cb, saveat=skip * Δt)

    return Matrix(reduce(hcat, sol.u)')
end

function exp_growth(S, y0, Δt, n; σ=1.0, p=0.95, ε=1e-6, skip=1)

    S, n = Int64(S), Int64(n)
    y = zeros(n,S)
    y[1,:] .= y0
    for i in 2:n
        gamma = abs.(1 .+ rand(Normal(0, σ), S))
        growth = y[i-1,:] .* exp.(log.(gamma) .* Δt)
        intro = zeros(S)
        if rand() > p
            intro .+= rand()
        end
        y[i,:] = growth .+ intro
        y[i,:][y[i,:] .< ε] .= 0.0
    end

    return y[1:skip:end, :]
end

### HELPER FUNCTIONS
function sparse_gaussian_matrix(K::Vector{Float64}, sparsity; μ=-1.0, σ=0.5, rng=Random.default_rng())

    S = length(K)
    D = Diagonal(K)
    J = zeros(S, S)

    for i in 1:S
        for j in 1:S
            if i == j
                # Diagonal: ensure negative real part (self-regulation)
                J[i, j] = -abs(μ + σ * randn(rng))
            elseif rand(rng) < sparsity
                # Off-diagonal: asymmetric interaction
                J[i, j] = μ + σ * randn(rng)
            end
        end
    end

    # Now recover A = - D^{-1} * J
    A = -inv(D) * J

    # Adjust diagonal to enforce A * K = 1
    # (This ensures K is an equilibrium after conversion)
    for i in 1:S
        A[i, i] = (1.0 - sum(A[i, :] .* K)) / K[i]
    end

    return sparse(A)
end


end