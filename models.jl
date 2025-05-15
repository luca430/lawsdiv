module GenerativeModels

using Random, Distributions, SpecialFunctions
using LinearAlgebra, SparseArrays
using DifferentialEquations, DiffEqCallbacks

function logistic_growth(S, y0, Δt, n; r=1.0, K=ones(S), σ=1.0, skip=1)
    # Logistic drift term
    function logistic!(du, u, p, t)
        r, K, σ = p
        du .= r .* u .* (1 .- u ./ K)
    end

    # Multiplicative noise
    function diffusion!(du, u, p, t)
        σ = p[3]
        du .= σ .* u
    end

    # Only project when needed (faster than every step)
    function condition(u, t, integrator)
        any(u .< 0)
    end

    function project_nonnegative!(integrator)
        @inbounds for i in eachindex(integrator.u)
            if integrator.u[i] < 0.0
                integrator.u[i] = 0.0
            end
        end
    end

    # Callback: only correct when needed
    cb = DiscreteCallback(condition, project_nonnegative!)

    # Problem with domain check: aborts if u ever goes negative
    tspan = (0.0, n * Δt)
    p = (r, K, σ)
    prob = SDEProblem(logistic!, diffusion!, y0, tspan, p;
                      isoutofdomain = (u, p, t) -> any(u .< 0))

    # Solve
    sol = solve(prob, SRIW1(), callback=cb, saveat=skip * Δt)

    return Matrix(reduce(hcat, sol.u)')
end

function lotka_volterra(S, y0, Δt, n; r=1.0, A=I(S), σ=1.0, skip=1)
    # Logistic drift term
    function logistic!(du, u, p, t)
        r, A, σ = p
        du .= r .* u .* (1 .- A * u)
    end

    # Multiplicative noise
    function diffusion!(du, u, p, t)
        σ = p[3]
        du .= σ .* u
    end

    # Only project when needed (faster than every step)
    function condition(u, t, integrator)
        any(u .< 0)
    end

    function project_nonnegative!(integrator)
        @inbounds for i in eachindex(integrator.u)
            if integrator.u[i] < 0.0
                integrator.u[i] = 0.0
            end
        end
    end

    # Callback: only correct when needed
    cb = DiscreteCallback(condition, project_nonnegative!)

    # Problem with domain check: aborts if u ever goes negative
    tspan = (0.0, n * Δt)
    p = (r, A, σ)
    prob = SDEProblem(logistic!, diffusion!, y0, tspan, p;
                      isoutofdomain = (u, p, t) -> any(u .< 0))

    # Solve
    sol = solve(prob, SRIW1(), callback=cb, saveat=skip * Δt)

    return Matrix(reduce(hcat, sol.u)')
end

function exp_growth(S, y0, Δt, n; σ=1.0, p=0.95, skip=1)

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