module GenerativeModels

using Random, Distributions, SpecialFunctions
using LinearAlgebra, SparseArrays
using DifferentialEquations, DiffEqCallbacks

function logistic_growth(S, y0, Δt, n; r=1.0, K=ones(S), σ=1.0, ε=1e-6, skip=1)

    function logistic!(du, u, p, t)
        r, K, σ = p
        @inbounds for i in 1:S
            # u_val = u[i] < ε ? 0.0 : u[i]
            u_val = max(0.0, u[i])
            du[i] = r * u_val * (1.0 - u_val / K[i])
        end
    end

    function diffusion!(du, u, p, t)
        σ = p[3]
        @inbounds for i in 1:S
            # u_val = u[i] < ε ? 0.0 : u[i]  # Clamp input to avoid negative stochastic pushes
            u_val = max(0.0, u[i])
            du[i] = σ * u_val
        end
    end

    function condition(u, t, integrator)
        any(u .< 0)
    end
    
    function apply_clamp!(integrator)
        @. integrator.u = max(0.0, integrator.u)
    end
    
    cb = DiscreteCallback(condition, apply_clamp!)

    tspan = (0.0, n * Δt)
    p = (r, K, σ)
    tstops = 0.0:Δt:(n * Δt)

    prob = SDEProblem(logistic!, diffusion!, y0, tspan, p)

    sol = solve(prob, EM(); callback=cb, saveat=skip * Δt, tstops=tstops)#, adaptive=false)

    return Matrix(reduce(hcat, sol.u)')[1:end-1,:]
end

function lotka_volterra(S, y0, Δt, n; r=1.0, A=I(S), σ=1.0, ε=1e-6, skip=1)

    extinct = falses(S)

    function logistic!(du, u, p, t)
        r, A, σ, extinct = p
        u_val = [max(0.0, u[i]) for i in 1:S]
        @inbounds for i in 1:S
            du[i] = extinct[i] ? 0.0 : r * u_val[i] * (1.0 + dot(view(A, i, :), u_val))
        end
    end

    function diffusion!(du, u, p, t)
        σ = p[3]
        extinct = p[4]
        @inbounds for i in 1:S
            u_val = max(0.0, u[i])
            du[i] = extinct[i] ? 0.0 : σ * u_val
        end
    end

    # Extinction condition: fire when anything < ε and not already extinct
    function condition(u, t, integrator)
        any((u .< ε) .& .!integrator.p[4])
    end

    # Apply extinction + clamp
    function apply_extinction!(integrator)
        u = integrator.u
        extinct = integrator.p[4]
        @inbounds for i in 1:S
            if u[i] < ε
                u[i] = 0.0
                extinct[i] = true
            end
        end
    end

    extinction_cb = DiscreteCallback(condition, apply_extinction!)

    # Positivity projection at every step
    function project_positivity!(integrator)
        @. integrator.u = max(integrator.u, 0.0)
    end

    positivity_cb = DiscreteCallback((u, t, integrator) -> true, project_positivity!; save_positions=(false, false))

    cb = CallbackSet(extinction_cb, positivity_cb)

    tspan = (0.0, n * Δt)
    p = (r, A, σ, extinct)
    prob = SDEProblem(logistic!, diffusion!, y0, tspan, p)
    
    sol = solve(prob, SOSRI(); callback=cb, saveat=skip * Δt, dt=Δt)

    return Matrix(reduce(hcat, sol.u)')[1:end-1, :]
end


function exp_growth(S, y0, Δt, n; σ=1.0, p=0.95, ε=1e-6, skip=1)

    S, n = Int64(S), Int64(n)
    y = zeros(n,S)
    y[1,:] .= y0
    for i in 2:n
        gamma = abs.(1 .+ rand(Normal(0, σ), S))
        growth = y[i-1,:] .* exp.(log.(gamma) .* Δt)
        intro = zeros(S)
        mask = rand(S) .> p
        intro[mask] .+= rand(Normal(), sum(mask))
        y[i, :] = growth .+ intro
        y[i, :] = ifelse.(y[i, :] .< ε, 0.0, y[i, :])
    end

    return y[1:skip:end, :]
end

function stat_model(S, n; β=1.0, mean_abs=rand(S), ε=1e-6, skip=1)

    θ = mean_abs ./ β
    y = zeros(n, S)
    for i in 1:S
        y[:, i] = rand(Gamma(β, θ[i]), n)
    end
    y[y .< ε] .= 0.0
    
    return y[1:skip:end, :]
end

function exp_growth(S, y0, Δt, n; σ=1.0, ε=1e-6, skip=1)

    S, n = Int64(S), Int64(n)
    y = zeros(n,S)
    y[1,:] .= y0
    for i in 2:n
        random_r = rand(Normal(1,σ), S)
        gamma = [max(0.0, r) for r in random_r]
        growth = y[i-1,:] .* exp.(log.(gamma) .* Δt)
        y[i, :] = ifelse.(y[i, :] .< ε, 0.0, y[i, :])
    end

    return y[1:skip:end, :]
end

### HELPER FUNCTIONS
function sparse_gaussian_matrix(K::Vector{Float64}, sparsity; μ=-1.0, σ=0.5)

    S = length(K)
    A = zeros(S,S)

    for i in 1:S
        A[i,i] = - 1 / K[i]
        for j in 1:S
            if (i != j) & (rand() < sparsity)
                # Off-diagonal: asymmetric interaction
                A[i, j] = μ + σ * randn()
            end
        end
    end

    return sparse(A)
end

end