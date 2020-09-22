using JuMP, Gurobi, Plots, LaTeXStrings, Distributions, LinearAlgebra

include("../src/discrete.jl")
include("../../BPUnkPr/StrangeSinDist.jl")

typedist = discretise(Uniform(0.0, 0.5), 1000)

function solve_discretised(dist::DiscreteNonParametric)
    # Create problem data
    prob = probs(dist)
    belief = support(dist)
    n = length(belief)

    # Initalise model
    BPPI = Model(Gurobi.Optimizer)

    # Create variables
    @variable(BPPI, 0 ≤ πG[1:n] ≤ 1)
    @variable(BPPI, 0 ≤ πB[1:n] ≤ 1)

    # Create objective
    @expression(BPPI, exppayoff, dot(prob, belief .* πG) + dot(prob, (1 .- belief) .* πB))
    @objective(BPPI, Max, exppayoff)

    # Create constraints
    @constraint(BPPI, obG, belief .* πG .≥ (1 .- belief) .* πB)
    @constraint(BPPI, obB, (1 .- belief) .* (1 .- πB) .≥ belief .* (1 .- πG))
    @constraint(BPPI,
                report[i = 1:n, j = 1:n; i ≠ j],
                belief[i] * πG[i] + (1 - belief[i]) * (1 - πB[i]) ≥ belief[i] * πG[j] + (1 - belief[i]) * (1 - πB[j]))

    # Solve model
    optimize!(BPPI)

    termstatus = termination_status(BPPI)
    termstatus == MOI.OPTIMAL || @warn("Termination status = $termstatus")

    # Generate plots
    plot(belief, value.(πG), label = L"\pi_G", legend = :outertopright, show = true)
    plot!(belief, value.(πB), label = L"\pi_B")

    return BPPI, πG, πB
end

function expected_payoff(cut, dist::DiscreteNonParametric)
    belief = support(dist)
    prob = probs(dist)
    cutindex = findfirst(≥(cut), belief)
    prob = prob[cutindex:end]
    belief = belief[cutindex:end]
    return dot(prob, belief) + (cut / (1 - cut)) * dot(prob, (1 .- belief))
end

function plot_exp_payoff(dist::DiscreteNonParametric)
    x = support(dist)
    plot!(x, t -> expected_payoff(t, dist), label = L"V", legend = :outertopright)
end

