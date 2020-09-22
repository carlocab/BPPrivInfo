using JuMP, Gurobi, Plots, LaTeXStrings, Distributions, LinearAlgebra
using BPPrivInfo


lb = 0.0 + eps(0.0)
ub = 1.0 - eps(1.0)
dist = MixtureModel([Uniform(lb, 1/2 + 1/100),
                     truncated(Normal(ub, 0.1), 1/2 - 1/100, ub)], [35/100, 65/100])

typedist = discretise(dist, 1000)
prob = probs(typedist)
belief = support(typedist)

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
    # plot(belief, -dual.(UpperBoundRef.(πG)), label = L"\pi_G \le 1", legend = :outertopright, show = true)

    return BPPI, πG, πB, report
end

function plot_exp_payoff(dist::DiscreteNonParametric)
    x = support(dist)
    plot!(x, t -> expected_payoff(t, dist), label = L"V", legend = :outertopright)
end

model, πG, πB, report = solve_discretised(typedist)

get_report_duals(report, i, n) = dual.(report[i,j] for j in 1:n if j ≠ i)

function plot_report_duals(dist, report)
    belief = support(dist)
    n = length(belief)
    plt = plot(belief[2:end], get_report_duals(report, 1, n), label = "1", legend = false)
    for j in 2:n
        plot!(plt, belief[1:end .≠ j], get_report_duals(report, j, n), label = "$j")
    end
    display(plt)
end
