module BPPrivInfo

export integrated_payoff
export expected_payoff
export V
export λ
export upper_integral_λ
export Λ
export μ
export dμdp
export optimise
export UnivariateDensity
export discretise

using LinearAlgebra
using Optim
using QuadGK
using Distributions
using NLsolve
using Quadrature
using Expectations
using ForwardDiff
using Zygote

# Calls to include need to come after `using`
include("UnivariateDensity.jl")
include("discrete.jl")

const derivative = ForwardDiff.derivative
const CUD = ContinuousUnivariateDistribution
const E(d::CUD, n::Integer=500) = expectation(d; n=n)
const Density = UnivariateDensity

likelihoodratio(p) = p / (1 - p)

# Assume sender chooses a cutoff type in [0,1/2]
const lb = 0.0
const ub = 0.5
# Assume Uniform Distribution for default arguements
const unidist = Uniform(lb, ub)
const f(t) = pdf(unidist, t)
const unidens = Density(f, lb, ub)

# Sender Ex-Post Payoff
function payoff(type::T, cutoff::T) where {T <: Real}
    if cutoff ≤ 1/2 && cutoff ≤ type
        return type + (1 - type) * likelihoodratio(cutoff)
    elseif cutoff ≤ type
        return one(type)
    else
        return zero(type)
    end
end
payoff(type::Real, cutoff::Real) = payoff(promote(type, cutoff)...)

# Sender Expected Payoffs
payoff_integrand(t, c, f::Function=f) = payoff(t, c) * f(t)
integrated_payoff(cutoff, density::Function=f, ub::Real=ub) =
        quadgk(t -> payoff_integrand(t, cutoff, density), cutoff, ub)
integrated_payoff(cutoff::Real, density::Density=unidens) = 
        integrate(t -> payoff(t, cutoff), density, cutoff)

"""
Sender expected payoff as a function of the cutoff and the type distribution
"""
expected_payoff(cutoff::T, dist::CUD=unidist; n::Integer=500) where {T <: Real} =
                                        E(dist, n)(t -> payoff(t, cutoff))
expected_payoff(cutoff, f::Function=f, ub::Real=ub) = integrated_payoff(cutoff, f, ub)[1]
expected_payoff(cutoff, density::Density) = expected_payoff(cutoff, density.f, density.ub)
function expected_payoff(cut, dist::DiscreteNonParametric; kwargs...)
    belief = support(dist)
    prob = probs(dist)
    cutindex = findfirst(≥(cut), belief)
    prob = prob[cutindex:end]
    belief = belief[cutindex:end]
    return dot(prob, belief) + (cut / (1 - cut)) * dot(prob, (1 .- belief))
end

"""
Alias for expected_payoff
"""
V(cutoff::Real, dist::CUD=unidist; n::Integer=500) = expected_payoff(cutoff, dist; n = n)
V(cutoff, f::Function, ub::Real) = expected_payoff(cutoff, f, ub)
V(cutoff, density::Density=unidens) = V(cutoff, density.f, density.ub)
V(cutoff, dist::DiscreteNonParametric) = expected_payoff(cutoff, dist)

# Lagrange Multipliers
series_integrand(p::T, t::T) where {T <: Real} = t ≥ p ? 1 - t : zero(t)
series_integrand(p::Real, t::Real) = series_integrand(promote(p, t)...)
series_integral(p, dist::CUD, n::Integer=500) = E(dist, n)(t -> series_integrand(p, t))
integral_term(p, dist::CUD, n::Integer=500) = series_integral(p, dist, n) / (1 - p)^2

series_integrand(t, f::Function) = (1 - t) * f(t)
series_integral(p, f::Function, ub::Real) = quadgk(t -> series_integrand(t, f), p, ub)
integral_term(p, f::Function, ub::Real) = series_integral(p, f, ub)[1] / (1 - p)^2

# Multiplier for IC constraint
"""
Lagrange multiplier for the IC (reporting) constraint
"""
λ(p::Real, dist::CUD=unidist, n::Integer=500) = pdf(dist, p) - integral_term(p, dist, n)
λ(p, f::Function, ub::Real) = f(p) - integral_term(p, f, ub)
λ(p, density::Density=unidens) = λ(p, density.f, density.ub)

upper_integral_λ(p, f::Function, ub::Real) = quadgk(t -> λ(t, f, ub), p, ub)
upper_integral_λ(p, density::Density=unidens) = upper_integral_λ(p, density.f, density.ub)

"""
``\\Lambda (p, dist) = \\int_p^{1/2} \\lambda(t, dist) \\, \\mathrm{d}t``
"""
Λ(p::Real, dist::CUD=unidist, n::Integer=500) =
                            E(Uniform(), n)(t -> t ≥ p ? λ(t, dist, n) : zero(t))
Λ(p, f::Function, ub::Real) = upper_integral_λ(p, f, ub)[1]
Λ(p, density::Density=unidens) = Λ(p, density.f, density.ub)

# Multiplier for bound constraint on π_G
"""
``\\mu`` is the Lagrange multiplier for the constraint that ``\\pi_G`` must be a probability.
"""
μ(p::Real, dist::CUD=unidist, n::Integer=500) = 2p * pdf(dist, p) - integral_term(p, dist, n)
μ(p, f::Function, ub::Real) = 2p * f(p) - integral_term(p, f, ub)
μ(p, density::Density=unidens) = μ(p, density.f, density.ub)

# ∂μ/∂p
# Use 5000 nodes for dμdp or else plot looks terrible
"""
``\\partial \\mu / \\partial p``
"""
dμdp(p::Real, dist::CUD=unidist, n::Integer=500) = 
                (2 + 1 / (1 - p)) * pdf(dist, p) +
                2p * derivative(t -> pdf(dist, t), p) +
                2integral_term(p, dist, n) / (1 - p)

dμdp(p, f::Function=f, ub::Real=ub) =
                (2 + 1 / (1 - p)) * f(p) +
                2p * derivative(f, p) +
                2integral_term(p, f, ub) / (1 - p)

dμdp(p, d::Density) = dμdp(p, d.f, d.ub)

function optimise(dist::T=unidist; n::Integer=5000, alg=Brent()) where {T <: Union{CUD, DiscreteNonParametric}}
    objective(x) = -expected_payoff(x, dist; n = n)
    lb = minimum(dist)
    ub = maximum(dist)
    return optimize(objective, lb, ub, alg)
end

function optimise(f::Function, lb::Real, ub::Real; alg=Brent())
    objective(x) = -V(x, f, ub)
    return optimize(objective, lb, ub, alg)
end

optimise(density::Density; alg=Brent()) = optimise(density.f, density.lb, density.ub; alg=alg)

end # module
