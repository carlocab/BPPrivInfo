module BPPrivInfo

export integrated_payoff
export expected_payoff
export V
export λ
export upper_integral_λ
export Λ
export μ
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
const UD = UnivariateDistribution
const E(d::UD, n::Integer=500) = expectation(d; n=n)
const Density = UnivariateDensity

const likelihoodratio(p) = p / (1 - p)
const LR = likelihoodratio
likelihood_ratio_multiplier(x, y) = (likelihoodratio(y) - 1) / (likelihoodratio(y) - likelihoodratio(x))
const LRM = likelihood_ratio_multiplier

# Assume sender chooses a cutoff type in [0,1/2]
const lb = 0.0
const ub = 0.5
# Assume Uniform Distribution for default arguements
const unidist = Uniform(lb, ub)
const f(t) = pdf(unidist, t)
const unidens = Density(f, lb, ub)

# Sender Ex-Post Payoff
function payoff(type::T, cutoff::T) where {T <: Real}
    pay = type + (1 - type) * likelihoodratio(cutoff)
    if cutoff ≤ 1/2 && cutoff ≤ type
        return pay
    elseif cutoff ≤ type
        return one(pay)
    else
        return zero(pay)
    end
end
payoff(type::Real, cutoff::Real) = payoff(promote(type, cutoff)...)

# Methods for two-cutoff case
function payoff(type::T, pL::T, pH::T) where {T <: Real}
    pay = type + LR(pL) * (1 - type)
    if pL ≤ type ≤ pH
        return pay
    else
        return zero(pay)
    end
end
payoff(type::Real, pL::Real, pH::Real) = payoff(promote(type, pL, pH)...)

# Sender Expected Payoffs
payoff_integrand(t, c, f::Function=f) = payoff(t, c) * f(t)
integrated_payoff(cutoff, density::Function=f, ub::Real=ub) =
        quadgk(t -> payoff_integrand(t, cutoff, density), cutoff, ub)
integrated_payoff(cutoff::Real, density::Density=unidens) =
        integrate(t -> payoff(t, cutoff), density, cutoff)

"""
Sender expected payoff as a function of the cutoff and the type distribution
"""
expected_payoff(cutoff::Real, dist::UD=unidist; n::Integer) = E(dist, n)(t -> payoff(t, cutoff))
expected_payoff(cutoff, f::Function=f, ub::Real=ub) = integrated_payoff(cutoff, f, ub)[1]
expected_payoff(cutoff, density::Density) = expected_payoff(cutoff, density.f, density.ub)

# Methods for the two-cutoff case
expected_payoff(pL::T, pH::T, dist::UD; n::Integer) where {T <: Real} = LRM(pL, pH) * E(dist, n)(t -> payoff(t, pL, pH)) + 1 - cdf(dist, pH)
expected_payoff(pL::T, pH::T, f::Function, ub::Real) where {T <: Real} = LRM(pL, pH) * expected_payoff(pL, f, pH) + quadgk(f, pH, ub)[1]
expected_payoff(pL::T, pH::T, density::Density) where {T<: Real} = expected_payoff(pL, pH, density.f, density.ub)

expected_payoff(pL::Real, pH::Real, dist::UD; n::Integer) = expectedpayoff(promote(pL, pH)..., dist; n = n)
expected_payoff(pL::Real, pH::Real, f::Function, ub::Real) = expectedpayoff(promote(pL, pH)..., f, ub)
expected_payoff(pL::Real, pH::Real, density::Density) = expectedpayoff(promote(pL, pH)..., density)

"""
Alias for expected_payoff
"""
V(cutoff::Real, dist::UD=unidist; n::Integer=500) = expected_payoff(cutoff, dist; n = n)
V(cutoff, f::Function, ub::Real) = expected_payoff(cutoff, f, ub)
V(cutoff, density::Density=unidens) = V(cutoff, density.f, density.ub)

V(pL, pH, dist::UD; n=Integer=500) = expected_payoff(pL, pH, dist; n = n)
V(pL, pH, f::Function, ub::Real) = expected_payoff(pL, pH, f, ub)
V(pL, pH, density::Density) = expected_payoff(pL, pH, density)

# Lagrange Multipliers
series_integrand(p::T, t::T) where {T <: Real} = t ≥ p ? 1 - t : zero(t)
series_integrand(p::Real, t::Real) = series_integrand(promote(p, t)...)
series_integral(p, dist::UD, n::Integer=500) = E(dist, n)(t -> series_integrand(p, t))
integral_term(p, dist::UD, n::Integer=500) = series_integral(p, dist, n) / (1 - p)^2

series_integrand(t, f::Function) = (1 - t) * f(t)
series_integral(p, f::Function, ub::Real) = quadgk(t -> series_integrand(t, f), p, ub)
integral_term(p, f::Function, ub::Real) = series_integral(p, f, ub)[1] / (1 - p)^2

# Multiplier for IC constraint
"""
Lagrange multiplier for the IC (reporting) constraint
"""
λ(p::Real, dist::UD=unidist, n::Integer=500) = pdf(dist, p) - integral_term(p, dist, n)
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

LR′(x) = 1 / (1 - x)^2
function dVdpH(pL, pH, dist::CUD; n::Integer=500)
    fpH = pdf(dist, pH)
    int_mul = LR′(pH) * (1 - LR(pL)) / (LR(pH) - LR(pL))^2
    term1 = fpH * (-1 + LRM(pL, pH) * (pH + LR(pL) * (1 - pH)))
    term2 = E(dist, n)(t -> payoff(t, pL, pH))
    return term1 + int_mul * term2
end

function dVdpH(pL, pH, f::Function)
    fpH = f(pH)
    int_mul = LR′(pH) * (1 - LR(pL)) / (LR(pH) - LR(pL))^2
    term1 = fpH * (-1 + LRM(pL, pH) * (pH + LR(pL) * (1 - pH)))
    term2 = quadgk(t -> payoff(t, pL, pH), pL, pH)[1]
    return term1 + int_mul * term2
end

dVdpH(pL, pH, dens::Density) = dVdpH(pL, pH, dens.f)

function dVdpL(pL, pH, dist::CUD; n::Integer=500)
    lrmul = LRM(pL, pH)
    fpL = pdf(dist, pL)
    integrand(x) = pL ≤ x ≤ pH ? 1 - x : zero(x)
    integral1 = E(dist, n)(integrand)
    term1 = -2pL * fpL + LR′(pL) * integral1
    term2 = LR′(pL) * E(dist, n)(t -> payoff(t, pL, pH)) / (LR(pH) - LR(pL))
    return lrmul *(term1 + term2)
end

function dVdpL(pL, pH, f::Function) 
    lrmul = LRM(pL, pH)
    fpL = f(pL)
    integrand(x) = pL ≤ x ≤ pH ? 1 - x : zero(x)
    integral1 = quadgk(integrand, pL, pH)[1]
    term1 = -2pL * fpL + LR′(pL) * integral1
    term2 = LR′(pL) * quadgk(t -> payoff(t, pL, pH), pL, pH)[1] / (LR(pH) - LR(pL))
    return lrmul *(term1 + term2)
end

dVdpL(pL, pH, dens::Density) = dVdpL(pL, pH, dens.f)

function ∇expected_payoff!(G, x, dist::CUD; n::Integer=500)
    G[1] = dVdpL(x[1], x[2], dist; n = n)
    G[2] = dVdpH(x[1], x[2], dist; n = n)
end

function ∇expected_payoff!(G, x, f::Function)
    G[1] = dVdpL(x[1], x[2], f)
    G[2] = dVdpH(x[1], x[2], f)
end

# This next function is not necessary, but keeping it helps standardise the
# structure of the code.
∇expected_payoff!(G, x, dens::Density) = ∇expected_payoff!(G, x, dens.f)

function optimise(dist::UD=unidist; n::Integer=5000, initx=[eps(0.0), 1 - eps(1.0)], alg=Brent(), inner_optimizer=BFGS())
    ub = maximum(dist)
    lb = minimum(dist)
    if ub > 1/2 && cdf(dist,ub) - cdf(dist,1/2) > 0
        lower = [lb, 1/2]
        upper = [1/2, ub]
        initx = [max(initx[1], lb + eps(lb)), min(initx[2], ub - eps(ub))]
        gradient!(G, x) = -∇expected_payoff!(G, x, dist; n = n)
        return optimize(x -> -expected_payoff(x[1], x[2], dist; n = n), gradient!, lower, upper, initx, Fminbox(inner_optimizer))
    else
        return optimize(x -> -expected_payoff(x, dist; n = n), lb, ub, alg)
    end
end

function optimise(f::Function, lb::Real, ub::Real; initx=[eps(0.0), 1 - eps(1.0)], alg=Brent(), inner_optimizer=BFGS())
    if ub > 1/2 && quadgk(f, 1/2, ub)[1] > 0
        lower = [lb, 1/2]
        upper = [1/2, ub]
        initx = [max(initx[1], lb + eps(lb)), min(initx[2], ub - eps(ub))]
        gradient!(G, x) = -∇expected_payoff!(G, x, f)
        return optimize(x -> -expected_payoff(x[1], x[2], f, ub), gradient!, lower, upper, initx, Fminbox(inner_optimizer))
    else
        objective(x) = -expected_payoff(x, f, ub)
        return optimize(x -> -expected_payoff(x, f, ub), lb, ub, alg)
    end
end

optimise(density::Density; initx=[eps(0.0), 1 - eps(1.0)], alg=Brent(), inner_optimizer=BFGS()) = optimise(density.f, density.ub; alg=alg)

end # module
