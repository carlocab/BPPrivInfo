module BPPrivInfo

using Optim
using QuadGK
using Distributions
using NLsolve
using Quadrature
using ForwardDiff
using Zygote

export integrated_payoff, expected_payoff, V, λ, up_integral_λ, Λ, μ, dμdp

const derivative = ForwardDiff.derivative

likelihoodratio(p) = p / (1 - p)

# Assume sender chooses a cutoff type in [0,1/2]
const lb = 0.0
const ub = 0.5
# Assume Uniform Distribution for Default
const unidist = Uniform(lb, ub)
f(t) = pdf(unidist, t)

# Sender Ex-Post Payoff
function payoff(type, cutoff)
    if cutoff ≤ 1/2
        return type + (1 - type) * likelihoodratio(cutoff)
    else
        return one(type)
    end
end

# Sender Expected Payoffs
payoff_integrand(type, cutoff, dist::Distribution) = payoff(type, cutoff) * pdf(dist, type)
payoff_integrand(t, c, f::Function=f) = payoff(t, c) * f(t)
integrated_payoff(cutoff, dist::Distribution=unidist) = quadgk(
                                              t -> payoff_integrand(t, cutoff, dist),
                                              cutoff,
                                              maximum(dist)
                                             )
integrated_payoff(cutoff, density::Function=f, ub::Real=ub) =
        quadgk(t -> payoff_integrand(t, cutoff, density), cutoff, ub)
"""
Sender expected payoff as a function of the cutoff and the type distribution
"""
expected_payoff(cutoff, dist::Distribution=unidist) = integrated_payoff(cutoff, dist)[1]
expected_payoff(cutoff, f::Function=f, ub::Real=ub) = integrated_payoff(cutoff, f, ub)[1]
"""
Alias for expected_payoff
"""
V(cutoff, dist::Distribution=unidist) = expected_payoff(cutoff, dist)
V(cutoff, f::Function=f, ub::Real=ub) = expected_payoff(cutoff, f, ub)

# Lagrange Multipliers
series_integrand(t, dist::Distribution) = (1 - t) * pdf(dist, t)
series_integrand(t, f::Function) = (1 - t) * f(t)
series_integral(p, dist::Distribution) = quadgk(t -> series_integrand(t, dist), p, maximum(dist))
series_integral(p, f::Function, ub::Real) = quadgk(t -> series_integrand(t, f), p, ub)
integral_term(p, dist::Distribution) = series_integral(p, dist)[1] / (1 - p)^2
integral_term(p, f::Function, ub::Real) = series_integral(p, f, ub)[1] / (1 - p)^2

# Multiplier for IC constraint
"""
Lagrange multiplier for the IC (reporting) constraint
"""
λ(p, dist::Distribution=unidist) = pdf(dist, p) - integral_term(p, dist)
λ(p, f::Function=f, ub::Real=ub) = f(p) - integral_term(p, f, ub)
up_integral_λ(p, dist::Distribution=unidist) = quadgk(t -> λ(t, dist), p, maximum(dist))
up_integral_λ(p, f::Function=f, ub::Real=ub) = quadgk(t -> λ(t, f, ub), p, ub)

"""
``\\Lambda (p, dist) = \\int_p^{1/2} \\lambda(t, dist) \\, \\mathrm{d}t``
"""
Λ(p, dist::Distribution=unidist) = up_integral_λ(p, dist)[1]
Λ(p, f::Function=f, ub::Real=ub) = up_integral_λ(p, f, ub)[1]

# Multiplier for bound constraint on π_G
"""
``\\mu`` is the Lagrange multiplier for the constraint that ``\\pi_G`` must be a probability.
"""
μ(p, dist::Distribution=unidist) = 2p * pdf(dist, p) - integral_term(p, dist)
μ(p, f::Function=f, ub::Real=ub) = 2p * f(p) - integral_term(p, f, ub)

# ∂μ/∂p
"""
``\\partial \\mu / \\partial p``
"""
dμdp(p, dist::Distribution=unidist) = (2 + 1 / (1 - p)) * pdf(dist, p) +
                2p * derivative(t -> pdf(dist, t), p) +
                2integral_term(p, dist) / (1 - p)

dμdp(p, f::Function=f, ub::Real=ub) =
                (2 + 1 / (1 - p)) * f(p) +
                2p * derivative(f, p) +
                2integral_term(p, f, ub) / (1 - p)

end # module
