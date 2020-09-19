module BPPrivInfo

using LaTeXStrings
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
# Uniform Density
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
payoff_integrand(type, cutoff, density) = payoff(type, cutoff) * density(type)
integrated_payoff(cutoff, density=f) = quadgk(
                                              t -> payoff_integrand(t, cutoff, density),
                                              cutoff,
                                              maximum(density)
                                             )
"""
Sender expected payoff as a function of the cutoff and the type density f
"""
expected_payoff(cutoff, density=f) = inteated_payoff(cutoff, density)[1]
"""
Alias for expected_payoff
"""
V(cutoff, f=f) = expected_payoff(cutoff, f)

# Lagrange Multipliers
series_integrand(t, f) = (1 - t) * f(t)
series_integral(p, f) = quadgk(t -> series_integrand(t, f), p, maximum(f))
integral_term(p, f) = series_integral(p, f)[1]

# Multiplier for IC constraint
"""
Lagrange multiplier for the IC (reporting) constraint
"""
λ(p, f=f) = f(p) - integral_term(p, f)
up_integral_λ(p, f=f) = quadgk(t -> λ(t, f), p, maximum(f))
"""
``\\Lambda (p, f) = \\int_p^{1/2} \\lambda(t, f) \\, \\mathrm{d}t``
"""
Λ(p, f=f) = up_integral_λ(p, f)[1]

# Multiplier for bound constraint on π_G
"""
``\\mu`` is the Lagrange multiplier for the constraint that ``\\pi_G`` must be a probability.
"""
μ(p, f=f, ub=ub) = 2p * f(p) - integral_term(p, f, ub)
# ∂μ/∂p
"""
``\\partial \\mu / \\partial p``
"""
dμdp(p, f=f, ub=ub) = (2 + 1 / (1 - p)) * f(p) +
                    2p * derivative(f, p) +
                    2integral_term(p, f, ub) / (1 - p)

end # module
