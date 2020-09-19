module BPPrivInfo

using Plots
using LaTeXStrings
using Optim
using QuadGK
using Distributions
using NLsolve
using Quadrature
using ForwardDiff
using Zygote

export integrated_payoff, expected_payoff, V, λ, Λ, μ, dμdp

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
integrated_payoff(cutoff, density=f, ub=ub) = quadgk(
                                                t -> payoff_integrand(t, cutoff, density),
                                                cutoff,
                                                ub
                                               )
expected_payoff(cutoff, density=f, ub=ub) = inteated_payoff(cutoff, density, ub)[1]
V(cutoff, f=f, ub=ub) = expected_payoff(cutoff, f, ub)

# Lagrange Multipliers
series_integrand(t, f) = (1 - t) * f(t)
series_integral(p, f, ub) = quadgk(t -> series_integrand(t, f), p, ub)
integral_term(p, f, ub) = series_integral(p, f, ub)[1]

# Multiplier for IC constraint
λ(p, f=f, ub=ub) = f(p) - integral_term(p, f, ub)
up_integral_λ(p, f, ub) = quadgk(t -> λ(t, f, ub), p, ub)
Λ(p, f=f, ub=ub) = up_integral_λ(p, f, ub)[1]

# Multiplier for bound constraint on π_G
μ(p, f=f, ub=ub) = 2p * f(p) - integral_term(p, f, ub)
# ∂μ/∂p
dμdp(p, f=f, ub=ub) = (2 + 1 / (1 - p)) * f(p) +
                    2p * derivative(f, p) +
                    2integral_term(p, f, ub) / (1 - p)

end # module
