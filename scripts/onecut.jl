module onecut

using BPPrivInfo
using Distributions
using Plots
using LaTeXStrings
using QuadGK

export plotfuncs

const lb = 0.0
const ub = 0.5
const tridist = TriangularDist(lb, ub) 
const sin_f = t -> sin(4π * t) + 1

function plotfuncs(n=501, dist::Distribution=tridist)
    x = range(minimum(dist), maximum(dist), length = n)
    y₁ = map(t -> V(t, dist), x)
    y₂ = map(t -> -μ(t, dist), x)
    y₃ = map(t -> μ(t, dist), x)
    y₄ = map(t -> λ(t, dist), x)
    plt = plot(x, [y₁ y₂ y₃ y₄], labels = [L"V" L"V^\prime" L"\mu" L"\lambda"])
    hline!([0.0], label = L"0")
    xlabel!("Cutoff")
    plot!(legend = :outertopright)
    return plt
end

function plotfuncs(n, f::Function, lb::Real=lb, ub::Real=ub)
    x = range(lb, ub, length = n)
    y₁ = map(t -> V(t, f, ub), x)
    y₂ = map(t -> -μ(t, f, ub), x)
    y₃ = map(t -> μ(t, f, ub), x)
    y₄ = map(t -> λ(t, f, ub), x)
    plt = plot(x, [y₁ y₂ y₃ y₄], labels = [L"V" L"V^\prime" L"\mu" L"\lambda"])
    hline!([0.0], label = L"0")
    xlabel!("Cutoff")
    plot!(legend = :outertopright)
    return plt
end

end # module
