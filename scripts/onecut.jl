module onecut

using BPPrivInfo
using Distributions
using Plots
using LaTeXStrings
using QuadGK

const lb = 0.0
const ub = 0.5
const tridist = TriangularDist(lb, ub) 
const sin_f(t) = 2sin(4π * t) + 2

function plotfuncs(dist::Distribution=tridist, n::Integer=501)
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

function plotfuncs(f::Function, n::Integer, lb::Real=lb, ub::Real=ub)
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
