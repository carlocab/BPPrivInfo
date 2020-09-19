module onecut

using BPPrivInfo
using Distributions
using Plots
using LaTeXStrings

export plotfuncs

lb = 0.0
ub = 0.5
dist = TriangularDist(lb, ub) 

function plotfuncs(n=501, dist=dist)
    x = range(minimum(dist), maximum(dist), length = n)
    y₁ = map(V, x)
    y₂ = map(t -> -μ(t), x)
    y₃ = map(μ, x)
    y₄ = map(λ, x)
    plt = plot(x, [y₁ y₂ y₃ y₄], labels = [L"V" L"V^\prime" L"\mu" L"\lambda"])
    hline!([0.0], label = L"0")
    xlabel!("Cutoff")
    plot!(legend = :outertopright)
    return plt
end

end
