
function discretise(dist::ContinuousUnivariateDistribution, n::Integer=10)
    F(t) = cdf(dist, t)
    lb = minimum(dist)
    ub = maximum(dist)
    nodes = range(lb, ub; length = n + 1)
    pmf = [F(nodes[i + 1]) - F(nodes[i]) for i in 1:n]
    return DiscreteNonParametric(nodes[2:end], pmf)
end

discretise(dist::ContinuousUnivariateDistribution, n::Number) = discretise(dist, Integer(n))
