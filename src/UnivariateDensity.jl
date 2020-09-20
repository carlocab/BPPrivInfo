# More accurately, this should be called `BoundedUnivariateDensity`
struct UnivariateDensity{F<:Function, T<:Real}
    f::F
    lb::T
    ub::T
    function UnivariateDensity{F,T}(f::F, lb::T, ub::T) where {F<:Function, T<:Real}
        area = quadgk(f, lb, ub)[1]
        minf = Optim.minimum(Optim.optimize(f, lb, ub, Brent()))
        if area ≈ 1.0 && lb < ub && (minf ≥ 0 || minf ≈ 0.0)
            g(t) = lb ≤ t ≤ ub ? f(t) : zero(t)
            return new{typeof(g), T}(g, lb, ub)
        else
            @error "Not a valid density!"
            # Maybe better to throw an argument exception?
        end
    end
end

UnivariateDensity(f::F, lb::T, ub::T) where {F<:Function, T <: Real} = UnivariateDensity{F, T}(f, lb, ub)
UnivariateDensity(f::Function, lb::Real, ub::Real) = UnivariateDensity(f, promote(lb, ub)...)
UnivariateDensity(f::Real, lb, ub) = UnivariateDensity(_ -> 1 / (ub - lb), lb, ub)

function integrate(g::Function, d::UnivariateDensity, bounds::Vararg{T}) where {T <: Real}
    integrand(t) = g(t) * d.f(t)
    if length(bounds) == 0
        return quadgk(integrand, d.lb, d.ub)
    elseif length(bounds) == 1
        return quadgk(integrand, bounds[1], d.ub)
    else
        return quadgk(integrand, bounds[1], bounds[2])
    end
end
