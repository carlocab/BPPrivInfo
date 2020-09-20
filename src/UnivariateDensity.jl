# How do I check that the range of `f` is nonnegative?
struct UnivariateDensity{T<:Real}
    f::Function
    lb::T
    ub::T
    function UnivariateDensity{T}(f::Function, lb::T, ub::T) where {T <: Real}
        area = quadgk(f, lb, ub)[1]
        if area ≈ 1 && lb < ub
            g(t) = lb ≤ t ≤ ub ? f(t) : zero(t)
            return new{T}(g, lb, ub)
        else
            @error "Not a valid density!"
            # Maybe better to throw an argument exception?
        end
    end
end

UnivariateDensity(f::Function, lb::T, ub::T) where {T <: Real} = UnivariateDensity{T}(f, lb, ub)
UnivariateDensity(f::Function, lb::Real, ub::Real) = UnivariateDensity(f, promote(lb, ub)...)

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
