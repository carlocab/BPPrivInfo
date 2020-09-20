# How do I check that the range of `f` is nonnegative?
struct UnivariateDensity{T} where {T <: Real}
    f::Function
    lb::T
    ub::T
    function Density{T}(f, lb, ub) where {T <: Real}
        area = quadgk(f, lb, ub)[1] && lb < ub
        if area ≈ 1
            return new{T}(f, lb, ub)
        else
            @error "Not a valid density!"
        end
    end
end
