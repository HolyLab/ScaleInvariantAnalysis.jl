# We could use WoodburyMatrices.jl, but we only need rank-1 and this reduces the dependency burden.
struct ShermanMorrisonMatrix{T<:Number, AT<:AbstractMatrix{T}, UT<:AbstractVector{T}, VT<:AbstractVector{T}} <: AbstractMatrix{T}
    A::AT
    u::UT
    v::VT
    A⁻¹u::UT
    A⁻¹v::VT
    vᵀA⁻¹u::T

    function ShermanMorrisonMatrix{T, AT, UT, VT}(A, u, v) where {T<:Number, AT<:AbstractMatrix{T}, UT<:AbstractVector{T}, VT<:AbstractVector{T}}
        axes(A, 1) == eachindex(u) || throw(ArgumentError("Dimensions of A and u must be compatible (got $(axes(A, 1)) and $(eachindex(u)))"))
        axes(A, 2) == eachindex(v) || throw(ArgumentError("Dimensions of A and v must be compatible (got $(axes(A, 2)) and $(eachindex(v)))"))
        A⁻¹u = A \ u
        A⁻¹v = A \ v
        vᵀA⁻¹u = dot(v, A⁻¹u)
        new{T, AT, UT, VT}(A, u, v, A⁻¹u, A⁻¹v, vᵀA⁻¹u)
    end
end
function ShermanMorrisonMatrix{T}(A, u, v) where T<:Number
    A = convert(AbstractMatrix{T}, A)::AbstractMatrix{T}
    u = convert(AbstractVector{T}, u)::AbstractVector{T}
    v = convert(AbstractVector{T}, v)::AbstractVector{T}
    return ShermanMorrisonMatrix{T, typeof(A), typeof(u), typeof(v)}(A, u, v)
end
ShermanMorrisonMatrix(A, u, v) = ShermanMorrisonMatrix{eltype(A)}(A, u, v)

Base.size(SM::ShermanMorrisonMatrix) = size(SM.A)
Base.IndexStyle(::Type{<:ShermanMorrisonMatrix}) = IndexCartesian()
Base.getindex(SM::ShermanMorrisonMatrix, i::Int, j::Int) = SM.A[i, j] + SM.u[i] * SM.v[j]
Base.setindex!(SM::ShermanMorrisonMatrix, val, i::Int, j::Int) = setindex!(SM.A, val - SM.u[i] * SM.v[j], i, j)

function LinearAlgebra.mul!(y::AbstractVector, SM::ShermanMorrisonMatrix, x::AbstractVector, α::Number, β::Number)
    mul!(y, SM.A, x, α, β)
    dotvx = α * dot(SM.v, x)
    y .+= dotvx .* SM.u
    return y
end

Base.:(*)(SM::ShermanMorrisonMatrix, x::AbstractVector) = mul!(similar(x), SM, x, true, false)

function LinearAlgebra.ldiv!(SM::ShermanMorrisonMatrix, y::AbstractVector)
    ldiv!(SM.A, y)
    α = dot(SM.v, y) / (1 + SM.vᵀA⁻¹u)
    y .-= α .* SM.A⁻¹u
    return y
end

Base.:(\)(SM::ShermanMorrisonMatrix, x::AbstractVector) = ldiv!(similar(x), SM, x)
