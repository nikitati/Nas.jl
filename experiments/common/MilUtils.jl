module MilUtils

import Base: *, +
import Base: show

using Flux
using Flux: @functor
import Flux.testmode!
using Mill
using Mill: BagChain
using Zygote: Buffer

*(c::Float32, a::ArrayNode) = ArrayNode(c .* a.data, a.metadata)
*(a::ArrayNode, c::Float32) = c * a
+(a::ArrayNode, b::ArrayNode) = ArrayNode(a.data + b.data, a.metadata)

show(io::IO, a::ArrayModel) = show(io, a.m)

testmode!(m::BagChain, mode = true) = (map(x -> testmode!(x, mode), m.layers); m)
testmode!(m::BagModel, mode = true) = (testmode!(m.im, mode); testmode!(m.bm, mode); m)
testmode!(m::ArrayModel, mode = true) = (testmode!(m.m, mode); m)


"""
    SetLayer(in::Integer, out::Integer, agg, σ)

Permutation-equivariant layer from the DeepSets paper.
Currently works only with batches of vectors.
"""
struct SetLayer{A, F, T<:AbstractArray, B<:AbstractArray}
    Γ::T
    β::B
    agg::A
    σ::F
end

function SetLayer(in::Integer, out::Integer, agg = sum, σ = identity;
               initW = Flux.glorot_uniform, initb = zeros)
    return SetLayer(initW(out, in), initb(out), agg, σ)
end

@functor SetLayer

function (s::SetLayer)(x::AbstractArray{T, 3}) where {T <: Union{Float32,Float64}}
    Γ, β, agg, σ = s.Γ, s.β, s.agg, s.σ
    bsize = size(x, 3)
    h = x .- agg(x, dims=2)
    a = NNlib.batched_mul(repeat(Γ, 1, 1, bsize), h) .+ β
    σ.(a)
end

function (s::SetLayer)(x::AbstractArray{T, 2}) where {T <: Union{Float32, Float64}}
    Γ, β, agg, σ = s.Γ, s.β, s.agg, s.σ
    h = x .- agg(x, dims=2)
    σ.(Γ*h .+ β)
end


end  # module MilUtils
