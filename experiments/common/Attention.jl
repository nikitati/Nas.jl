module Attention

using Flux
using Flux: @functor
using Mill
using Zygote: Buffer

abstract type AttentionFunction end

struct LinearAttention{F} <: AttentionFunction
    f::F
end

LinearAttention(in::Integer) = LinearAttention(Dense(in, 1))

@functor LinearAttention

(att::LinearAttention)(x) = att.f(x)

struct MLPAttention{F} <: AttentionFunction
    f::F
end

@functor MLPAttention

MLPAttention(in::Integer, hidden::Integer) = MLPAttention(Chain(Dense(in, hidden, Flux.tanh), Dense(hidden, 1)))

(att::MLPAttention)(x) = att.f(x)

struct GatedAttention{H, G, F} <: AttentionFunction
    h::H
    g::G
    f::F
end

@functor GatedAttention

GatedAttention(in::Integer, hidden::Integer) = GatedAttention(Dense(in, hidden, Flux.tanh), Dense(in, hidden, Flux.sigmoid), Dense(hidden, 1))

(att::GatedAttention)(x) = att.f(att.h(x) .* att.g(x))



"""
    AttentionPooling(AttentionFunction)
"""
struct AttentionPooling{T <: AttentionFunction} <: AggregationFunction
    att::T
end

@functor AttentionPooling

(m::AttentionPooling)(x::MaybeMatrix, bags::AbstractBags) = attention_pool_forw(x, bags)

function attention_pool_forw(x::AbstractMatrix, bags::AbstractBags)
    y = Buffer(zeros(eltype(x), size(x, 1), length(bags)))
    cs = att(x)
    @inbounds for (bi, b) in enumerate(bags)
        ws = Flux.softmax(cs[:, b])
        for j in b
            y[:, bi] += ws[j] .* x[:, j]
        end
    end
    copy(y)
end

# function attention_pool_back(Δ, y, x, bags, w)
#     dx = similar(x)
#     dw = zero(w)
#     @inbounds for (bi, b) in enumerate(bags)
#         for j in b
#             dx[:, j] = w[j] * Δ[i, bi]
#             ∇dw_segmented_sum!(dw, Δ, x, y, w, i, j, bi)
#         end
#     end
#     dx, nothing, dw
# end

end  # module Attention
