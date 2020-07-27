"""
    StatefulSoftmax(n)

`StatefulSoftmax` layer provides a parametrized `n`-dimensional softmax
function. Outputs can be used as coefficients for the differentiable convex
combination or for the complementary gate.
"""
struct StatefulSoftmax{L}
    logits::L
end

StatefulSoftmax(n::Integer) = StatefulSoftmax(log.(ones(Float32, n) ./ n))

Flux.@functor StatefulSoftmax

function (ss::StatefulSoftmax)()
    Flux.softmax(ss.logits)
end

function Base.show(io::IO, ss::StatefulSoftmax)
  print(io, "StatefulSoftmax(", "logits=", ss.logits)
  print(io, ")")
end


"""
    GumbelSoftmax(n)

`GumbelSoftmax` layer allows to sample from a categorical distribution in a
differentiable way with a reparametrization trick. Using distribution's class
probabilities `p`, Gumbel random variable realization `g` and temperature
parameter `t`, a sample can be obtained as:

    z = softmax((log.(p) .+ g)./t)

"""
mutable struct GumbelSoftmax{L, T}
    logits::L
    t::T
end

Flux.@functor GumbelSoftmax

Flux.trainable(gs::GumbelSoftmax) = (gs.logits,)

GumbelSoftmax(n::Integer) = GumbelSoftmax(log.(ones(Float32, n) ./ n), one(Float32))
GumbelSoftmax(n::Integer, temp) = GumbelSoftmax(log.(ones(Float32, n) ./ n), Float32(temp))

function (gs::GumbelSoftmax)()
    logits, t = gs.logits, gs.t
    gumbelsoftmax(logits, t)
end

function Base.show(io::IO, gs::GumbelSoftmax)
  print(io, "GumbelSoftmax(", "n=", length(gs.logits), ", ", "temp=", gs.t)
  print(io, ")")
end


"""
    STGumbelSoftmax(n)

`STGumbelSoftmax` is a modification of the `GumbelSoftmax` which makes a
strict sample from a categorical distribution:

    z = onehot(argmax((log.(p) .+ g)./t))

However, backward pass is the same as in the `GumbelSoftmax` layer. This
estimator is called Straight-Through Gumbel-Softmax in literature.
"""
mutable struct STGumbelSoftmax{L, T}
    logits::L
    t::T
end

Flux.@functor STGumbelSoftmax

Flux.trainable(gs::STGumbelSoftmax) = (gs.logits,)

STGumbelSoftmax(n::Integer) = STGumbelSoftmax(log.(ones(Float32, n) ./ n), one(Float32))
STGumbelSoftmax(n::Integer, temp) = STGumbelSoftmax(log.(ones(Float32, n) ./ n), Float32(temp))

function stgumbelforw(logits, t)
    y = gumbelsoftmax(logits, t)
    Flux.onehot(argmax(y), 1:length(y))
end

Zygote.@adjoint function stgumbelforw(logits, t)
    y, back = Zygote.pullback(gumbelsoftmax, logits, t)
    Flux.onehot(argmax(y), 1:length(y)), back
end

function (stgs::STGumbelSoftmax)()
    logits, t = stgs.logits, stgs.t
    stgumbelforw(logits, t)
end

function Base.show(io::IO, gs::STGumbelSoftmax)
  print(io, "Straight-Through GumbelSoftmax(", "n=", length(gs.logits), ", ", "temp=", gs.t)
  print(io, ")")
end


"""
    EmbeddingLayer(hs, vs)

`EmbeddingLayer` is a trainable lookup table which maps a finite discrete variable to a
real vector of dimension `hs`.
"""
struct EmbeddingLayer{T}
    W::T
    vocab
    unk
end

Flux.@functor EmbeddingLayer

Flux.trainable(emb::EmbeddingLayer) = (emb.W,)

EmbeddingLayer(hs::Integer, vs::Integer, vocab, unk) = EmbeddingLayer(Flux.glorot_normal(hs, vs), vocab, unk)
EmbeddingLayer(hs::Integer, vs::Integer) = EmbeddingLayer(hs, vs+1, 0:vs, 0)

function (emb::EmbeddingLayer)(x)
    emb.W * Flux.onehotbatch(x, emb.vocab, emb.unk)
end


"""
    SetLayer(in::Integer, out::Integer, agg, σ)

Permutation-equivariant layer from the DeepSets paper.
Currently works only with batches of vectors.
"""
struct SetLayer
    Γ
    β
    agg
    σ
end

Flux.@functor SetLayer

function SetLayer(in::Integer, out::Integer, agg = sum, σ = identity;
               initW = glorot_uniform, initb = zeros)
    return SetLayer(initW(out, in), initb(out), agg, σ)
end

function (s::SetLayer)(x)
    Γ, β, agg, σ = s.Γ, s.β, s.agg, s.σ
    bsize = size(x, 3)
    h = x .- agg(x, dims=2)
    a = NNlib.batched_mul(repeat(Γ, 1, 1, bsize), h) .+ β
    σ(a)
end

struct ZeroOp
    dims
    z
end

ZeroOp(dims...) = ZeroOp(dims, zeros(Float32, dims...))

function (zo::ZeroOp)(x)
    bsize = size(x, 3)
    repeat(zo.z, bsize)
end
