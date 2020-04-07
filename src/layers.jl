"""
    GumbelSoftmax(p, t)

`GumbelSoftmax` layer allows to sample from a categorical distribution in a
differentiable way with a reparametrization trick. Using distribution's class
probabilities `p`, gumbel random variable realization `g` and temperature
parameter `t`:

    z(g) = softmax((log.(p) .+ g)./t)

"""
mutable struct GumbelSoftmax{P, T}
    p::P
    t::T
end

Flux.@functor GumbelSoftmax (p,)

GumbelSoftmax(n::Integer) = GumbelSoftmax(ones(n) ./ n, 1.0)

function (gs::GumbelSoftmax)(g)
    p, t = gs.p, gs.t
    glogits = (log.(p) .+ g) ./ t
    softmax(glogits)
end

function (gs::GumbelSoftmax)()
    g = gumbelrand(size(gs.p)...)
    gs(g)
end

function Base.show(io::IO, gs::GumbelSoftmax)
  print(io, "GumbelSoftmax(", "probs=", gs.p, ", ", "temp=", gs.t)
  print(io, ")")
end

"""
    StatefulSoftmax(n)

`StatefulSoftmax` layer provides a parametrized `n`-dimensional softmax
function. Outputs can be used as coefficients for the differentiable convex
combination.
"""
struct StatefulSoftmax{T}
    p::AbstractArray{T}
end

Flux.@functor StatefulSoftmax

StatefulSoftmax(n::Integer) = StatefulSoftmax(zeros(n))

function (ss::StatefulSoftmax)()
    softmax(ss.p)
end

function Base.show(io::IO, ss::StatefulSoftmax)
  print(io, "StatefulSoftmax(", "probs=", ss.p)
  print(io, ")")
end
