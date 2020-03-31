using Random
using Flux

"""
    ChoiceNode(layers...)

Create a search space choice between several layers.
"""
struct ChoiceNode{T}
    ms::AbstractArray{T, 1}
end

ChoiceNode(ms...) = ChoiceNode([ms...])

functor(cn::ChoiceNode) = cn.ms, ms -> ChoiceNode(ms...)

function (cn::ChoiceNode)(x::AbstractArray)
    map(m -> m(x), cn.ms)
end

function (cn::ChoiceNode)(x::AbstractArray, z::AbstractArray)
    sum(cn(x) .* z)
end

applicative(cn::ChoiceNode, α::AbstractArray) = x -> cn(x, α)
applicative(cn::ChoiceNode, α) = x -> cn(x, α())


Base.getindex(cn::ChoiceNode, i::Integer) = cn.ms[i]
Base.getindex(cn::ChoiceNode, i::AbstractArray) = cn.ms[i]

function Base.show(io::IO, cn::ChoiceNode)
  print(io, "Choice Node(")
  join(io, cn.ms, ", ")
  print(io, ")")
end


"""
    SearchSpace{T}(choices::AbstractArray{{ChoiceNode, T}, 1})

`SearchSpace` maintains all choice nodes `choices` and their respective
architecture parameters `alpha`.
Architecture parameters can be
"""
struct SearchSpace
    choices
end

Base.getindex(ss::SearchSpace, i::Integer) = ss.choices[i]

function gumbelrand(dims...)
    -log.(-log.(rand(dims...)))
end

"""
    GumbelSoftmax(p::Vector{Float32}, t::Float32)

`GumbelSoftmax` layer allows to sample from a categorical distribution in a
differentiable way with a reparametrization trick from a vector of class
probabilities `p`, gumbel random variable `g` and temperature parameter `t`:

    z(g) = σ((log(p) + g)/t)

where σ is a softmax function.
"""
mutable struct GumbelSoftmax
    p::AbstractArray{Float64, 1}
    t::Float64
end

Flux.@functor GumbelSoftmax (p,)

function (gs::GumbelSoftmax)(g::AbstractArray)
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


struct StatefulSoftmax
    p::AbstractArray{Float64, 1}
end

Flux.@functor StatefulSoftmax

function (ss::StatefulSoftmax)()
    softmax(ss.p)
end

function Base.show(io::IO, ss::StatefulSoftmax)
  print(io, "StatefulSoftmax(", "probs=", ss.p)
  print(io, ")")
end
