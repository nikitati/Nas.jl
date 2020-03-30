using Random
using Flux

"""
    ChoiceNode(layers...)

Create a search space choice between several layers.
"""
struct ChoiceNode{T}
    ms::Vector{T}
end

ChoiceNode(ms...) = ChoiceNode([ms...])

functor(cn::ChoiceNode) = cn.ms, ms -> ChoiceNode(ms...)

function (cn::ChoiceNode)(x::AbstractArray)
    map(m -> m(x), cn.ms)
end

function (cn::ChoiceNode)(x::AbstractArray, z::AbstractArray)
    sum(cn(x) .* z)
end

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

function applicative(ss::SearchSpace, i::Integer)
    cn, α = ss[i]
    x -> cn(x, α)
end

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
    p::Vector{Float64}
    t::Float64
end

Flux.@functor GumbelSoftmax (p,)

function (gs::GumbelSoftmax)(g::AbstractArray)
    p, t = gs.p, gs.t
    glogits = (log.(p) .+ g) ./ t
    softmax(glogits)
end

function Base.show(io::IO, gs::GumbelSoftmax)
  print(io, "GumbelSoftmax(", "probs=", gs.p, ", ", "temp=", gs.t)
  print(io, ")")
end
