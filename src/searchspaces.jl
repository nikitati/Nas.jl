
"""
    ChoiceNode(layers...)
    ChoiceNode(layers, architecture)

`ChoiceNode` contains several layers from which architecture optimization
algorithm is able to choose from. Input and output dimensions should be
compatible for each layer. It is also possible to provide architecture model
which parametrizes the choice node or architecture coefficients directly.
"""
struct ChoiceNode{C, A}
    choices::C
    architecture::A
end

Flux.@functor ChoiceNode (choices,)

ChoiceNode(ms::AbstractArray, arch) = ChoiceNode(tuple(ms...), arch)
ChoiceNode(ms::AbstractArray) = ChoiceNode(tuple(ms...), zeros(length(ms)))
ChoiceNode(ms...) = ChoiceNode(ms, zeros(length(ms)))

model_params(cn::ChoiceNode) = Flux.params(cn.choices)

function architecture_params(cn::ChoiceNode{C, A}) where {C, A <: Union{AbstractArray{<:Number}, Nothing}}
    Flux.Params()
end

architecture_params(cn::ChoiceNode) = Flux.params(cn.architecture)

mapdata(cn::ChoiceNode, x) = map(f -> f(x), cn.choices)

function (cn::ChoiceNode)(x, z)
    xs = mapdata(cn, x)
    sum(xs .* z)
end

function (cn::ChoiceNode{C, Nothing})(x) where {C}
    arr = "Cannot implicitly call ChoiceNode without an architecture"
    throw(MethodError(xn, err))
end

function (cn::ChoiceNode{C, A})(x) where {C, A <: AbstractArray{<:Number}}
    z = cn.architecture
    cn(x, z)
end

function (cn::ChoiceNode)(x)
    z = cn.architecture()
    cn(x, z)
end

Base.length(cn::ChoiceNode) = length(cn.choices)

Base.getindex(cn::ChoiceNode, i::Integer) = cn.choices[i]
Base.getindex(cn::ChoiceNode, i::AbstractArray) = cn.choices[i]

function Base.show(io::IO, cn::ChoiceNode)
  print(io, "Choice Node(")
  join(io, cn.choices, ", ")
  print(io, ")")
end

"""
    SearchSpace(model)

`SearchSpace` encapsulates the Flux model (functor) equipped with `ChoiceNode`
layers.
"""
struct SearchSpace{T}
    model::T
end

choices!(p::AbstractArray, x::ChoiceNode, seen = IdSet()) = push!(p, x)

function choices!(p::AbstractArray, x, seen = IdSet())
  x in seen && return
  push!(seen, x)
  for child in trainable(x)
    choices!(p, child, seen)
  end
end

function choices(searchspace::SearchSpace)
    m = searchspace.model
    cs = []
    choices!(cs, m)
    return tuple(cs...)
end

function params(searchspace::SearchSpace)
  m = searchspace.model
  ps = Flux.params(m)
  archps = Params()
  cs = choices(searchspace)
  for choice in cs
      push!(archps, architecture_params(choice)...)
  end
  return ps, archps
end
