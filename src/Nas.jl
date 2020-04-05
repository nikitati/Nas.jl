module Nas

using Random
using Flux
using Flux: trainable
using Zygote: IdSet

include("utils.jl")
include("layers.jl")
export GumbelSoftmax, StatefulSoftmax
include("searchspaces.jl")
export ChoiceNode, SearchSpace, choices
# include("strategies.jl")
# export optimize!, RandomSearch

end
