module Nas

using Random
using StatsBase: sample, ProbabilityWeights
using Flux
using Flux: trainable
using Flux.Optimise: update!
using Zygote: Params, IdSet

include("utils.jl")
include("layers.jl")
export GumbelSoftmax, StatefulSoftmax
include("searchspaces.jl")
export ChoiceNode, SearchSpace, choices
# include("strategies.jl")
# export optimize!, RandomSearch

end
