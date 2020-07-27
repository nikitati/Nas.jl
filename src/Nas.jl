module Nas

using Random
using StatsBase: sample, ProbabilityWeights
using Flux
using Flux.Optimise: update!
using Zygote
using Zygote: Params, IdSet

include("utils.jl")
include("layers.jl")
export StatefulSoftmax, GumbelSoftmax, STGumbelSoftmax
include("searchspaces.jl")
export ChoiceNode, archparams, set_choice!, choices, archfix,
       sample_architecture, set_architecture!, set_random_architecture!,
       SPChoiceNode
include("strategies.jl")
export RandomSearch, DARTSearch, ENASearch, SNASearch, optimize, optimize!

end
