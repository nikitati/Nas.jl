module Nas

using Flux, Zygote


include("searchspaces/hyperparameters.jl")
export Assignment, Domain, Hyperparameter, DependentParameter,
       getvalue, assign!, hyperparams
include("searchspaces/nodes.jl")
export ChoiceNode, InputNode,
       nchoices


end
