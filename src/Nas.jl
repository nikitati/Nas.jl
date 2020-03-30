module Nas


include("searchspaces.jl")
export ChoiceNode, SearchSpace, applicative, GumbelSoftmax, gumbelrand
include("strategies.jl")
export optimize!, RandomSearch

end
