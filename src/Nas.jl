module Nas

using Flux, Zygote

export ParameterValue, ParameterDomain, Hyperparameter, DependentParameter,
       hyperparams, getvalue, assign!, compile, compose,
       TemplateDense, TemplateChain
include("searchspaces/basic.jl")


end
