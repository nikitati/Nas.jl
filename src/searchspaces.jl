abstract type AbstractChoiceNode end

"""
    ChoiceNode(layers...)
    ChoiceNode(architecture, layers)

`ChoiceNode` contains several layers from which architecture optimization
algorithm is able to choose from. Input and output dimensions should be
compatible for each layer. It is also possible to provide architecture model
which parametrizes the choice node or architecture coefficients directly.
"""
mutable struct ChoiceNode{A, C <: Tuple} <: AbstractChoiceNode
    architecture::A
    choices::C
    choice
end

ChoiceNode(archtype, ms::AbstractArray) = ChoiceNode(archtype(length(ms)), tuple(ms...), 0)
ChoiceNode(ms...) = ChoiceNode(nothing, ms, 0)

Flux.@functor ChoiceNode (choices,)

function (cn::ChoiceNode{Nothing, C})(x) where {C <: Tuple}
    f = cn.choices[cn.choice]
    f(x)
end

function (cn::ChoiceNode)(x)
    a = cn.architecture()
    choices = cn.choices
    # TODO avoid unnecessary evaluation
    # choices = cn.choices[a .!= 0]
    # a = [a .!= 0]
    # TODO figure out why it is neccessary for Zygote to work
    y = map(f -> f(x), choices)
    y = [y...]
    sum(y .* a)
end

function archparams(cn::ChoiceNode{Nothing, C}) where {C <: Tuple}
    Flux.Params()
end

archparams(cn::AbstractChoiceNode) = Flux.params(cn.architecture)

function set_choice!(cn::AbstractChoiceNode, choice)
    cn.choice = choice
end

function getchoice(cn::AbstractChoiceNode)
    cn.choice
end

Base.length(cn::AbstractChoiceNode) = length(cn.choices)

Base.getindex(cn::AbstractChoiceNode, i::Integer) = cn.choices[i]
Base.getindex(cn::AbstractChoiceNode, i::AbstractArray) = cn.choices[i]

function Base.show(io::IO, cn::ChoiceNode)
    print(io, "Choice Node(")
    join(io, cn.choices, ", ")
    print(io, ")")
end

"""
    SPChoiceNode

Single path mode.
"""
struct SPChoiceNode{A, C} <: AbstractChoiceNode
    architecture::A
    choices::C
    choice
end

SPChoiceNode(archtype, ms::AbstractArray) = SPChoiceNode(archtype(length(ms)), tuple(ms...), 0)

Flux.@functor SPChoiceNode (choices,)

function (cn::SPChoiceNode)(x)
    a = cn.architecture()
    i = argmax(a)
    cn[i](x)
end

choices!(p::AbstractArray, x::T, seen = IdSet()) where {T <: AbstractChoiceNode} = push!(p, x)

function choices!(p::AbstractArray, x, seen = IdSet())
    x in seen && return
    push!(seen, x)
    fs = (getfield(x, name) for name in  fieldnames(typeof(x)))
    for child in fs
      choices!(p, child, seen)
    end
end

function choices(searchspace)
    cs = []
    choices!(cs, searchspace)
    return tuple(cs...)
end

function archparams(searchspace)
    archps = Params()
    cs = choices(searchspace)
    for choice in cs
        push!(archps, archparams(choice)...)
    end
    return archps
end

function fix_(x::AbstractChoiceNode, a)
    x[a[x]]
end

function fix_(x::Chain, a)
    Chain((fix_(layer, a) for layer in x.layers)...)
end

function fix_(x::T, a) where {T}
    fs = fieldnames(T)
    isempty(fs) && return x
    c = getproperty(parentmodule(T), nameof(T))
    c((fix_(getfield(x, n), a) for n in fs)...)
end

function archfix(searchspace, architecture::IdDict)
    fix_(searchspace, architecture)
end

function get_choice_probs(choicenode::ChoiceNode{Nothing, C}) where {C <: Tuple}
    n = length(choicenode)
    return [1 / n for i = 1:n]
end

function get_choice_probs(choicenode::AbstractChoiceNode)
    choicenode.architecture()
end

function get_probs(choice_nodes)
    IdDict(cn => ProbabilityWeights(get_choice_probs(cn)) for cn in choice_nodes)
end

function sample_choices(prob_weights)
    IdDict(k => sample(pw) for (k, pw) in pairs(prob_weights))
end

function sample_architecture(searchspace, probabilites)
    probabilites = IdDict(k => ProbabilityWeights(probs) for (k, probs) in pairs(probabilites))
    archchoices = sample_choices(probabilites)
    archfix(searchspace, archchoices)
end

function sample_architecture(searchspace)
    choice_nodes = choices(searchspace)
    probabilites = get_probs(choice_nodes)
    sample_architecture(searchspace, probabilites)
end

function set_architecture!(searchspace, archchoices)
    for (choice_node, choice) in pairs(archchoices)
        set_choice!(choice_node, choice)
    end
end

function set_random_architecture!(searchspace)
    choice_nodes = choices(searchspace)
    probabilites = get_probs(choice_nodes)
    archchoices = sample_choices(probabilites)
    set_architecture!(searchspace, archchoices)
end

function set_random_architecture!(searchspace, probabilities)
    probabilites = IdDict(k => ProbabilityWeights(probs) for (k, probs) in pairs(probabilities))
    archchoices = sample_choices(probabilites)
    set_architecture!(searchspace, archchoices)
end
