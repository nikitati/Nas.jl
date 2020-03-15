
mutable struct ParameterValue{T}
    value::Union{Nothing, T}
end

struct ParameterDomain{T}
    values::AbstractArray{T, 1}
end

struct Hyperparameter{T}
    assignment::ParameterValue{T}
    domain::ParameterDomain{T}
end

function Hyperparameter(domain::ParameterDomain{T}) where {T}
    Hyperparameter(ParameterValue{T}(nothing), domain)
end

function assign!(hyperparameter::Hyperparameter{T}, value::T) where {T}
    hyperparameter.assignment.value = value
end

function getvalue(hyperparameter::Hyperparameter{T}) where {T}
    hyperparameter.assignment.value
end

function hyperparamfields(mt)
    filter(fname -> fieldtype(mt, fname) <: Hyperparameter,
           fieldnames(mt))
end

function hyperparams(m)
    paramfields = hyperparamfields(typeof(m))
    (map(fname -> getfield(m, fname), paramfields)...,)
end

struct DependentParameter
    parents::AbstractArray{Hyperparameter, 1}
    binding::Function
end

function getvalue(dp::DependentParameter)
    dp.binding(dp.parents...)
end

struct TemplateDense
    hidden::Hyperparameter{<:Integer}
    activation::Hyperparameter{Function}
    input::DependentParameter
end

function TemplateDense(
    hd::ParameterDomain{<:Integer},
    ad::ParameterDomain{Function},
    in::DependentParameter
    )
    TemplateDense(Hyperparameter(hd),
                  Hyperparameter(ad),
                  in)
end

function TemplateDense(
    hd::Hyperparameter{<:Integer},
    ad::Hyperparameter{Function},
    in::TemplateDense
    )
    TemplateDense(hd, ad, DependentParameter([in.hidden], getvalue))
end

function compile(td::TemplateDense)
    Dense(map(getvalue, (td.input, td.hidden, td.activation))...)
end

struct TemplateChain{T<:Tuple}
    modules::T
    TemplateChain(ms...) = new{typeof(ms)}(ms)
end

function compile(tc::TemplateChain)
    Chain(map(compile, tc.modules)...)
end
