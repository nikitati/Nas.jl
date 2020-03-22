mutable struct Assignment{T}
    value::Union{Nothing, T}
end

struct Domain{T}
    values::AbstractArray{T, 1}
end

struct Hyperparameter{T}
    assignment::Assignment{T}
    domain::Domain{T}
end

function Hyperparameter(domain::Domain{T}) where {T}
    Hyperparameter(Assignment{T}(nothing), domain)
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
