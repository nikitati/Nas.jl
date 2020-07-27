call(f, xs...) = f(xs...)
runall(f) = f
runall(fs::AbstractVector) = () -> foreach(call, fs)


function gumbelrand(T, dims)
    -log.(-log.(rand(T, dims) .+ T(1e-20)) .+ T(1e-20))
end

function gumbelsoftmax(logits, t)
    g = gumbelrand(typeof(t), size(logits))
    log_probs = Flux.logsoftmax(logits)
    Flux.softmax((log_probs .+ g) ./ t)
end


mutable struct Accumulator{P <: Number, V}
    priorities::AbstractArray{P}
    values::AbstractArray{V}
    minpriorityidx::Integer
end

Accumulator(c::Integer) = Accumulator(fill(-Inf, c), Array{Any}(undef, c), 1)

function add!(a::Accumulator{P, V}, priority::P, item::V) where {P, V}
    minpriority = a.priorities[a.minpriorityidx]
    if priority > minpriority
        a.priorities[a.minpriorityidx] = priority
        a.values[a.minpriorityidx] = item
        a.minpriorityidx = argmin(a.priorities)
    end
end

function peek(a::Accumulator)
    maxpriorityidx = argmax(a.priorities)
    a.values[maxpriorityidx]
end

function toarray(a::Accumulator)
    order = sortperm(a.priorities)
    return (a.priorities[order], a.values[order])
end
