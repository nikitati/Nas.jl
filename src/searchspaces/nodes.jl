
struct ChoiceNode{T}
    functors::AbstractArray{T, 1}
end

function (cn::ChoiceNode)(x)
    [f(x) for f in cn.functors]
end

function (cn::ChoiceNode)(i, x)
    cn.functors[i](x)
end

function nchoices(cn::ChoiceNode)
    length(cn.functors)
end

struct InputNode{T}
    inputs::AbstractArray{T, 1}
end

function (in::InputNode)(ix, x)
    in.inputs[ix](x)
end

function nchoices(cn::InputNode)
    length(cn.inputs)
end
