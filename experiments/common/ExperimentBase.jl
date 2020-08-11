module ExperimentBase

using Random
using Base.Iterators: partition
using StatsBase
using Flux
using Flux: testmode!
using Flux.Losses
using Nas


accuracy(ŷ, y) = mean((ŷ .> oftype(ŷ[1], 0.5)) .== y)

bcentropyloss(ŷ, y) = Losses.logitbinarycrossentropy(ŷ, y)

sqnorm(x) = sum(abs2, x)
l2regularization(params) = sum(sqnorm, params)

function bagbatches(bagnode, targets; batchsize = 1, shuffle = true)
    idx = randperm(length(bagnode.bags))
    batches = partition(idx, batchsize)
    return ((bagnode[batch], targets[batch]) for batch in batches)
end

function evaluate(model, testset, metric; modelcall = (m, x) -> m(x))
    (x, y) = testset
    testmode!(model)
    return metric(modelcall(model, x), y)
end

function evaluate_samples(searchspace, nsamples, evaluator)
    metric = -Inf
    arch = nothing
    for j = 1:nsamples
        a = sample_architecture(searchspace)
        m = evaluator(a)
        (metric, arch) = (m > metric) ? (m, a) : (metric, arch)
    end
    return (metric, arch)
end

function from_config(::Type{RandomSearch}, config)
    rsconf = config["search"]["random"]
    RandomSearch(rsconf["n"], rsconf["k"])
end

function from_config(::Type{DARTSearch}, config)
    dartsconf = config["search"]["darts"]
    (w_opt, α_opt) = Descent(dartsconf["lr_w"]), ADAM(dartsconf["lr_alpha"])
    DARTSearch(dartsconf["ksi"], w_opt, α_opt)
end

function from_config(::Type{SNASearch}, config)
    snasconf = config["search"]["snas"]
    opt = ADAM(snasconf["lr"])
    SNASearch(snasconf["epochs"], snasconf["decay"], opt)
end

function from_config(::Type{ENASearch}, config, searchspace)
    enasconf = config["search"]["enas"]
    opt = ADAM(enasconf["lr"])
    ENASearch(searchspace, opt, enasconf["controller_epochs"], enasconf["controller_hidden"])
end

end # module ExperimentBase
