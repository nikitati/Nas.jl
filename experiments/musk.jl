import YAML
using ArgParse
using Random
using FileIO, JLD2

using StatsBase: mean, std, median
using Nas
using Mill
using Mill: BagChain
using Flux
using Flux: onehotbatch, onecold, crossentropy, @epochs, throttle
using Flux.Optimise: Optimiser
using Flux.Losses
using MLDataPattern: kfolds, splitobs

include("common/ExperimentBase.jl")
using .ExperimentBase: bagbatches, accuracy, bcentropyloss, l2regularization,
                        evaluate, evaluate_samples, from_config
include("common/Attention.jl")
using .Attention: LinearAttention, MLPAttention, GatedAttention, AttentionPooling
include("common/MilUtils.jl")
import .MilUtils

function load_data()
    fMat = load("data/musk/musk.jld2", "fMat")
    bagids = load("data/musk/musk.jld2", "bagids")
    x = BagNode(ArrayNode(Flux.normalise(fMat, dims=2)), bagids)
    y = load("data/musk/musk.jld2", "y")
    y = map(i -> maximum(y[i]), x.bags)
    idx = randperm(length(y))
    return x[idx], y[idx]
end

function make_baseline()
    model = BagModel(
    ArrayModel(Chain(
        Dense(166, 128),
        BatchNorm(128, Flux.relu),
        Dense(128, 128),
        BatchNorm(128, Flux.relu),
        Dense(128, 128, Flux.relu),
        Dropout(0.5)
        )
        ),
    SegmentedMax(128),
    ArrayModel(Dense(128, 1))
    )
    return model
end

struct SearchSpace
    ϕ
    aggs
    cs
    ρ
end

function SearchSpace(archtype)
    SearchSpace(
    (ArrayModel(Chain(
        Dense(166, 128),
        BatchNorm(128),
        ChoiceNode(archtype, [x -> Flux.relu.(x), x -> Flux.tanh.(x), x -> Flux.elu.(x)]))),
    ArrayModel(Chain(
        Dense(128, 128),
        BatchNorm(128),
        ChoiceNode(archtype, [x -> Flux.relu.(x), x -> Flux.tanh.(x), x -> Flux.elu.(x)]))),
    ArrayModel(Chain(
        Dense(128, 128),
        BatchNorm(128),
        ChoiceNode(archtype, [x -> Flux.relu.(x), x -> Flux.tanh.(x), x -> Flux.elu.(x)]),
        Dropout(0.5)))
    ),
    ntuple(_ -> ChoiceNode(archtype, [SegmentedMax(128), SegmentedMean(128), SegmentedSum(128)]), 3),
    ntuple(_ -> ChoiceNode(archtype, [ArrayModel(identity), ArrayModel(zero)]), 3),
    ArrayModel(Dense(128, 1))
    )
end

Flux.@functor SearchSpace

function Flux.testmode!(m::SearchSpace, mode = true)
    testmode!(m.ϕ)
    foreach(x -> testmode!(x, mode), m.aggs)
    foreach(x -> testmode!(x, mode), m.cs)
    testmode!(m.ρ.m)
    m
end

function (s::SearchSpace)(x::BagNode)
    ϕ, aggs, cs, ρ = s.ϕ, s.aggs, s.cs, s.ρ
    h₁ = ϕ[1](x.data)
    h₂ = ϕ[2](h₁)
    h₃ = ϕ[3](h₂)
    (a₁, a₂, a₃) = aggs[1](h₁, x.bags), aggs[2](h₂, x.bags), aggs[3](h₃, x.bags)
    z = ArrayNode(cs[1](a₁).data + cs[2](a₂).data + cs[3](a₃).data, a₁.metadata)
    ρ(z)
end

function Base.show(io::IO, s::SearchSpace)
    print(io, "SearchSpace(\n")
    for (i, f) in enumerate(s.ϕ)
        print(io, "ϕ[$i] = ")
        print(io, f)
        print(io, "\n")
    end
    for (i, f) in enumerate(s.aggs)
        print(io, "agg[$i] = ")
        print(io, f)
        print(io, "\n")
    end
    for (i, c) in enumerate(s.cs)
        print(io, "c[$i] = ")
        print(io, c)
        print(io, "\n")
    end
    print(io, ")\n")
end

function train_model!(model, trainset; batchsize=1, epochs=10)
    Flux.trainmode!(model)
    (x, y) = trainset
    ps = params(model)
    loss(x, y) = bcentropyloss(model(x).data, y) + 0.0005*l2regularization(ps)
    optimizer = Optimiser(InvDecay(1e-4), Nesterov(5e-4, 0.9))
    for i = 1:epochs
        data = bagbatches(x, y; batchsize = batchsize)
        Flux.train!(loss, ps, data, optimizer)
    end
end

function parse_args()
    s = ArgParseSettings()

    @add_arg_table! s begin
        "--config"
            help = "path to the experiment configuration file"
            default = "configs/musk.yaml"
    end

    return parse_args(s)
end

function main()
    args = parse_args()
    config = YAML.load_file(args["config"])
    Random.seed!(config["seed"])
    (x, y) = load_data()
    K = config["kfolds"]
    train_folds, val_folds = kfolds(length(y), K)
    accs = zeros(K)
    @info("Baseline")
    for (i, (trn_idx, val_idx)) in enumerate(zip(train_folds, val_folds))
        model = make_baseline()
        trainset = (x[trn_idx], y[trn_idx])
        testset = (x[val_idx], y[val_idx])
        train_model!(model, trainset)
        accs[i] = evaluate(model, testset, accuracy; modelcall = (m, x) -> m(x).data[1, :])
        @info("Fold $(i)/$(K): accuracy = $(accs[i])")
    end
    @info("Baseline accuracy $(mean(accs)) +- $(std(accs)) std")
    @info("Search Methods")
    nsamples = config["nsamples"]
    @info("DARTS")
    dartsconf = config["search"]["darts"]
    epochs = dartsconf["epochs"]
    α_reg = dartsconf["regularization"]
    darts = from_config(DARTSearch, config)
    accs = zeros(K)
    elapsed = zeros(K)
    for (i, (trn_idx, val_idx)) in enumerate(zip(train_folds, val_folds))
        trainset = (x[trn_idx], y[trn_idx])
        testset = (x[val_idx], y[val_idx])
        searchspace = SearchSpace(StatefulSoftmax)
        ps = params(searchspace)
        loss(x, y) = bcentropyloss(searchspace(x).data, y) + α_reg * l2regularization(ps)
        start_time = time_ns()
        for j = 1:epochs
            dtrn_idx, dval_idx = splitobs(length(trn_idx); at = 0.8)
            data = (bagbatches(map(d -> d[dtrn_idx], trainset)...),
                    bagbatches(map(d -> d[dval_idx], trainset)...))
            optimize!(darts, searchspace, loss, data)
        end
        evaluator(model) = (train_model!(model, trainset; epochs=10);
                            evaluate(model, testset, accuracy; modelcall = (m, x) -> m(x).data[1, :]))
        (accs[i], arch) = evaluate_samples(searchspace, nsamples, evaluator)
        print(arch)
        @info("$(i)/$(K): accuracy = $(accs[i])")
    end

    @info("DARTS accuracy $(mean(accs)) +- $(std(accs)) std")
    @info("DARTS elapsed time $(median(elapsed)) s")
    @info("SNAS")
    snasconf = config["search"]["snas"]
    epochs = snasconf["epochs"]
    α_reg = snasconf["regularization"]
    snas = from_config(SNASearch, config)
    accs = zeros(K)
    elapsed = zeros(K)
    for (i, (trn_idx, val_idx)) in enumerate(zip(train_folds, val_folds))
        trainset = (x[trn_idx], y[trn_idx])
        testset = (x[val_idx], y[val_idx])
        searchspace = SearchSpace(GumbelSoftmax)
        ps = params(searchspace)
        loss(x, y) = bcentropyloss(searchspace(x).data, y) + α_reg * l2regularization(ps)
        dataloader() = bagbatches(trainset...)
        start_time = time_ns()
        optimize!(snas, searchspace, loss, dataloader)
        end_time = time_ns()
        elapsed[i] = (end_time - start_time)*1e-9
        evaluator(model) = (train_model!(model, trainset; epochs=10);
                            evaluate(model, testset, accuracy; modelcall = (m, x) -> m(x).data[1, :]))
        (accs[i], arch) = evaluate_samples(searchspace, nsamples, evaluator)
        print(arch)
        @info("$(i)/$(K): accuracy = $(accs[i])")
    end
    end_time = time_ns()
    elapsed = (end_time - start_time)*1e-9
    @info("SNAS accuracy $(mean(accs)) +- $(std(accs)) std")
    @info("SNAS elapsed time $(median(elapsed)) s")
    @info("ENAS")
    enasconf = config["search"]["enas"]
    α_reg = snasconf["regularization"]
    enas = from_config(ENASearch, config, SearchSpace(nothing))
    accs = zeros(K)
    elapsed = zeros(K)
    for (i, (trn_idx, val_idx)) in enumerate(zip(train_folds, val_folds))
        trainset = (x[trn_idx], y[trn_idx])
        testset = (x[val_idx], y[val_idx])
        searchspace = SearchSpace(nothing)
        dtrn_idx, dval_idx = splitobs(length(trn_idx); at = 0.8)
        dtrainset = (trainset[1][dtrn_idx], trainset[2][dtrn_idx])
        valset = (trainset[1][dval_idx], trainset[2][dval_idx])
        evaluator(model) = (train_model!(model, trainset; epochs=1);
                            evaluate(model, testset, accuracy; modelcall = (m, x) -> m(x).data[1, :]))
        start_time = time_ns()
        optimize!(enas, searchspace, evaluator)
        end_time = time_ns()
        elapsed[i] = (end_time - start_time)*1e-9
        evaluator(model) = (train_model!(model, trainset; epochs=10);
                            evaluate(model, testset, accuracy; modelcall = (m, x) -> m(x).data[1, :]))
        (accs[i], arch) = evaluate_samples(searchspace, nsamples, evaluator)
        print(arch)
        @info("$(i)/$(K): accuracy = $(accs[i])")
    end

    @info("ENAS accuracy $(mean(accs)) +- $(std(accs)) std")
    @info("ENAS elapsed time $(median(elapsed)) s")
    @info("Random search with weight sharing")
    random_search = from_config(RandomSearch, config)
    random_search = RandomSearch(rswsconf["n"], rswsconf["k"])
    accs = zeros(K)
    elapsed = zeros(K)
    for (i, (trn_idx, val_idx)) in enumerate(zip(train_folds, val_folds))
        trainset = (x[trn_idx], y[trn_idx])
        testset = (x[val_idx], y[val_idx])
        searchspace = SearchSpace(nothing)
        evaluator(model) = (train_model!(model, trainset; epochs=1);
                        evaluate(model, testset, accuracy; modelcall = (m, x) -> m(x).data[1, :]))
        start_time = time_ns()
        losses, models = optimize!(random_search, searchspace, evaluator, nothing)
        end_time = time_ns()
        elapsed[i] = (end_time - start_time)*1e-9
        evaluator(model) = (train_model!(model, trainset; epochs=10);
                            evaluate(model, testset, accuracy; modelcall = (m, x) -> m(x).data[1, :]))
        arch = archfix(searchspace, models[1])
        accs[i] = evaluator(arch)
        print(arch)
        @info("$(i)/$(K): accuracy = $(accs[i])")
    end
    @info("Random search with weight sharing accuracy $(mean(accs)) +- $(std(accs)) std")
    @info("Random search with weight sharing elapsed time $(median(elapsed)) s")
    @info("Random search with weight sharing")
    random_search = from_config(RandomSearch, config)
    random_search = RandomSearch(rswsconf["n"], rswsconf["k"])
    accs = zeros(K)
    elapsed = zeros(K)
    for (i, (trn_idx, val_idx)) in enumerate(zip(train_folds, val_folds))
        trainset = (x[trn_idx], y[trn_idx])
        testset = (x[val_idx], y[val_idx])
        searchspace = SearchSpace(nothing)
        evaluator(model) = (train_model!(model, trainset; epochs=10);
                            evaluate(model, testset, accuracy; modelcall = (m, x) -> m(x).data[1, :]))
        start_time = time_ns()
        losses, models = optimize(random_search, searchspace, evaluator, nothing)
        end_time = time_ns()
        elapsed[i] = (end_time - start_time)*1e-9
        accs[i] = losses[1]
        print(archfix(searchspace, models[1]))
        @info("$(i)/$(K): accuracy = $(accs[i])")
    end
    @info("Random search with weight sharing accuracy $(mean(accs)) +- $(std(accs)) std")
    @info("Random search with weight sharing elapsed time $(median(elapsed)) s")
end

main()
