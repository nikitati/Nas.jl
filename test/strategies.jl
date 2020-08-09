Random.seed!(13)

@testset "Random" begin
    searchspace = ChoiceNode([x -> sin.(x), x -> x .^ 3, x -> 3 .* x])
    strategy = RandomSearch(100, 3)
    x = range(0,stop=2*pi,length=100)
    y = sin.(x)
    data = zip(x, y)
    evaluator(model) = -Flux.mse(model(x), y)
    losses, models = optimize(strategy, searchspace, evaluator, data)
    @test isapprox(evaluator(models[1]), 0)
    losses, archs = optimize!(strategy, searchspace, evaluator, data)
    # fixed = archfix(searchspace, archs[1])
    @test isapprox(evaluator(archs[1]), 0)
end

@testset "DARTS" begin
    searchspace = ChoiceNode(StatefulSoftmax, [x -> sin.(x), x -> x .^ 3, x -> 3 .* x])
    x = collect(range(0,stop=2*pi,length=100))
    y = sin.(x)
    loss(x) = Flux.mse(searchspace(x[1]), y[2])
    strategy = DARTSearch(0.0, Descent(0.01), Descent(0.01))
    data = ([x y], [x y])
    optimize!(strategy, searchspace, loss, data)
    @test argmax(searchspace.architecture()) == 1
end

@testset "SNAS" begin
    searchspace = ChoiceNode(GumbelSoftmax, [x -> sin.(x), x -> x .^ 3, x -> 3 .* x])
    strategy = SNASearch(Descent(0.01))
    x = range(0,stop=2*pi,length=100)
    y = sin.(x)
    loss(x, y) = Flux.mse(searchspace(x), y)
    data = zip(x, y)
    optimize!(strategy, searchspace, loss, data)
    @test argmax(searchspace.architecture()) == 1
end

@testset "ENAS" begin
    true_fun = x -> sin.(x) .^ 2
    searchspace = Chain(
        ChoiceNode([x -> x .^ 3, x -> sin.(x), x -> 3 .* x]),
        ChoiceNode([x -> x .^ 2, x -> 0.5 .* x, x -> -1.0 .* x])
    )
    strategy = ENASearch(searchspace, ADAM(), 500, 100)
    x = range(0,stop=2*pi,length=100)
    y = true_fun(x)
    data = zip(x, y)
    evaluator(m) = -Flux.mse(m(x), y)
    _, a, _ = Nas.sample_controller(strategy.controller, choices(searchspace))
    # @info "Pre-training architecture probabilities = $(a)"
    optimize!(strategy, searchspace, evaluator, data)
    logps, ps, actions = Nas.sample_controller(strategy.controller, choices(searchspace))
    # @info "Sampled probabilities = $(ps)"
    # @info "Sampled actions = $(actions)"
    @test actions[1] == 2
    @test actions[2] == 1
    set_architecture!(searchspace, IdDict(zip(choices(searchspace), actions)))
    @test isapprox(evaluator(searchspace), 0.0)
end
