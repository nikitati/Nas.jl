@testset "ChoiceNode" begin
    c = ChoiceNode(
        Chain(Dense(10, 100), Dense(100, 2)),
        Chain(Dense(10, 150), Dense(150, 2)),
        Chain(Dense(10, 200), Dense(200, 2))
    )
    @test length(c) == 3
    x = randn(10, 500)
    set_choice!(c, 2)
    y = c(x)
    @test size(y) == (2, 500)
    @test isequal(c[2](x), y)
    ps = archparams(c)
    @test isempty(ps)
    c = ChoiceNode(StatefulSoftmax, [
        Chain(Dense(10, 100), Dense(100, 2)),
        Chain(Dense(10, 150), Dense(150, 2)),
        Chain(Dense(10, 200), Dense(200, 2))
        ]
    )
    y = c(x)
    @test size(y) == (2, 500)
    ps = archparams(c)
    @test !isempty(ps)
    c = SPChoiceNode(STGumbelSoftmax, [
        Chain(Dense(10, 100), Dense(100, 2)),
        Chain(Dense(10, 150), Dense(150, 2)),
        Chain(Dense(10, 200), Dense(200, 2))
    ])
    @test length(c) == 3
    x = randn(10, 500)
    y = c(x)
    @test size(y) == (2, 500)
end

@testset "Searchspace" begin
    c1 = ChoiceNode(
        Chain(Dense(128, 300), Dense(300, 64)),
        Chain(Dense(128, 400), Dense(400, 64)),
        Chain(Dense(128, 500), Dense(500, 64))
    )
    c2 = ChoiceNode(
        Chain(Dense(64, 100), Dense(100, 10)),
        Chain(Dense(64, 150), Dense(150, 10))
    )
    model = Chain(c1, c2)
    cs = choices(model)
    @test length(cs) == 2
    architecture = IdDict(cs[1] => 2, cs[2] => 1)
    set_architecture!(model, architecture)
    x = randn(128, 500)
    y = model(x)
    @test size(y) == (10, 500)
    @test isequal(c2[1](c1[2](x)), y)
    archps = archparams(model)
    @test isempty(archps)
    set_random_architecture!(model)
    @test size(model(x)) == (10, 500)
    fixed = archfix(model, architecture)
    expected_fixed = Chain(c1[2], c2[1])
    @test typeof(fixed) == typeof(expected_fixed)
    x = randn(128, 500)
    @test isequal(fixed(x), expected_fixed(x))
    ps = Flux.params(fixed)
    expected_ps = Flux.params(expected_fixed)
    @test length(ps) == length(expected_ps)
    c1 = ChoiceNode(GumbelSoftmax, [
        Chain(Dense(128, 300), Dense(300, 64)),
        Chain(Dense(128, 400), Dense(400, 64)),
        Chain(Dense(128, 500), Dense(500, 64))
        ],
    )
    c2 = ChoiceNode(GumbelSoftmax, [
        Chain(Dense(64, 100), Dense(100, 10)),
        Chain(Dense(64, 150), Dense(150, 10))
        ],
    )
    model = Chain(c1, c2)
    y = model(x)
    @test size(y) == (10, 500)
    archps = archparams(model)
    @test !isempty(archps)
    fixed = sample_architecture(model)
    @test size(fixed(x)) == (10, 500)
end
