@testset "ChoiceNode" begin
    c = ChoiceNode(
        Chain(Dense(10, 100), Dense(100, 2)),
        Chain(Dense(10, 150), Dense(150, 2)),
        Chain(Dense(10, 200), Dense(200, 2))
    )
    @test length(c) == 3
    x = randn(10, 500)
    z = [0, 1, 0]
    y = c(x, z)
    @test size(y) == (2, 500)
    @test isequal(c[2](x), y)
    ps = Nas.architecture_params(c)
    @test isempty(ps)
    c = ChoiceNode([
        Chain(Dense(10, 100), Dense(100, 2)),
        Chain(Dense(10, 150), Dense(150, 2)),
        Chain(Dense(10, 200), Dense(200, 2))
        ],
        z
    )
    y = c(x)
    @test size(y) == (2, 500)
    @test isequal(c[2](x), y)
    ps = Nas.architecture_params(c)
    @test isempty(ps)
    copyto!(z, [1, 0, 0])
    y = c(x)
    @test size(y) == (2, 500)
    @test isequal(c[1](x), y)
    c = ChoiceNode([
        Chain(Dense(10, 100), Dense(100, 2)),
        Chain(Dense(10, 150), Dense(150, 2)),
        Chain(Dense(10, 200), Dense(200, 2))
        ],
        StatefulSoftmax(3)
    )
    y = c(x)
    @test size(y) == (2, 500)
    ps = Nas.architecture_params(c)
    @test !isempty(ps)
end

@testset "Searchspace" begin
    α = [[0, 1, 0], [0, 2]]
    c1 = ChoiceNode([
        Chain(Dense(128, 300), Dense(300, 64)),
        Chain(Dense(128, 400), Dense(400, 64)),
        Chain(Dense(128, 500), Dense(500, 64))
        ],
        α[1]
    )
    c2 = ChoiceNode([
        Chain(Dense(64, 100), Dense(100, 10)),
        Chain(Dense(64, 150), Dense(150, 10))
        ],
        α[2]
    )
    model = Chain(c1, c2)
    x = randn(128, 500)
    @test size(model(x)) == (10, 500)
    space = SearchSpace(model)
    cs = choices(space)
    @test length(cs) == 2
    ps, archps = Nas.params(space)
    @test isempty(archps)
    c1 = ChoiceNode([
        Chain(Dense(128, 300), Dense(300, 64)),
        Chain(Dense(128, 400), Dense(400, 64)),
        Chain(Dense(128, 500), Dense(500, 64))
        ],
        StatefulSoftmax(3)
    )
    c2 = ChoiceNode([
        Chain(Dense(64, 100), Dense(100, 10)),
        Chain(Dense(64, 150), Dense(150, 10))
        ],
        StatefulSoftmax(2)
    )
    model = Chain(c1, c2)
    y = model(x)
    @test size(y) == (10, 500)
    space = SearchSpace(model)
    ps, archps = Nas.params(space)
    @test !isempty(archps)
end
