using Test, Nas, Flux

@testset "ChoiceNode" begin
    c = ChoiceNode(
        Chain(Dense(10, 100), Dense(100, 2)),
        Chain(Dense(10, 150), Dense(150, 2)),
        Chain(Dense(10, 200), Dense(200, 2))
    )
    x = randn(10, 500)
    y = c(x)
    @test size(y[1]) == (2, 500)
    z = [0, 1, 0]
    y = c(x, z)
    @test size(y) == (2, 500)
    @test isequal(c[2](x), y)
end

@testset "Gumbel" begin
    s = GumbelSoftmax([0.2, 0.3, 0.5], 1.0)
    g = gumbelrand(3)
    z = s(g)
    @test size(z) == (3,)
    @test all(0<= z_i <= 1 for z_i in z)
    @test isequal(sum(z), 1)
    c = ChoiceNode(
        Chain(Dense(10, 100), Dense(100, 2)),
        Chain(Dense(10, 150), Dense(150, 2)),
        Chain(Dense(10, 200), Dense(200, 2))
    )
    x = randn(10, 500)
    y = c(x, z)
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
        Chain(Dense(64, 150), Dense(150, 10)),
    )
    α = [[0, 1, 0], [0, 2]]
    x = randn(128, 500)
    model = Chain(applicative(c1, α[1]), applicative(c2, α[2]))
    y = model(x)
    @test size(y) == (10, 500)
    α[1] = [0, 0, 1]
    y = model(x)
    @test size(y) == (10, 500)
    gs = [GumbelSoftmax([0.2, 0.3, 0.5], 1.0), GumbelSoftmax([0.2, 0.8], 1.0)]
    model = Chain(applicative(c1, gs[1]), applicative(c2, gs[2]))
    y = model(x)
    @test size(y) == (10, 500)
    ss = [StatefulSoftmax([0.2, 0.3, 0.5]), StatefulSoftmax([0.2, 0.8])]
    model = Chain(applicative(c1, ss[1]), applicative(c2, ss[2]))
    y = model(x)
    @test size(y) == (10, 500)
end

@testset "Gradients" begin
    c1 = ChoiceNode(
        Chain(Dense(128, 300), Dense(300, 64)),
        Chain(Dense(128, 400), Dense(400, 64)),
        Chain(Dense(128, 500), Dense(500, 64))
    )
    c2 = ChoiceNode(
        Chain(Dense(64, 100), Dense(100, 10)),
        Chain(Dense(64, 150), Dense(150, 10)),
    )
end
