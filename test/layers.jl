@testset "Gumbel" begin
    s = GumbelSoftmax([0.2, 0.3, 0.5], 1.0)
    z = s()
    @test size(z) == (3,)
    @test all(0 <= z_i <= 1 for z_i in z)
    @test isapprox(sum(z), 1.0)
    c = ChoiceNode(
        Chain(Dense(10, 100), Dense(100, 2)),
        Chain(Dense(10, 150), Dense(150, 2)),
        Chain(Dense(10, 200), Dense(200, 2))
    )
    x = randn(10, 500)
    @test size(c(x, z)) == (2, 500)
end

@testset "Softmax" begin
    s = StatefulSoftmax(3)
    z = s()
    @test size(z) == (3,)
    @test all(0 <= z_i <= 1 for z_i in z)
    @test isapprox(sum(z), 1.0)
    c = ChoiceNode(
        Chain(Dense(10, 100), Dense(100, 2)),
        Chain(Dense(10, 150), Dense(150, 2)),
        Chain(Dense(10, 200), Dense(200, 2))
    )
    x = randn(10, 500)
    y = c(x, z)
    @test size(y) == (2, 500)
end
