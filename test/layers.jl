@testset "Gumbel" begin
    s = GumbelSoftmax(3)
    z = s()
    @test size(z) == (3,)
    @test all(0 <= z_i <= 1 for z_i in z)
    @test isapprox(sum(z), 1.0)
    s = GumbelSoftmax(1000)
    z = s()
    @test size(z) == (1000,)
    @test all(0 <= z_i <= 1 for z_i in z)
    @test isapprox(sum(z), 1.0)
    st = STGumbelSoftmax(5)
    z = st()
    @test size(z) == (5,)
    @test sum(z .== ones(size(z))) == 1
end

@testset "Softmax" begin
    s = StatefulSoftmax(3)
    z = s()
    @test size(z) == (3,)
    @test all(0 <= z_i <= 1 for z_i in z)
    @test isapprox(sum(z), 1.0)
end

@testset "Embedding" begin
    emb = Nas.EmbeddingLayer(3, 5)
    @test size(emb([3]), 1) == 3
end
