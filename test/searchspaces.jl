using Test, Nas, Flux

@testset "Hyperparameters" begin
    phidden = Hyperparameter(Domain([80, 100, 120]))
    pactiv = Hyperparameter(Domain([sigmoid, relu]))
    pinit = DependentParameter([], () -> 200)
    pdep = DependentParameter([phidden], getvalue)
    @test getvalue(phidden) == nothing
    @test getvalue(pactiv) == nothing
    assign!(phidden, 100)
    @test getvalue(phidden) == 100
    @test getvalue(pdep) == 100
    assign!(pactiv, relu)
    @test getvalue(pactiv) == relu
    @test getvalue(pinit) == 200
end

@testset "Nodes" begin
    sdense = Dense(50, 100)
    choices = [Dense(100, 200), Dense(100, 300), Dense(100, 400)]
    choiceout = DependentParameter()
    cdense = ChoiceNode(choices)
    i = 2
    x = randn(50, 100)
    f = Chain(sdense, cdense, ys -> ys[i])
    y = choices[i](sdense(x))
    @test isequal(f(x), y)
end
