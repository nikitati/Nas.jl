using Nas
using Flux
using Test

@testset "Nas.jl" begin
    phidden = Hyperparameter(ParameterDomain([80, 100, 120]))
    pactiv = Hyperparameter(ParameterDomain([sigmoid, relu]))
    template = TemplateDense(phidden, pactiv, DependentParameter([], () -> 200))
    @test phidden in hyperparams(template)
    @test pactiv in hyperparams(template)
    @test getvalue(phidden) == nothing
    @test getvalue(pactiv) == nothing
    assign!(phidden, 100)
    @test getvalue(phidden) == 100
    assign!(pactiv, relu)
    dense = compile(template)
    @test size(dense.W) == (100, 200)
    @test dense.σ == relu
    template_2 = TemplateDense(ParameterDomain([20, 40, 50]),
                               ParameterDomain([sigmoid, relu]),
                               DependentParameter([phidden], getvalue))
    assign!(template_2.hidden, 20)
    assign!(template_2.activation, relu)
    dense_2 = compile(template_2)
    @test size(dense_2.W) == (20, 100)
    assign!(phidden, 120)
    @test size(compile(template_2).W) == (20, 120)
    ta = TemplateDense(Hyperparameter(ParameterDomain([20, 25, 30])),
                       Hyperparameter(ParameterDomain([sigmoid, relu])),
                       DependentParameter([], () -> 50))
    tb = TemplateDense(Hyperparameter(ParameterDomain([20, 25, 30])),
                       Hyperparameter(ParameterDomain([sigmoid, relu])),
                       ta)
    tc = TemplateDense(Hyperparameter(ParameterDomain([20, 25, 30])),
                       Hyperparameter(ParameterDomain([sigmoid, relu])),
                       tb)
    tchain = TemplateChain(ta, tb, tc)
    assign!(ta.hidden, 30)
    assign!(tb.hidden, 25)
    assign!(tc.hidden, 20)
    chain = compile(tchain)
    @test size(chain[1].W) == (30, 50)
    @test size(chain[2].W) == (25, 30)
    @test size(chain[3].W) == (20, 25)
end
