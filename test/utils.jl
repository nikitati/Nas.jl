@testset "Accumulator" begin
    a = Nas.Accumulator(3)
    Nas.add!(a, 10.0, "a")
    Nas.add!(a, 5.5, "b")
    Nas.add!(a, 5.0, "c")
    @test a.values[a.minpriorityidx] == "c"
    Nas.add!(a, 7.0, "d")
    @test a.values[a.minpriorityidx] == "b"
    Nas.add!(a, 3.0, "x")
    @test a.values[a.minpriorityidx] == "b"
end
