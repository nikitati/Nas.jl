
"""
    RandomSearch(n::Integer)

Random search with number of iterations `n`.
"""
struct RandomSearch
    n::Integer
end

function samplearchitecture()
    nothing
end

function optimize!(rs::RandomSearch, evalcb::Function)
    n = rs.n
    for i = 1:n
        samplearchitecture()
    end
end
