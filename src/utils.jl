function gumbelrand(dims...)
    -log.(-log.(rand(dims...)))
end
