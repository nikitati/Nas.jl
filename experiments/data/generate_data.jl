using Random
using JLD2
using FileIO
using Distributions: Multinomial
using Flux.Data.MNIST
using MLDataUtils

function sine(out_path, n=100, σ=0.1, split=0.7)
    rng = Random.seed!(13)
    x = range(-pi, pi, length=n)
    y = sin.(x) + σ * randn(rng, n)
    x, y = getobs(shuffleobs((x, y)))
    train_idx, test_idx = splitobs(n, at = split)
    names = Dict("x" => x, "y" => y,
                 "train_idx" => train_idx, "test_idx" => test_idx)
    save("$out_path/synthetic/sine.jld2", names)
end

function freidman(out_path, n=500, σ=0.1, split=0.7)
    rng = Random.seed!(13)
    x = randn(rng, 10, n)
    y = 10 .* sin.(pi .* x[1, :] .* x[2, :]) .+
     .+ 20 .* (x[3, :] .- 0.5).^2 .+
     .+ 10 .* x[4, :] .+ 5 .* x[5, :] .+
     .+ σ .* randn(rng, n)
    x, y = getobs(shuffleobs((x, y)))
    train_idx, test_idx = splitobs(n, at = split)
    names = Dict("x" => x, "y" => y,
                 "train_idx" => train_idx, "test_idx" => test_idx)
    save("$out_path/synthetic/freidman.jld2", names)
end

function cricles(out_path, n=500, p=0.09, split=0.7)
    rng = Random.seed!(13)
    in_circle(x, y; r=1) = x^2 + y^2 <= r^2
    x = randn(rng, 2, n)
    y = in_circle.(x[1, :], x[2, :])
    y = [rand() > p ? y_i : !y_i for y_i in y]
    x, y = getobs(shuffleobs((x, y)))
    train_idx, test_idx = splitobs(n, at = split)
    names = Dict("x" => x, "y" => y,
                 "train_idx" => train_idx, "test_idx" => test_idx)
    save("$out_path/synthetic/circles.jld2", names)
end

function musk2(raw_data, out_path)
    zeros(3, 3)
end

function digits_text(out_path, n=100, max_train=10, max_test=100)
    rng = Random.seed!(13)
    function sample(maxn, k)
        n = rand(1:maxn)
        rand(Multinomial(n, k))
    end
    function to_mill(x)
        nobs = size(x, 2)
        b = vcat([repeat([j-1 i], k, 1) for i = 1:nobs for (j, k) in enumerate(x[:, i])]...)
        return b[:, 1], b[:, 2]
    end
    train_x = hcat([sample(max_train, 10) for i = 1:n]...)
    test_x = hcat([sample(max_test, 10) for i = 1:n]...)
    save("$out_path/digits_text.jld2", Dict("train_x" => train_x, "test_x" => test_x))
    train_features, train_bagids = to_mill(train_x)
    test_features, test_bagids = to_mill(test_x)
    names = Dict("train_features" => train_features,
                 "train_bagids" => train_bagids,
                 "test_features" => test_features,
                 "test_bagids" => test_bagids)
    save("$out_path/mnist/digits_text_mill.jld2", names)
end

function digits_mnist(out_path, train_n=10000, test_n=5000, max_train=10, max_test=50)
    rng = Random.seed!(13)
    function to_mill(y, bag_size, n_bags)
        nsamples = size(y, 1)
        bags = [rand(1:nsamples, rand(1:bag_size)) for i = 1:n_bags]
        labels = [sum(y[b]) for b in bags]
        tmp = vcat([[b repeat([i], (size(b, 1)))] for (i, b) in enumerate(bags)]...)
        feature_idx = tmp[:, 1]
        bagids = tmp[:, 2]
        return feature_idx, bagids, labels
    end
    y = MNIST.labels()
    y_test = MNIST.labels(:test)
    feature_idx_train, bagids_train, labels_train = to_mill(y, max_train, train_n)
    feature_idx_test, bagids_test, labels_test = to_mill(y, max_test, test_n)
    names = Dict("feature_idx_train" => feature_idx_train,
                 "bagids_train" => bagids_train, "labels_train" => labels_train,
                 "feature_idx_test" => feature_idx_test,
                 "bagids_test" => bagids_test, "labels_test" => labels_test,
                 )
    save("$out_path/mnist/digits_mnist_mill.jld2", names)
end

function iot(raw_path, out_path)
    @info "Not implemented"
end

function modelnet(raw_path, out_path)
    @info "Not implemented"
end

function main(data_dir)
    sine(data_dir)
    freidman(data_dir)
    cricles(data_dir)
    digits_text(data_dir)
    digits_mnist(data_dir)
    musk2("data/musk/raw", data_dir)
    modelnet("data/modelnet/raw", data_dir)
    iot("data/iot/raw", data_dir)
end

main("data")
