
"""
    RandomSearch(n::Integer)

Random search with number of iterations `n`.
"""
struct RandomSearch
    n::Integer
end

function sample_onehot(k::Integer)
    a = zeros(k)
    a[rand(1:k)] = 1
    return a
end

function sample_uniform(searchspace::SearchSpace)
    [sample_onehot(length(choice)) for choice in choices(searchspace)]
end

function sample_uniform!(searchspace::SearchSpace)
    for choice in choices(searchspace)
        a = sample_onehot(length(choice))
        copyto!(choice.architecture, a)
    end
    return searchspace
end

function optimize!(strategy::RandomSearch, searchspace, evaluator, data; cb = () -> ())
    n = strategy.n
    cb = runall(cb)
    for i = 1:n
        sample_uniform!(searchspace)
        evaluator(searchspace)
        cb()
    end
end

struct DARTSearch{W, A, L}
    ξ
    w_opt::W
    α_opt::A
    loss::L
end

macro inner_step(inner_opt, w, loss, expr)
    decorated = quote
        ξ = $inner_opt.eta
        if ξ > 0
            w_gs = gradient($w) do
                $loss()
            end
            update!($inner_opt, $w, w_gs)
            $expr
            update!($inner_opt, $w, -w_gs)
        else
            $expr
        end
    end
    return decorated
end

function optimize!(strategy::DARTSearch, searchspace, evaluator, data; cb = () -> ())
    ξ, w_opt, α_opt = strategy.ξ, strategy.w_opt, strategy.α_opt
    loss = strategy.loss
    inner_opt = Descent(ξ)
    w, α = params(searchspace)
    cb = runall(cb)
    trainset, validset = data
    for (train, valid) in zip(trainset, validset)
        α_gs = gradient(α) do
            @inner_step inner_opt w evaluator loss(valid)
            return arch_loss
        end
        update!(α_opt, α, α_gs)
        w_gs = gradient(w) do
            loss(train)
        end
        update!(w_opt, w, w_gs)
    end
end

struct ENASearch
    controller
    controller_opt
    controller_step
    controller_batch
    child_opt
    child_loss
    child_step
end

function sample_probs(controller, searchspace::SearchSpace)
    c = s.controller
    choices = choices(searchspace)
    n = length(choices[0])
    x = zeros(n)
    reset!(c)
    probs = [zeros(n) for i in 1:(length(choices)+1)]
    log_probs = [zeros(n) for i in 1:(length(choices)+1)]
    for i = 2:(length(choices)+1)
        # assuming the last layer is a softmax
        x = c[1:end-1](probs[i-1])
        probs[i] = softmax(x)
        log_probs[i] = logsoftmax(x)
    end
    return probs[2:end], log_probs[2:end]
end

function set_architecture!(searchspace::SearchSpace, probs::AbstractArray)
    for (p, choice) in zip(probs, searchspace)
        k = sample(1:length(choice), ProbabilityWeights(p))
        a = zeros(length(choice))
        a[k] = 1
        copyto!(choice.architecture, a)
    end
    return searchspace
end

function sample_controller!(controller, searchspace::SearchSpace)
    choices = choices(searchspace)
    n = length(choices[0])
    p = zeros(n)
    reset!(controller)
    for choice in choices
        p = controller(p)
        i = sample(1:length(choice), ProbabilityWeights(p))
        a[i] = 1
        copyto!(choice.architecture, a)
    end
    return searchspace
end

function train_controller!(strategy::ENASearch, searchspace, evaluator, data)
    ps = params(strategy.controller)
    for d in data
        gs = gradient(ps) do
            probs, log_probs = sample_probs(controller, searchspace)
            set!(searchspace, probs)
            reward = evaluator(data)
            training_loss = -sum(log_probs) * reward
            return training_loss
        end
        update!(strategy.controller_opt, ps, gs)
    end
end

function optimize!(strategy::ENASearch, searchspace, evaluator, data; cb = () -> ())
    controller, controller_opt = strategy.controller, strategy.controller_opt
    child_opt, child_loss = strategy.child_opt, strategy.child_loss
    controller_step, child_step = strategy.controller_step, strategy.child_step
    controller_batch = strategy.controller_batch
    w, _ = params(searchspace)
    trainset, validset = data
    for i = 1:child_step
        sample_controller!(controller, searchspace)
        train!(child_loss, w, trainset, child_opt)
    end
    for i = 1:controller_step
        train_controller(controller, searchspace, evaluator, validset)
    end
end

struct SNASearch{T}
    joint_opt::T
end

function optimize!(strategy::SNASearch, searchspace, evaluator, data; cb = () -> ())
    opt = strategy.joint_opt
    ps, α = params(searchspace)
    push!(ps, α...)
    Flux.train!(evaluator, ps, data, opt, cb = cb)
end
