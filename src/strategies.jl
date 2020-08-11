
"""
    RandomSearch(n::Integer, k::Integer)

Random search with number of iterations `n` which keeps track of `k` best
parameter combinations.
"""
struct RandomSearch
    n::Integer
    accumulator::Accumulator
    RandomSearch(n::Integer, k::Integer) = new(n, Accumulator(k))
end

function optimize(strategy::RandomSearch, searchspace, evaluator, data = nothing; cb = () -> ())
    n = strategy.n
    accum = strategy.accumulator
    cb = runall(cb)
    for i = 1:n
        model = sample_architecture(searchspace)
        performance = evaluator(model)
        add!(accum, performance, model)
        cb()
    end
    return toarray(accum)
end

function optimize!(strategy::RandomSearch, searchspace, evaluator, data = nothing; cb = () -> ())
    n = strategy.n
    accum = strategy.accumulator
    choice_nodes = choices(searchspace)
    probabilities = get_probs(choice_nodes)
    cb = runall(cb)
    for i = 1:n
        set_random_architecture!(searchspace, probabilities)
        performance = evaluator(searchspace)
        curr_choice = IdDict(c => getchoice(c) for c in choice_nodes)
        add!(accum, performance, curr_choice)
        cb()
    end
    return toarray(accum)
end

struct DARTSearch
    ξ
    w_opt
    α_opt
end

function optimize!(strategy::DARTSearch, searchspace, evaluator, data; cb = () -> ())
    ξ, w_opt, α_opt = strategy.ξ, strategy.w_opt, strategy.α_opt
    loss = evaluator
    inner_opt = Descent(ξ)
    w, α = params(searchspace), archparams(searchspace)
    cb = runall(cb)
    train, valid = data
    for (train_batch, valid_batch) in zip(train, valid)
        if ξ > 0
            α_gs = gradient(α) do
                w_gs = gradient(w) do
                    loss(train)
                end
                Flux.update!(inner_opt, w, w_gs)
                arch_loss = loss(valid)
                Flux.update!(inner_opt, w, -w_gs)
                arch_loss
            end
            Flux.update!(α_opt, α, α_gs)
        else
            Flux.train!(loss, α, [valid_batch], α_opt)
        end
        Flux.train!(loss, w, [train_batch], w_opt)
        cb()
    end
end

struct SNASearch{T}
    epochs::Integer
    joint_opt::T
end

function apply!(d::InvDecay, gs::S) where {S <: Union{STGumbelSoftmax, GumbelSoftmax}}
    γ = d.gamma
    n = get!(d.state, gs, 1)
    gs.t *= 1 / (1 + γ * n)
    d.state[gs] = n + 1
end

function optimize!(strategy::SNASearch, searchspace, evaluator, dataloader; cb = () -> ())
    epochs, opt = strategy.epochs, strategy.joint_opt
    ps, α = params(searchspace), archparams(searchspace)
    choicenodes = choices(searchspace)
    decay = InvDecay()
    push!(ps, α...)
    for e = 1:epochs
        data = dataloader()
        Flux.train!(evaluator, ps, data, opt, cb = cb)
        foreach(c -> apply!(decay, c.architecture), choicenodes)
    end
end


mutable struct Controller
    embedding
    recurrent
    decoders
    step
end

function Controller(searchspace, hidden_dim)
    tokens = [length(choice) for choice in choices(searchspace)]
    vs = sum(tokens)
    decoders = tuple((Dense(hidden_dim, t) for t in tokens)...)
    Controller(
        EmbeddingLayer(hidden_dim, vs),
        GRU(hidden_dim, hidden_dim),
        decoders,
        1
    )
end

Controller(embedding, recurrent, decoders) = Controller(embedding, recurrent, decoders, 1)

Flux.@functor Controller

Flux.trainable(c::Controller) = (c.embedding, c.recurrent, c.decoders...)

function (c::Controller)(x)
    h = c.recurrent(c.embedding(x))
    y = c.decoders[c.step](h)
    c.step += 1
    return y
end

function reset!(c::Controller)
    Flux.reset!(c.recurrent)
    c.step = 1
end

function sample_controller(controller, choices)
    reset!(controller)
    action = 0
    nchoices = length(choices)
    actionlogprobs = Zygote.Buffer(Array{Array{Float32, 1}, 1}(undef, nchoices))
    actionprobs = Zygote.Buffer(Array{Array{Float32, 1}, 1}(undef, nchoices))
    actions = Array{Integer, 1}(undef, nchoices)
    offset = 0
    for (i, choice) in enumerate(choices)
        logits = controller([action + offset])[:]
        logprobs = logsoftmax(logits)
        probs = softmax(logits)
        Zygote.ignore() do
            action = sample(ProbabilityWeights(probs))
            actions[i] = action
        end
        actionlogprobs[i] = logprobs
        actionprobs[i] = probs
        offset += length(choice)
    end
    copy(actionlogprobs), copy(actionprobs), actions
end

function controller_step!(controller, optim, searchspace, evaluator)
    controller_params = params(controller)
    choice_nodes = choices(searchspace)
    local R
    gs = gradient(controller_params) do
        actionlogprobs, actionprobs, actions = sample_controller(controller, choice_nodes)
        Zygote.ignore() do
            archchoices = IdDict(zip(choice_nodes, actions))
            set_architecture!(searchspace, archchoices)
            baseline = 0
            R = evaluator(searchspace) - baseline
        end
        loss = sum(-actionlogprobs[i][a] for (i, a) in enumerate(actions)) * R
        return loss
    end
    update!(optim, controller_params, gs)
end

struct ENASearch
    optimizer
    epochs
    controller
end

function ENASearch(searchspace, optimizer, epochs, controller_hidden)
    ENASearch(optimizer, epochs, Controller(searchspace, controller_hidden))
end

function optimize!(strategy::ENASearch, searchspace, evaluator, data = nothing; cb = () -> ())
    controller, epochs, opt = strategy.controller, strategy.epochs, strategy.optimizer
    cb = runall(cb)
    for i = 1:epochs
        controller_step!(controller, opt, searchspace, evaluator)
        cb()
    end
end
