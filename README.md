# Nas.jl
Programmable Neural Architecture Search

The work is done as part of the Master's Thesis "Architecture Optimization for Multiple Instance Learning Neural Networks". The repo is currently under refactoring and improvement process and the code may be unstable.

[![Build Status](https://travis-ci.com/nikitati/Nas.jl.svg?branch=master)](https://travis-ci.com/nikitati/Nas.jl)


## Neural Architecture Search library

This library contains a collection of state-of-the-art Neural Architecture Search algorithm implementations and a minimalistic interface to create flexible search spaces based on Flux.jl modules.

Currently implemented search algorithms:

* Random Search
* Random Search with Weight Sharing
* ENAS [[1](#cite1)]
* DARTS [[2](#cite2)]
* SNAS [[3](#cite3)]

## Example usage:

```julia
using Flux
using Nas

searchspace = Chain(
  Dense(28*28, 256, Flux.relu),
  ChoiceNode(GumbelSoftmax, [
    Dense(256, 128, Flux.relu),
    Dense(256, 128, Flux.tanh)
  ]),
)

snas = SNASearch(epochs=10, tmpdecay=0.01, opt=ADAM())
optimize!(snas, searchspace, loss, data)

model = sample_architecture(searchspace)
Flux.train!(model, ...)
```


## References

<a name="cite1"><b>1</b></a> *Pham, H., Guan, M.Y., Zoph, B., Le, Q.V. and Dean, J., 2018. Efficient neural architecture search via parameter sharing. arXiv preprint arXiv:1802.03268.*, https://arxiv.org/abs/1802.03268  
<a name="cite2"><b>2</b></a> *Liu, H., Simonyan, K. and Yang, Y., 2018. Darts: Differentiable architecture search. arXiv preprint arXiv:1806.09055.*, https://arxiv.org/abs/1806.09055  
<a name="cite3"><b>3</b></a> *Xie, S., Zheng, H., Liu, C. and Lin, L., 2018. SNAS: stochastic neural architecture search. arXiv preprint arXiv:1812.09926.*, https://arxiv.org/abs/1812.09926  
