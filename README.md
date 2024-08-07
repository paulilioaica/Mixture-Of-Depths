﻿# Mixture-Of-Depths

![Mixture of Depths](https://miro.medium.com/v2/resize:fit:1010/0*Gp38u4wBSdVe4sMz.png)
[Paper link](https://arxiv.org/html/2404.02258v1)

This repository contains an implementation of the **Mixture of Depths** paradigm with a standard transformer decoder in PyTorch.

## Introduction

Transformer-based language models distribute computation uniformly across input sequences. However, not all tokens require the same amount of computation for accurate predictions. This work explores dynamically allocating computation to specific positions in a sequence, optimizing allocation across layers in the model depth2.

There is a key parameter  in this concept:
- `top_k` which is the top k tokens (e.g. 128 out of 256) determined by fit (given by the gating/routing mechanism) which selects only important tokens which require more compute as you get deeper into the layers.

Essentially, the proposed method enforces a fixed compute budget by capping the number of tokens that participate in self-attention and MLP computations at each layer3. Tokens are routed to either full computation (standard transformer block) or a residual connection (saving compute).


To run the sparse Mixture of Experts model, follow the same steps as for the sequential model mentioned above.

[Find more about this in the original paper](https://arxiv.org/html/2404.02258v1)
