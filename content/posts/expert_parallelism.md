---
author: [""]
title: "Ultrascale Playbook - Expert Parallelism"
date: 2025-12-13
summary: "Notes on training LLMs using expert parallelism"
description: ""
tags: ["llm", "llm-training", "expert-parallelism"]
series: ["Ultra-scale Playbook"]
ShowToc: true
ShowBreadCrumbs: true
math: true
---

To compute the next token, large language models (LLMs) typically use dense layers where all parameters are involved in the computation for every token. The idea of mixture of expert models (MoE) is to divide the model into multiple "experts" (sub-model) and only activate a subset of these experts for each input token. Expert parallelism refers to the way these experts are distributed across multiple GPUs and how tokens are routed between them.

The question arises: what are these experts and who determines which tokens go to which experts?

## Mixture of Experts (MoE)

The transformer architecture consists of multi-head attention (MHA), feed forward networks (FFNN) and normalization layers. Mixture of Experts architectures introduce experts and routers that choose suitable experts for each token to the transformer architecture. MoE architectures increase the model's capacity without increasing the computation required per token. This is achieved by introducing many experts but activating only a few for each token.  Because MoE activates only a small subset of experts for each token, the compute cost per token is much lower than in dense models. This makes it possible to scale model or dataset size under the same training budget, and MoE models usually reach dense-model quality in fewer training steps.

{{< figure align=center src="/images/moe.png" attr="[A Visual Guide to Mixture of Experts (MoE)](https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-mixture-of-experts)">}}

### Experts

Each expert typically contains its own FFNN layers. The FFNN layers in the transformer architecture are replaced by these experts, which can range from a small number of experts to hundreds or even thousands. Although this increases the total number of parameters, only a small number of experts are activated per token. As shown in the diagram below, we replace the dense FFNN layers with sparse experts, so not all experts are involved in the forward pass for a given token.

{{< figure align=center src="/images/experts.png" attr="[A Visual Guide to Mixture of Experts (MoE)](https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-mixture-of-experts)">}}

In the [Mixtral of Experts paper](https://arxiv.org/abs/2401.04088), the authors investigated whether there are specialized experts for specific domains such as physics, mathematics and coding. They observed that experts naturally learn to route similar types of tokens to the same subset of experts. For example, `self` keyword in Python and `Question` in English often get routed through the same expert.

### Gate network (or Router)

The gating network consists of a small MLP. The router takes a token representation as input and produces a probability distribution over experts. This probability distribution is used to select one or more experts to route the input token. There are different approaches such as top-1, where the expert with the highest probability is selected, or top-k, where the top k experts are chosen. The value of k is set to 1 or 2 in practice for many models to avoid the computation complexity of using all the experts.

{{< figure align=center src="/images/router.png" attr="[A Visual Guide to Mixture of Experts (MoE)](https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-mixture-of-experts)">}}

There can be two types of MoE: dense MoE and sparse MoE. In dense MoE, the router sends each token to all experts, whereas in the sparse MoE setting only a few experts are selected. Current LLMs typically use sparse MoE because it is computationally cheaper.

{{< figure align=center src="/images/dense_sparse_moe.png" attr="[A Visual Guide to Mixture of Experts (MoE)](https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-mixture-of-experts)">}}

In many models the MoE layers are mixed with standard dense transformer blocks rather than replacing every FFNN layer. Some architectures also include shared experts where the router can send tokens that do not fit well into any specialized expert. Both the experts and the gating network (or routers) are trained jointly, where the gating algorithm selects an expert to forward the token to. There are several issues with MoE training particularly:

* **Router collapse**: where the same experts gets selected most of the time. There are various gating mechanisms proposed to address this problem. One approach proposed in [Gshard paper](https://arxiv.org/abs/2006.16668) is to use random top-2 routing where first expert is selected using highest probability and second is selected with probability proportional to its weight.
* **Balancing experts**, or **load balancing**, to ensure tokens are distributed fairly across experts so that no expert becomes overloaded or underutilized. Expert capacity is proposed in the [Switch Transformer paper](https://arxiv.org/abs/2101.03961) to address this. Each expert has a maximum number of tokens it can process per batch, referred to as expert capacity. Another approach is to use auxiliary loss to encourage giving all experts equal importance.
* **Communication overhead** which we will discuss below.

Together these issues make MoE training more complex than training dense models, and most recent work focuses on improving routing stability, reducing communication cost and keeping experts evenly utilized during training.

## Inference

Inference for MoE models works differently from inference in standard dense transformer architectures. In a dense transformer, every feed forward layer uses all of its parameters for every token. In an MoE architecture, only a small set of experts is activated for each token, so the compute cost depends on the number of selected experts rather than the total number of parameters.

To see why this matters, consider a dense model with 40B parameters and a MoE model with 120B total parameters. The MoE model has 8 experts in each MoE layer with top-2 routing, and each expert contains about 10B parameters. Although the MoE model contains far more parameters overall, only two experts are used for a given token. For that token, the dense model uses all 40B parameters, while the MoE model uses only 20B parameters from the two selected experts. The remaining experts stay inactive and do not contribute to the computation. One caveat is that the full 120B parameters still need to be loaded into memory, even though only a small portion is used for each token.

This allows MoE models to offer the quality benefits of a much larger model while keeping the inference cost closer to that of a smaller dense model. These benefits come with a trade off. As soon as experts are spread across GPUs, the model must move tokens between devices, which introduces additional communication. We explore this next.

## Communication

In expert parallelism, the experts are distributed across multiple GPUs. If the experts are split across devices, what communication is required during a forward pass?

Let's take a look at what a forward pass for MoE looks like. Consider, for example, a setup with 4 GPUs and 8 experts, with 2 experts per GPU. So GPU 0 holds E0 and E1, GPU 1 holds E2 and E3 and so on. For a given input of 4 tokens per GPU,

* GPU 0 performs forward pass as is until the router. The router decides which experts should process each of the 4 tokens. For example, t0 gets assigned expert E2, t1 gets assigned E7, t2 gets assigned E0 and t3 gets assigned E3.
* Now since experts live on separate GPUs, these tokens must be sent to the GPU corresponding to their expert. So, t0 goes to GPU 1, t1 goes to GPU 3 and so on. The same process takes place on GPUs 1, 2 and 3 for their input tokens. This process of moving tokens to the GPUs that host their assigned experts is called dispatching and it involves all-to-all communication within the expert parallel group. All GPUs are involved as each GPU must send their tokens to various GPUs holding the experts.
* Once expert computation is performed, another all-to-all communication takes place to send the tokens back to their original GPU, where the model proceeds to the next stage of computation, such as the self-attention layer.

Because each GPU hosts different experts, every GPU must send the tokens assigned to those experts to the corresponding GPU. This creates a dense communication pattern between all GPUs. A MoE layer under expert parallelism performs two all-to-all communication rounds: one for dispatch and one for combine.

{{< figure align=center src="/images/expert_parallelism.png" attr="Expert parallelism communication">}}

Expert parallelism (EP) is often used in conjunction with other parallelism techniques. EP with data parallelism (DP) allows training to scale across many GPUs by replicating the model while distributing experts within each replica. This keeps the non-MoE layers replicated as usual while the experts remain sharded across GPUs inside each data parallel group.

Similarly, combining EP with tensor parallelism (TP) and data parallelism (DP) allows scaling to very large models where dense layers are split across GPUs, experts are distributed across another set of GPUs, and the entire model is replicated across data parallel groups. This multi-dimensional strategy makes it possible to train extremely large MoE models that would not fit or compute efficiently under a single form of parallelism.

{{< figure align=center src="/images/expert_parallelism_all.png" attr="[A Survey on Mixture of Experts in Large Language Models paper](https://arxiv.org/abs/2407.06204)">}}

## Wrap up

Mixture of Experts architectures increase model capacity by introducing many experts and a router that selects only a few of them per token. Expert parallelism distributes these experts across multiple GPUs and relies on all-to-all communication to move tokens to the appropriate experts.

This completes our series on parallelism techniques used for training and inference LLMs where we started with

* [Data parallelism](https://dudeperf3ct.github.io/posts/ultrascale_data_parallelism/) with [sharding](https://dudeperf3ct.github.io/posts/ultrascale_zero_deepspeed/)
* [Pipeline parallelism](https://dudeperf3ct.github.io/posts/pipeline_parallelism/)
* [Tensor parallelism and sequence parallelism](https://dudeperf3ct.github.io/posts/tensor_sequence_parallelism/)
* [Context parallelism](https://dudeperf3ct.github.io/posts/context_parallelism/)
* [Expert parallelism](https://dudeperf3ct.github.io/posts/expert_parallelism/)

In the next on the series, I will look into implementing these using PyTorch to understand it better.