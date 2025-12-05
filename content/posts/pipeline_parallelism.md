---
author: [""]
title: "Ultrascale Playbook - Pipeline Parallelism"
date: 2025-10-25
summary: "Notes on training LLMs using pipeline parallelism"
description: ""
tags: ["llm", "llm-training", "pipeline-parallelism"]
ShowToc: true
ShowBreadCrumbs: true
math: true
---

The previous blogs in [the series](https://dudeperf3ct.github.io/tags/llm-training/) introduced the first and simplest degree of parallelism - [data parallelism](https://dudeperf3ct.github.io/posts/ultrascale_data_parallelism/). Along with optimization from sharding techniques such as [ZeRO](https://dudeperf3ct.github.io/posts/ultrascale_zero_deepspeed/), LLMs could be trained on a large number of GPUs. But what if LLMs are so large (70B+ parameters) that they cannot fit within the memory of single GPU or even multiple GPUs. This is where pipeline parallelism becomes essential.

The core idea is simple: split the model's sequential layers across multiple GPUs. For a 24-layer model and 4 GPUs, you might assign:
* GPU 1: Layers 1-6
* GPU 2: Layers 7-12
* GPU 3: Layers 13-18
* GPU 4: Layers 19-24

Each GPU is now a "stage" in the pipeline, responsible for storing and computing only its assigned layers. During the forward pass, activations flow from one stage to the next. During the backward pass, gradients flow in the reverse. However this introduces communication overhead between GPUs to send and recieve activations and gradients. The key challenge in distributed training is to minimize this communication cost or overlap communication with computation.


## Naive model parallelism

The naive approach to model parallelism would be to distribute the layers across GPU and run the batches of data sequentially. This is how it looks for a single input batch

{{< figure align=center src="/images/naive_pp.png" attr="HuggingFace [blog](https://huggingface.co/spaces/nanotron/ultrascale-playbook?section=splitting_layers_on_various_nodes_-_all_forward,_all_backward). Here the numbers indicate the layers processed for a single batch.">}}

> [!INFO] Side note
> The backward pass (pink boxes) is typically about 2× longer than the forward pass (teal boxes).
> This is because the backward pass involves two matrix multiplications — one to compute gradients w.r.t. activations, and another for gradients w.r.t. weights whereas the forward pass requires only one.

The way it works is

1. **Split the model layers across GPUs**: For example in the figure above model has 16 layers and there are 4 GPUs: GPU 1 holds layers 1-4, GPU 2 holds layers 5-8, GPU 3 holds layers 9-12 and GPU 4 holds layers 13-16.
2. **Forward pass**: Batch of inputs is sent to GPU 1. GPU 1 computes its layers and produces activations. Once GPU 1 finishes computation, these activations are transferred GPU 2. GPU 2 then performs its layers computations and this continues sequentially across all GPUs. Each stage must wait for the previous stage to finish before starting.
3. **Backward pass**: The gradients computed during backward pass flow in the opposite direction, from the last GPU to the first. Each GPU must wait for the gradient from the next stage before starting its backward computation.

There are several inefficiencies with this naive approach

1. **Low GPU Utilization (Pipeline Bubbles)**: At any given time, only one GPU is active while others wait for input from previous stage. The idle time is indicated in gray in the figure above. 
2. **Lack of Computation and Communication Interleaving**: GPUs wait for intermediate activations or gradients to be transferred before starting their own computation. Ideally, we’d like to interleave communication (sending data) with computation to hide latency.
3. **High Memory Demand (Activation Accumulation)**: Each GPU must cache all intermediate activations for the entire minibatch until the backward pass begins. For large batch size, this quickly becomes a memory bottleneck.

{{< collapse summary="**Measuring bubble time**" >}}

Let \(t_f\) and \(t_b\) be the time to execute forward and backward pass for a single minibatch. The number of micro-batches is \(m\) (for naive case \(m = 1\)), the number of pipeline stages (number of devices used for pipeline parallelism or degree of pipeline parallelism) is denoted as \(p\).

The ideal time per iteration is \(t_{id}\). In the case of naive parallelism, the pipeline bubble consists of \(p-1\) forward passes at the start of a batch and \(p-1\) backward passes at the end. The total amount of time spent in the pipeline bubble is 
$$
t_{pb} = (p-1) * (t_f + t_b) 
$$

The ideal processing time for all the samples in the batch is the following. Here, the number of micro-batches \(m = 1\). 
$$
t_{id} = m * (t_f + t_b) = (t_f + t_b), \text{as m=1}
$$
Therefore, the fraction of ideal computation time spent in the pipeline bubble is
$$
\text{Pipeline bubble size} = \frac{t_{pb}}{t_{id}} = \frac{(p-1) * (t_f + t_b)}{(t_f + t_b)} = p-1
$$

So as we increase the number of GPUs \(p\), idle time grows linearly.

{{</ collapse >}}

## GPipe


[GPipe](https://arxiv.org/pdf/1811.06965) paper tackled the bubble problem with a key insight: split the mini-batch into smaller micro-batches.

{{< figure align=center src="/images/afab.png" attr="HuggingFace [blog](https://huggingface.co/spaces/nanotron/ultrascale-playbook?section=splitting_layers_on_various_nodes_-_all_forward,_all_backward). Here the numbers indicate the micro-batches.">}}

The diagram above shows 8 micro batches being proccessed concurrently, keeping every GPU occupied with either forward or backward work. For example, when the first GPU processes a micro-batch, it immediately starts processing second micro-batch. The second GPU starts processing first micro-batch as soon as the first GPU completes processing it.

By keeping the pipeline full of multiple micro-batches, GPipe dramatically improves GPU utilization. However, while GPipe reduces **pipeline bubbles**, it **does not address** the other two drawbacks:
- It still **does not overlap computation with communication**, and  
- It **retains all activations in memory** until the backward phase begins.

{{< collapse summary="**Measuring bubble time**" >}}

Let \(t_f\) and \(t_b\) be the time to execute forward and backward pass for a single minibatch. The number of micro-batches is \(m\), the number of pipeline stages (number of devices used for pipeline parallelism or degree of pipeline parallelism) is denoted as \(p\).

The ideal time per iteration is \(t_{id}\). The total amount of time spent in the pipeline bubble is 
$$
t_{pb} = (p-1) * (t_f + t_b) 
$$

The ideal processing time for the \(m\) micro-batches is the following
$$
t_{id} = m * (t_f + t_b)
$$

Therefore, the fraction of ideal computation time spent in the pipeline bubble is
$$
\text{Pipeline bubble size} = \frac{t_{pb}}{t_{id}} = \frac{(p-1) * (t_f + t_b)}{m * (t_f + t_b)} = \frac{p-1}{m}
$$

Thus, increasing the number of micro-batches \(m\) reduces bubble size proportionally. However, a larger \(m\) also increases memory usage because all microbatch activations must be stored.
{{</ collapse >}}
 
    
## 1F1B

[One forward one backward](https://arxiv.org/pdf/1806.03377) (1F1B) schedule helps reduce the activation memory by alternating between forward and backward passes.

As soon as the first micro-batch completes its forward pass through the entire pipeline, the last GPU can immediately start its backward pass. Once the backward computation for a micro-batch is done, its cached activations can be **freed**, reducing the total memory footprint.

{{< figure align=center src="/images/1f1b.png" attr="HuggingFace [blog](https://huggingface.co/spaces/nanotron/ultrascale-playbook?section=one_forward,_one_backward_and_llama_3.1_schemes)">}}

Looking at the figure above, only 4 micro-batches (degree of parallelism) of activation needs to be stored. In contrast, GPipe must store activations for all 8 micro-batches at once. Both approaches process the same number of micro-batches (8), but 1F1B significantly reduces activation memory. This approach does not improve the pipeline bubble observed in the GPipe approach. The bubble still has the same size.

The alternating forward and backward pass means the training loop starts getting complicated as scheduler has to keep track of all the micro-batches and their corresponding forward and backward stages. The 1F1B helps reduce the activation memory but still has the same pipeline bubble time as the GPipe approach. 

## Interleaved 1F1B

To further reduce the bubble size, instead of linear splitting layers across GPUs for the model, the layers of the model are interleaved across devices to form a ring, connecting first and last GPU creating a loop.

If we have a 16-layer model and 4 GPUs, using a modulo scheme based on 4 GPUs:
* GPU 1 → layers (0, 4, 8, 12)
* GPU 2 → layers (1, 5, 9, 13)
* GPU 3 → layers (2, 6, 10, 14)
* GPU 4 → layers (3, 7, 11, 15)

This splits the model into 4 interleaved chunks.

{{< figure align=center src="/images/looping.png" attr="[Breadth-First Pipeline Parallelism](https://arxiv.org/pdf/2211.05953) Figure 3">}}

Looping introduced in [Megatron-LM](https://www.arxiv.org/pdf/2104.04473) paper improves interleaving and reduces bubble time. Comparing the figures for 1F1B and interleaving 1F1B, the same 8 micro-batches complete faster in interleaving 1F1B compared to only 1F1B. Why is that? How does looping help?

{{< figure align=center src="/images/interleaved_1f1b.png" attr="HuggingFace [blog](https://huggingface.co/spaces/nanotron/ultrascale-playbook?section=interleaving_stages) A bug in this diagram where backward first layers should be swapped with last layers.">}}

Compared to standard 1F1B, the same 8 micro-batches finish faster because each GPU computes only a small chunk at a time and immediately forwards it to the next GPU. The backward pass works similarly in reverse.

{{< collapse summary="**Napkin calculation when interleaved 1F1B is faster compared to 1F1B**" >}}

Consider the following:

- \(K\) = number of pipeline stages (GPUs),
- \(T_{chunk}\) = time to compute one contiguous chunk in vanilla 1F1B,
- \(v\) = number of sub-chunks you split each chunk into,
- \(T_{sub}\) = \(\frac{T_{chunk}}{v}\) = time per sub-chunk,
- \(C_{comm}\) = extra communication/overhead introduced by interleaving

Rough bubble estimates for 1F1B and interleaved 1F1B:
$$
\text{1F1B} = (K - 1) * T_{chunk}
$$

$$
\text{Interleaved 1F1B} = (K - 1) * T_{sub} + C_{comm} = (K - 1) * \frac{T_{chunk}}{v} + C_{comm}
$$

So interleaving reduces the bubble by roughly (old time minus new time)
$$
\Delta_{bubble} = (K - 1) * T_{chunk} * (1 - \frac{1}{v}) - C_{comm}
$$

Interleaving wins if \(\Delta_{bubble} > 0\) i.e.
$$
C_{comm} < (K - 1) * T_{chunk} * (1 - \frac{1}{v})
$$

Plugging some toy numbers in the above equation,
\(K = 4\), \(T_{chunk} = 100 ms\), \(v = 4\) gives RHS = 3 * 100 * (1 − 1/4) = 225 ms.
So if the extra comm/overhead per iteration is \(<\) 225 ms, interleaving should reduce the bubble and improve throughput. If \(C_{comm}\) is larger (e.g., due to very high message latency or many tiny messages), interleaving can lose.

{{</ collapse >}}

The training loop is getting more complicated and now model placement requires careful consideration, since layers are no longer sequential. Interleaved 1F1B keeps GPUs busy more consistently, cutting idle time and improving throughput, while still maintaining low activation memory like 1F1B.

{{< collapse summary="**Measuring bubble time**" >}}


The number of micro-batches is \(m\), the number of pipeline stages (number of devices used for pipeline parallelism or degree of pipeline parallelism) is denoted as \(p\). The number of interleaved stages per GPU is \(v\). Each GPU now executes \(v\) smaller stages, so per-stage compute time becomes \(\frac{t_f}{v}\) and \(\frac{t_b}{v}\).

The ideal time per iteration is \(t_{id}\). The total amount of time spent in the pipeline bubble is 
$$
t_{pb} = \frac{(p-1) * (t_f + t_b)}{v}
$$

The ideal processing time for the \(m\) micro-batches is the following
$$
t_{id} = \frac{m * (t_f + t_b)}{v}
$$

Therefore, the fraction of ideal computation time spent in the pipeline bubble is
$$
\text{Pipeline bubble size} = \frac{(p-1) * (t_f + t_b)}{v * m * (t_f + t_b)} = \frac{p-1}{m * v}
$$
Thus, interleaving reduces bubble size by an additional factor of \(v\), at the cost of increased communication.

{{</ collapse >}}

[Breadth-First Pipeline Parallelism](https://arxiv.org/pdf/2211.05953) paper introduces two schedule breadth-first pipeline (BFS) and depth-first pipeline (DFS) -- similar to the above interleaved 1F1B. Extracting following excerpt from paper on performance of different schedules

> [!QUOTE] Paper excerpt
> For smaller batches, the breadth-first schedule is by far the most efficient, minimizing both the bubble and network overheads. The depth-first schedule also reduces the pipeline bubble, but its high network overhead makes the performance worse than than the non-looped configurations in most cases. For larger batches, the pipeline bubble is small in all cases, and 1F1B is the fastest because of its lower pipeline-parallel network overhead and memory usage."

[Llama 3.1](https://www.arxiv.org/pdf/2407.21783) paper provides insights on how they combine DFS and BFS schedule to optimize the memory and communication efficiently. They also balance the pipeline by reducing transformers layers from first and last stages. First layer is responsible for embedding lookup which increases memory and last layer is used to calculate output and loss which increases the latency.

> [!TASK] TODO
> I don't understand DFS and BFS schedule clearly. I will revisit those and rewrite for clarity.

## Zero Bubble Pipeline Schedule

[Zero Bubble](https://www.arxiv.org/pdf/2401.10241) paper proposes a clever strategy that eliminates pipeline bubbles by performing fine-grained computation scheduling for the backward pass. 

In naive parallelism, we hinted at how backward pass takes roughly about twice as long as the forward pass. This is because the backward pass involves two matrix multiplications — one to compute gradients w.r.t. activations, and another for gradients w.r.t. weights whereas the forward pass requires only one.

The Zero Bubble schedule exploits the fact that weight gradients are not sequentially dependent and can therefore be computed whenever idle compute resources are available. By filling these idle slots with weight-gradient computation, Zero Bubble keeps all GPUs busy and eliminates the pipeline “bubbles.”

{{< collapse summary="**Forward and backward computation**" >}}

Let's consider two consecutive linear layers
$$
A_{L} = W_{L}X_{L}
$$

$$
A_{L+1} = W_{L+1}A_{L}
$$

Here,
- \(X_{L}\) - input activations to layer \(L\)
- \(A_{L}\) - output activations of layer \(L\)
- \(W_{L}\) - weight matrix of layer \(L\)

Let's denote \(G_{L+1}\) as upstream gradient from the next layer 
$$
G_{L+1} = \frac{\delta{L}}{\delta{A_{L+1}}}
$$

Backward for layer \(L+1\) consists of two gradients: inputs (B) and weights (W)
$$
\text{Input-grad (B)} = dA_{L} = \frac{\delta{L}}{\delta{A_{L}}} = W^{T}_{L+1}G_{L+1}
$$

$$
\text{Weights-grad (W)} = dW_{L+1} = \frac{\delta{L}}{\delta{W_{L+1}}} = G_{L+1}A^{T}_{L}
$$

Backward for layer \(L\) consists of two gradients: inputs (B) and weights (W)
$$
\text{Input-grad (B)} = dX_{L} = \frac{\delta{L}}{\delta{X_{L}}} = W^{T}_{L}dA_{L}
$$
$$
\text{Weights-grad (W)} = dW_{L} = \frac{\delta{L}}{\delta{W_{L}}} = dA_{L}X^{T}_{L}
$$

To compute \(\delta{X_{L}}\) i.e \(B_{L}\), we first need \(dA_{L}\), and to get \(dA_{L}\), we need \(G_{L+1}\). This creates a strict sequential dependency chain \(G_{L+1} -> dA_{L} -> dX_{L} ...\) where Bs depend on the next layer's B.

However, \(dW_{L+1}\) needs only (\(G_{L+1}, A_{L}\)) and \(dW_{L}\) needs only (\(dA_{L}, X_{L}\)). These weights gradients are not required to produce any \(dA\) or \(X\) for earlier layers, so they are off the critical path of the backward chain, and can be scheduled later (any time after their inputs are ready and before optimizer step) to fill the bubbles.

{{</ collapse >}}

{{< figure align=center src="/images/zero_bubble1.png" attr="[Zero bubble paper](https://www.arxiv.org/pdf/2401.10241) Figure 2">}}

The figure above shows 1F1B scheduling where there are lot of idle times (white squares). The figure below shows Zero Bubble schedule where nearly all the idle time is eliminated. There's a complexity here in how these schedules are designed heuristically and synchronizing the gradients for the optimizer step.

{{< figure align=center src="/images/zero_bubble2.png" attr="[Zero bubble paper](https://www.arxiv.org/pdf/2401.10241) Figure 3 Zero Bubble Schedule">}}


## DeepSeek v3 DualPipe

The [DeepSeek v3 paper](https://www.arxiv.org/pdf/2412.19437) introduced DualPipe, a technique that overlaps forward and backward computation–communication phases to further reduce pipeline bubbles.

The key idea behind DualPipe is to overlap computation and communication within each pair of forward and backward chunks. Each chunk includes four components — Attention, All-to-All Dispatch, MLP, and All-to-All Combine. The all-to-all operations are introduced by expert parallelism to send and receive data. Similar to the Zero Bubble approach above, the backward chunks for Attention and MLP are split into two parts: backward pass for inputs (B) and backward pass for weights (W).

The figure below illustrates how overlapping computation effectively hides communication latency.

{{< figure align=center src="/images/deepseek_overlap.png" attr="[DeepSeek v3 technical paper](https://www.arxiv.org/pdf/2412.19437) Figure 4">}}

Figure 5 from the paper illustrates the DualPipe scheduling for 8 pipeline parallelism (PP) ranks and 20 micro-batches, where micro-batches are processed bidirectionally. Until now in all parallelism strategies, micro-batches have been processed in one direction (first microbatch starts at GPU 0, then GPU 1 and so on). In DualPipe, one stream processes forward passes from left to right (GPU 0 -> GPU 1 -> ... -> GPU 7). The other, concurrent stream processes forward passes from right to left (GPU 7 -> GPU 6 -> ... -> GPU 0).

The schedule is meticulously crafted so that when one microbatch is in its forward pass on a GPU, another microbatch can be in its backward pass on the same GPU. This overlap of forward and backward computations (and their associated communications) is what dramatically reduces the "pipeline bubble" and increases GPU utilization.

This efficiency gain comes at the cost of maintaining two copies of certain model parameters. These duplicate copies are strategically used to facilitate the efficient computation of gradients during the backward pass and enable further overlapping of operations. Why 2x parameters are required?

To run two independent forward passes simultaneously in opposite directions, each device needs to hold two sets of weights for its layer chunks:
* One set for the "forward" pipeline (left-to-right).
* One set for the "reverse" pipeline (right-to-left).

These two sets of parameters are synchronized (their gradients are combined during the backward pass), but they exist as separate copies in memory to allow for the simultaneous, bidirectional computation that defines DualPipe. This memory overhead is the trade-off for the significantly reduced pipeline bubbles and higher training throughput.

{{< figure align=center src="/images/deepseek3.png" attr="[DeepSeek v3 technical paper](https://www.arxiv.org/pdf/2412.19437) Figure 5">}}

By running two synchronized pipelines in opposite directions, DualPipe achieves:

* Full overlap of forward and backward communication-computation phases
* Near-zero pipeline bubbles and higher hardware utilization
* Better scaling across many GPUs, especially when combined with expert and tensor parallelism

> [!IDEA] DeepSeek blog
> I would like to explore DeepSeek v3 in depth as part of separate post.

## Wrap up

This post introduced a series of pipeline parallelism strategies, each addressing different bottlenecks:

* **Naive model parallelism**: basic layer-splitting, high idle time.
* **GPipe**: reduced bubbles via micro-batching.
* **1F1B**: reduced activation memory with alternating passes.
* **Interleaved 1F1B**: finer scheduling, fewer bubbles.
* **Zero Bubble**: filled idle slots using gradient independence.
* **DualPipe**: full overlap with bidirectional scheduling.

In practice, pipeline parallelism is almost always combined with data and/or tensor parallelism to achieve truly massive scale. While this post focused on training, pipeline parallelism is also highly effective for inference. Without the need for a backward pass, the scheduling becomes much simpler, allowing for efficient execution of large models across multiple devices.

Pipeline parallelism is particularly valuable for cross-node training, where inter-node bandwidth is often limited. Unlike data parallelism, which requires synchronizing gradients across all nodes, pipeline parallelism primarily passes activations between consecutive stages. This reduces the volume of data sent over the network at any given time, making it easier to utilize bandwidth efficiently and keep GPUs busy even when communication is slower.