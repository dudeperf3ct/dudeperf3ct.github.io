---
author: [""]
title: "Ultrascale Playbook - Context Parallelism"
date: 2025-11-22
summary: "Notes on training LLMs using context parallelism"
description: ""
tags: ["llm", "llm-training", "context-parallelism"]
ShowToc: true
ShowBreadCrumbs: true
math: true
---

Before diving into context parallelism, let’s briefly recap the parallelism strategies covered in earlier posts:

* [Data Parallelism](https://dudeperf3ct.github.io/posts/ultrascale_data_parallelism/) with [Sharding](https://dudeperf3ct.github.io/posts/ultrascale_zero_deepspeed/): In this technique, the model is sharded across multiple devices, and each device processes a different batch of data. Gradients are synchronized across devices after each batch using [distributed communication](https://dudeperf3ct.github.io/posts/distributed_communication_part2/). This is effective for scaling training across many devices, but can be limited by the size of the model that can fit on each device.
* [Pipeline Parallelism](https://dudeperf3ct.github.io/posts/pipeline_parallelism/): In this technique, the model is divided into stages, and each stage is assigned to a different device. Data flows through the pipeline, with each device processing its assigned stage. This allows for larger models to be trained, but can introduce latency due to the sequential nature of the pipeline. There are several scheduling strategies to minimize the pipeline bubbles such as 1F1B, Interleaved 1F1B and Zero-Bubble schedule.
* [Tensor Parallelism](https://dudeperf3ct.github.io/posts/tensor_sequence_parallelism/) with [Sequence Parallelism](https://dudeperf3ct.github.io/posts/tensor_sequence_parallelism/): In this technique, individual layers of the model are split across multiple devices. For example, in a transformer block, the attention heads in MHA and linear layers in MLP are split using tensor parallelism either using row-wise sharding or column-wise sharding. The remaining layers such as normalization and dropout are split using sequence parallelism. This communication overhead is manageable within a single node (NVLink), but becomes expensive across nodes (InfiniBand), especially due to K/V replication.

The attention KV tensors starts dominating for large sequence sizes, TP + SP replicates this matrix on every device. The combination of TP + SP works well within a single node using NVLink communication. However when scaling across multiple nodes, the communication overhead can become a bottleneck. The following diagram shows a significant performance drop going from TP=8 to TP=16. 

{{< figure align=center src="/images/tp_sp_bottleneck.png" attr="HuggingFace [blog](https://huggingface.co/spaces/nanotron/ultrascale-playbook?section=context_parallelism)">}}

Context Parallelism (CP) is a new technique that directly addresses these limits. Instead of splitting the hidden dimension (like TP) or merely sharding activations (like SP), CP distributes the sequence length itself across devices.

## Transformer Architecture

In the previous blog on tensor parallelism and sequence parallelism, we discussed how TP + SP can be used to train large language models efficiently.

{{< figure align=center src="/images/sequence_parallelism.png" attr="Adapted from Reducing Activation Recomputation in Large Transformer Models [paper](https://www.arxiv.org/abs/2205.05198)">}}

* TP shards the linear layers (QKV projections, attention output, and MLP) across the hidden dimension. Each GPU holds a slice of the weight matrices and contributes to the overall computation.
* SP shards token activations across the sequence dimension to reduce activation memory for layers that do not require communication across tokens (LayerNorm, dropout, residual connections).

However, the KV matrix is not sharded in TP + SP. Every device stores a full copy of K and V for all tokens in the sequence. As sequence lengths grow, this quickly becomes the dominant memory consumer. The KV matrix per layer requires: `seq_len × hidden_dim × 2 (K and V) × bytes_per_element`. For example, with a sequence length of 128k, hidden size 16,384, and using fp16 (2 bytes per element), the memrory required for KV matrix per layer is: `128,000 × 16,384 × 2 × 2 bytes = 8.4 GB`. With 24 layers, this amounts to over 200 GB of KV matrix alone, which is impractical for training.

Context Parallelism (CP) addresses this by sharding the sequence itself across devices. Each GPU holds only a fraction of the tokens. Most transformer sublayers such as MLP, normalization, residuals are per-token operations and can be computed entirely locally.

The only exception is self-attention. Each token still needs global context, so the attention block must exchange K and V across devices. Once attention is done, computation returns to local per-token operations.

{{< figure align=center src="/images/context_parallelism.png">}}

In the figure above, for a context parallelism of degree 2, only the input is split across 2 devices. GPU 0 processes the first half of the sequence, while GPU 1 processes the second half. The attention block requires all tokens to compute attention scores. Therefore, we need to communicate the key and value tensors across devices before computing attention. After the attention block, the MLP and LayerNorm can be computed independently on each device without further communication. To conserve the memory, we can discard the communicated key and value tensors after the attention block. The final output of the transformer block is then a concatenation of the outputs from both devices.

Context Parallelism does not change the fact that self-attention is quadratic in sequence length. Each query still needs to attend to all keys. Instead, CP distributes the work across devices. With CP = N:
* Each GPU holds L/N tokens (where L is the total sequence length)
* Each GPU computes attention for its own Q chunk
* All GPUs exchange K/V slices so every device has the full K/V needed for its queries

So total compute remains the same, but the per-GPU cost and memory footprint drop by a factor of N.

In all, one communication step is required for the forward pass of each transformer block. The backward pass requires two communication steps: one to gather the key and value tensors across devices to calculate the gradients and another to scatter the reduced gradients.

{{< collapse summary="**Attention and Multi-head Attention Refresher**" >}}
</br>

**Attention Mechanism**

{{< figure align=center src="/images/attn.png" attr="Attention is all you need [paper](https://arxiv.org/pdf/1706.03762)">}}

Attention function can be described as mapping a query and a set of key-value pairs to an output, where the query, keys, values, and output are all vectors. The output is computed as a weighted sum of the values, where the weight assigned to each value is determined by a compatibility function of the query with the corresponding key. The most commonly used compatibility function is the dot product, scaled by the square root of the dimension of the key vectors to prevent large dot product values.

$$
Attention(Q, K, V) = softmax\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

**Multi-Head Attention**

Multi-head attention allows the model to jointly attend to information from different representation subspaces at different positions. Instead of performing a single attention function with d_model-dimensional keys, values, and queries, multi-head attention projects the queries, keys, and values h times with different learned linear projections to d_k, d_k, and d_v dimensions, respectively. The attention function is then performed in parallel on each of these projected versions of queries, keys, and values, yielding h outputs. These are concatenated and once again projected, resulting in the final values.

{{< figure align=center src="/images/multi_head.png" attr="Attention is all you need [paper](https://arxiv.org/pdf/1706.03762)">}}

$$
Multihead(Q,K,V) = Concat(head_1, ..., head_h)W^O 
$$

where 

$$
head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)
$$

and following are the parameter matrices

$$
W_i^Q \in R^{d_{model} × d_k} \\
W_i^K \in R^{d_{model} × d_k} \\
W_i^V \in R^{d_{model} × d_v} \\
W^O \in R^{hd_v × d_{model}} 
$$

{{< figure align=center src="/images/attention_explained.png" attr="[Tensor Parallelism and Sequence Parallelism: Detailed Analysis](https://insujang.github.io/2024-01-11/tensor-parallelism-and-sequence-parallelism-detailed-analysis/)">}}

The diagram above shows the detailed computation of multi-head attention. The input hidden states are projected into Q, K, and V using linear layers. The Q, K, and V matrices are then split into multiple heads. Each head computes attention scores using the scaled dot-product attention mechanism. The outputs from all heads are concatenated and passed through a final linear layer to produce the final output of the multi-head attention block.

{{< /collapse >}}


## Optimizing Attention

A straightforward implementation of context parallelism would require an all-gather of K and V across all devices before the attention computation. For long sequences, this would involve communicating GBs of data per transformer block every iteration of the forward and backward pass. One way to optimize this is to use a ring-based communication pattern. 

### Ring Attention

In ring attention, each device only sends and receives data from its two neighboring devices in a ring topology. This reduces the communication overhead compared to an all-gather operation, as each device only needs to communicate with two other devices instead of all devices.

The following animation from Hugging Face playbook shows how the [ring attention](https://www.arxiv.org/abs/2310.01889) works for a context parallelism of degree 4 across 4 devices. Each device starts with its own slice of the sequence. In each step, each device sends its current K and V tensors to the next device in the ring and receives K and V tensors from the previous device. After receiving the K and V tensors, each device computes attention for its own slice of the sequence using the received K and V tensors. This process is repeated until each device has received K and V tensors from all other devices in the ring.

{{< figure align=center src="/images/ring_attention.gif" attr="HuggingFace [blog](https://huggingface.co/spaces/nanotron/ultrascale-playbook?section=ring_attention)">}}

The drawback of uneven load distribution across devices while computing the attention for causal transformers is addressed [DistFlashAttn](https://arxiv.org/abs/2310.03294) and [Striped Attention](https://arxiv.org/abs/2311.09431). Ring attention reduces the cost of gathering KV, but KV still scales with the number of heads. This leads us to Ulysses, which shards attention heads themselves.

### Ulysses Attention

In the naive implementation of CP, each GPU holds all attention heads for its local slice of the sequence. The [DeepSpeed-Ulysses](https://www.arxiv.org/abs/2309.14509) approach addresses this by further sharding the attention heads across devices. Instead of every GPU storing all heads for its token chunk, each GPU stores only a subset of heads. This reduces KV-matrix size, Q/K/V projection size, and overall memory footprint unlocking larger models and longer context lengths training. This approach supports any attention mechanism such as self-attention, cross-attention.

{{< figure align=center src="/images/cp_sp.png" attr="Arctic Ulysses [blog](https://www.snowflake.com/en/engineering-blog/ulysses-low-latency-llm-inference/)">}}

The figure above shows 3 transitions for a transformer block

1. Context parallelism is applied in the first part here degree of 2 splitting the sequence across GPU until the attention block.
2. All the KV tensors are gathered using all-to-all communication, and at the multi-head attention CP switches to attention head parallelism (HP).
3. Once attention block is complete, the remaining computation proceeds similarly to the context parallelism

In the example above, context parallelism (CP) splits the sequence across two GPUs: GPU 0 processes the first half of the tokens, and GPU 1 processes the second half. Each GPU runs the Q/K/V projections on its own tokens, so initially:
* GPU 0 has Q/K/V for its tokens, but with all 4 attention heads
* GPU 1 has Q/K/V for its tokens, also with all 4 attention heads

Ulysses then introduces head parallelism, assigning:
* GPU 0: heads 0 and 1
* GPU 1: heads 2 and 3

To make this distribution work, each GPU must hold the Q/K/V tensors only for the heads it owns, but for all tokens. That means GPU 0 must receive heads 0–1 from GPU 1's tokens, and GPU 1 must receive heads 2–3 from GPU 0's tokens.

This requires a redistribution of Q/K/V by head groups. An all-gather would replicate all heads on every GPU, which is wasteful and incorrect. Instead, an all-to-all is used so each GPU receives exactly the head slices it needs.

After this all-to-all:
* GPU 0 owns heads 0–1 for both halves of the sequence
* GPU 1 owns heads 2–3 for both halves of the sequence

Each GPU can now compute attention for its assigned heads independently. However, before entering the MLP block, we must reconstruct the full hidden dimension. Each token's output must contain all four heads concatenated: `output = concat(head0, head1, head2, head3)`

But after head-parallel attention:
* GPU 0 only has head0 and head1 outputs
* GPU 1 only has head2 and head3 outputs

So a second all-to-all is performed to return the missing head outputs to the GPU that owns the corresponding token shard. After this exchange:
* GPU 0 has all four heads for the tokens in the first half
* GPU 1 has all four heads for the tokens in the second half

Deepspeed-Ulysses involves lots of communication. Its degree of CP is limited to the number of attention heads. For example, with 128 attention heads, the maximum CP degree is 128 (each GPU gets 1 head). In practice, a lower CP degree is used to balance communication overhead and memory savings.

{{< figure align=center src="/images/loongtrain.png" attr="LoongTrain [paper](https://arxiv.org/abs/2406.18485)">}}

{{< collapse summary="**Comparing CP vs Ulysses + CP**" >}}
Suppose we are training a large Transformer with:
* Sequence length: 128k tokens
* Hidden size: 16,384
* Number of heads: 128

**Context parallelism (CP) = 4**

Context parallelism splits the sequence across 4 GPUs:

* Each GPU receives 128k / 4 = 32k tokens
* But each GPU still stores all 128 attention heads
* And each GPU holds the full hidden dimension = 16,384

**Ulysses head parallelism (HP = 8) and Context parallelism (CP = 4)**

Ulysses additionally partitions the attention heads across 8 GPUs. Now each GPU receives:

* 32k tokens (from CP)
* 128 / 8 = 16 attention heads (from head parallelism)
* 16,384 / 8 = 2,048 hidden-dim slice (since head count + hidden dim scale together)

{{< /collapse >}}

## Arctic Long Sequence Training (ALST)

ALST [paper](https://www.arxiv.org/pdf/2506.13996) extends DeepSpeed-Ulysses approach along with various optimization tricks such as sequence tiling to scale the training for multi-million sequence lengths.

CP does not take into consideration the problem of exploding activation memory for large sequences. Sequence Tiling approach helps reduce the peak activation memory by tiling forward and backward computations along sequence dimension. Tiling means partitioning an extremely long sequence into smaller, manageable segments ("tiles") and executing the forward and backward passes tile-by-tile rather than on the full sequence at once. By only keeping activations for a single tile in memory at any moment, the peak activation footprint is drastically reduced, allowing training on multi-million–token sequences without running out of memory.

{{< figure align=center src="/images/alst.png" attr="Arctic Long Sequence Training [paper](https://www.arxiv.org/pdf/2506.13996)">}}

Next, it extends the [DeepSpeed-Ulysses](#ulysses-attention) to support modern attention approaches such as Grouped-query attention (GQA) and Multi-query attention (MQA) in addition to Multi-head attention (MHA).

Using these techniques along with various PyTorch optimisations such as activation checkpointing, fused kernels, ALST demonstrates training of models with multi-million sequence lengths. [ArcticTraining](https://github.com/snowflakedb/ArcticTraining) framework and [DeepSpeed](https://www.deepspeed.ai/tutorials/ulysses-alst-sequence-parallelism/)) library implement these optimizations to enable training of LLMs with extremely long context lengths.


## Wrap Up

With CP, we now have:
* DP to scale batch
* PP to scale depth
* TP/SP to scale width
* CP to scale sequence length

The only remaining frontier is scaling the number of parameters, especially in Mixture-of-Experts architectures. This is exactly the domain of Expert Parallelism, which we explore in the next post.