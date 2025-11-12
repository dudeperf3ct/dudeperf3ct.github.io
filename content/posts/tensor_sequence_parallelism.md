---
author: [""]
title: "Ultrascale Playbook - Tensor and Sequence Parallelism"
date: 2025-11-11
summary: "Notes on training LLMs using tensor and sequence parallelism"
description: ""
tags: ["llm", "llm-training", "tensor-parallelism", "sequence-parallelism"]
ShowToc: true
ShowBreadCrumbs: true
math: true
---

Let's recap what we have seen so far in scaling LLM training

* [Data Parallelism](https://dudeperf3ct.github.io/posts/ultrascale_data_parallelism/) with [Sharding](https://dudeperf3ct.github.io/posts/ultrascale_zero_deepspeed/) : This approach replicates and shards the model across multiple GPUs, each device processing a different portion of the batch. It scales effectively to hundreds of GPUs but eventually suffers from heavy communication overhead particularly in gradient synchronization and optimizer states in ZeRO-based setups.
* [Pipeline Parallelism](https://dudeperf3ct.github.io/posts/pipeline_parallelism/) : This approach splits the model layers into stages distributed across multiple GPUs. Each stage processes a micro-batch and passes intermediate activations to the next stage. Pipeline parallelism scales across the model depth, but introduces pipeline bubbles—idle time while waiting for other stages—and requires careful tuning of micro-batch sizes to balance GPU utilization.

In LLMs certain layers such as large transformer blocks or embedding matrices may not fit within a single GPU's memory. This is where tensor parallelism and sequence parallelism come into play. These approaches split the computation within a layer itself across multiple GPUs. 

An input to LLM consists of a shape `(batch_size, sequence_length, hidden_dim)`. 
* Data parallelism strategy splits along the batch size dimension (different examples to different GPUs). 
* Pipeline parallelism splits the model depth (different layers to different GPU)
* Tensor and sequence parallelism split within a layer, distributing computation along the hidden and sequence dimensions respectively

Tensor parallelism processes same batch across multiple GPUs simultaneously, avoiding the pipeline bubbles of pipeline parallelism. However, it introduces frequent inter-GPU communication between devices to exchange partial results.

## Tensor Parallelism

It’s well known that LLMs are dominated by matrix multiplications. As model dimensions (hidden size, attention heads, FFN intermediate sizes) grow, the matrices can become too large to fit on a single GPU. 

Consider a simple matrix multiplication for a linear layer \(Y = XW\) where \(X\) is the input or activation values and \(W\) represents the weight of the linear layers.

{{< figure align=center src="/images/matmul.png" attr="HuggingFace [blog](https://huggingface.co/spaces/nanotron/ultrascale-playbook?section=tensor_parallelism)">}}

Tensor parallelism distributes this computation across multiple GPUs by sharding the weight and activation tensors and performing the matrix multiplication in parallel. There are two primary ways to split this operation while achieving the same final result:

**Row-wise sharding**

In row-wise sharding,  
- The weights \(W\) are split by rows (along the input dimension).  
- The input \(X\) must be scattered across devices so that each GPU receives the portion relevant to its shard of \(W\).

Each GPU performs its local matrix multiplication, producing a partial output. The partial outputs are then combined using an All-Reduce. All-Reduce is a collective operation that sums results across GPUs.

{{< figure align=center src="/images/row_matmul.png" attr="HuggingFace [blog](https://huggingface.co/spaces/nanotron/ultrascale-playbook?section=tensor_parallelism)">}}

**Column-wise sharding**

In column-wise sharding,

- The weights \(W\) are split by columns (along the output dimension). 
- Each GPU computes a partial output projection using the full input \(X\) (which is broadcasted to all devices).  

After local computations, GPUs perform an All-Gather operation to concatenate the partial outputs along the feature dimension and reconstruct the full output \(Y\). All-Gather is a collective operation that gathers results from all GPUs.

{{< figure align=center src="/images/col_matmul.png" attr="HuggingFace [blog](https://huggingface.co/spaces/nanotron/ultrascale-playbook?section=tensor_parallelism)">}}

Each of these strategies trades off communication and memory differently:
- **Row-wise sharding** reduces memory usage for the weights but requires an All-Reduce after the layer.
- **Column-wise sharding** simplifies the input distribution but requires an All-Gather to reassemble outputs.

A natural question is: when should each strategy be used? The choice depends on the input and output dimensions, with the general principle being to minimize the communication volume.

* Row-wise sharding is preferable when the output dimension is small (less data to transfer during reduction).
* Column-wise sharding is preferable when the input dimension is small (less data to broadcast).

## Tensor parallelism in Transformers

The original transformer block consists of two main components relevant for TP: the MLP block and Multi-head Attention (MHA).

{{< figure align=center src="/images/transformer_arch.png" >}}

### Multi-layer perceptron {#multi-layer-perceptron}

The MLP block is a feed-forward network with two linear layers and a non-linear activation between them. The first linear layer expands the hidden dimension (up-projection going from h to 4h). The activation function, usually ReLU or GeLU, is applied. The second linear layer projects the data back down to its original dimensionality (down-projection going from 4h to h).

{{< figure align=center src="/images/mlp_block.png">}}

Because the first linear layer expands the output size, the preferred sharding differs between the two linear layers:

In tensor-parallel MLP:

* First linear layer (up-projection): column-wise sharding; each GPU holds a slice of the weight columns, computes partial outputs locally, and the full input is broadcast.
* Activation: applied locally on each GPU (no communication needed).
* Second linear layer (down-projection): row-wise sharding; partial outputs are combined via a single All-Reduce to produce the final output.

{{< figure align=center src="/images/mlp_tp.png">}}

 This column-then-row ordering avoids an All-Gather between the two linear layers. Overall, only one collective communication (an All-Reduce) is required to produce the final output. If you instead used row-then-column, you would need an extra synchronization before the activation because activation functions are not additive.
 
### Multi-Head Attention

{{< figure align=center src="/images/attention_block.png">}}

Attention computes query (Q), key (K), and value (V) projections followed by scaled dot-product attention. In multi-head attention, the input features are projected into multiple heads. Each head attends independently and the outputs of all heads are concatenated back to the original feature size, followed by a final linear projection.

{{< figure align=center src="/images/mha.png" attr="Attention is all you need [paper](https://arxiv.org/pdf/1706.03762)">}}

In tensor-parallel MHA, the heads are split across GPUs, typically using column-wise sharding on the weight matrices for the Q, K, and V projections. Each GPU handles a subset of heads:

* Q, K, V projections: column-wise sharding; each GPU handles a subset of attention heads locally. No communication is required at this stage.
* Scaled dot-product attention: computed locally on each GPU for assigned heads.
* Final linear projection: row-wise sharding; each GPU multiplies its local attention output with its slice of the weight matrix. Only a single All-Reduce is needed to produce the final output.

{{< figure align=center src="/images/mha_tp.png" attr="HuggingFace [blog](https://huggingface.co/spaces/nanotron/ultrascale-playbook?section=tensor_parallelism_in_a_transformer_block)">}}

This approach combines the head-wise attention and final projection GEMMs, removes an intermediate synchronization, and minimizes communication. In multi-query attention (shared K and V but separate Q), column-wise sharding is applied only to Q, while K and V can be broadcast or replicated, further reducing memory usage and communication.

Across schemes, the principle is the same: distribute independent computations across GPUs, minimize synchronization, and fuse GEMMs wherever possible to achieve better scaling and efficiency.

## Async Tensor Parallelism (AsyncTP)

In a straightforward TP implementation, a single transformer block requires multiple collective ops across forward and backward. For example, a typical block may need:

- 2 All-Reduce operations during the forward pass (one after self-attention, one after MLP)
- 2 All-Reduce operations during the backward pass

{{< figure align=center src="/images/transformer_tp.png" attr="Adapted from Reducing Activation Recomputation in Large Transformer Models [paper](https://www.arxiv.org/abs/2205.05198)">}}

As model size increases, frequent switches between computation and communication create GPU idle time. AsyncTP techniques aim to overlap communication with computation so communication latency is hidden and GPUs remain productive.

### PyTorch AsyncTP

PyTorch AsyncTP decomposes large matmuls into finer sub-matmuls so you can start computing on parts of the input while other parts are still arriving. That enables overlap of communication and compute and prevents communication from blocking GPU compute. This approach achieves 8% speedup for training Llama 7B and 70B LLMs.

> Detailed walkthrough on the implementation of AsyncTP in PyTorch and `torchtitan` library: https://discuss.pytorch.org/t/distributed-w-torchtitan-introducing-async-tensor-parallelism-in-pytorch/209487

{{< figure align=center src="/images/naive_asynctp.png" attr="Pytorch [forum](https://discuss.pytorch.org/t/distributed-w-torchtitan-introducing-async-tensor-parallelism-in-pytorch/209487 )">}}

There are multiple ways to implement overlap. Consider the example where an All-Gather is required to gather all outputs from previous layer to perform matmul \(A @ B\). Here, \(A\) is sharded across 4 devices. To perform matmul in the non-tp approach all chunks have to be gathered using all-gather communication and then perform the computation.

Comparing this to the AsyncTP approach, each sharded \(A0\), \(A1\), \(A2\) and \(A3\) compute partial results. The overlap between communication and computation is hidden using two streams - one for computation and another for computation. One stream computes a sub-matmul while the other fetches the next shard. However, this can create many partial waves and leads to inefficiency: partial waves (small final waves that don't fully occupy all SMs) occur for each sub-matmul and those partial waves accumulate, increasing total execution time and leaving SMs idle.

In the diagram above,

* Compute \(A0 @ B\) in stream 0 which leads to full waves + partial wave at the end.
* Wait for communication to fetch A1 in stream 1.
* Compute \(A1 @ B\) in stream 0 which leads to full waves + partial wave at the end.

The partial waves in the compute don’t overlap with anything. Streaming Multiprocessors (SMs) sit idle each time.

{{< collapse summary="**Wave**" >}}
In GPU computing, work is distributed across Streaming Multiprocessors (SM) in waves. Consider for example, Hopper GPU (H100) has 132 SMs. A large multiplication is broken down into chunks and each wave uses all available SMs to process batches to chunk.

**Partial Wave:** When the number of thread blocks doesn't divide evenly, the final wave only uses some SMs (e.g., 76 out of 132) while the rest sit idle until that wave completes. For a large matmul with 10 full waves + 1 partial wave, this final partial wave is negligible (amortized across the whole operation). However, when decomposing into sub-matmuls, each sub-matmul has its own partial wave, multiplying the inefficiency.

{{</ collapse >}}

{{< figure align=center src="/images/twostream_asynctp.png" attr="Pytorch [forum](https://discuss.pytorch.org/t/distributed-w-torchtitan-introducing-async-tensor-parallelism-in-pytorch/209487 )">}}

A better approach is the alternating-stream approach shown above. Now, instead of having dedicated streams for compute and communication, there are two symmetric streams that swap roles. Let's compare what is happening in the two AsyncTP implementations above

In the diagram above,

* Launch computation for \(A0 @ B\) on stream 0.
* While it’s running, start fetching A1 in stream 1.
* When A0 reaches its final partial wave, stream 1 can already start computing the first blocks of \(A1 @ B\).

This alternating streams approach overlaps the computation and computation for the partial waves. The approach also relies on CUDA P2P mechanisms to avoid copying through host memory explained in the writeup. I have few questions that I haven't found a good explanation for

* Does above approach work only for All-gather followed by matmul setups?
* There is a question around the diagram which compares original matmul with naive implementation: why there is Send A0 in Stream 1 at all places?
* In the alternating stream approach what is Post A0 and why are there barrier for synchronization in Stream 1 for A3 when the computation is happening on the same stream (Stream 1)? For example, there's no such barrier in between Fetch A1 and it's computation.

### DeepSpeed AsyncTP

The [Domino paper](https://www.arxiv.org/abs/2409.15241) by the folks behind the DeepSpeed library uses a different approach to hide the communication latency. While PyTorch AsyncTP focuses on decomposing individual matmuls, Domino redesigns the entire tensor parallelism strategy through multi-dimensional tensor slicing.

It uses combination of row-wise splitting on input and column-wise splitting on weights to overlap computation and communication. This approach achieves 1.3x speedup for a Megatron-LLM training on DGX-H100 GPU.

{{< figure align=center src="/images/domino_asynctp.png" attr="Domino [paper](https://www.arxiv.org/abs/2409.15241)">}}

In the diagram above for the forward pass, the input is sliced into two partitions μ-batch 0 and μ-batch 1.
* First, compute self-attention on μ-batch 0
* Launch AllReduce(attn0) asynchronously (non-blocking)
* Immediately start self-attention on μ-batch 1 while AllReduce(attn0) is running
* The AllReduce(attn1) communication overlaps with layerNorm, residual, and dropout operations
* Similarly for MLP: compute MLP on μ-batch 0, launch AllReduce(MLP0) asynchronously
* Start MLP on μ-batch 1 immediately, overlapping with AllReduce(MLP0)
* AllReduce(MLP1) overlaps with μ-batch 0's computation in the next transformer block

By grouping operations across micro-batches (like layerNorm, residual, dropout), Domino creates "overlapping space" where communication from one micro-batch hides behind computation from another. This achieves both **intra-layer** (within the same layer) and **inter-layer** (across successive layers) overlapping.

## Sequence Parallelism

Tensor parallelism split the computation in the MLP and MHA blocks across the GPUs. The layers that are typically not split are normalization and dropout; although these layers are not compute-heavy, they require considerable activation memory. Sequence parallelism parallelizes those layers along the sequence dimension to reduce activation memory usage.

> There is sometimes confusion in literature (also noted in the HuggingFace playbook) where sequence parallelism techniques are referred to as enabler for the training longer sequences. To make terminology explicit, we'll call those techniques context parallelism, a topic for the next post.

{{< figure align=center src="/images/sequence_parallelism.png" attr="Adapted from Reducing Activation Recomputation in Large Transformer Models [paper](https://www.arxiv.org/abs/2205.05198)">}}

{{< collapse summary="**Deriving communication operations \(g\) and \(\bar{g}\)**" >}}

Let's look at how \(g\) and \(\bar{g}\) communication blocks can be derived for the MLP block.

The MLP block consists of the following

$$
\begin{aligned}
Y &= \text{LayerNorm}(X) \\
Z &= \text{GeLU}(YA) \\
W &= ZB \\
V &= \text{Dropout}(W)
\end{aligned}
$$

The input \(X\) is of shape `(b,s,h)` and \(A\) and \(B\) weight matrices are of size \(h \times 4h\) and \(4h \times h\) split column-wise and row-wise for tensor parallelism respectively.

{{< figure align=center src="/images/mlp_sp.png" attr="Reducing Activation Recomputation in Large Transformer Models [paper](https://www.arxiv.org/abs/2205.05198)">}}

Here, input to layer-norm is parallelized on the sequence dimension \(X = [X_{1}^{s}, X_{2}^{s}]\). The output from the layer will also be parallel along the sequence dimension \(Y = [Y_{1}^{s}, Y_{2}^{s}]\). From [MLP block](#multi-layer-perceptron) section on the tensor parallelism, it expects a column-wise sharding followed by row-wise sharding. To assemble entire input \(Y\), an all-gather operation has to be performed. Hence, \(g\) will be an all-gather communication operation for the forward pass.

The MLP produces outputs \(W_{1}\) and \(W_{2}\) on different devices. To feed the next layer - dropout in this case -, these outputs must be combined and redistributed along the sequence dimension. This is done using a reduce-scatter operation, which performs a reduction (sum) across devices and splits the result, ensuring each device has the correct portion of the output. This communication corresponds to \(\bar{g}\) in the forward pass.

{{</ collapse >}}


## Communication volume

Tensor parallelism requires four all-reduce operations in a single forward and backward pass for a single transformer block. 

Tensor parallelism along with sequence parallelism requires four all-gather and four reduce-scatter operations in a single forward and backward pass. 

The all-reduce operations - usually implemented as ring all-reduce - consists of two steps: a reduce-scatter followed by all-gather. The TP+SP strategy has the same volume as TP approach. SP does not introduce any communication overhead and helps reducing the activation memory further. 

## Wrap up

This post introduced two new degrees of parallelism: tensor parallelism and sequence parallelism. 
* Tensor parallelism splits computation by sharding weights or activations across GPUs.
* Sequence parallelism splits normalization and dropout along the sequence dimension to reduce activation memory.

 By effectively using both TP and SP together, the entire transformer block and its individual layers are partitioned such that all computations are efficiently distributed across multiple GPUs. TP and SP setup requires higher inter-GPU communication hence are optimal over GPUs connected with high speed connections like NVLink.

TP and SP are optimal for both the training and inference of LLMs as they reduce the activation memory and distribute the computation across devices. 