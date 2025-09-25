---
author: [""]
title: "Pydata MCR talk on training LLMs"
date: 2025-09-25
summary: "My talk on training LLMs at Pydata MCR"
description: ""
tags: ["pydata-mcr", "llm-training", "llm"]
ShowToc: false
ShowBreadCrumbs: true
---

I ~will be giving~ gave a talk at [Pydata MCR](https://www.linkedin.com/company/pydatamcr/) on How to train your LLMs?.

The talk ~will cover~ covered parallelism strategies and memory optimization techniques for scaling the training of large language models. Here's a short summary of the key topics,

* [Training on single GPU](https://dudeperf3ct.github.io/posts/ultrascale_one_gpu/) : This shows a breakdown of memory consumption of components such as parameters, gradients, optimiser states and activation memory. These are stored during training.
* [Activation Recomputation](https://dudeperf3ct.github.io/posts/ultrascale_one_gpu/#activation-recomputation): A memory optimisation technique for activation memory.
* [Gradient Accumulation](https://dudeperf3ct.github.io/posts/ultrascale_one_gpu/#gradient-accumulation): Gradient accumulation helps train on larger batches without using extra memory.
* [Data parallelism](https://dudeperf3ct.github.io/posts/ultrascale_data_parallelism/): First and simplest degree of parallelism strategy. It speeds up training throughput by using multiple GPUs for training but does not reduce per-GPU memory requirements.
* [ZeRO Sharding](https://dudeperf3ct.github.io/posts/ultrascale_zero_deepspeed/): Shards parameters, gradients, and optimizer states across GPUs, dramatically lowering per-GPU model state memory.

Slides from the talk are available here:

{{< gslides src="https://docs.google.com/presentation/d/e/2PACX-1vSPjnmd-Jyp10DJT6_m5N9bglP2zb1SvuqFK0UmtByLvkzJCcApieNe9rgvwg6ZJK0tSpVxaxs-d2dI/embed?start=false&loop=false&delayms=3000" >}}

