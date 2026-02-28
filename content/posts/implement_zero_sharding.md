---
author: [""]
title: "ZeRO Sharding Revisited - Implementations using PyTorch"
date: "2026-02-28"
tags: ["llm", "llm-training", "zero", "sharding", "pytorch"]
description: ""
summary: "Implement ZeRO sharding strategies using PyTorch"
ShowToc: true
ShowBreadCrumbs: true
---

The [previous blog](https://dudeperf3ct.github.io/posts/implement_data_parallelism/) on the implementation series explored distributed data parallelism (DDP). It started with a simple DDP setup and introduced complexities such as async communication and overlapping communication with computation to make training efficient. On top of the strategies that optimize compute, sharding is another strategy used with DDP to reduce memory consumption.

DDP approach usually involves replicating the model across all participating GPUs. This means that each GPU holds a complete copy of the model parameters, gradients, and optimizer states. As the model size increases, this replication can lead to significant memory consumption on each GPU, which can limit the maximum model size that can be trained or require more expensive hardware.

Sharding techniques like ZeRO (Zero Redundancy Optimizer) help reduce this memory consumption by partitioning the model states across the GPUs. Instead of each GPU holding a complete copy of the model, each GPU holds only a portion of the model states, which allows for training larger models without running out of memory.

In this post, we will look into implementing sharding techniques for a simple DDP setup for training an LLM across multiple GPUs using PyTorch. We will compare memory efficiency and trade-off for these techniques.

* Simple DDP (Baseline)
* Simple DDP + ZeRO-1/ZeRO-2/ZeRO-3
* PyTorch DDP + ZeRO-1/ZeRO-2/ZeRO-3

## Refresher on Sharding

> [!TIP]
> Here's the [blog](https://dudeperf3ct.github.io/posts/ultrascale_zero_deepspeed/) on sharding that introduces various sharding techniques applied in conjunction with data parallelism. 

Sharding techniques like Zero Redundancy Optimizer (ZeRO) help reduce the memory consumed for training. Typically, during training of any large model, memory is dominated by model states -- model parameters, gradients, and optimizer states. There are 3 ZeRO techniques:

* ZeRO-1 stage reduces memory by sharding only the optimizer states across multiple GPUs
* ZeRO-2 stage shards the gradient and optimizer states
* ZeRO-3 stage shards all the 3 model states -- model parameters, gradients and optimizer providing maximum memory reduction.

## Setup

> [!CODE]
> All the code snippets shown in this post are available at the github repo: [llm-parallelism-pytorch](https://github.com/dudeperf3ct/llm-parallelism-pytorch).

The same setup for model, data, profiling, and training loop is reused.

* Model: [`SmolLM2-360M-Instruct`](https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct) model 
* Data: [`Yelp Review`](https://huggingface.co/datasets/Yelp/yelp_review_full) dataset
* The code for data pipeline that takes care of tokenization and batching to create training and validation data loaders is the same.
* PyTorch profiling is used to record the memory and time spent in computation and communication time 
* Training loop is the same with each batch performing a training step of forward pass, calculating loss, backward pass and optimizer step 

## Baseline

Simple DDP is the DDP approach used to establish a baseline for the memory consumed and time taken for training across 10 epochs on a subset of the dataset. This baseline approach is exactly the same as the one introduced and implemented in the data parallelism writeup.

> [!IMPORTANT]
> Simple DDP can be replaced with optimized implementations such as DDP with backward hooks or asynchronous gradient synchronization introduced in the [data parallelism implementation](https://dudeperf3ct.github.io/posts/implement_data_parallelism/) blog. To keep the benchmark simple, I have used Simple DDP. Using the bucketing gradients variant of DDP requires significant changes to the ZeRO implementation discussed below.

PyTorch model is wrapped using a custom implementation that overrides two functions

* `sync_parameters`: It broadcasts the initial parameters from rank 0 to all other ranks. Ranks here refer to participating GPUs for the training job.
* `sync_gradients`: After performing forward pass and calculating loss on the local batch, the gradients across all GPUs are gathered and averaged. The averaged gradients are sent back to all GPUs. This ensures all the parameters are in sync for the next iteration across all the GPUs. 

The training setup looks like

* Each GPU takes a batch of different data
* It performs a training step that consists of forward pass, calculating loss, and backward pass
* All gradients across each GPU are gathered, averaged and distributed back
* Optimizer takes a step of adjusting the parameters

{{< collapse summary="**Implementation**" >}}

```python
import torch
import torch.distributed as dist
from utils.ddp_utils import get_dist_info

class SimpleDDP(torch.nn.Module):
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model
        self.rank, self.world_size, self.local_rank = get_dist_info()
        # broadcast parameters from rank 0 to all other ranks
        self.sync_parameters()

    def sync_parameters(self):
        """Broadcast parameters from rank 0 to all other ranks."""
        for param in self.model.parameters():
            # distributed comm: broadcast
            # broadcast parameter values across all ranks
            # to be same as that of rank 0
            dist.broadcast(tensor=param.data, src=0)

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def sync_gradients(self):
        """Average gradients across ranks."""
        # Synchronize gradients across all ranks
        for param in self.model.parameters():
            # distributed comm: All Reduce
            # To perform synchronization, we
            # first need to gather gradients from all ranks
            # sum all the gathered gradients
            # broadcast the summed results to all ranks
            # All this can be performed using single all_reduce operation
            if param.grad is not None:
                dist.all_reduce(tensor=param.grad, op=dist.ReduceOp.SUM)
                # Average the gradients by all ranks
                param.grad /= self.world_size
```
{{< /collapse >}}


{{< collapse summary="**Result**" >}}

{{< plotly file="static/images/ddp_sharding_plots/baseline.html" >}}

{{< /collapse >}}

Next, let's take a look at how sharding techniques might reduce memory consumption and what the trade-offs are.

## ZeRO-1

ZeRO-1 sharding technique shards only the optimizer states across all the GPUs used for training. 

The training setup looks almost similar to the Simple DDP wrapper until the optimizer step where

* **Optimizer step (sharded)**: Inside optimizer step, each GPU has a portion of optimizer states. It takes a step to update parameters belonging to only its portion.
* **Parameter sync**: Once the local optimizer step is performed, a `torch.dist.broadcast` operation is used to exchange updated parameters from the rank that contains the updated parameters to the rest of the GPUs. This way, all GPUs end up with identical model weights before the next iteration.

A custom wrapper is built around the optimizer that takes care of sharding optimizer states, taking an optimizer step, and synchronizing the updated parameters. `ZeroOneSharding` class wraps a PyTorch optimizer to implement ZeRO-1 style optimizer-state sharding and parameter synchronization. This is achieved through the following functions:

* In the `__init__`, get list of all parameters from optimizer's parameter groups. [PyTorch optimizer](https://github.com/pytorch/pytorch/blob/v2.10.0/torch/optim/optimizer.py#L399) maintains param groups, each with `"params"` and corresponding hyperparameters like learning rate, momentum.
* `_shard_optimizer_states`: Performs parameter-level sharding using these parameters where each GPU contains a fixed-slice subset of parameters. Each rank's optimizer owns a disjoint subset of parameters. If parameters are not divided equally among all GPUs, it might lead to imbalance. Thus in practice, tensor-level sharding is preferred. 
* `step`: This function mirrors the wrapped original optimizer's `step` function but runs it only on the local parameter subset. Once parameters are updated locally on that rank, a `broadcast` distributed operation takes care of exchanging the parameters belonging only to a particular rank to all other ranks. 
* `zero_grad`: This function mimics the original optimizer's `zero_grad` function to reset the gradients for all model parameters.
 
{{< collapse summary="**Implementation**" >}}

```python
from collections.abc import Callable
from typing import Any

import torch
import torch.distributed as dist

from utils.ddp_utils import get_dist_info


class ZeroOneSharding:
    """Zero-1 sharding optimizer.

    In Zero-1 sharding, optimizer states are sharded across available GPUs.
    Model parameters remain replicated on all ranks.
    During the optimizer step, each GPU only updates its local parameters and their corresponding states.
    After the local optimizer step, we need to synchronize the updated parameters across all ranks
    to ensure that all ranks have the same parameter values for the next iteration.

    Note:
        This reference implementation uses parameter-level partitioning: whole ``Parameter`` objects
        are assigned to owners. It does not split individual tensors across ranks.
    """

    def __init__(self, optimizer: torch.optim.Optimizer):
        self.optimizer = optimizer
        self.rank, self.world_size, self.local_rank = get_dist_info()
        # Get parameter groups stored in the optimizer
        # Parameter group contains the parameters and
        # their corresponding hyperparameters (like learning rate, momentum, etc.)
        self.original_param_groups = self.optimizer.param_groups
        # Get list of all parameters
        self._all_params = [
            parameter
            for param_group in self.original_param_groups
            for parameter in param_group["params"]
        ]
        # Shard optimizer states across ranks
        self._shard_optimizer_states()

    @staticmethod
    def _build_shard_bounds(total: int, world_size: int) -> list[tuple[int, int]]:
        """Build contiguous [start, end) shard bounds for each rank."""
        base, remainder = total // world_size, total % world_size
        shard_sizes = [base + int(rank < remainder) for rank in range(world_size)]

        bounds: list[tuple[int, int]] = []
        start = 0
        for size in shard_sizes:
            end = start + size
            bounds.append((start, end))
            start = end
        return bounds

    def _shard_optimizer_states(self):
        """Shard optimizer states across ranks.

        Each rank keeps only a portion of the optimizer states, determined by the rank and world size.
        """
        total_params = len(self._all_params)
        shard_bounds = self._build_shard_bounds(total=total_params, world_size=self.world_size)

        start_idx, end_idx = shard_bounds[self.rank]
        self.local_param_indices = list(range(start_idx, end_idx))
        self.local_params = [self._all_params[i] for i in self.local_param_indices]
        local_param_ids = {id(parameter) for parameter in self.local_params}
        self._param_owner = {}
        # Build a mapping from parameter ID to the rank that owns it, based on the shard bounds
        for owner_rank, (owner_start_idx, owner_end_idx) in enumerate(shard_bounds):
            for parameter in self._all_params[owner_start_idx:owner_end_idx]:
                self._param_owner[id(parameter)] = owner_rank

        # Keep only the local parameters in the optimizer's param_groups for the current rank
        for group in self.optimizer.param_groups:
            group["params"] = [
                parameter for parameter in group["params"] if id(parameter) in local_param_ids
            ]

    def step(self, closure: Callable[[], float] | None = None, **kwargs: Any):
        """Perform single optimizer step and sync parameters across all ranks.

        Parameter update:
            In this implementation, we use per-parameter ``broadcast`` from each
            owner rank to synchronize updated parameters after the local optimizer step.

            ZeRO Stage-1 is often described as an all-gather style synchronization
            of updated parameter partitions. With this code's parameter-level ownership
            and no bucketization, repeated broadcasts are a simpler equivalent.

        Arguments:
            closure (Callable): a closure that re-evaluates the model and
                returns the loss; optional for most optimizers.
        """
        # Perform optimization only on the local parameters
        self.optimizer.step(closure=closure, **kwargs)
        # After the local optimizer step, synchronize the updated parameters across all ranks
        # This ensures that all ranks have the same parameter values for the next iteration
        with torch.profiler.record_function("zero1_param_broadcast"):
            for param in self._all_params:
                # Broadcasting each parameter separately is simple but not communication-optimal.
                # Broadcast the updated parameter values from the rank that owns them to all other ranks
                owner_rank = self._param_owner[id(param)]
                dist.broadcast(tensor=param.data, src=owner_rank)

    def zero_grad(self, set_to_none: bool = True):
        # Clear gradients for all model params, not only the local optimizer shard.
        for parameter in self._all_params:
            parameter.grad = None
```
{{< /collapse >}}


{{< collapse summary="**Result**" >}}
{{< plotly file="static/images/ddp_sharding_plots/zero1.html" >}}
{{< /collapse >}}

## ZeRO-2

ZeRO-2 builds on ZeRO-1 by sharding gradients in addition to optimizer states. 

The training loop for ZeRO-2 sharding is similar to ZeRO-1 with respect to optimizer states, with the main change being how gradients are reduced and stored. 

* Forward, loss, and backward compute are the same as DDP/ZeRO-1; each rank initially computes grads for all parameters, but after synchronization it retains only its owned gradient shard.
* **Gradients sync (sharded)**: In DDP/ZeRO-1, `all_reduce` is used such that every rank ends up with the same reduced gradient for every parameter. In ZeRO-2, gradients are partitioned: `reduce_scatter` is used such that only the owning rank receives the reduced gradient for its shard.
* **Optimizer step (sharded) and parameter synchronization**: As in ZeRO-1, each rank updates only its owned parameters using its local optimizer-state shard (and now also its local gradient shard). After the local update, updated parameters are synchronized across ranks so all ranks begin the next iteration with identical model weights.

Similar to the implementation `ZeroOneSharding`, `Zero2Sharding` creates a custom wrapper around PyTorch's optimizer to implement ZeRO-2 style sharding. All functions are the same, with the addition of a new function that implements how gradients are sharded.

* `shard_gradients`: In the Simple DDP approach, gradient synchronization uses `all_reduce` to gather gradients from all ranks, average them, and distribute them back. Since gradients in ZeRO-2 are sharded as well, the averaged gradients should instead be passed to the GPU that contains each sharded portion.

Because gradient synchronization is handled in `SimpleDDP.sync_gradients()` in our baseline, `SimpleDDP` is extended to support a pluggable gradient sync function:

* `_gradient_sync_fn` selects the gradient synchronization strategy.
* `sync_gradients` uses the default all_reduce behavior unless `_gradient_sync_fn` is set, in which case it calls the custom `shard_gradients()` implementation.


{{< collapse summary="**Implementation**" >}}

Simple DDP modification

```python

import torch
import torch.distributed as dist

from utils.ddp_utils import get_dist_info


class SimpleDDP(torch.nn.Module):
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model
        self.rank, self.world_size, self.local_rank = get_dist_info()
        # Options for custom gradient synchronization function,
        # used in ZeRO2 and ZeRO3 where gradients are sharded across ranks
        self._gradient_sync_fn = None
        # broadcast parameters from rank 0 to all other ranks
        self.sync_parameters()

    def set_gradient_sync_fn(self, gradient_sync_fn):
        """Set a custom gradient synchronization function."""
        self._gradient_sync_fn = gradient_sync_fn

    def sync_parameters(self):
        """Broadcast parameters from rank 0 to all other ranks."""
        for param in self.model.parameters():
            # distributed comm: broadcast
            # broadcast parameter values across all ranks
            # to be same as that of rank 0
            dist.broadcast(tensor=param.data, src=0)

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def sync_gradients(self):
        """Average gradients across ranks."""
        # These are used to override the default gradient sync behaviour
        # For example, in ZeRO2 and ZeRO-3, the gradients are sharded across ranks
        # and the synchronization is performed using reduce-scatter semantics.
        if self._gradient_sync_fn is not None:
            self._gradient_sync_fn()
            return

        # Synchronize gradients across all ranks
        for param in self.model.parameters():
            # distributed comm: All Reduce
            # To perform synchronization, we
            # first need to gather gradients from all ranks
            # sum all the gathered gradients
            # broadcast the summed results to all ranks
            # All this can be performed using single all_reduce operation
            if param.grad is not None:
                dist.all_reduce(tensor=param.grad, op=dist.ReduceOp.SUM)
                # Average the gradients by all ranks
                param.grad /= self.world_size
```

ZeRO-2 sharding implementation

```python
from collections.abc import Callable
from typing import Any

import torch
import torch.distributed as dist

from utils.ddp_utils import get_dist_info


class Zero2Sharding:
    """Zero-2 sharding optimizer.

    In Zero-2 sharding, both model gradients and optimizer states are sharded across available GPUs.
    Each GPU handles a portion of model gradients and their corresponding optimizer states.

    During the backward pass, in a DDP setup gradients are synchronized using all_reduce
    but in this case since gradients are sharded, we only need to synchronize the local gradients on each GPU.
    To update only the local gradients, reduce_scatter can be used to sum the gradients across all ranks
    and scatter the results back to the local shards.

    Note:
        The behaviour for optimizer states is the same as Zero-1 sharding
        where each GPU only updates its local parameters and their corresponding states.

        This reference implementation uses parameter-level partitioning: whole ``Parameter`` objects
        are assigned to owners. It does not split individual tensors across ranks.
    """

    def __init__(self, optimizer: torch.optim.Optimizer):
        self.optimizer = optimizer
        self.rank, self.world_size, self.local_rank = get_dist_info()
        # Get parameter groups stored in the optimizer
        # Parameter group contains the parameters and
        # their corresponding hyperparameters (like learning rate, momentum, etc.)
        self.original_param_groups = self.optimizer.param_groups
        # Get list of all parameters
        self._all_params = [
            parameter
            for param_group in self.original_param_groups
            for parameter in param_group["params"]
        ]
        # Shard optimizer states across ranks
        self._shard_optimizer_states()

    @staticmethod
    def _build_shard_bounds(total: int, world_size: int) -> list[tuple[int, int]]:
        """Build contiguous [start, end) shard bounds for each rank."""
        base, remainder = total // world_size, total % world_size
        shard_sizes = [base + int(rank < remainder) for rank in range(world_size)]

        bounds: list[tuple[int, int]] = []
        start = 0
        for size in shard_sizes:
            end = start + size
            bounds.append((start, end))
            start = end
        return bounds

    def _shard_optimizer_states(self):
        """Shard optimizer states across ranks.

        Each rank keeps only a portion of the optimizer states, determined by the rank and world size.
        """
        total_params = len(self._all_params)
        shard_bounds = self._build_shard_bounds(total=total_params, world_size=self.world_size)

        start_idx, end_idx = shard_bounds[self.rank]
        self.local_param_indices = list(range(start_idx, end_idx))
        self.local_params = [self._all_params[i] for i in self.local_param_indices]
        local_param_ids = {id(parameter) for parameter in self.local_params}
        self._param_owner = {}
        # Build a mapping from parameter ID to the rank that owns it, based on the shard bounds
        for owner_rank, (owner_start_idx, owner_end_idx) in enumerate(shard_bounds):
            for parameter in self._all_params[owner_start_idx:owner_end_idx]:
                self._param_owner[id(parameter)] = owner_rank

        # Keep only the local parameters in the optimizer's param_groups for the current rank
        for group in self.optimizer.param_groups:
            group["params"] = [
                parameter for parameter in group["params"] if id(parameter) in local_param_ids
            ]

    def shard_gradients(self):
        """Shard gradients across ranks using reduce-scatter.

        This function is used to override the default gradient sync behaviour in DDP
        where all gradients are synchronized across ranks using all_reduce.

        Gradient update:
            We use zero tensors as placeholders when a local grad is missing so that
            all ranks still execute the same collectives in the same order. This is
            not the same as a true `None` grad semantically: `None` means "unused this
            step", while zero means "used, but gradient value is 0".

            ``reduce_scatter`` is used to sum gradients and return owner-local shards.
            Here it is done per parameter with explicit zero placeholders for non-owners,
            which is simple but less efficient than bucketized implementations.
        """
        with torch.profiler.record_function("zero2_reduce_scatter"):
            for parameter in self._all_params:
                owner_rank = self._param_owner[id(parameter)]
                # Not optimal to zero out the gradients on non-owner ranks,
                # but it keeps the implementation simple as zero tensor instead of None
                grad = (
                    parameter.grad.detach().contiguous()
                    if parameter.grad is not None
                    else torch.zeros_like(parameter.data)
                )
                output = torch.empty_like(grad)
                input_list = []
                for rank in range(self.world_size):
                    if rank == owner_rank:
                        input_list.append(grad)
                    else:
                        input_list.append(torch.zeros_like(grad))

                dist.reduce_scatter(output, input_list, op=dist.ReduceOp.SUM)
                if self.rank == owner_rank:
                    output /= self.world_size
                    parameter.grad = output
                else:
                    parameter.grad = None

    def step(self, closure: Callable[[], float] | None = None, **kwargs: Any):
        """Perform single optimizer step and sync parameters across all ranks.

        Parameter update:
            In this implementation, we use per-parameter ``broadcast`` from each
            owner rank to synchronize updated parameters after the local optimizer step.

            ZeRO Stage-2 is often described as synchronizing updated parameter partitions
            with all-gather style communication. With this code's parameter-level ownership
            and no bucketization, repeated broadcasts are a simpler equivalent.

        Arguments:
            closure (Callable): a closure that re-evaluates the model and
                returns the loss; optional for most optimizers.
        """
        # Perform optimization only on the local parameters
        self.optimizer.step(closure=closure, **kwargs)
        # After the local optimizer step, synchronize the updated parameters across all ranks
        # This ensures that all ranks have the same parameter values for the next iteration
        with torch.profiler.record_function("zero2_param_broadcast"):
            for param in self._all_params:
                # Broadcasting each parameter separately is simple but not communication-optimal.
                # Broadcast the updated parameter values from the rank that owns them to all other ranks
                owner_rank = self._param_owner[id(param)]
                dist.broadcast(tensor=param.data, src=owner_rank)

    def zero_grad(self, set_to_none: bool = True):
        # Clear gradients for all model params, not only the local optimizer shard.
        for parameter in self._all_params:
            parameter.grad = None
```

{{< /collapse >}}


{{< collapse summary="**Result**" >}}

{{< plotly file="static/images/ddp_sharding_plots/zero2.html" >}}

{{< /collapse >}}


## ZeRO-3

ZeRO-3 sharding shards all three model states -- model parameters, gradients and optimizer states across GPUs. It builds on ZeRO-2 with support for sharding the model parameters as well.

The training loop has overlap with ZeRO-2 except for how the forward and backward pass use the sharded model parameters,
* **Forward pass**: Since model parameters are sharded, before performing forward pass, these parameters must be gathered before the forward pass. Once forward pass is completed, the gathered parameters can be released freeing the memory
* **Backward pass**: Similar to forward pass, the backward needs to gather the sharded parameters to perform backward pass. It releases the memory taken by gathered parameters once computation is complete.
* The remaining steps, sharded gradients and sharded optimizer step, are the same as in ZeRO-2. Here, syncing of parameters is not required because parameters remain sharded.

`Zero3Sharding` builds on `Zero2Sharding` implementation by adding new functions to perform model parameter sharding. In particular:

* `gather_full_parameters`: Gather parameters from the other ranks so each rank temporarily materializes the full parameters for forward and backward computation 
* `reshard_model_parameters`: Free memory by discarding the non-owned parameter storage on each rank
* `shard_gradients`: performs sharded gradient synchronization using reduce-scatter similar to ZeRO-2 and then reshards parameters to free memory.
* `step`: runs the optimizer update only on locally owned parameters; no post-step parameter broadcast is needed since parameters remain sharded.

In addition to plugging the `sharded_gradient` function as the gradient synchronization strategy for the Simple DDP implementation, a `_pre_forward_fn` function takes care of gathering full parameters before performing the forward pass.

{{< collapse summary="**Implementation**" >}}

Simple DDP with modifications

```python
import torch
import torch.distributed as dist

from utils.ddp_utils import get_dist_info


class SimpleDDP(torch.nn.Module):
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model
        self.rank, self.world_size, self.local_rank = get_dist_info()
        # Options for custom gradient synchronization function,
        # used in ZeRO2 and ZeRO3 where gradients are sharded across ranks
        self._gradient_sync_fn = None
        # Used in Zero3 to materialize full parameters before forward via owner broadcast
        self._pre_forward_fn = None
        # broadcast parameters from rank 0 to all other ranks
        self.sync_parameters()

    def set_gradient_sync_fn(self, gradient_sync_fn):
        """Set a custom gradient synchronization function."""
        self._gradient_sync_fn = gradient_sync_fn

    def set_pre_forward_fn(self, pre_forward_fn):
        """Set a callback to run before each forward pass."""
        self._pre_forward_fn = pre_forward_fn

    def sync_parameters(self):
        """Broadcast parameters from rank 0 to all other ranks."""
        for param in self.model.parameters():
            # distributed comm: broadcast
            # broadcast parameter values across all ranks
            # to be same as that of rank 0
            dist.broadcast(tensor=param.data, src=0)

    def forward(self, *args, **kwargs):
        # These are used to override the default forward behaviour in DDP
        # For example, in ZeRO3, the full parameters are materialized on each rank
        # before forward pass
        if self._pre_forward_fn is not None:
            self._pre_forward_fn()
        return self.model(*args, **kwargs)

    def sync_gradients(self):
        """Average gradients across ranks."""
        # These are used to override the default gradient sync behaviour
        # For example, in ZeRO2 and ZeRO-3, the gradients are sharded across ranks
        # and the synchronization is performed using reduce-scatter semantics.
        if self._gradient_sync_fn is not None:
            self._gradient_sync_fn()
            return

        # Synchronize gradients across all ranks
        for param in self.model.parameters():
            # distributed comm: All Reduce
            # To perform synchronization, we
            # first need to gather gradients from all ranks
            # sum all the gathered gradients
            # broadcast the summed results to all ranks
            # All this can be performed using single all_reduce operation
            if param.grad is not None:
                dist.all_reduce(tensor=param.grad, op=dist.ReduceOp.SUM)
                # Average the gradients by all ranks
                param.grad /= self.world_size
```

Zero-3 implementation

```python
from collections.abc import Callable
from typing import Any

import torch
import torch.distributed as dist

from utils.ddp_utils import get_dist_info


class Zero3Sharding:
    """Zero-3 sharding optimizer.

    In Zero-3 sharding, parameters, gradients, and optimizer states are sharded
    across available GPUs.
    - Parameters are sharded by ownership (non-owner ranks keep empty tensors).
      This is parameter-level partitioning: whole ``Parameter`` objects are assigned by index.
      It does not split each parameter tensor across ranks.
    - Full parameters are materialized before forward via owner broadcast.
    - Gradients are sharded using reduce-scatter similar to Zero2.
    - Full parameters are resharded after backward to free memory.
    - Optimizer step updates only local-owner parameters/states similar to Zero1 and Zero2.

    Note:
        The behaviour for optimizer states is the same as Zero-1 and Zero-2 sharding
        where each GPU only updates its local parameters and their corresponding states.

        Native PyTorch FSDP/FSDP2 uses tensor-level (intra-parameter) sharding semantics,
        while this class is intentionally a simpler parameter-ownership reference.
    """

    def __init__(self, optimizer: torch.optim.Optimizer):
        self.optimizer = optimizer
        self.rank, self.world_size, self.local_rank = get_dist_info()
        # Get parameter groups stored in the optimizer
        # Parameter group contains the parameters and
        # their corresponding hyperparameters (like learning rate, momentum, etc.)
        self.original_param_groups = self.optimizer.param_groups
        # Get list of all parameters
        self._all_params = [
            parameter
            for param_group in self.original_param_groups
            for parameter in param_group["params"]
        ]
        # Keep track of parameter shapes for reshaping during gather/scatter steps
        self._param_shapes = {
            id(parameter): tuple(parameter.shape) for parameter in self._all_params
        }
        # Shard optimizer states across ranks
        self._shard_optimizer_states()
        # Shard model parameters across ranks
        # Free memory of non-owner parameters by keeping empty tensors,
        # and only keep local shard of parameters on each rank
        self.reshard_model_parameters()

    @staticmethod
    def _build_shard_bounds(total: int, world_size: int) -> list[tuple[int, int]]:
        """Build contiguous [start, end) shard bounds for each rank."""
        base, remainder = total // world_size, total % world_size
        shard_sizes = [base + int(rank < remainder) for rank in range(world_size)]

        bounds: list[tuple[int, int]] = []
        start = 0
        for size in shard_sizes:
            end = start + size
            bounds.append((start, end))
            start = end
        return bounds

    def _shard_optimizer_states(self):
        """Shard optimizer states across ranks.

        Each rank keeps only a portion of the optimizer states, determined by the rank and world size.
        """
        total_params = len(self._all_params)
        shard_bounds = self._build_shard_bounds(total=total_params, world_size=self.world_size)

        start_idx, end_idx = shard_bounds[self.rank]
        self.local_param_indices = list(range(start_idx, end_idx))
        self.local_params = [self._all_params[i] for i in self.local_param_indices]
        local_param_ids = {id(parameter) for parameter in self.local_params}
        self._param_owner = {}
        # Build a mapping from parameter ID to the rank that owns it, based on the shard bounds
        for owner_rank, (owner_start_idx, owner_end_idx) in enumerate(shard_bounds):
            for parameter in self._all_params[owner_start_idx:owner_end_idx]:
                self._param_owner[id(parameter)] = owner_rank

        # Keep only the local parameters in the optimizer's param_groups for the current rank
        for group in self.optimizer.param_groups:
            group["params"] = [
                parameter for parameter in group["params"] if id(parameter) in local_param_ids
            ]

    def gather_full_parameters(self):
        """Materialize full parameters on all ranks before forward.

        Parameter gather:
            Each rank starts with only its local shard of parameters.
            We broadcast the full parameters from their owner ranks to all other ranks before forward pass,
            so that the full model is materialized on each rank for the forward and backward computations.

        Note:
            For this parameter-ownership layout, ``broadcast`` is the natural primitive because each
            full parameter lives on one owner rank. If each rank held tensor slices of every parameter
            (tensor-level partitioning), ``all_gather`` of slices would be the natural primitive.
        """
        with torch.profiler.record_function("zero3_param_gather"):
            for parameter in self._all_params:
                owner_rank = self._param_owner[id(parameter)]
                full_shape = self._param_shapes[id(parameter)]
                if self.rank != owner_rank and tuple(parameter.data.shape) != full_shape:
                    parameter.data = torch.empty(
                        full_shape, device=parameter.device, dtype=parameter.dtype
                    )
                dist.broadcast(tensor=parameter.data, src=owner_rank)

    def reshard_model_parameters(self):
        """Keep only owner parameters materialized; free non-owner parameter storage.

        This is used to free the memory of non-owner parameters after forward and backward communication steps.
        """
        for parameter in self._all_params:
            owner_rank = self._param_owner[id(parameter)]
            if self.rank != owner_rank:
                parameter.data = torch.empty(0, device=parameter.device, dtype=parameter.dtype)
                parameter.grad = None

    def shard_gradients(self):
        """Shard gradients across ranks using reduce-scatter.

        This function is used to override the default gradient sync behaviour in DDP
        where all gradients are synchronized across ranks using all_reduce.

        Gradient update:
            We use zero tensors as placeholders when a local grad is missing so that
            all ranks still execute the same collectives in the same order. This is
            not the same as a true `None` grad semantically: `None` means "unused this
            step", while zero means "used, but gradient value is 0".

            Reduce scatter is used to sum the gradients across all ranks and scatter the results
            back to the local shards.
            After backward communication, we free the memory of non-owner parameters and gradients.
        """
        with torch.profiler.record_function("zero3_reduce_scatter"):
            for parameter in self._all_params:
                owner_rank = self._param_owner[id(parameter)]
                grad = (
                    parameter.grad.detach().contiguous()
                    if parameter.grad is not None
                    else torch.zeros(
                        self._param_shapes[id(parameter)],
                        device=parameter.device,
                        dtype=parameter.dtype,
                    )
                )
                output = torch.empty_like(grad)
                input_list = []
                for rank in range(self.world_size):
                    if rank == owner_rank:
                        input_list.append(grad)
                    else:
                        input_list.append(torch.zeros_like(grad))

                dist.reduce_scatter(output, input_list, op=dist.ReduceOp.SUM)
                if self.rank == owner_rank:
                    output /= self.world_size
                    parameter.grad = output
                else:
                    parameter.grad = None

        # After backward communication, free the memory of non-owner parameters and gradients
        self.reshard_model_parameters()

    def step(self, closure: Callable[[], float] | None = None, **kwargs: Any):
        """Perform optimizer step on local shards only."""
        self.optimizer.step(closure=closure, **kwargs)

    def zero_grad(self, set_to_none: bool = True):
        for parameter in self._all_params:
            parameter.grad = None
```

{{< /collapse >}}


{{< collapse summary="**Result**" >}}

{{< plotly file="static/images/ddp_sharding_plots/zero3.html" >}}

{{< /collapse >}}


## PyTorch

The sharding experiments are repeated using PyTorch's implementation of [`ZeroRedundancyOptimizer`](https://docs.pytorch.org/docs/main/distributed.optim.html#torch.distributed.optim.ZeroRedundancyOptimizer). Equivalent components to the Simple DDP wrapper are PyTorch's [FSDP2](https://docs.pytorch.org/tutorials/intermediate/FSDP_tutorial.html) and [DDP](https://docs.pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html).


### PyTorch ZeRO-1

To implement ZeRO-1 using PyTorch, the model is wrapped using `DDP` and the optimizer is wrapped using `ZeroRedundancyOptimizer`.

```python
model = DDP(model, device_ids=[local_rank], output_device=local_rank)
optim = ZeroRedundancyOptimizer(model.parameters(), optimizer_class=torch.optim.AdamW, lr=5e-5, overlap_with_ddp=False)
```

{{< collapse summary="**Result**" >}}
{{< plotly file="static/images/ddp_sharding_plots/pt_zero1.html" >}}
{{< /collapse >}}

### PyTorch ZeRO-2

To implement ZeRO-2 using PyTorch, the model is wrapped using `FSDP` and the optimizer is initialized after wrapping the FSDP model.

```python
def wrap_with_fsdp2(model, reshard_after_forward: bool):
    """Apply FSDP2 fully_shard per decoder block, then on the full module."""
    for layer in model.model.layers:  # decoder blocks
        fully_shard(layer, reshard_after_forward=reshard_after_forward)
    return fully_shard(model, reshard_after_forward=reshard_after_forward)
```

The important parameter here is `reshard_after_forward` which is set to `False` for ZeRO-2 and `True` for ZeRO-3. Setting `reshard_after_forward=False` means that FSDP2 will keep the full parameters in memory after forward pass, which is required for ZeRO-2 where parameters are not sharded. 

```python
model = wrap_with_fsdp2(model, reshard_after_forward=False)
optim = torch.optim.AdamW(model.parameters(), lr=5e-5)
```

{{< collapse summary="**Result**" >}}
{{< plotly file="static/images/ddp_sharding_plots/pt_zero2.html" >}}
{{< /collapse >}}

### PyTorch ZeRO-3

Setting `reshard_after_forward=True` means that FSDP2 will free the memory of non-owner parameters after forward pass, which is required for ZeRO-3 where parameters are sharded.

```python
model = wrap_with_fsdp2(model, reshard_after_forward=True)
optim = torch.optim.AdamW(model.parameters(), lr=5e-5)
```

FSDP2 shards parameters as [DTensor](https://docs.pytorch.org/docs/stable/distributed.tensor.html) performing tensor-level sharding. This means for example if we have 2 layer model across 2 GPUs, each GPU will have half of the parameters for each layer. This is different from our reference implementation where we have parameter-level sharding.


{{< collapse summary="**Result**" >}}
{{< plotly file="static/images/ddp_sharding_plots/pt_zero3.html" >}}
{{< /collapse >}}


## Results

Here's a comparison of memory consumption across model states for baseline vs all ZeRO stages across 2 GPUs.

| Mode | Model MB | Grad MB | Optim MB | Total state MB | vs baseline |
|---|---:|---:|---:|---:|---:|
| baseline | 1380.26 | 1380.26 | 2760.51 | 5521.03 | 1.000 |
| zero1 | 1380.26 | 1380.26 | 1380.26 | 4140.78 | 0.750 |
| zero2 | 1380.26 | 690.12 | 1380.26 | 3450.65 | 0.625 |
| zero3 | 690.12 | 690.12 | 1380.26 | 2760.51 | 0.500 |
| pytorch_zero2 | 690.13 | 690.13 | 1380.26 | 2760.52 | 0.500 |
| pytorch_zero3 | 690.13 | 690.13 | 1380.26 | 2760.52 | 0.500 |

The memory is reduced as expected according to the [ZeRO paper](https://arxiv.org/abs/1910.02054),

- Stage 1: optimizer states shard -> ~75% of baseline model-state memory.
- Stage 2: optimizer + gradients shard -> ~62.5%
- Stage 3: optimizer + gradients + parameters shard -> ~50%

Runtime tradeoff (avg batch time):
  - faster: `pytorch_zero1` (~1.701 s), `baseline`/`zero1` (~1.785 s)
  - slower: `zero2` (~2.321 s), `zero3` (~2.458 s), `pytorch_zero3` (~2.553 s)

ZeRO stage-2 and stage-3 allow lower memory consumption at the expense of slower training. The runtime overhead of ZeRO-2 and ZeRO-3 is largely due to the communication cost of synchronizing gradients and parameters across ranks. ZeRO-3 has higher overhead than ZeRO-2 because it also needs to gather and reshard parameters in addition to sharding gradients.