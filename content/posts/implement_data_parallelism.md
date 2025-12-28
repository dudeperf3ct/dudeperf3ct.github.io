---
author: [""]
title: "Data Parallelism Revisited - Implementations using PyTorch"
date: "2025-12-28"
tags: ["llm", "llm-training", "data-parallelism", "pytorch"]
description: ""
summary: "Implement various data parallelism strategies using PyTorch"
ShowToc: true
ShowBreadCrumbs: true
---

We have looked at data parallelism - one of the simplest parallelism techniques - previously. It enables scaling the training on LLMs across multiple GPUs by distributing data.

In this post, we look at implementing different techniques for training an LLM across multiple GPUs using PyTorch.
* Simple DDP
* Simple DDP with gradient accumulation
* Using backward hooks to overlap gradient synchronization
* Asynchronous gradient synchronization
* Bucketing gradients

## Refresher on DDP

> [!TIP]
> Here's the [blog](https://dudeperf3ct.github.io/posts/ultrascale_data_parallelism/) on data parallelism that outlines various data parallelism techniques that we will be implementing in following section. 

In Distributed Data Parallelism (DDP), the data is distributed across multiple GPUs. Each GPU performs the training using a distinct batch of data. A synchronization step is required before the optimizer step that gathers all gradients from all GPUs, averages the gradients and distributes the same gradient across all the GPUs.

The figure below shows how the data parallelism works across 2 GPUs.

{{< figure align=center src="/images/dp.png" attr="Data parallelism across 2 GPUs">}}

## Setup

> [!CODE]
> All the code snippets shown in this post are available at the github repo: [llm-parallelism-pytorch](https://github.com/dudeperf3ct/llm-parallelism-pytorch).

Before we start training using various DDP strategies, we need to set up the dataset and model. The [`Yelp Review`](https://huggingface.co/datasets/Yelp/yelp_review_full) dataset is used to train a [`SmolLM2-360M-Instruct`](https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct) model from Hugging Face model hub. The data pipeline takes care of tokenizing the dataset and creating data loaders for training and validation.

{{< collapse summary="**Data pipeline**" >}}

The [data pipeline](https://github.com/dudeperf3ct/llm-parallelism-pytorch/blob/main/data.py) consists of following steps:

1. Download the yelp review dataset from Hugging Face hub.
2. Select a small subset of the dataset for quick experiments.
3. Tokenize the dataset using the tokenizer from SmolLM2-360M-Instruct model.
4. Split the dataset into training and evaluation sets.
5. Create data loaders with `DistributedSampler` for distributed training.

```python
def prepare_data(batch_size: int, rank: int, world_size: int):
    """Prepare the dataset for training and evaluation.

    Returns:
        train_loader: DataLoader for training dataset.
        eval_loader: DataLoader for evaluation dataset.
    """
    # Download raw dataset
    raw_dataset = get_dataset()
    # For quick experiments, sample a subset of dataset
    raw_dataset["train"] = raw_dataset["train"].shuffle(seed=42).select(range(32))
    raw_dataset["test"] = raw_dataset["test"].shuffle(seed=42).select(range(16))
    tokenized_dataset = tokenize_data(raw_dataset)
    train_ds, eval_ds = split_dataset(tokenized_dataset)

    if rank == 0:
        print(
            f"Dataset sizes -> train: {len(train_ds)} samples, "
            f"eval: {len(eval_ds)} samples (world size={world_size})"
        )
    # Create dataloaders using distributed samplers
    collator = get_data_collator()
    num_workers = min(8, os.cpu_count() // max(1, world_size))
    use_workers = num_workers > 0
    train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True)
    eval_sampler = DistributedSampler(eval_ds, num_replicas=world_size, rank=rank, shuffle=False)
    train_loader = DataLoader(
        train_ds,
        shuffle=False,
        batch_size=batch_size,
        sampler=train_sampler,
        pin_memory=True,
        collate_fn=collator,
        num_workers=num_workers,
        persistent_workers=use_workers,
        prefetch_factor=2 if use_workers else None,
    )
    eval_loader = DataLoader(
        eval_ds,
        batch_size=batch_size,
        sampler=eval_sampler,
        pin_memory=True,
        collate_fn=collator,
        num_workers=num_workers,
        persistent_workers=use_workers,
        prefetch_factor=2 if use_workers else None,
    )
    return train_loader, eval_loader
```

{{< /collapse >}}

Next, once the data pipeline is ready, we set up the model and optimizer. We load the pre-trained SmolLM2-360M-Instruct model and move it to the appropriate device. AdamW optimizer with fixed learning rate is used for training.

{{< collapse summary="**Training loop**" >}}

A typical [training loop](https://github.com/dudeperf3ct/llm-parallelism-pytorch/blob/main/utils/train_utils.py) consists of

1. Iterate over training data loader to get all batches of the data
2. For each batch, we perform 
  - forward pass
  - calculate loss
  - backward pass
  - optimizer step
3. Record everything to get the statistics of training
4. Repeat steps 1 to 3 for a fixed number of epochs
5. At the end measure the performance on evaluation dataset.

For evaluation, we iterate through all the evaluation samples and report the accuracy on the evaluation dataset.

```python
def train_loop(...):
    # Prepare various events we need to capture during training
    with profiler_cm as profiler:
        ...
        for batch_idx, batch in enumerate(data, start=1):
            # perform a training step of forward, loss and backward pass
            model, optimizer, loss = train_step(batch, model, optimizer)
            profiler.step()
        # Print memory consumed by optimizer, gradients and parameters
        print_memory_stats(...)
    
def evaluate(model, dataloader, device):
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            # move batch to GPU
            outputs = model(**batch)
            # get the prediction
            # compare the prediction and ground truth
    # calculate overall accuracy 
```

{{< /collapse >}}

A [PyTorch profiler](https://docs.pytorch.org/tutorials/recipes/recipes/profiler_recipe.html) provides in-depth breakdown on memory consumption and time taken by all the operators during training. This provides insights into how long training spends in computation compared to communication, how long GPU performed computation versus how long it was idle or waiting for computation, which operations are the most expensive. Profiler helps find the bottleneck in the training and utilization of GPUs.

Finally, bringing everything together we are all set to perform distributed training. Before we start the distributed training, it is important to setup all the participating GPUs in training. This is taken care by `torchrun` and distributed `init_process_group`. The `torchrun` CLI script starts same training script across all the GPUs. It can be used in both single node (consisting of multiple GPUs) or multiple nodes distributed training scenarios. The `init_process_group` makes sure each GPU is initialized properly and can discover its peers. This is important as we will be running distributed communication operations such as All Reduce to gather the gradients from all GPUs and distribute the final results to the same.

The initialization of distributed process group is taken care by `ddp_initialize` function. It discovers how many GPUs are participating in the training. It assigns a local rank (`local_rank`) to each of these GPUs. The world size (`world_size`) refers to the total number of GPUs used for training.

{{< collapse summary="**Everything at one place**" >}}

```python
if __name__ == "__main__":
    set_seed(SEED)
    ddp_initialize()
    rank, world_size, local_rank = get_dist_info()
    per_device_batch = GLOBAL_BATCH_SIZE // world_size
    device = torch.device(f"cuda:{local_rank}")
    
    train_loader, eval_loader = prepare_data(per_device_batch, rank, world_size)
    model = get_model()
    model.to(device)

    # Select which DDP wrapper to use around model for distributed training
    model = SimpleDDP(model)
    optim = torch.optim.AdamW(model.parameters(), lr=5e-5)
    model = train_loop(...)
    evaluate(model, eval_loader, device=device)
    
    ddp_cleanup()
```

{{< /collapse >}}


## Simple DDP

A custom wrapper around the PyTorch model is created. This custom wrapper implements the following

1. `sync_parameters`: A function that ensures all ranks (or GPUs) are initialized with same parameter values
2. `forward`: This shows how the forward pass using this custom wrapper looks like. Nothing fancy, we call the model as is.
3. `sync_gradients`: This function implements how to calculate the gradients across all GPUs

The steps for synchronizing gradients are 

1. Iterate through the gradients of all the model parameters on each GPU
2. Invoke a distributed communication `all_reduce` to sum all the gradients across all GPUs
3. Average the gradients by dividing it using number of total GPUs used for training.

The `all_reduce` operation takes care of gathering and distributing the final results to all the participating GPUs. During the training step, we call the `model.sync_gradients()` function before `optimizer.step()` for each batch.

```python
def train_step(batch, model, optimizer):
    """Perform a single training step: forward, backward, optimizer step."""
    optimizer.zero_grad(set_to_none=True)
    outputs = model(**batch)
    loss = outputs.loss
    loss.backward()

    # Important step for SimpleDDP to sync gradients
    model.sync_gradients()

    optimizer.step()
    return model, optimizer, loss
```

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

The results in drop down below show a plotly figure of how simple DDP performs in terms of compute vs non-compute time, overlap and number of all-reduce calls made during training.

The baseline results for this simple DDP are : ~67% compute vs ~31% non-compute, no overlap, and 582 all-reduce calls (about 798 ms spent in comms). In Simple DDP, every parameter's gradient is synchronized after backward pass, leading to many small all-reduce calls that are latency-bound and don't overlap with computation. In next section, we will look into how we can reduce the communication overhead and improve compute share by using gradient accumulation technique.

{{< collapse summary="**Result**" >}}

{{< plotly file="static/images/ddp_plots/simple_ddp.html" >}}

{{< /collapse >}}


## Simple DDP with gradient accumulation

[Gradient Accumulation](https://dudeperf3ct.github.io/posts/ultrascale_one_gpu/#gradient-accumulation) technique helps training on larger batch sizes and reduce the peak activation memory usage during forward pass (only if we break down the batch into smaller micro-batches). The idea behind gradient accumulation is to perform gradient synchronization only after specific N steps. Unlike simple DDP above which perform gradient synchronization every batch, the gradients are accumulated across multiple batches and then a synchronization is invoked. This saves the number of communication calls we are making synchronization call every N step thus delaying the communication.

Building on the implementation of `SimpleDDP`, two new methods are introduced

- `should_sync`: This function handles enabling and disabling the gradient synchronization
- `no_sync`: This context manager disables gradient synchronization
- `sync_gradients`: This function is same as used in the `SimpleDDP` implementation but now we only sync if `should_sync` is set to `True`.

In training loop, we want to decide when to enable the `should_sync` to trigger the synchronization. This is done using the `should_sync = batch_idx % grad_accum_steps == 0` logic which enables the synchronization every `grad_accum_steps` steps. In case the synchronization is disabled, the `no_sync` context manager is used to perform backward pass and accumulate the gradients.

```python
def train_step_with_ga(...):
    # Zero gradients at the start of accumulation
    if (batch_idx - 1) % grad_accum_steps == 0:
        optimizer.zero_grad(set_to_none=True)
    outputs = model(**batch)
    loss = outputs.loss
    loss = loss / grad_accum_steps  # Scale loss for gradient accumulation
    # Determine if we should sync gradients this step
    should_sync = batch_idx % grad_accum_steps == 0
    # Perform backward and optimizer step based on accumulation
    if should_sync:
        loss.backward()
        model.sync_gradients()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    # Use no_sync context to skip gradient sync during accumulation steps
    # This skips optimizer step and gradient sync until accumulation is done
    else:
        with model.no_sync():
            loss.backward()
    return model, optimizer, loss * grad_accum_steps  # Return original loss value
```

{{< collapse summary="**Implementation**" >}}

```python
from contextlib import contextmanager
import torch
from ddp.simple_ddp import SimpleDDP

class SimpleDDPWithGA(SimpleDDP):
    """GradientAccumulation version of SimpleDDP."""

    def __init__(self, model: torch.nn.Module):
        super().__init__(model)
        self.do_sync = True

    @property
    def should_sync(self):
        """Indicate that gradient synchronization is needed."""
        return self.do_sync

    @contextmanager
    def no_sync(self):
        """Context manager to disable gradient synchronization across ranks."""
        prev = self.do_sync
        self.do_sync = False
        try:
            yield
        finally:
            self.do_sync = prev

    def sync_gradients(self):
        """Synchronize gradients across ranks if enabled."""
        if not self.should_sync:
            return
        super().sync_gradients()
```

{{< /collapse >}}

Relative to the baseline, `simple_ddp_ga` already lifted compute share to ~81.6% and halved all-reduce calls (582 -> 291). In the next section, PyTorch backward hooks are used to further overlap gradient synchronization with backward computation. 

{{< collapse summary="**Result**" >}}

{{< plotly file="static/images/ddp_plots/simple_ddp_ga.html" >}}

{{< /collapse >}}


## Using backward hooks to overlap gradient synchronization

PyTorch uses hooks as a callback functions that are used to intercept and execute custom code during forward and backward pass. In this implementation, we use backward hooks to trigger gradient synchronization as soon as the gradients for a parameter are computed during backward pass. This allows overlapping the gradient synchronization with the backward computation of other parameters, reducing the overall training time. 

In the previous implementation of `SimpleDDPWithGA`, the gradient synchronization was performed after the entire backward pass was completed for all parameters. This meant that all GPUs had to wait until all gradients were computed before starting the synchronization, leading to idle time. Using hooks, the last layer's gradient can be synchronized while the earlier layers are still computing their gradients.

To implement this, we register a backward hook on each parameter of the model using [`register_post_accumulate_grad_hook`](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.register_post_accumulate_grad_hook.html). This hook is called after the gradients for that parameter have been fully accumulated. In the hook function `_sync_gradient`, we perform the all-reduce operation to synchronize the gradients across all ranks. Extending the `SimpleDDPWithGA`, we create a new class `SimpleDDPHookGA` that registers the backward hooks during initialization. The `_sync_gradient` method performs the all-reduce operation and averages the gradients. The new flow is:

1. `register_backward_hook` registers `self._sync_gradient` on each `Parameter`.
2. During backward, once a param's grad is fully accumulated, PyTorch invokes the hook with that grad tensor.
3. `_sync_gradient` runs in-place on that tensor, `all_reduce`s, and divides by `world_size`, so `p.grad` ends up averaged across ranks.

The training step with gradient accumulation remains similar to the previous implementation, but we no longer need to call `model.sync_gradients()` explicitly. The backward hooks will handle the gradient synchronization automatically as gradients are computed.

```python
def train_step_with_ga(...):
    # Zero gradients at the start of accumulation
    if (batch_idx - 1) % grad_accum_steps == 0:
        optimizer.zero_grad(set_to_none=True)
    outputs = model(**batch)
    loss = outputs.loss
    loss = loss / grad_accum_steps  # Scale loss for gradient accumulation
    # Determine if we should sync gradients this step
    should_sync = batch_idx % grad_accum_steps == 0
    # Perform backward and optimizer step based on accumulation
    if should_sync:
        loss.backward()
        # We don't need to call sync_gradients explicitly here
        # Backward hooks will handle it before optimizer step
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    # Use no_sync context to skip gradient sync during accumulation steps
    # This skips optimizer step and gradient sync until accumulation is done
    else:
        with model.no_sync():
            loss.backward()
    return model, optimizer, loss * grad_accum_steps  # Return original loss value
```

{{< collapse summary="**Implementation**" >}}

```python
import torch
import torch.distributed as dist
from ddp.simple_ddp_ga import SimpleDDPWithGA

class SimpleDDPHookGA(SimpleDDPWithGA):
    """GradientAccumulation version of SimpleDDP using backward hooks.

    The flow is:
    - `register_backward_hook` registers `self._sync_gradient` on each `Parameter`.
    - During backward, once a param’s grad is fully accumulated,
      PyTorch invokes the hook with the parameter.
    - `_sync_gradient` runs on `param.grad`, `all_reduce`s, and divides by `world_size`,
      so `p.grad` ends up averaged across ranks.
    """

    def __init__(self, model: torch.nn.Module):
        super().__init__(model)
        self.register_ga_hook()

    def _sync_gradient(self, param):
        """Hook called after a param's grad is accumulated."""
        if not self.should_sync or param.grad is None:
            return

        # Sum the gradient across all ranks and then average.
        dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
        param.grad /= self.world_size

    def register_ga_hook(self):
        # Keep track of hooks to remove them later if needed.
        self.sync_hooks = []
        for p in self.model.parameters():
            if p.requires_grad:
                # Register a hook per parameter.
                # The hook will be called after all gradients for a tensor have been accumulated
                h = p.register_post_accumulate_grad_hook(self._sync_gradient)
                self.sync_hooks.append(h)
```

{{< /collapse >}}

Using hooks alone does not materially change the performance profile: ~80% compute, ~16% non-compute, zero overlap, and the same 291 all-reduce calls. While hooks allow us to interleave communication with backward computation, they do not by themselves create overlap. This is because the underlying `all_reduce` calls are still blocking. Asynchronous operations allow the program to continue executing while the communication is still in progress, enabling better utilization of GPU resources. 

{{< collapse summary="**Result**" >}}

{{< plotly file="static/images/ddp_plots/simple_ddp_hook.html" >}}

{{< /collapse >}}


## Asynchronous gradient synchronization

The `all_reduce` operation used in the previous implementations is a blocking call, meaning that the execution waits until the operation is complete before proceeding. This can lead to idle time on the GPU while waiting for the communication to finish. To further optimize the training process, we can use asynchronous communication to overlap gradient synchronization with ongoing computation.

To implement asynchronous gradient synchronization, we modify the `_sync_gradient` method to use the `async_op=True` flag in the `dist.all_reduce` call. This allows the all-reduce operation to be initiated without blocking the execution. The method returns a work handle that can be used to wait for the operation to complete later. We store these work handles along with their corresponding gradient tensors in a list called `handles`. After the backward pass is complete, we call a new method `finish_gradient_synchronization` that iterates through the stored handles, waits for each operation to complete using `work.wait()`, and then averages the gradients in place.

The new flow is:
1. `register_backward_hook` registers `_sync_gradient` on each param.
2. Each hook fires after a param's grad is accumulated; it kicks off an async all-reduce and records the work handle plus the grad view.
3. Call `finish_gradient_synchronization` after backward to wait on all pending reductions and average the grads in place.

The training step with gradient accumulation is updated to call `model.finish_gradient_synchronization()` after the backward pass when synchronization is needed.

```python
def train_step_with_ga(...):
    # Zero gradients at the start of accumulation
    if (batch_idx - 1) % grad_accum_steps == 0:
        optimizer.zero_grad(set_to_none=True)
    outputs = model(**batch)
    loss = outputs.loss
    loss = loss / grad_accum_steps  # Scale loss for gradient accumulation
    # Determine if we should sync gradients this step
    should_sync = batch_idx % grad_accum_steps == 0
    # Perform backward and optimizer step based on accumulation
    if should_sync:
        loss.backward()
        # For async, we don't sync gradients here
        # Wait for all async gradient syncs to complete
        model.finish_gradient_synchronization()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    # Use no_sync context to skip gradient sync during accumulation steps
    # This skips optimizer step and gradient sync until accumulation is done
    else:
        with model.no_sync():
            loss.backward()
    return model, optimizer, loss * grad_accum_steps  # Return original loss value
```


{{< collapse summary="**Implementation**" >}}

```python
import torch
import torch.distributed as dist
from ddp.simple_ddp_hook import SimpleDDPHookGA

class SimpleDDPAsyncHookGA(SimpleDDPHookGA):
    """Asynchronous GradientAccumulation version of SimpleDDP using backward hooks.

    The flow is:

    - `register_backward_hook` registers `_sync_gradient` on each param.
    - Each hook fires after a param's grad is accumulated; it kicks off an async all-reduce
      and records the work handle plus the grad view.
    - Call `finish_gradient_synchronization` after backward to wait on all pending
      reductions and average the grads in place.
    """

    def __init__(self, model: torch.nn.Module):
        super().__init__(model)
        self.handles = []

    def _sync_gradient(self, param):
        """Hook called after a param's grad is accumulated.
        Use asynchronous all-reduce to overlap communication with computation.
        """
        if not self.should_sync or param.grad is None:
            return

        # Asynchronously sum the gradient across all ranks and then average.
        handle = dist.all_reduce(param.grad, op=dist.ReduceOp.SUM, async_op=True)
        self.handles.append((handle, param.grad))

    def finish_gradient_synchronization(self) -> None:
        """Block until all outstanding gradient all‑reduces have completed."""
        for work, grad in self.handles:
            work.wait()
            grad.div_(self.world_size)
        self.handles.clear()

    def sync_gradients(self) -> None:
        """Synchronize gradients for last step if needed."""
        if not self.should_sync:
            return
        for p in self.model.parameters():
            if p.grad is not None:
                handle = dist.all_reduce(p.grad, op=dist.ReduceOp.SUM, async_op=True)
                self.handles.append((handle, p.grad))
        self.finish_gradient_synchronization()
```

{{< /collapse >}}

Async hooks squeeze out a tiny bit of overlap (~0.07%) while keeping compute high (~81.9%), but with 291 small all-reduces we're still latency-bound. The next step is to coalesce gradients into buckets to increase per-call payloads and cut call count.

{{< collapse summary="**Result**" >}}

{{< plotly file="static/images/ddp_plots/simple_ddp_async.html" >}}

{{< /collapse >}}


## Bucketing gradients

Bucketing gradients approach further optimizes the asynchronous gradient synchronization by grouping gradients into buckets before performing the all-reduce operation. This reduces the number of all-reduce calls and increases the payload size for each call, which can help mitigate the latency overhead associated with many small communication operations.

To implement bucketing, we modify the `_sync_gradient` method to accumulate gradients into a bucket until it reaches a specified size limit (in bytes). Once the bucket is full, we perform an asynchronous all-reduce on the entire bucket and store the work handle along with the gradients in the bucket. After the backward pass, we call `finish_gradient_synchronization` to flush any remaining gradients in the bucket and wait for all outstanding reductions to complete.

The new flow is:
1. Register post-accumulate hooks per parameter.
2. Accumulate gradients into size-limited buckets.
3. Launch async all-reduce for each full bucket.
4. Call `finish_gradient_synchronization()` after backward to flush the remaining partial bucket and wait for all reductions to complete.

The training step with gradient accumulation remains similar to the previous implementation, but we now rely on the bucketing mechanism for gradient synchronization.

> [!WARNING]
> Important correctness caveat: This implementation assumes that hook firing order is identical across ranks. In practice, this is not guaranteed and can silently corrupt gradients if parameters are bucketed differently.

{{< collapse summary="**Implementation**" >}}

```python
from contextlib import contextmanager
import torch
import torch.distributed as dist
from utils.ddp_utils import get_dist_info

class BucketDDPAsyncHookGA(torch.nn.Module):
    """Bucketed async DDP with gradient accumulation support.

    The flow is:
    - Register post-accumulate hooks per parameter.
    - Accumulate gradients into size-limited buckets.
    - Launch async all-reduce for each full bucket.
    - Call `finish_gradient_synchronization()` after backward to flush the
      remaining partial bucket and wait for all reductions to complete.

    A major concern here is we are building bucket on each rank by hook firing order
    Hook order can differ across each rank, so flat buffers line up different parameters
    on different ranks. The all-reduce then sums mismatched params and might corrupt grads silently.
    """

    def __init__(self, model: torch.nn.Module, bucket_cap_mb: int = 25):
        super().__init__()
        self.model = model
        self.handles = []
        self.do_sync = True
        self.bucket, self.bucket_size = [], 0
        self.bucket_cap_bytes = bucket_cap_mb * 1024 * 1024
        self.rank, self.world_size, self.local_rank = get_dist_info()
        # broadcast parameters from rank 0 to all other ranks
        # This ensures all models start with the same parameters
        self.sync_parameters()
        # Register backward hooks to handle gradient synchronization
        self.register_bucket_hook()

    def sync_parameters(self):
        """Broadcast parameters from rank 0 to all other ranks."""
        for param in self.model.parameters():
            torch.distributed.broadcast(tensor=param.data, src=0)

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def _sync_gradients(self):
        """Asynchronously all-reduce the gradients in the current bucket."""
        grads = [g for g in self.bucket if g is not None]
        if not grads:
            return
        # Concatenate gradients into a single tensor for all-reduce
        # This reduces the overhead of multiple small all-reduce calls
        # Perform asynchronous all-reduce
        flat_grad = torch.cat([g.reshape(-1) for g in grads])
        handle = dist.all_reduce(flat_grad, op=torch.distributed.ReduceOp.SUM, async_op=True)
        self.handles.append((handle, grads, flat_grad))

    def _fill_buckets(self, param):
        """Fill buckets with gradients and trigger async all-reduce when full."""
        if not self.should_sync or param.grad is None:
            return

        # Fill the bucket with the current gradient
        # Calculate the size of the gradient in bytes
        grad = param.grad
        grad_size = grad.numel() * grad.element_size()
        self.bucket.append(grad)
        self.bucket_size += grad_size

        # If the bucket is full, sync the gradients
        if self.bucket_size >= self.bucket_cap_bytes:
            self._sync_gradients()
            # Clear the bucket
            self.bucket = []
            self.bucket_size = 0

    def register_bucket_hook(self):
        # Keep track of hooks to remove them later if needed.
        self.sync_hooks = []
        for p in self.model.parameters():
            if p.requires_grad:
                # Register a hook per parameter.
                # The hook will be called after all gradients for a tensor have been accumulated
                h = p.register_post_accumulate_grad_hook(self._fill_buckets)
                self.sync_hooks.append(h)

    def finish_gradient_synchronization(self) -> None:
        """Block until all outstanding gradient all‑reduces have completed.
        Also flushes any remaining partial bucket.
        """
        # Ensure the final partial bucket is also synchronized.
        self.flush_buckets()
        for work, grads, flat_grad in self.handles:
            work.wait()
            # Unflatten the gradients back to their original shapes
            offset = 0
            for g in grads:
                numel = g.numel()
                g.copy_(flat_grad[offset : offset + numel].view_as(g))
                g.div_(self.world_size)
                offset += numel
        self.handles.clear()

    def flush_buckets(self):
        """Flush any remaining gradients in the bucket."""
        if self.bucket:
            self._sync_gradients()
            self.bucket = []
            self.bucket_size = 0

    @property
    def should_sync(self):
        """Indicate that gradient synchronization is needed."""
        return self.do_sync

    @contextmanager
    def no_sync(self):
        """Context manager to disable gradient synchronization across ranks."""
        prev = self.do_sync
        self.do_sync = False
        try:
            yield
        finally:
            self.do_sync = prev

    def sync_gradients(self) -> None:
        """Synchronize gradients for last step if needed."""
        if not self.should_sync:
            return
        # Hooks were skipped during no_sync; bucket the current grads manually.
        for p in self.model.parameters():
            if p.requires_grad and p.grad is not None:
                self._fill_buckets(p)
        self.finish_gradient_synchronization()
```

{{< /collapse >}}

With buckets, all-reduce calls drop from 291 to 44, overlap jumps to ~9.6%, and compute stays solid (~80.6%).

{{< collapse summary="**Result**" >}}

{{< plotly file="static/images/ddp_plots/bucket_ddp_async.html" >}}

{{< /collapse >}}


## PyTorch DDP

PyTorch offers a native [`torch.nn.parallel.DistributedDataParallel`](https://docs.pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html#torch.nn.parallel.DistributedDataParallel) API that implements optimised data parallelism for distributed training. This built-in DDP handles gradient synchronization, communication optimizations, and overlapping computation with communication internally, providing a high-performance solution for distributed training.

[`DistributedDataParallel`](https://docs.pytorch.org/docs/main/notes/ddp.html#internal-design) design document details how PyTorch DDP works under the hood. The key features of PyTorch DDP include:

* **Constructor**: DDP constructor takes care of making sure all model parameters are synchronized across all processes at the start of training. The `Reducer` component in each DDP process is responsible for managing gradient synchronization. And mapping each parameter and its gradient to a bucket that groups gradients for efficient communication.
* **Backward Pass**: During the backward pass, DDP uses hooks to monitor when gradients are computed for each parameter. Once a gradient is computed, it is added to its corresponding bucket. When a bucket is full, DDP initiates an asynchronous all-reduce operation to synchronize the gradients across all processes. This allows overlapping communication with ongoing computation.
* **Overlapping Communication and Computation**: DDP is designed to overlap gradient synchronization with the backward computation. While the backward pass is still computing gradients for other parameters, the all-reduce operation for the filled buckets can proceed in the background. This helps to hide communication latency and improve overall training efficiency.

Native `torch.nn.parallel.DistributedDataParallel` brings a tuned bucketed, overlapping implementation. In this run we see ~57% overlap and the lowest idle time (1.9%), though non-compute is still sizable because the batches are small relative to comm time.

DDP shows how aggressive overlap can be: ~57% overlap with only 1.9% idle. The trade-off here is a still-hefty non-compute share (~27%) driven by the tiny batch size. Larger batches and enabling gradient accumulation would likely push more time into useful compute.

{{< collapse summary="**Result**" >}}

{{< plotly file="static/images/ddp_plots/pytorch_ddp.html" >}}

{{< /collapse >}}


## Results comparison

| Approach            | Compute% | Non_Compute% | Idle% | Overlap% | Comm_overhead% | all_reduce_calls | all_reduce_ms |
|---------------------|---------:|-------------:|------:|---------:|---------------:|-----------------:|--------------:|
| simple_ddp          | 67.14    | 30.54        | 2.33  | 0.00     | 30.78          | 582.0            | 797.84        |
| simple_ddp_ga       | 81.58    | 14.64        | 3.79  | 0.00     | 14.73          | 291.0            | 358.79        |
| simple_ddp_hook     | 80.30    | 15.93        | 3.78  | 0.00     | 16.05          | 291.0            | 414.57        |
| simple_ddp_async    | 81.89    | 14.32        | 3.79  | 0.07     | 14.43          | 291.0            | 352.03        |
| bucket_ddp_async    | 80.59    | 15.73        | 3.68  | 9.63     | 16.60          | 44.0             | 438.03        |
| pytorch_ddp         | 71.13    | 26.95        | 1.92  | 57.16    | 35.83          | 90.5             | 1161.42       |

- Gradient accumulation delivers the biggest immediate win: halves all-reduce calls and slashes non-compute time (~16 pts drop) while boosting compute to ~82%.
- Hooks alone don't change the picture; async hooks add negligible overlap because the per-param calls are too small to amortize latency.
- Bucketing is the first meaningful overlap boost (~9.6%) by trading many tiny calls for 44 larger ones.
- PyTorch DDP shows how much overlap is possible (~57%), but with small batches communication still dominates non-compute; bigger batches would likely convert that overlap into more compute time.

## Wrap up

In this blog, we explored various techniques to implement data parallelism using PyTorch. Starting from a simple DDP implementation, we progressively enhanced it with gradient accumulation, backward hooks, asynchronous communication, and bucketing. Finally, we compared our custom implementations with PyTorch's native `DistributedDataParallel`.

In the next blog on the implementation series, we will look into implementing the Zero Redundancy Optimizer (ZeRO) techniques to further optimize distributed training by reducing memory footprint and improving scalability.
