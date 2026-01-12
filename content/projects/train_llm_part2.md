---
title: "Mini StarCoder2 - Pretraining (TorchTitan)"
tags: ["llm", "pretraining", "torchtitan"]
ShowToc: true
---

It's time to get the wallets out :money_with_wings::money_with_wings::money_with_wings::money_with_wings:

## Useful Links

> [!CODE]
> Github: https://github.com/dudeperf3ct/minicode-llm/tree/main/codellm_pretrain/torch_titan

> [!EXPERIMENT]
> W&B for project: https://wandb.ai/dudeperf3ct/torchtitan

> [!NOTE]
> Trained checkpoints: https://huggingface.co/dudeperf3ct/codellm_pretrain


## Getting Started

I am using [TorchTitan](https://github.com/pytorch/torchtitan) library for pretraining. TorchTitan library builds on PyTorch and provides first class support for all N-D parallelism. There is a paper [TorchTitan](https://arxiv.org/abs/2410.06511) that details all the work in creating a scalable and production ready distributed system. 

To get started on pretraining on our dataset, two main components are required: data and model. Rest of components such as optimizer, scheduler, training loop can be configured using a configuration file.

>[!NOTE]
> I have written a blog on config driven design in Python [here](https://dudeperf3ct.github.io/posts/python_in_practice/). TorchTitan uses a similar approach to configure the training components.

These are the components that I am using for pretraining

* Custom tokenizer: https://dudeperf3ct.github.io/projects/train_llm_part1/
* Dataset: [`tokyotech-llm/swallow-code-v2`](https://huggingface.co/datasets/tokyotech-llm/swallow-code-v2)
* Model Architecture: Llama 3.2 1B (1 billion parameter)

## Data

For this experiment, I am using [`tokyotech-llm/swallow-code-v2`](https://huggingface.co/datasets/tokyotech-llm/swallow-code-v2) dataset. The pretraining task will be a Fill-in-the-middle (FIM) task where a random span of code is masked and the model is trained to predict the missing span given the surrounding context. This enables model to perform code infilling tasks.

> [!TIP]
> More details about the pretraining dataset in this previous post [here](https://dudeperf3ct.github.io/projects/train_llm_part0/#pretraining-dataset).

The StarCoder2 paper section 5 Data Formatting provides templates for FIM task. It uses repository context where files from same repository are grouped together. It arranges them in random order as a grouping strategy. For FIM formatting, repositories are selected with 50% probability. The selected repository examples are split by `|<endoftext>|` and `<|file_sep|>` tokens. FIM transform shown below is applied to each chunk with 50% probability. FIM is not applied to repository metadata (`<reponame>reponame`).

```
<repo_name>reponame<file_sep>filepath0\ncode0<file_sep><fim_prefix>filepath1\n code1_pre<fim_suffix>code1_suf<fim_middle>code1_mid<file_sep> ...<|endoftext|>
```

I have disgressed from how StarCoder2 prepares its data format and applied FIM masking to individual files instead of grouping them by repository. This is because I don't have access to the full repository data. There are three strategies to apply FIM masking: Prefix-Suffix-Middle (PSM), Suffix-Prefix-Middle (SPM) and Middle-only. StarCoder2 paper uses 50% PSM and 50% SPM only for training. I am using 50% PSM, 25% SPM and 25% M-only probabilties for these strategies. The objective is to predict next token using language model but the sequence is permuted so the model learns to generate the hidden span given prefix + suffix (PSM), suffix + prefix (SPM), or no surrounding context (M).

We have to create a data pipeline that reads the data from HF dataset, applies FIM masking and creates batches for training. TorchTitan provides a support for Hugging Face datasets via [`HuggingFaceTextDataset`](https://github.com/pytorch/torchtitan/blob/9f211ec199bc887901b874edd6af5a20527a4175/torchtitan/hf_datasets/text_datasets.py#L71C7-L71C29) class. This class prepares data to generate pair of input and label. TorchTitan provides [`ParallelAwareDataloader`](https://github.com/pytorch/torchtitan/blob/9f211ec199bc887901b874edd6af5a20527a4175/torchtitan/components/dataloader.py#L46) class that creates dataloader with support for distributed training.

To support FIM masking, I created a custom dataset class that preprocesses the data in format required for FIM task.

```python
class ProcessSwallowCodeDataset:
    def __init__(
        self, rank: int, seed: int = 42, fim_rate: float = 0.5, min_code_length: int = 100
    ):
        self.rng = random.Random(seed + rank)
        self.fim_rate = fim_rate
        self.min_code_length = min_code_length

        # FIM tokens
        self.fim_prefix = "<|fim_prefix|>"
        self.fim_middle = "<|fim_middle|>"
        self.fim_suffix = "<|fim_suffix|>"
        self.endoftext = "<|endoftext|>"

    def _select_fim_format(self) -> str:
        """
        Select FIM format according to StarCoder2 strategy:
        - PSM (Prefix-Suffix-Middle): 50%
        - SPM (Suffix-Prefix-Middle): 25%
        - Middle-only: 25%
        """
        rand = self.rng.random()
        if rand < 0.5:
            return "PSM"  # Prefix-Suffix-Middle
        if rand < 0.75:
            return "SPM"  # Suffix-Prefix-Middle
        return "M"  # Middle-only

    def apply_fim_to_text(self, code):
        # No FIM (50% of time)
        if self.rng.random() > self.fim_rate:
            return code + self.endoftext

        fim_type = self._select_fim_format()

        # Select span (character-based)
        code_len = len(code)
        if code_len < self.min_code_length:
            return code + self.endoftext

        # Middle span: 10-50% of code
        min_middle = code_len // 10
        max_middle = code_len // 2

        middle_start = self.rng.randint(0, code_len - min_middle)
        middle_len = self.rng.randint(min_middle, min(max_middle, code_len - middle_start))
        middle_end = middle_start + middle_len

        prefix = code[:middle_start]
        middle = code[middle_start:middle_end]
        suffix = code[middle_end:]

        # Format based on type
        if fim_type == "PSM":
            return (
                f"{self.fim_prefix}{prefix}"
                f"{self.fim_suffix}{suffix}"
                f"{self.fim_middle}{middle}"
                f"{self.endoftext}"
            )
        if fim_type == "SPM":
            return (
                f"{self.fim_suffix}{suffix}"
                f"{self.fim_prefix}{prefix}"
                f"{self.fim_middle}{middle}"
                f"{self.endoftext}"
            )
        return f"{self.fim_middle}{middle}{self.endoftext}"
```

## Model

There are [not lot of PyTorch models](https://github.com/pytorch/torchtitan/tree/main/torchtitan/models) supported in TorchTitan library. TorchTitan has support for any Hugging Face transformers model out-of-the-box in experimental mode. There are few [known issues](https://github.com/pytorch/torchtitan/tree/main/torchtitan/experiments/transformers_modeling_backend#known-issues-to-address-later) and [future work](https://github.com/pytorch/torchtitan/tree/main/torchtitan/experiments/transformers_modeling_backend#further-work) for a stable support using transformers backend.

I decided to use Llama 3.2 1B, 1 billion parameter model, from Hugging Face as the model architecture. It will use the custom tokenizer trained [here](https://dudeperf3ct.github.io/projects/train_llm_part1/) as tokenizer and we have to patch Llama 3.2 1B architecture to use the new vocabulary size. The composability of different components makes it easy to configure this easily.

```python
import torchtitan.experiments.transformers_modeling_backend as base_backend
from torchtitan.experiments.transformers_modeling_backend.model.args import (
    HFTransformerModelArgs,
    TitanDenseModelArgs,
)
from torchtitan.protocols.train_spec import TrainSpec, register_train_spec

TRAIN_SPEC_NAME = "transformers_modeling_backend_custom"
FLAVOUR_NAME = "llama32_1b_tok32k"
VOCAB_SIZE = 32768


def _vocab_only_args(vocab_size: int) -> TitanDenseModelArgs:
    # Let HF config define core shape params; only override vocab size (and MLP width).
    args = TitanDenseModelArgs()
    for attr in (
        "dim",
        "n_layers",
        "n_heads",
        "n_kv_heads",
        "norm_eps",
        "rope_theta",
        "max_seq_len",
    ):
        setattr(args, attr, None)
    args.vocab_size = vocab_size
    # Llama 3.2 1B uses intermediate_size=4*hidden_size; this matches via 1.5x on 2/3*4.
    args.ffn_dim_multiplier = 1.5
    return args


base_spec = base_backend.get_train_spec()
custom_model_args = dict(base_spec.model_args)
custom_model_args[FLAVOUR_NAME] = HFTransformerModelArgs(
    titan_dense_args=_vocab_only_args(VOCAB_SIZE)
)

custom_spec = TrainSpec(
    model_cls=base_spec.model_cls,
    model_args=custom_model_args,
    parallelize_fn=base_spec.parallelize_fn,
    pipelining_fn=base_spec.pipelining_fn,
    build_optimizers_fn=base_spec.build_optimizers_fn,
    build_lr_schedulers_fn=base_spec.build_lr_schedulers_fn,
    build_dataloader_fn=base_spec.build_dataloader_fn,
    build_tokenizer_fn=base_spec.build_tokenizer_fn,
    build_loss_fn=base_spec.build_loss_fn,
    build_validator_fn=base_spec.build_validator_fn,
    build_metrics_processor_fn=base_spec.build_metrics_processor_fn,
    state_dict_adapter=base_spec.state_dict_adapter,
)

register_train_spec(TRAIN_SPEC_NAME, custom_spec)
```

## Smoke Testing

Before running everything end to end, I have to be careful with the money being at the stake. Smoke test help get refined estimates and flagging mistakes early on.

>[!IMPORTANT]
> I performed smoke testing on 2 x H100 instance (80 GB) on Lambda Labs. This run costed me about $8. The cost of each H100 was $3.19/GPU/hr on January 9.

Something I realized after running smoke test is how slow copying over files from remote instance using `scp`. The smoke tests saves bunch of wandb log files, checkpoint files and profiling traces. There are 2 checkpoint files for each rank saved taking upto 3 GB of space. I almost spent twice as long waiting for 12 GB to get all these files transferred from remote machine to my local machine. The transfer process took about 30 mins. The smoke test run lasted for 15 mins.

With the two main components configured, the first step is to run a smoke test to make sure everything is working before proceeding to full run. The full configuration file for smoke test can be found [here](https://github.com/dudeperf3ct/minicode-llm/blob/main/codellm_pretrain/torch_titan/train_configs/smoke_llama32_1b_swallowcode_tok32k.toml). Highlighting few note-worthy configuration below:

Since our model is small ~ 1B and plenty of memory, we can afford to trade-off speed for saving memory by disabling activation checkpointing.

```toml
[activation_checkpoint]
mode = "none"  # ["none", "selective", "full"]
selective_ac_option = '2'  # 'int' = ac every positive int layer or 'op', ac based on ops policy
```

Configuring various parallelism strategies using `torchtitan` is as easy as tweaking the number below. Since we have larger memory and smaller model, only using DDP should speed up the training. The other strategy would be to use FSDP by setting `data_parallel_shard_degree = -1`, this will split/shard the model parameters across GPU saving memory.

```toml
[parallelism]
data_parallel_replicate_degree = 2
data_parallel_shard_degree = 1
fsdp_reshard_after_forward = "default" # default / never / always
tensor_parallel_degree = 1
enable_async_tensor_parallel = false
pipeline_parallel_degree = 1
pipeline_parallel_schedule = "Interleaved1F1B"
context_parallel_degree = 1
```

The training configuration below decides how long the training job should run for. The maximum sequence length is set to 8K. The `local_batch_size` should be increased until we hit OOM to increase the utilization of GPU. Since this is a smoke run, the total number of steps are set to 1000. No gradient accumulation is enabled. The training is performed using mixed precision where weights are stored in `bfloat16` and gradients are stored in `float32`.

```toml
# Training config: https://github.com/pytorch/torchtitan/blob/81af8833ddeff9b5f1874dc7e20594aa17da6b86/torchtitan/config/job_config.py#L235
[training]
local_batch_size = 6
seq_len = 8192
max_norm = 1.0  # grad norm clipping
# global_batch_size defaults to local_batch_size * data_parallel_degree.
# If training.global_batch_size is set, TorchTitan derives gradient accumulation steps.
steps = 1000
dataset = "swallowcode"
dtype = "bfloat16"
mixed_precision_param = "bfloat16" # only works if when data_parallel_shard_degree > 1 or context_parallel_degree > 1
mixed_precision_reduce = "float32"
```

Profiling traces and memory snapshots is enabled as well to understand if all resources are being utilized correctly. These profiling trace files can be used at [perfetto UI](https://ui.perfetto.dev/). This provides detailed breakdown of CUDA streams and CPU threads. It shows the compute time for all the operations taking place on GPU and CPU. The memory snapshots can be viewed at [memory_viz](https://docs.pytorch.org/memory_viz).

```toml
# Profiling config:
[profiling]
enable_profiling = true
save_traces_folder = "profile_trace"
profile_freq = 400
enable_memory_snapshot = true
save_memory_snapshot_folder = "memory_snapshot"
```

Results from smoke tests can be viewed at Weights and Biases dashboard. Summary of smoke test results:

- Completed 1k steps / 98.3M tokens in ~15.4 minutes.
- Loss fell from 10.88 -> 4.69; LR warmed from 3e-7 to 3e-4 by step 1k (warmup covered the full run).
- Step time stabilized around ~0.90s; per-rank throughput ~54-55k tps (global ~109k), MFU ~41% (~405 TFLOPS).
- Data loading was negligible (~0.03% of step time) and memory stayed flat at ~70.2 GiB active / 72.8 GiB reserved, with 0 OOMs or alloc retries.
- Token budget: `steps * global_batch_size * seq_len`. With `local_batch_size=6`, `NGPU=2`, `global_batch_size=12`, `seq_len=8192` -> 98,304 tokens/step (98.3M at 1k steps).

> [!EXPERIMENT]
> W&B plots for experiment: [Plots](https://wandb.ai/dudeperf3ct/torchtitan/groups/Smoke%20run%20-%2098M%20tokens/workspace?nw=nwuserdudeperf3ct), [Logs](https://wandb.ai/dudeperf3ct/torchtitan/groups/Smoke%20run%20-%2098M%20tokens/logs), [Summary](https://wandb.ai/dudeperf3ct/torchtitan/groups/Smoke%20run%20-%2098M%20tokens/overview) and [Report](https://wandb.ai/dudeperf3ct/torchtitan/reports/Pretraining-LLM-experiment--VmlldzoxNTU4NTA1NQ) <br>
> Detailed summary: [Smoke test summary W&B](https://github.com/dudeperf3ct/minicode-llm/blob/main/codellm_pretrain/torch_titan/docs/smoke_test_wandb_summary.md)

Here is an example log from running smoke test 

```bash
2026-01-09 19:01:15
[titan] 2026-01-09 19:01:15,186 - root - INFO - CUDA capacity: NVIDIA H100 80GB HBM3 with 79.18GiB memory
2026-01-09 19:01:15
[titan] 2026-01-09 19:01:15,198 - root - INFO - Model transformers_modeling_backend_custom llama32_1b_tok32k size: 1,040,254,976 total parameters
2026-01-09 19:01:15
[titan] 2026-01-09 19:01:15,198 - root - INFO - Compiling the loss function with torch.compile
2026-01-09 19:01:15
[titan] 2026-01-09 19:01:15,214 - root - INFO - Compiling each TransformerBlock with torch.compile
2026-01-09 19:01:15
[titan] 2026-01-09 19:01:15,216 - root - INFO - Applied DDP to the model
2026-01-09 19:01:15
[titan] 2026-01-09 19:01:15,281 - root - INFO - Peak FLOPS used for computing MFU: 9.890e+14
2026-01-09 19:01:15
[titan] 2026-01-09 19:01:15,282 - root - INFO - CUDA memory usage for model: 2.07GiB(2.62%)
...
2026-01-09 19:02:58
[titan] 2026-01-09 19:02:58,105 - root - INFO - step: 100  loss:  9.4939  grad_norm: 18.2500  memory: 72.82GiB(91.97%)  tps: 55,314  tflops: 412.06  mfu: 41.66%
2026-01-09 19:03:41
[titan] 2026-01-09 19:03:41,340 - root - INFO - [GC] Performing periodic GC collection took 0.00 seconds
2026-01-09 19:03:42
[titan] 2026-01-09 19:03:42,734 - root - INFO - step: 150  loss:  8.5498  grad_norm: 70.0000  memory: 72.82GiB(91.97%)  tps: 55,069  tflops: 410.23  mfu: 41.48%
2026-01-09 19:04:26
[titan] 2026-01-09 19:04:26,189 - root - INFO - [GC] Performing periodic GC collection took 0.00 seconds
2026-01-09 19:04:27
[titan] 2026-01-09 19:04:27,595 - root - INFO - step: 200  loss:  8.1258  grad_norm: 208.0000  memory: 72.82GiB(91.97%)  tps: 54,783  tflops: 408.11  mfu: 41.26%
2026-01-09 19:05:11
[titan] 2026-01-09 19:05:11,142 - root - INFO - [GC] Performing periodic GC collection took 0.00 seconds
```

The smoke test provides some insights that can be used before starting a full run

* Utilization: The important metrics when it comes to training are memory and mfu. It shows how efficiently the GPU resources are utilized. I have written a blog on [calculating MFU](https://dudeperf3ct.github.io/posts/llm_batch_size_provider/#compute). Looking at the [tflops comparison table](https://github.com/stas00/ml-engineering/tree/master/compute/accelerator#tflops-comparison-table) compiled by Stas Beckmann in the ML Engineering book, H100 has theoretical peak TFLOPs of 989. But the observed, tflops is around ~ 412 for each step. Dividing theoretical value by observed value, we get the actual utilization MFU which sits around 41% can be considered good without any performance tuning. 
* Training Time: Total time taken per step and entire run of 1k steps. 
* Token Throughput: Each training step process about 55k tokens on H100.
* Loss: The trend we expect to see for a loss curve is for it go downwards
* Errors: Flagging any OOM or data related errors

I used 2 x H100 for smoke testing as I planned to perform full run on same hardware but increasing the count of GPUs to 4. Insights for full training from smoke testing (for 4x H100):
- `tokens = steps * global_batch_size * seq_len`
- With `local_batch_size=6`, `NGPU=4`, `seq_len=8192`: `global_batch_size=24`
- With `steps=50000`: `tokens ~ 24 * 8192 * 50000 ~ 9.83B`
- Using smoke-test throughput (~55k tokens/sec/GPU on 2x H100), estimate step time as `(global_batch_size * seq_len) / (tps_per_gpu * NGPU)` -> ~0.9s/step on 4 GPUs
- That puts 50k steps at ~ 12.5 hours (~$144 at $12.36/hr)
- Compute-optimal for a 1B model is ~20B tokens (Chinchilla), so this run is still undertrained.

## Full Run

Once smoke test is complete we get rough idea about what to expect from full run.

>[!IMPORTANT]
> I performed full testing on 4 x H100 instance (80 GB) on Lambda Labs. This run costed me about $150. The cost of each H100 was $3.09/GPU/hr on January 9 ($12.36/hr) :money_with_wings::money_with_wings: .

The only change for full runs are number of steps to run training for, how often to checkpoint, disable profiling and memory traces and setting `data_parallel_replicate_degree=4`. 

[Here](https://github.com/dudeperf3ct/minicode-llm/blob/main/codellm_pretrain/torch_titan/train_configs/full_llama32_1b_swallowcode_tok32k.toml) is the link to the configuration for running training for ~ 10 billion tokens

This runs training for 50k steps compared to 1k for smoke test.

```toml
# Training config: https://github.com/pytorch/torchtitan/blob/81af8833ddeff9b5f1874dc7e20594aa17da6b86/torchtitan/config/job_config.py#L235
[training]
local_batch_size = 6
seq_len = 8192
max_norm = 1.0  # grad norm clipping
# global_batch_size defaults to local_batch_size * data_parallel_degree.
# If training.global_batch_size is set, TorchTitan derives gradient accumulation steps.
# tokens = steps * global_batch_size * seq_len
steps = 50000
dataset = "swallowcode"
dtype = "bfloat16"
mixed_precision_param = "bfloat16" # only works if when data_parallel_shard_degree > 1 or context_parallel_degree > 1
mixed_precision_reduce = "float32"
```

I was not really sure about the hyperparameters for optimizers and schedulers for full run. I took Codex's advice on summary of what others have recommended for pretraining. Is there a good intuition or default values when it comes to thinking about these parameters?

```toml
# Optimiser config: https://github.com/pytorch/torchtitan/blob/81af8833ddeff9b5f1874dc7e20594aa17da6b86/torchtitan/config/job_config.py#L134
[optimizer]
name = "AdamW"
lr = 3e-4
beta1 = 0.9
beta2 = 0.95
eps = 1e-8
weight_decay = 0.1
implementation = "fused"

# LR scheduler config: https://github.com/pytorch/torchtitan/blob/81af8833ddeff9b5f1874dc7e20594aa17da6b86/torchtitan/config/job_config.py#L169
[lr_scheduler]
warmup_steps = 800
decay_ratio = 1.0
decay_type = "cosine"
min_lr_factor = 0.0
```

The important learning from smoke test was syncing the output folder that contains the checkpoint folder. Every 5k steps, a checkpoint is created across all ranks which is saved to the disk. I used `rclone` command and ran it in infinite loop sleeping every 2 minutes as a background process. It syncs the remote ssh machine with my local folder. This saved a lot of cost I would have had to spend syncing or copying all checkpoints after training (as I had to do for smoke testing).

Similar to smoke testing results, the results for full run be viewed at Weights and Biases dashboard. Summary of training on 10B tokens:
* Completed 50k steps / 9.83B tokens in ~ 12.26 hours (~$152 at $12.36/hr).
* Loss fell from 10.86 -> 2.91; LR warmed up to 3e-4 by step 800 then cosine-decayed to ~0 by the end.
* Step time stabilized around ~0.88s; per-rank throughput ~55.8k tps (global ~223k), MFU ~42% (~416 TFLOPS).
* Data loading was negligible (~0.03% of step time) and memory stayed flat at ~70.2 GiB active / 70.9 GiB reserved (88-89% of 80GB), with 0 OOMs or alloc retries.

> [!EXPERIMENT]
> W&B plots for experiment: [Plots](https://wandb.ai/dudeperf3ct/torchtitan/groups/Full%20run%20-%209.8B%20tokens/workspace), [Logs](https://wandb.ai/dudeperf3ct/torchtitan/groups/Full%20run%20-%209.8B%20tokens/logs), [Summary](https://wandb.ai/dudeperf3ct/torchtitan/groups/Full%20run%20-%209.8B%20tokens/overview) and [Report](https://wandb.ai/dudeperf3ct/torchtitan/reports/Pretraining-LLM-experiment--VmlldzoxNTU4NTA1NQ) <br>
> Detailed summary: [Full run W&B summary](https://github.com/dudeperf3ct/minicode-llm/blob/main/codellm_pretrain/torch_titan/docs/full_training_wandb_summary.md)

## Evaluation

> [!NOTE]
> All the checkpoints are uploaded to Hugging Face repository: https://huggingface.co/dudeperf3ct/codellm_pretrain

The last weights saved at step 50k was used to perform basic evaluation. For evaluation, I tested two modes: 

* Language Modelling: This is standard completion task, given a code as prompt fill until end of text token or max number of token stopping criteria is reached.
* Code infilling task: This uses the FIM tokens to create evaluate how model would perform a code completion task given for example start and end, and ask it to complete what would the middle part of code would look like.

The code repository's [evaluation section](https://github.com/dudeperf3ct/minicode-llm/tree/main/codellm_pretrain/torch_titan#evaluation) provides command on how to run these for different tasks. Below are two example evaluation output using the last saved model at 50k.

{{< collapse summary="**Evaluation run on last checkpoint (50k step) full run**" >}}

Using LM mode to predict next token as completion task

```bash
python eval/eval_generate.py \
  --config ./train_configs/full_llama32_1b_swallowcode_tok32k.toml \
  --checkpoint ./checkpoint/step-50000 \
  --prompt "def count_vowels(s):\n    \"\"\"Count vowels in a string.\"\"\"\n    vowels = set(\"aeiouAEIOU\")\n" \
  --mode lm \
  --max_new_tokens 64 \
  --temperature 0.8 \
  --top_k 50 \
  --stop_at_eos \
  --custom_import custom_spec
[titan] 2026-01-10 22:03:55,522 - root - INFO - Loading tokenizer from tokenizer.json
[titan] 2026-01-10 22:03:55,791 - root - INFO - Applying Llama-like patch for Llama
[titan] 2026-01-10 22:04:08,618 - root - INFO - Loading checkpoint: ./checkpoint/step-50000
/home/dudeperf3ct/projects/mini-codellm/codellm_pretrain/torch_titan/.venv/lib/python3.12/site-packages/torch/distributed/checkpoint/utils.py:483: UserWarning: torch.distributed is disabled, unavailable or uninitialized, assuming the intent is to load in a single process.
  return func(*args, **kwargs)
[titan] 2026-01-10 22:04:10,153 - root - INFO - Checkpoint loaded in 1.53 seconds
[titan] 2026-01-10 22:05:15,528 - root - INFO - [prompt] prompt:
def count_vowels(s):
    """Count vowels in a string."""
    vowels = set("aeiouAEIOU")

[titan] 2026-01-10 22:05:15,529 - root - INFO - [prompt] completion:
#
 def print _ from _ with _ with : Tuple _ name == "__ _ path _ title : ") : List and _ on __
    : Path : List of 2 . append
 def count : List [ str ( B :
     start _ path : List [ str , str = [
     root


 def
[titan] 2026-01-10 22:05:15,529 - root - INFO - [prompt] tokens: prompt=27 completion=64 time=65.37s
```

Greedy decoding with `temperature=0` and `top_k=1`

```bash
python eval/eval_generate.py \
  --config ./train_configs/full_llama32_1b_swallowcode_tok32k.toml \
  --checkpoint ./checkpoint/step-50000 \
  --prompt "def count_vowels(s):\n    \"\"\"Count vowels in a string.\"\"\"\n    vowels = set(\"aeiouAEIOU\")\n" \
  --mode lm \
  --max_new_tokens 64 \
  --temperature 0 \
  --top_k 1 \
  --stop_at_eos \
  --custom_import custom_spec

[titan] 2026-01-10 22:07:58,523 - root - INFO - Loading tokenizer from tokenizer.json
[titan] 2026-01-10 22:07:58,787 - root - INFO - Applying Llama-like patch for Llama
[titan] 2026-01-10 22:08:10,971 - root - INFO - Loading checkpoint: ./checkpoint/step-50000
/home/dudeperf3ct/projects/mini-codellm/codellm_pretrain/torch_titan/.venv/lib/python3.12/site-packages/torch/distributed/checkpoint/utils.py:483: UserWarning: torch.distributed is disabled, unavailable or uninitialized, assuming the intent is to load in a single process.
  return func(*args, **kwargs)
[titan] 2026-01-10 22:08:12,580 - root - INFO - Checkpoint loaded in 1.61 seconds
[titan] 2026-01-10 22:09:13,698 - root - INFO - [prompt] prompt:
def count_vowels(s):
    """Count vowels in a string."""
    vowels = set("aeiouAEIOU")

[titan] 2026-01-10 22:09:13,699 - root - INFO - [prompt] completion:

 def print ( f . get _ to _ to _ to _ to _ to _ to _ to _ to _ to _ to _ to _ to - 8 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
[titan] 2026-01-10 22:09:13,699 - root - INFO - [prompt] tokens: prompt=27 completion=64 time=61.12
```

{{</ collapse >}}

The results don't look. Even with greedy decoding, the model outputs were still noisy and didn’t consistently complete simple Python snippets. The “extra spaces” are a byte‑level BPE artifact (low‑confidence tokens). This likely means the model is still under‑trained for clean code and FIM exposure is relatively sparse (PSM/SPM are a fraction of the data, and short snippets rarely see FIM due to `min_code_length=100`).  

If I run this again, I'll do two quick sanity checks before any expensive run:
- Overfit a tiny dataset (1–5 files) with `fim_rate=1.0` to confirm the pipeline.
- Run greedy LM/FIM eval on a long snippet with a large missing middle.

> [!NOTE]
> [Eval debugging notes document](https://github.com/dudeperf3ct/minicode-llm/blob/main/codellm_pretrain/torch_titan/docs/eval_debugging_notes.md) provides high level overview from various evaluation results, patterns of what's going wrong and likely hypothesis.


## Thoughts on working with torchtitan

Everything was not a walk in the park. Below I describe the errors I encountered and my experience when running torchtitan training

* Data loading timed out during couple of smoke test runs. The solution for this was to disable XNET `export HF_HUB_DISABLE_XET=1`. 
* Silent errors when `local_batch_size` far exceeds the memory limit. The training run silently failed without starting any runs. It does not throw any OOM errors.
* TorchTitan supports only next token prediction as it's objective. All the data loading and datasets assume this is the only task.
* Lack of support for additional PyTorch-native models. It only supports models under [`model`](https://github.com/pytorch/torchtitan/tree/main/torchtitan/models) folder which includes llama 3.1, llama 4, DeepSeek v3 and gpt_oss.
* Experimental support for `transformers` backend. Since there was not Llama 3.2 native PyTorch model, I used one from Hugging Face hub. This meant I had to hack around making sure that this new model uses custom tokenizer and new vocabulary size.
* Configuring or enabling different parallelism is as easy as changing the number in the config file.
* Evaluation: I did not implement any evaluation but it would be good to see the progress of the model as it is being trained.
* Training provides useful metrics such as mfu, memory and loss. It makes it easy to keep an eye on utilization. 
* Support for W&B experiment tracker allows ease of monitoring the logs, loss, and all the metrics exposed as part of training including number of tokens seen.

## Next Steps

What's next? I am not sure. I half expected throwing $150 (lol) would be enough to get at least a basic autocomplete model. I don't have any good ideas (other than throwing more money and running the pretraining for entire 50B tokens further) on how to improve these so may be explore or research on these (continual learning, mid-training). I also wanted to explore SFT and RL training if base model is decent.

* [Nvidia's NeMo 2.0](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemo-2.0/index.html) framework looks interesting. I wonder how easy is it to get started and kick of a smoke test with it compared to torchtitan.
* Gather insights from community on the results of this experiment.
* RL or SFT training

