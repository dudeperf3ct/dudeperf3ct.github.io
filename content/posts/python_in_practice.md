---
author: [""]
title: "Python in practice - Config driven design"
date: 2025-01-11
summary: "Patterns in python code design"
description: ""
tags: ["python", "best-practices"]
ShowToc: false
ShowBreadCrumbs: true
---

Modern ML systems are increasingly declarative: you describe what you want to run, not how to wire it together. The [`ArcticTraining`](https://github.com/snowflakedb/ArcticTraining) library is a clean, well-designed example of this idea in Python. Under the hood, it combines registries, factories, callbacks, and configuration files to create a flexible and extensible training framework. The same declarative pattern is used by other libraries such as [torchtitan](https://github.com/pytorch/torchtitan/), [NeMo](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemo-2.0/index.html) framework for training LLMs.
 
This post breaks down those patterns and explains why they work so well. Jump to wrap-up section on how to identify various patterns. 

# Declarative > Imperative 

Instead of imperative pipelines like:

```python
dataset = load_data(...)
dataset = process(...)
dataset = split(...)
model = build_model(...)
trainer = Trainer(...)
trainer.train()
```

you describe everything in YAML and run:

```bash
arctic_training run-causal.yml
```

What is special here is how cleanly the pattern is implemented in plain Python.

## The Big Picture Architecture

At a high level, ArcticTraining is built around three ideas:

1. Factories for major components (data, model, optimizer, trainer, …)
2. Registries to map string names → concrete implementations
3. Callbacks to inject behavior without subclass explosion

All of this is wired together through configuration files.

Let's unpack the two most interesting pieces: callbacks and registries + factories.

### Callbacks: Extending Behavior Without Overriding Methods

Consider a data pipeline with stages like:
* load
* process
* split
* create_dataloader

Different training recipes often want slightly different behavior at these stages:
* filter long samples
* pack sequences
* pad batches
* slice datasets

Subclassing and overriding each method quickly becomes brittle. ArcticTraining solves this with method-level callbacks. There are two core abstractions:

* `Callback`: represents a single pre- or post-hook
* `CallbackMixin`: manages registration and execution

The [`Callback`](https://github.com/snowflakedb/ArcticTraining/blob/cefd6ee525ce54750edc33d6016a52b0461118e6/arctic_training/callback/callback.py#L26) class explicitly distinguishes **pre** and **post** callbacks and validates function signatures:

```python
class Callback:
    ...
    
    def _validate_pre_fn_sig(self):
        """Validate if pre callback has correct function signature."""
        
    def _validate_post_fn_sig(self):
        """Validate if post callback has correct function signature."""
        
    def _run_pre_callback():
        """Run the pre callback function."""
    
    def _run_post_callback():
        """Run the post callback function."""
```

The callback mixin keeps track of registered callbacks and provides a `callback_wrapper` decorator to wrap methods with pre and post callback invocations.

```python
def callback_wrapper(name: str):
    """A decorator to wrap a method with pre- and post-callbacks."""

    def decorator(method):
        if hasattr(method, WRAPPER_NAME_ATTR):
            return method

        @functools.wraps(method)
        def wrapper(self, *args, **kwargs):
            args, kwargs = self._run_callbacks(f"pre-{name}", args, kwargs)
            return_val = method(self, *args, **kwargs)
            return_val, _ = self._run_callbacks(f"post-{name}", return_val, {})
            return return_val

        setattr(wrapper, WRAPPER_NAME_ATTR, name)
        return wrapper

    return decorator
```

To register a pre-callback or post-callback, the [`CallbackMixin`](https://github.com/snowflakedb/ArcticTraining/blob/cefd6ee525ce54750edc33d6016a52b0461118e6/arctic_training/callback/mixin.py#L55) can be used as one of the base classes. 

**How Derived Classes Use Callbacks**

Here's where it becomes elegant. Example: [`CausalDataFactory`](https://github.com/snowflakedb/ArcticTraining/blob/cefd6ee525ce54750edc33d6016a52b0461118e6/arctic_training/data/causal_factory.py#L150). 


```python
class CausalDataFactory(DataFactory):
    name = "causal"
    config: CausalDataConfig
    callbacks = [
        ("post-load", slice_and_pack_dataset),
    ]
```

Let's unpack what happens when we call `load` on `CausalDataFactory` to get dataset:

* `DataFactory.load` (discussed in next section) is decorated with `@callback_wrapper("load")`
* During initialization, `CallbackMixin` sees "post-load". It registers `slice_and_pack_dataset` as a post-callback
* When load() runs:
  * pre-load → nothing
  * load → returns dataset
  * post-load → modifies dataset

No overrides. No super calls. No fragile inheritance.

Similarly, [`SFTDataFactory`](https://github.com/snowflakedb/ArcticTraining/blob/cefd6ee525ce54750edc33d6016a52b0461118e6/arctic_training/data/sft_factory.py#L357), creates two callbacks, that are useful for filtering and padding dataset.

```python
class SFTDataFactory(DataFactory):
    name = "sft"
    config: SFTDataConfig
    default_source_cls = HFDataSourceInstruct
    callbacks = [
        ("post-load", filter_dataset_length),
        ("post-load", pack_dataset),
    ]
```

**Why callbacks instead of overriding methods?**

Without callbacks, extending behavior often means subclassing and overriding methods like `load()` or `process()`. As the number of variations grows, this quickly leads to deep inheritance trees and duplicated logic.

Callbacks flip this model:
- the base class owns the lifecycle
- extensions declare *where* they hook in
- behavior composes instead of overrides

This keeps control flow centralized while still allowing fine-grained customization.

### Registry and Factory pattern

There is a `RegistryMeta` class that follows the [registry](https://github.com/faif/python-patterns/blob/master/patterns/behavioral/registry.py) pattern to keep track of all subclasses of a base class.

```python
class RegistryMeta(ABCMeta):
    _registry: dict[str, type] = {}
    
    def __new__(mcs: Type["RegistryMeta"], name: str, bases: Tuple, class_dict: Dict) -> Type:
        """Creates a new class, validates it, and registers it."""
        cls: Type = super().__new__(mcs, name, bases, class_dict)
        ...
        # Register subclass if not in the registry already
        mcs._registry[base_type][registry_name] = cls
```

Any subclass that defines a `name` is automatically discoverable. This pattern is reused across:

* data factories
* model factories
* optimizers
* schedulers
* trainers

Each of these components uses the `RegistryMeta` class and callback mixin `CallbackMixin` to create a base factory class.

**Data**

```python
class DataFactory(ABC, CallbackMixin, metaclass=RegistryMeta):
    """Base DataFactory class for loading training and evaluation data."""
    name: str
    config: DataConfig
    default_source_cls: Optional[Type] = None
    
    # implement training and evaluation data loading methods
    @callback_wrapper("load")
    def load(self, data_sources: List["DataSources"]):
        """Loads data from one or more data sources and concatenates into a single dataset."""
        ...
    
    @callback_wrapper("process")
    def process(self, dataset):
        """Process the dataset (tokenization, formatting, etc.)."""
        ...
    
    @callback_wrapper("split")
    def split(self, training_data):
        """Split the data into training and evaluation sets."""
        ...

    @callback_wrapper("create_dataloader")
    def create_dataloader(self, dataset: DatasetType) -> DataLoader:
        """Create a torch DataLoader from the dataset."""
        return DataLoader(
            dataset,
            batch_size=self.micro_batch_size,
            sampler=DistributedSampler(dataset, num_replicas=self.world_size, rank=self.global_rank),
            num_workers=self.config.dl_num_workers,
            persistent_workers=True,
            drop_last=True,
        )
    
    def __call__(self):
        """Main method to load, process, split data and create dataloaders."""
        ...
```

**Model**

Similar to data, models also have a [base factory class](https://github.com/snowflakedb/ArcticTraining/blob/main/arctic_training/model/factory.py).

```python
class ModelFactory(ABC, CallbackMixin, metaclass=RegistryMeta):
    """Base class for model creation."""

    name: str
    config: ModelConfig

    def __call__(self) -> PreTrainedModel:
        config = self.create_config()
        model = self.create_model(model_config=config)
        return model
    
    # Create trainer

    @abstractmethod
    @callback_wrapper("create-config")
    def create_config(self) -> Any:
        """Creates the model config (e.g., huggingface model config)."""
        raise NotImplementedError("create_config method must be implemented")

    @abstractmethod
    @callback_wrapper("create-model")
    def create_model(self, model_config) -> PreTrainedModel:
        """Creates the model."""
        raise NotImplementedError("create_model method must be implemented")
```

At runtime, the configuration file is parsed and used to look up concrete implementations from the registries.
For example, when the config specifies:

```yaml
type: causal
```

the framework uses `RegistryMeta` to resolve "causal" to the registered `CausalDataFactory` class, instantiate it with the parsed config, and then execute its lifecycle via `__call__`.

## Config-Driven Design: Everything Comes Together

All of this abstraction pays off in the configuration file. Here's an [example configuration](https://github.com/snowflakedb/ArcticTraining/blob/main/projects/causal/run-causal.yml) for post training.

```yaml
type: causal
micro_batch_size: 1
exit_iteration: 10
min_iterations: 10

deepspeed:
  zero_optimization:
    stage: 3

optimizer:
  learning_rate: 1e-5

model:
  #type: "liger"
  name_or_path: hf-internal-testing/tiny-random-LlamaForCausalLM
  #name_or_path: TinyLlama/TinyLlama_v1.1
  #name_or_path: meta-llama/Llama-3.1-8B

  attn_implementation: flash_attention_2
  #attn_implementation: sdpa

  dtype: bf16

data:
  sources:
    - type: huggingface_causal
      # the first dataset is tiny but fast to download to try it out
      name_or_path: stas/gutenberg-100:train[:100]
      # this is 14GB-large
      # name_or_path: manu/project_gutenberg:en[:100]
      # split: en
      # sample_count: 100_000

  cache_dir: /tmp/data-cache
  num_proc: 16
  dl_num_workers: 1

  max_length: 2048

logger:
  level: WARNING
#  level: INFO

  output_dir: "logs"
  #file_output_ranks: [0,1]
  print_output_ranks: [0,1,2,3,4,5,6,7]

checkpoint:
  - type: huggingface
    save_every_n_steps: 300
    #save_end_of_training: true
    output_dir: /tmp/ft-model
```

To run this recipe, it's as easy as running a single command `arctic_training run-causal.yml`. It also allows configuring different parameters using a single file.

## Why this pattern works?

What I like most about ArcticTraining’s design is that it:
* Encourages composition over inheritance
* Keeps control flow explicit
* Makes configuration executable
* Scales well as the system grows

## Wrap up

This codebase is a great reminder that good architecture doesn’t require exotic patterns or heavy frameworks. A small set of ideas — registries, factories, callbacks, and configuration — applied consistently can go a long way.

Here are a few practical heuristics for spotting when these patterns might be useful.

---

### Factory smell: conditionals on configuration

If you see code like this:

```python
if config.type == "causal":
    data = CausalDataFactory(...)
elif config.type == "sft":
    data = SFTDataFactory(...)
elif config.type == "rlhf":
    ...
```

this is a factory smell. You're selecting behavior based on runtime configuration. A factory pattern helps turn names into objects and removes conditional logic from your application code.

### Registry smell: manual bookkeeping

If you start maintaining a dictionary like this:

```python
FACTORIES = {
    "causal": CausalDataFactory,
    "sft": SFTDataFactory,
}
```

this is where a registry becomes useful. A registry allows implementations to register themselves automatically, avoiding centralized bookkeeping and making it easier to extend the system without modifying existing code.

### Callback smell: subclassing to tweak behavior

If you find yourself writing subclasses like:


```python
class MyDataFactory(BaseDataFactory):
    def load(self):
        data = super().load()
        data = slice(data)
        data = pack(data)
        return data
```

and another subclass like:

```python
class MyOtherDataFactory(BaseDataFactory):
    def load(self):
        data = super().load()
        data = filter(data)
        return data
```

this is a signal that inheritance is being used to compose behavior. Callbacks solve this by allowing extensions to hook into well-defined lifecycle points without overriding methods. Instead of choosing one subclass, multiple behaviors can be composed declaratively and applied in a predictable order.