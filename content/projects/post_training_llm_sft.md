---
title: "Post training - Supervised Finetuning"
tags: ["llm", "sft"]
ShowToc: true
---

In previous posts, I explored how to pretrain a language model. The setup for pretraining used Llama 3.2 1B base model trained on [`tokyotech-llm/swallow-code-v2`](https://huggingface.co/datasets/tokyotech-llm/swallow-code-v2) dataset for Fill-in-Middle (FIM) task. 

The mistake here was using narrow code-only corpus as pretraining dataset. A more practical pipeline would be to begin with a broadly pretrained base model, optionally continue pretraining it on high-quality code, and then apply SFT using verified prompt–response demonstrations. Pretraining task are used to learn language representation across various tasks.

The story so far has been (if pretrained correctly) our base model acts as a reliable completion model. The next transition would be to create a instruction-following model. This is where post-training stage is helpful. Modern post-training consists of multiple stages of supervised finetuning (SFT) and RL training using [favorite policy optimisation (PO) algorithm](https://xcancel.com/agarwl_/status/1981518825007853891). A good analogy I think of different stages is

* Pretraining: Learning language, knowledge and task representations through next-token prediction.
* SFT: Adapting a pretrained model to imitate desired responses, follow instructions and produce task-specific output formats.
* RL or verifiable RL: It takes one step further, optimizing model behaviour using rewards, preferences or verifiable outcomes rather than only imitating reference responses.

{{< figure align=center src="/images/post-training-adventure.png" attr="[The Smol Training Playbook](https://huggingface.co/spaces/HuggingFaceTB/smol-training-playbook#beyond-base-modelspost-training-in-2025)">}}

SFT is usually performed after pretraining stage. There's also a mid-training stage (or continued-pretraining). This stage uses domain-specific dataset to further train the pretrained model before SFT. It is particularly useful when SFT task shares same finetuning domain. SFT models don't have to learn the core domain specific skills from scratch as pretrained model is warmed up. It is also used for training on large context gradually increasing the context length. 

Continued pretraining may be planned in advance when domain adaptation is clearly required, or introduced after early SFT experiments reveal gaps in the base model's underlying knowledge. If SFT experiment identifies any weak areas for particular task, a targeted mid-training is performed. Mid-training is less useful if base model already has the skill.

The SFT stage (aka Instruction Finetuning [IFT]) may begin from a pretrained base model to teach instruction-following behaviour, or from an existing post-trained model to specialize it further for a domain or task. It consists of training on pair of input consisting of prompt and output where output contains a conversation style or reasoning trace responses. An example from [Tulu-3 SFT personas](https://huggingface.co/datasets/allenai/tulu-3-sft-personas-instruction-following) dataset is following

| id | prompt | messages | constraints |
| -------- | ------- | -------- | ------- |
| `personas_IF_m3burbc7lf7ttw3mzthb6xpo` | List some popular JavaScript libraries or frameworks for front-end development that have been widely used in the last five years. Provide a short answer, not exceeding 75 words. | **User:** List some popular JavaScript libraries or frameworks for front-end development that have been widely used in the last five years. Provide a short answer, not exceeding 75 words.<br><br>**Assistant:** Some popular JavaScript libraries and frameworks for front-end development in the last five years include React, Angular, Vue.js, Svelte, and jQuery. These tools have been widely adopted for building dynamic user interfaces and web applications, each offering unique features and benefits to developers. | `length constraints:number of words` |


## Setup

Keeping the same theme of building a specialised Python coding LLM, the next tasks are looking for dataset and model. I found [KodCode](https://huggingface.co/datasets/KodCode/KodCode-V1-SFT-R1) SFT code dataset. KodCode contains approximately 447K synthetic coding question–solution–test triplets spanning multiple domains and difficulty levels. The [technical report](https://arxiv.org/pdf/2503.02951) details how they constructed synthetic dataset using DeepSeek-R1 and GPT-4o.

As base model, I will be using [Qwen3.5-4B-Base](https://huggingface.co/Qwen/Qwen3.5-4B-Base) dense model to perform SFT. Before getting excited which framework to use to train SFT, it's helpful to get an idea on how would we evaluate whether SFT helps the base model. For this, I picked 3 evaluation benchmarks similar to the KodCode technical paper. The paper evaluated their approach on `Qwen2.5-Coder-32B-Instruct` on the dataset.

1. LiveCodeBench v5 (3 modes - Easy, Medium, Hard variants)
2. HumanEval (Original and Plus version)
3. MBPP (Original and Plus variant)

The idea here is to first evaluate the performance of the base model on these benchmarks and compare it to the performance of SFT trained model.

> [!CODE]
> The code for evaluation: https://github.com/dudeperf3ct/minicode-llm/tree/main/codellm_sft/eval/README.md

Following are the various experiments I will look at and what we are going to learn from these experiments.

| Experiments | What we are going to learn |
| -------- | ------- | 
| `Qwen3.5-4B-Base` base model before SFT | Raw pretrained Python coding capability on held-out benchmarks |
| Base -> direct SFT  | Whether verified final implementations teach concise coding-assistant behaviour |
| Base -> reasoning SFT  | Whether explicit reasoning demonstrations improve coding problem-solving|
| `Qwen3.5-4B`, post-trained model | How narrow coding SFT compares with broad professional post-training |

The direct vs reasoning SFT is interesting as it provides insights whether reasoning helps provide any benefits to coding problem-solving tasks.

## Training SFT

Pretraining and SFT use the same underlying next-token prediction objective, but expose the model to very different data distributions. Pretraining teaches model to learn the language syntax and code structure. SFT changes the conditional behaviour of the model by training it on curated instruction–response demonstrations. It can teach instruction-following, response structure, task strategies and domain-specific behaviour.

There are two versions of KodCode SFT dataset: [`KodCode-V1-SFT-R1`](https://huggingface.co/datasets/KodCode/KodCode-V1-SFT-R1) (DeepSeek-R1 reasoning traces) and [`KodCode-V1-SFT-4o`](https://huggingface.co/datasets/KodCode/KodCode-V1-SFT-4o) (solution generated by GPT-4o).

For the SFT experiment, I will train two variants of SFT on [`KodCode-V1-SFT-R1`](https://huggingface.co/datasets/KodCode/KodCode-V1-SFT-R1) dataset:

1. **Direct SFT:** the assistant target contains only `r1_solution`, the final verified implementation.
2. **Reasoning SFT:** the assistant target contains the complete R1 response, including its reasoning trace and final implementation.

There are [many frameworks](https://huggingface.co/spaces/HuggingFaceTB/smol-training-playbook#tools-of-the-trade) that provide ease of finetuning

* Axolotl (I am using this config driven approach)
* Snowflake Arctic Training
* Unsloth
* NeMo
* TRL from HuggingFace

> [!CODE]
> The code for SFT: https://github.com/dudeperf3ct/minicode-llm/tree/main/codellm_sft/sft/README.md

To further branch on different approaches, there are multiple options to perform finetuning.

1. Parameter Efficient Finetuning (PEFT): LoRA and QLoRA like approaches that add small trainable weight/parameters alongside base model weights. The base model weights are frozen and only the small newly added weights are trained. 
2. Full-finetuning (FT): FT updates all language-model and output-head parameters while leaving Qwen3.5’s unused vision encoder frozen.

> [!TIP]
> The Smol Training Playbook: https://huggingface.co/spaces/HuggingFaceTB/smol-training-playbook#beyond-base-modelspost-training-in-2025

The Smol Training Playbook provides one of the best and comprehensive guide for training state-of-the-art language models. The [post training section](https://huggingface.co/spaces/HuggingFaceTB/smol-training-playbook#beyond-base-modelspost-training-in-2025) recommends following best practice for SFT

1. Pick base model : Model size, Architecture (Dense vs MoE), Post training record
2. Pick datasets specific to targeted domain
3. Pick good chat template (or customize your own)
4. Train and vibe-test baselines

The playbook covers excellent ablation studies on [which hyperparameters matter](https://huggingface.co/spaces/HuggingFaceTB/smol-training-playbook#which-hyperparameters-actually-matter) noting the primary of them being learning rate, batch size, packing and epochs. For my case, since we want to go with both LoRA and FT, it introduces few more parameter specific to LoRA-based like `rank` and `target_modules`.

The implementation plan to perform SFT using `Qwen3.5-4B-Base` base model experiment proceeds in the following steps,

1. Prepare dataset: This will use KodCode R1 dataset to create a sample of 10K training, 500 validation and 500 test samples. Further, 1k and 32 subsets are created from 10k training samples. A stratification is performed using `(gpt_difficulty, subset, style)` tuple.
2. Decontaminate dataset: To avoid benchmaxxing, a decontamination is performed using [Open-R1 decontamination](https://github.com/huggingface/open-r1/blob/main/scripts/decontaminate.py) strategy. It remove any code that shares n-gram (8-gram) overlap with benchmark datasets.
3. Incremental baselines: Test the models incrementally first with 32 samples overfitting to check if chat templates and data is parsed as expected, 1k samples as pilot to review syntax and 10k sample experiment as final LoRA once both the previous experiments work fine. The samples can be increased from 10k to 50k-100k on verification.
4. Full finetuning: Once LoRA finetuning is complete, a text-only full-finetuning is performed on the same 10k samples.
5. Evaluation: Both checkpoints from LoRA and FT are used to perform evaluation on the benchmark.


## Eval Results

All results are reported as percentages. HumanEval, HumanEval+, MBPP and MBPP+ use `pass@1`, while LiveCodeBench uses accuracy. These benchmarks check if the generated code passes the tests for each task.

| Experiments | HumanEval | HumanEval+ | MBPP | MBPP+ | LiveCodeBench Easy | LiveCodeBench Medium | LiveCodeBench Hard |
| -------- | ------- | -------- | -------- | -------- | -------- | -------- | -------- |
| `Qwen3.5-4B-Base` base model before SFT | 75.00% | 68.29% | 67.46% | 56.88% | **74.91%** | **35.35%** | **8.15%** |
| `Qwen3.5-4B`, post-trained model non-thinking | **82.32%** | **75.00%** | **75.40%** | **61.64%** | 55.56% | 27.19% | 4.81% |

These baselines show how Qwen's pretrained and post-trained checkpoints perform for the benchmark datasets.

SFT models

| Experiments | HumanEval | HumanEval+ | MBPP | MBPP+ | LiveCodeBench Easy | LiveCodeBench Medium | LiveCodeBench Hard |
| -------- | ------- | -------- | -------- | -------- | -------- | -------- | -------- |
| Base → direct SFT (LoRA) | | | | | | | |
| Base → direct SFT (FT) | | | | | | | |
| Base → reasoning SFT (LoRA) | | | | | | | |
| Base → reasoning SFT (FT) | | | | | | | |

In the next project, we will look into RL post training.
