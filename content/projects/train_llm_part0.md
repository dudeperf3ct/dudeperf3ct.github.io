---
title: "Mini StarCoder2 - Data"
tags: ["llm", "data"]
ShowToc: true
---

One of the projects I am looking forward to training an LLM from scratch. For this project, I am creating a mini LLM version of StarCoder2 model. [StarCoder2](https://arxiv.org/pdf/2402.19173) is a CodeLLM.

To limit the scope here's a brief outline of the project

* Mini LLM, maybe 200M parameter model as a start
* Focusing only on Python as a language

The Chinchilla paper suggests using roughly 20x the number of parameters in tokens for optimal training. For my 200M parameter model, this means ~4B tokens would be ideal. To start, I'll experiment with a smaller slice of 500M tokens.

 I will document my process here as I make progress.

---

The section 2 in the paper outlines the various datasources used. I will start with a similar approach.

## Data Sources

The authors use the code repositories from various sources 
* [Software Heritage](https://docs.softwareheritage.org/devel/api-reference.html). They use 2023-09-06 version of the SH graph dataset as the primary source. 
* Issues, and pull requests from the [GHArchive](https://www.gharchive.org/)
* Kaggle and Jupyter Notebooks
* Scraping the documentation for various packages
* Lots of small high-quality datasets mentioned in Section 2.7 of the paper such as GSM8K, Deepmind Mathematics, Rosetta Code and Code Contest
* Stack Overflow, ArXiv, Wikipedia and OpenWebMath dataset
* Intermediate representation

Taking a look at the [datasets](https://docs.softwareheritage.org/devel/swh-export/graph/dataset.html) on the Software Heritage, I chose to use a particularly smaller dataset [`2019-01-28-popular-3k-python`](https://docs.softwareheritage.org/devel/swh-export/graph/dataset.html#graph-dataset-2019-01-28-popular-3k-python). This dataset contains 3k most popular Python projects from Github, Gitlab, PyPI and Debian.

Is this enough to train from scratch? Probably not. 

{{< collapse summary="**Nerd-sniped on Merkle DAGs**" >}}

While reading the documentation on the Software Archive, I came across Merkle Direct Acyclic Graph (DAG) - data structure used to create a graph of the source code. The entire dataset is a big Merkle DAG. A Merkle tree is a hash where leaf nodes consists of the data and non-leaf nodes are the hash of the child nodes. Hash trees are useful in peer-to-peer networks where data received from peers is verified. 

Software Heritage stores archived code in a Merkle DAG (a content-addressed graph). When it visits a repository over time it adds content-addressed nodes for file contents, directories, and commits. Identical artifacts produce the same cryptographic identifier, so duplicates across versions, forks or different hosting places are stored only once. Here’s the documentation diagram illustrating three visits to the [Hello World](https://forge.softwareheritage.org/source/helloworld/) example repo.

{{< figure align=center src="/images/swh-merkle-dag.svg" attr="Software Heritage [Data Model](https://docs.softwareheritage.org/devel/swh-model/data-model.html)">}}

{{</ collapse >}}

The dataset consists of about 3k files in parquet format. There's no graph dataset and it's the only format dataset is available in.

```bash

$ tree -L 1
.
├── content
├── directory
├── directory_entry_dir
├── directory_entry_file
├── directory_entry_rev
├── origin
├── origin_visit
├── person
├── release
├── revision
├── revision_history
├── skipped_content
├── snapshot
├── snapshot_branch
└── snapshot_branches

16 directories, 0 files

$ find . -type f | wc -l
2851
```
Luckily there's a [schema](https://docs.softwareheritage.org/devel/swh-storage/db-schema.html) showing how all these data are connected. Next steps,

1. Filter the dataset to get only the repositories with `main` or `master` branch. We are not interested in all the branches and code across all these branches.
2. Traverse the directories and sub-directories to get the git SHA stored for each content inside the `content` table.
3. The output will be a file that contains mapping for `repo_url`, `sha1_git` and extra information regarding the code `path` and `name`.

> The code for this logic is implemented by the [`SWHDatasetParser`](https://github.com/dudeperf3ct/minicode-llm/blob/main/codellm_data/dataset/parser.py#L14) class.

To parallelize this processing, [Ray](https://docs.ray.io/en/latest/index.html) is used. A Ray Actor class is created that parallelizes the parsing for batches of data.

Once all the repositories are traversed, next task is to download the content for all the files. Herein lies the problem. To get the content, the request in form of `https://archive.softwareheritage.org/api/1/content/[ (hash_type) :] (hash) /` must be sent to the Software Heritage API. The API is [rate-limited](https://archive.softwareheritage.org/api/#rate-limiting) providing 120 requests per hour for anonymous users and 1200 for the authenticated users.

> The code for this logic is implemented by the [`SWHContentDownloader`](https://github.com/dudeperf3ct/minicode-llm/blob/main/codellm_data/content/downloader.py#L27) class.

The trick here is to create a schedule that sends requests without hammering the API.

**Example run statistics**

Example run by traversing 50 repos (out of 4k repos) gives 25k files. The time taken to run for parsing 50 repos on my local machine with 14-core Intel Core Ultra 7 CPUs is approximately 1 hour. It uses batch size of 10, creating 5 actors. This task is **compute bound**.

After filtering for only Python files and License files, the number is down to 10k. The time taken to download these 10k files with rate-limited SWH API is about 7 hours. It makes requests using bearer token as an authenticated user. This task is **network bound**. 

These 10k files contain 2220518 lines of code. Next task would be preprocessing and filtering the raw dataset further.

## Preprocessing Pipeline

The first task would be to filter out the repos that don't have a permissive license. In addition to license filtering, the data pipeline should also cover

1. PII Redaction
2. Decontamination (Remove any overlap dataset with the [pretrained data](#pretraining-dataset))
3. Malware Removal
4. (Stretch) Removing Near-Duplicates.

> The code: TBA

--- 

## Initial Plan vs. Reality Check

My initial plan was to use Software Heritage's 3k Python dataset for pretraining. However, after performing an initial experiment run to parse and download this data, the numbers revealed a problem:

**Scale Analysis:**

- 3k repos → ~800k Python files (estimated) → ~5B tokens after filtering
- This meets the *minimum* Chinchilla recommendation (4B tokens for 200M params)
- Reality: This dataset is better suited for finetuning than pretraining

**Bottlenecks:**

- Even with authenticated API access (1200 req/hour), downloading the full dataset would take 23+ days of continuous requests
- Development iteration becomes impractically slow

---

## Pretraining dataset

I'm pivoting to use [`tokyotech-llm/swallow-code-v2`](https://huggingface.co/datasets/tokyotech-llm/swallow-code-v2) while repurposing python dataset for finetuning phase. [Swallow code](https://arxiv.org/abs/2505.02881) is a refined version of [The-Stack-v2](https://huggingface.co/datasets/bigcode/the-stack-v2) dataset containing about 50B tokens of high quality Python code. All data is publicly available under the Apache 2.0 License. 

They rewrite the `The-Stack-v2` dataset which is a 900B token dataset by cleaning the dataset and rewriting it using [Qwen3-235B-A22B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-235B-A22B-Instruct-2507). The data curation performed on top of stack-v2 dataset focusing exclusively on Python code consists of 5 steps

1. Auto-Formatting – Standardize code style using [ruff formatter](https://docs.astral.sh/ruff/).
2. Length Filtering – Remove excessively long or truncated samples.
3. LLM Quality Scoring – Rate each snippet for readability and style compliance (0–10 scale) using [SeedCoder](https://arxiv.org/abs/2506.03524) prompt for quality scoring.
4. LLM Rewriting Phase – Use `Qwen3-235B-A22B-Instruct` to rewrite and enhance code for clarity, structure, and algorithmic soundness.
5. Post-Formatting – Apply a final ruff pass to ensure uniform formatting and compliance.

{{< figure align=center src="/images/swallowcode_datapipeline.png" attr="[Swallow code](https://huggingface.co/datasets/tokyotech-llm/swallow-code-v2)">}}

Thanks to the authors[^1] for releasing an open-source version of the dataset.

[^1]: Kazuki Fujii and Yukito Tajima and Sakae Mizuki and Hinari Shimada and Taihei Shiotani and Koshiro Saito and Masanari Ohi and Masaki Kawamura and Taishi Nakamura and Takumi Okamoto and Shigeki Ishida and Kakeru Hattori and Youmi Ma and Hiroya Takamura and Rio Yokota and Naoaki Okazaki
